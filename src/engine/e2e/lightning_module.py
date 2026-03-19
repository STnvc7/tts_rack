import functools
import os
import socket
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, cast

import lightning as L
import numpy as np
import torch
from dsp_board.processor import Processor
from lightning.pytorch.core.optimizer import LightningOptimizer
from metric_board import Evaluator, MeanMetric, MetricOutput
from metric_board.interface import MSEMetric
from torch.optim.lr_scheduler import _LRScheduler as LRScheduler
import wandb

from interface.data import DataLoaderOutput
from interface.loggable import tensor_to_loggable
from interface.loss import DiscriminatorLoss, E2EModelLoss
from interface.model import Discriminators, E2EModel, E2EModelOutput
from engine._common.tensor import slice_segment_by_id
from utils.io.wav import save_wav
from utils.io.yaml import save_as_yaml
from utils.tensor import fix_length, to_numpy

class E2EEngine(L.LightningModule):
    def __init__(
        self,
        model: E2EModel,
        discriminator: Optional[Discriminators],
        loss: E2EModelLoss,
        discriminator_loss: Optional[DiscriminatorLoss],
        optimizer: functools.partial,
        scheduler: functools.partial,
        dsp_processor: Processor,
        valid_metrics: Evaluator,
        test_metrics: Evaluator,
        exp_path: str,
        controls: Optional[List[Dict[str, Any]]] = None,
        n_valid_log_samples: int = 8,
        n_test_save_samples: int = 20,
        gradient_clip_value: float = 1.0,
    ):
        super().__init__()
        self.model = model
        self.discriminator = discriminator
        self.loss = loss
        self.discriminator_loss = discriminator_loss
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.automatic_optimization = False
        self.gradient_clip_value = gradient_clip_value
        self.controls = controls
        self.dsp_processor = dsp_processor
        self.sample_rate = dsp_processor.sample_rate

        self.valid_metrics = valid_metrics
        self.valid_metrics_features = {}
        self.test_metrics = test_metrics
        self.rtf = MeanMetric()
        self.memory = MeanMetric()

        self.exp_path = exp_path
        self.n_valid_log_samples = n_valid_log_samples
        self.n_test_save_samples = n_test_save_samples

    def configure_optimizers(self):
        optim_gen = self.optimizer(params=self.model.parameters())
        sched_gen = self.scheduler(optimizer=optim_gen)
        if self.discriminator is None:
            return [optim_gen], [sched_gen]
        optim_disc = self.optimizer(params=self.discriminator.parameters())
        sched_disc = self.scheduler(optimizer=optim_disc)
        return [optim_gen, optim_disc], [sched_gen, sched_disc]

    # :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    # TRAINING
    # :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    def training_step(self, batch: Optional[DataLoaderOutput], batch_idx):
        # when dataloader failed
        if batch is None:
            return
            
        target = batch.wav  # (batch, 1, time)
        if batch.segment_id_wav is not None:
            target = slice_segment_by_id(batch.wav, batch.segment_id_wav.unsqueeze(1), dim=-1)
            batch.wav = target
        
        # forward process -----------------------
        model_output = self.model(batch)

        # ---------------------------------------
        if self.discriminator and self.discriminator_loss:
            optims = cast(List[LightningOptimizer], self.optimizers())
            optim_gen = optims[0]
            optim_disc = optims[1]

            # train discriminator -------------------
            optim_disc.zero_grad()
            disc_outputs_disc = self.discriminator(target, model_output, mode="discriminator")
            loss_output_disc = self.discriminator_loss(disc_outputs_disc)
            total_loss_disc = loss_output_disc.total_loss
            self.manual_backward(total_loss_disc)
            self.clip_gradients(optim_disc.optimizer, self.gradient_clip_value, "norm")
            optim_disc.step()

            # for generator training ---------------
            disc_outputs_gen = self.discriminator(target, model_output, mode="generator")
        else:
            optim_gen = cast(LightningOptimizer, self.optimizers())
            total_loss_disc = 0
            disc_outputs_gen = {}

        # train generator -----------------------
        optim_gen.zero_grad()
        loss_output_gen = self.loss(batch, model_output, disc_outputs_gen)
        total_loss_gen = loss_output_gen.total_loss
        self.manual_backward(total_loss_gen)
        self.clip_gradients(optim_gen.optimizer, self.gradient_clip_value, "norm")
        optim_gen.step()

        # logging ------------------------------
        loss_components = loss_output_gen.loss_components if loss_output_gen.loss_components else {}
        log_dict = {
            "train/epoch": self.current_epoch,
            "train/learning_rate": optim_gen.param_groups[0]["lr"],
            "train/discriminator_loss": total_loss_disc,
            "train/generator_loss": total_loss_gen,
            **{f"train/{k}": v for k, v in loss_components.items()},
        }

        wandb.log(log_dict, step=self.global_step)
        self.log_dict(log_dict)
        self.log("train_loss", total_loss_gen, prog_bar=True)
        return

    def on_train_epoch_end(self):
        scheduler = self.lr_schedulers()
        if isinstance(scheduler, list):
            scheduler = cast(List[LRScheduler], scheduler)
            for _s in scheduler:
                _s.step()
        else:
            scheduler = cast(LRScheduler, scheduler)
            scheduler.step()

    # :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    # VALIDATION
    # :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    def validation_step(self, batch: Optional[DataLoaderOutput], batch_idx):
        if batch is None:
            return
        
        if batch.segment_id_wav is not None:
            batch.wav = slice_segment_by_id(batch.wav, batch.segment_id_wav.unsqueeze(1), dim=-1)
        
        # forward process -----------------------
        model_output: E2EModelOutput = self.model(batch)
        
        target = batch.wav.squeeze(1)
        pred = fix_length(model_output.pred.squeeze(1), target.shape[-1], dim=-1)
        
        # compute validation loss -----------
        self.valid_metrics.update(pred, target)

        if model_output.pred_features is not None:
            # initialize metrics at first validation step
            if not self.valid_metrics_features:
                keys = model_output.pred_features.keys()
                self.valid_metrics_features = {key: MSEMetric(dim=1) for key in keys}
            
            # update metrics
            for _key, _pred in model_output.pred_features.items():
                _gt = batch.features[_key]
                _pred = fix_length(_pred, _gt.shape[-1], dim=-1)
                self.valid_metrics_features[_key].update(_pred, _gt)

        # add samples to table --------------
        if (self.n_valid_log_samples is None) or (batch_idx < self.n_valid_log_samples):
            self._add_valid_sample_to_table(batch, model_output)

        return

    def on_validation_epoch_start(self):
        self.valid_audio_table = wandb.Table(columns=["target", "predicted"])
        self.valid_spc_table = wandb.Table(columns=["target", "predicted"])
        self.valid_mel_table = wandb.Table(columns=["target", "predicted"])
        self.valid_pred_table = {}
        self.valid_loggable_table = None

    def on_validation_epoch_end(self):
        results = self.valid_metrics.compute()
        valid_loss = sum([v.mean for v in results.values()])

        loss_dict = {
            "validation/loss": valid_loss,
            **{f"validation/metrics/{m}": r.mean for m, r in results.items()},
            **{f"validation/metrics_feature/{k}": v.compute().mean for k, v in self.valid_metrics_features.items()},
            "validation/vocoder/audio": self.valid_audio_table,
            "validation/vocoder/spectrogram": self.valid_spc_table,
            "validation/vocoder/mel_spectrogram": self.valid_mel_table,
            **{f"validation/acoustic_model/{k}": v for k, v in self.valid_pred_table.items()},
            "validation/loggable_outputs": self.valid_loggable_table,
        }
        wandb.log(loss_dict, step=self.global_step)
        self.log("valid_loss", valid_loss)

        self.valid_metrics.reset()
        for k in self.valid_metrics_features.keys():
            self.valid_metrics_features[k].reset()
            
        return

    # local function for validation steps =============================
    def _add_valid_sample_to_table(self, batch: DataLoaderOutput, model_output: E2EModelOutput):
        target = batch.wav.squeeze()
        pred = model_output.pred.squeeze()

        # add audio -----------------------------
        self.valid_audio_table.add_data(
            wandb.Audio(to_numpy(target, np.float32), self.sample_rate),
            wandb.Audio(to_numpy(pred, np.float32), self.sample_rate),
        )

        # add spectrogram -----------------------
        spc_target = self.dsp_processor.log_spectrogram(target)
        spc_pred = self.dsp_processor.log_spectrogram(pred)
        self.valid_spc_table.add_data(
            tensor_to_loggable(spc_target, "target").to_wandb_media(),
            tensor_to_loggable(spc_pred, "predicted").to_wandb_media(),
        )

        # add mel spectrogram -------------------
        mel_target = self.dsp_processor.mel_spectrogram(target)
        mel_pred = self.dsp_processor.mel_spectrogram(pred)
        self.valid_mel_table.add_data(
            tensor_to_loggable(mel_target, "target").to_wandb_media(),
            tensor_to_loggable(mel_pred, "predicted").to_wandb_media(),
        )

        # add predicted features ----------------
        if model_output.pred_features is not None:
            # initialize table at first loop
            if self.valid_pred_table == {}:
                self.valid_pred_table = {
                    k: wandb.Table(columns=["target", "predicted"])
                    for k in model_output.pred_features.keys()
                }
            
            # add elements
            for _key, _pred in model_output.pred_features.items():
                _gt = batch.features[_key].squeeze()
                self.valid_pred_table[_key].add_data(
                    tensor_to_loggable(_gt, _key).to_wandb_media(),
                    tensor_to_loggable(_pred, _key).to_wandb_media(),
                )

        # add intermediate feature --------------
        if model_output.loggable_outputs is not None:
            loggable_keys = list(model_output.loggable_outputs.keys())
            loggable_values = [
                v.to_wandb_media() for v in model_output.loggable_outputs.values()
            ]
            if self.valid_loggable_table is None:
                self.valid_loggable_table = wandb.Table(columns=loggable_keys)
            self.valid_loggable_table.add_data(*loggable_values)

    # :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    # TEST
    # :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    def on_test_epoch_start(self):
        date_str = datetime.now().strftime("%y%m%d_%H%M")
        hostname = socket.gethostname().split(".")[0]
        self.result_dir = os.path.join(
            self.exp_path, "results", f"{date_str}_{hostname}"
        )

    def test_step(self, batch: Optional[DataLoaderOutput], batch_idx):
        if batch is None:
            return
        
        # forward process -----------------------
        target = batch.wav  # (batch, 1, time)
        if batch.segment_id_wav is not None:
            target = slice_segment_by_id(batch.wav, batch.segment_id_wav.unsqueeze(1), dim=-1)
            batch.wav = target

        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)

        torch.cuda.synchronize()
        start = time.perf_counter()
        model_output = self.model.inference(batch)
        torch.cuda.synchronize()
        end = time.perf_counter()

        if self.device.type == "cuda":
            memory_usage = torch.cuda.max_memory_allocated(self.device) / (1024**2)
        else:
            memory_usage = 0.0

        target = target.squeeze()
        pred = model_output.pred.squeeze()
        
        if target.shape[-1] > pred.shape[-1]:
            pred = fix_length(pred, target.shape[-1])
        else:
            target = fix_length(target, pred.shape[-1])

        if self.test_metrics:
            self.test_metrics.update(pred, target)

        sec = pred.shape[-1] / self.sample_rate
        self.rtf.update(float((end - start) / sec))
        self.memory.update(float(memory_usage / sec))

        # save samples --------------------------
        if self.n_test_save_samples is not None and self.n_test_save_samples <= batch_idx:
            return

        filename = batch.filename[0]
        gt_path = os.path.join(self.result_dir, "wav", "ground_truth", f"{filename}.wav")
        pred_path = os.path.join(self.result_dir, "wav", "pred", f"{filename}.wav")
        os.makedirs(os.path.dirname(gt_path), exist_ok=True)
        os.makedirs(os.path.dirname(pred_path), exist_ok=True)
        save_wav(target, gt_path, sample_rate=self.sample_rate)
        save_wav(pred, pred_path, sample_rate=self.sample_rate)

        # save predicted features -----------------
        features = model_output.pred_features if model_output.pred_features else {}
        for k, v in features.items():
            _gt = batch.features[k]
            self._save_feature(_gt.squeeze(), os.path.join("feature", "ground_truth", k), filename)
            self._save_feature(v.squeeze(), os.path.join("feature", "pred", k), filename)
        
        # predict controlled output ------
        controls = self.controls if self.controls else []
        for c in controls:
            control_output = self.model.inference(batch, control=c)
            label = "-".join([f"{k}_{v}" for k, v in c.items()])
            path = os.path.join(self.result_dir, "wav", "pred_control", label, f"{filename}.wav")
            os.makedirs(os.path.dirname(path), exist_ok=True)
            save_wav(control_output.pred.squeeze(), path, self.sample_rate)
            
            features = control_output.pred_features if control_output.pred_features else {}
            for k, v in features.items():
                self._save_feature(v.squeeze(), os.path.join("feature", "pred_control", k), filename)

        return

    def _save_feature(self, data: torch.Tensor, category: str, filename: str):
        tensor_path = os.path.join(self.result_dir, category, "tensor", f"{filename}.pt")
        img_path = os.path.join(self.result_dir, category, "img", f"{filename}.png")
        os.makedirs(os.path.dirname(tensor_path), exist_ok=True)
        os.makedirs(os.path.dirname(img_path), exist_ok=True)

        torch.save(data, tensor_path)
        tensor_to_loggable(data, filename).save(img_path)

    def on_test_epoch_end(self):
        result = {}
        if self.test_metrics:
            result = self.test_metrics.compute()
        params = 0
        for param in self.model.parameters():
            params += param.numel()
        result["Params"] = MetricOutput(mean=params, std=0, min=params, max=params)
        result["RTF"] = self.rtf.compute()
        result["Memory"] = self.memory.compute()

        save_as_yaml(os.path.join(self.result_dir, "result.yaml"), result)
        wandb.log({f"evaluation/{k}": v.mean for k, v in result.items()})
        return

    # :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    # PREDICTION
    # :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    def predict_step(self, batch: DataLoaderOutput, batch_idx):
        model_output = self.model.inference(batch)
        pred = model_output.pred.squeeze()
        filename = batch.filename[0]
        return pred, filename

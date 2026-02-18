import functools
import os
import socket
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, cast

import lightning as L
import numpy as np
import torch
from torch.optim.lr_scheduler import _LRScheduler as LRScheduler
from dsp_board.processor import Processor
from lightning.pytorch.core.optimizer import LightningOptimizer
from metric_board import Evaluator, MeanMetric, MetricOutput

import wandb
from interface.data import DataLoaderOutput
from interface.loggable import tensor_to_loggable
from interface.loss import DiscriminatorLoss, GeneratorLoss
from interface.model import Discriminators, Generator, GeneratorOutput
from engine._common.tensor import slice_segment_by_id
from utils.io.wav import save_wav
from utils.io.yaml import save_as_yaml
from utils.tensor import fix_length, to_numpy

class VocoderEngine(L.LightningModule):
    def __init__(
        self,
        generator: Generator,
        discriminator: Optional[Discriminators],
        generator_loss: GeneratorLoss,
        discriminator_loss: Optional[DiscriminatorLoss],
        optimizer: functools.partial,
        scheduler: functools.partial,
        dsp_processor: Processor,
        valid_metrics: Evaluator,
        test_metrics: Optional[Evaluator],
        exp_path: str,
        controls: Optional[List[Dict[str, Any]]] = None,
        n_valid_log_samples: int = 8,
        n_test_save_samples: int = 20,
        gradient_clip_value: float = 1.0,
    ):
        super().__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.generator_loss = generator_loss
        self.discriminator_loss = discriminator_loss
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.automatic_optimization = False
        self.gradient_clip_value = gradient_clip_value
        self.controls = controls
        self.dsp_processor = dsp_processor
        self.sample_rate = dsp_processor.sample_rate

        self.valid_metrics = valid_metrics
        self.test_metrics = test_metrics
        self.rtf = MeanMetric()
        self.memory = MeanMetric()

        self.exp_path = exp_path
        self.n_valid_log_samples = n_valid_log_samples
        self.n_test_save_samples = n_test_save_samples

    def configure_optimizers(self):
        optim_gen = self.optimizer(params=self.generator.parameters())
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
        if batch is None:
            return
            
        # slice acoustic features and waveform -----
        if batch.segment_id_feats is not None and batch.segment_id_wav is not None:
            batch.features = {
                k: slice_segment_by_id(v, batch.segment_id_feats.unsqueeze(1).expand(-1,v.shape[1],-1), dim=-1)
                for k, v in batch.features.items()
            }
            batch.wav = slice_segment_by_id(batch.wav, batch.segment_id_wav.unsqueeze(1), dim=-1)
        
        # forward process ---------------------------
        generator_output = self.generator(batch.features, batch.wav)
        
        # train discriminator for GAN taining--------
        if self.discriminator is not None and self.discriminator_loss is not None:
            optims = cast(List[LightningOptimizer], self.optimizers())
            optim_gen = optims[0]
            optim_disc = optims[1]
            
            optim_disc.zero_grad()
            disc_outputs_disc = self.discriminator(batch.wav, generator_output, mode="discriminator")
            loss_output_disc = self.discriminator_loss(disc_outputs_disc)
            total_loss_disc = loss_output_disc.total_loss
            self.manual_backward(total_loss_disc)
            self.clip_gradients(optim_disc.optimizer, self.gradient_clip_value, "norm")
            optim_disc.step()
            
            # for generator training ---------------
            disc_outputs_gen = self.discriminator(batch.wav, generator_output, mode="generator")
        else:
            optim_gen = cast(LightningOptimizer, self.optimizers())
            total_loss_disc = 0
            disc_outputs_gen = {}
        
        # train generator -----------------------
        optim_gen.zero_grad()
        loss_output_gen = self.generator_loss(batch, generator_output, disc_outputs_gen)
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
            **{f"train/{k}": v for k, v in loss_components.items()}
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
        
        # slice acoustic features and waveform -----
        if batch.segment_id_feats is not None and batch.segment_id_wav is not None:
            batch.features = {
                k: slice_segment_by_id(v, batch.segment_id_feats.unsqueeze(1).expand(-1,v.shape[1],-1), dim=-1)
                for k, v in batch.features.items()
            }
            batch.wav = slice_segment_by_id(batch.wav, batch.segment_id_wav.unsqueeze(1), dim=-1)
        
        # forward process ---------------------------
        generator_output = self.generator(batch.features)

        # compute validation loss -----------
        self.valid_metrics.update(generator_output.pred.squeeze(), batch.wav.squeeze())

        # add samples to table --------------
        if (self.n_valid_log_samples is None) or (batch_idx < self.n_valid_log_samples):
            self._add_valid_sample_to_table(batch.wav.squeeze(), generator_output)

        return

    def on_validation_epoch_start(self):
        self.valid_audio_table = wandb.Table(columns=["target", "predicted"])
        self.valid_spectrogram_table = wandb.Table(columns=["target", "predicted"])
        self.valid_mel_spectrogram_table = wandb.Table(columns=["target", "predicted"])
        self.valid_loggable_table = None

    def on_validation_epoch_end(self):
        results = self.valid_metrics.compute()
        valid_loss = sum([v.mean for v in results.values()])
        loss_dict = {
            "validation/loss": valid_loss,
            **{f"validation/metrics/{m}": r.mean for m, r in results.items()},
            "validation/audio": self.valid_audio_table,
            "validation/spectrogram": self.valid_spectrogram_table,
            "validation/mel_spectrogram": self.valid_mel_spectrogram_table,
            "validation/loggable_outputs": self.valid_loggable_table,
        }
        wandb.log(loss_dict, step=self.global_step)
        self.log("valid_loss", valid_loss)
        
        self.valid_metrics.reset()
        return

    # local function for validation steps =============================
    def _add_valid_sample_to_table(self, target: torch.Tensor, gen_output: GeneratorOutput):
        target = target.squeeze()
        pred = gen_output.pred.squeeze()

        # add audio -----------------------------
        self.valid_audio_table.add_data(
            wandb.Audio(to_numpy(target, np.float32), self.sample_rate),
            wandb.Audio(to_numpy(pred, np.float32), self.sample_rate),
        )

        # add spectrogram -----------------------
        spc_target = self.dsp_processor.log_spectrogram(target)
        spc_pred = self.dsp_processor.log_spectrogram(pred)
        self.valid_spectrogram_table.add_data(
            tensor_to_loggable(spc_target, "target").to_wandb_media(),
            tensor_to_loggable(spc_pred, "predicted").to_wandb_media(),
        )

        # add mel spectrogram -------------------
        mel_target = self.dsp_processor.mel_spectrogram(target)
        mel_pred = self.dsp_processor.mel_spectrogram(pred)
        self.valid_mel_spectrogram_table.add_data(
            tensor_to_loggable(mel_target, "target").to_wandb_media(),
            tensor_to_loggable(mel_pred, "predicted").to_wandb_media(),
        )

        # add intermediate feature --------------
        if gen_output.loggable_outputs is None:
            return

        loggable_keys = list(gen_output.loggable_outputs.keys())
        loggable_values = [v.to_wandb_media() for v in gen_output.loggable_outputs.values()]

        if self.valid_loggable_table is None:
            self.valid_loggable_table = wandb.Table(columns=loggable_keys)
        self.valid_loggable_table.add_data(*loggable_values)

    # :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    # TEST
    # :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    def on_test_epoch_start(self):
        date_str = datetime.now().strftime("%y%m%d_%H%M")
        hostname = socket.gethostname().split(".")[0]
        self.result_dir = os.path.join(self.exp_path, "results", f"{date_str}_{hostname}")

    def test_step(self, batch: Optional[DataLoaderOutput], batch_idx):
        if batch is None:
            return
        
        if batch.segment_id_feats is not None and batch.segment_id_wav is not None:
            batch.features = {
                k: slice_segment_by_id(v, batch.segment_id_feats.unsqueeze(1).expand(-1,v.shape[1],-1), dim=-1)
                for k, v in batch.features.items()
            }
            batch.wav = slice_segment_by_id(batch.wav, batch.segment_id_wav.unsqueeze(1), dim=-1)

        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        
        torch.cuda.synchronize()
        start = time.perf_counter()
        generator_output = self.generator.inference(batch.features)
        torch.cuda.synchronize()
        end = time.perf_counter()

        if self.device.type == "cuda":
            memory_usage = torch.cuda.max_memory_allocated(self.device) / (1024**2)
        else:
            memory_usage = 0.0

        target = batch.wav.squeeze()
        pred = generator_output.pred.squeeze()
        pred = fix_length(pred, target.shape[-1])

        if self.test_metrics:
            self.test_metrics.update(pred, target)

        sec = pred.shape[-1] / self.sample_rate
        self.rtf.update(float((end - start) / sec))
        self.memory.update(float(memory_usage / sec))

        # save samples --------------------------
        if self.n_test_save_samples is not None and self.n_test_save_samples <= batch_idx:
            return

        gt_path = os.path.join(self.result_dir, "ground_truth", f"{batch.filename[0]}.wav")
        pred_path = os.path.join(self.result_dir, "pred", f"{batch.filename[0]}.wav")
        os.makedirs(os.path.dirname(gt_path), exist_ok=True)
        os.makedirs(os.path.dirname(pred_path), exist_ok=True)
        save_wav(target, gt_path, sample_rate=self.sample_rate)
        save_wav(pred, pred_path, sample_rate=self.sample_rate)

        # predict controlled features and save them ------
        if self.controls is None:
            return
        for c in self.controls:
            control_output = self.generator.inference(batch.features, control=c)
            label = "-".join([f"{k}_{v}" for k, v in c.items()])
            path = os.path.join(self.result_dir, "pred_control", label, f"{batch.filename[0]}.wav")
            os.makedirs(os.path.dirname(path), exist_ok=True)
            save_wav(control_output.pred.squeeze(), path, self.sample_rate)

        return

    def on_test_epoch_end(self):
        result = {}
        if self.test_metrics is not None:
            result = self.test_metrics.compute()
        params = 0
        for param in self.generator.parameters():
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
        input_feature = batch.features
        generator_output: GeneratorOutput = self.generator.inference(input_feature)

        pred = generator_output.pred.squeeze()
        filename = batch.filename[0]
        return pred, filename

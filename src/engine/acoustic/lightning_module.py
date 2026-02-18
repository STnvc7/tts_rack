import functools
import os
import socket
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import lightning as L
from metric_board.interface import MSEMetric
import numpy as np
import torch
from dsp_board import Processor

import wandb
from interface.data import DataLoaderOutput
from interface.loggable import tensor_to_loggable
from interface.loss import AcousticModelLoss, LossOutput
from interface.model import AcousticModel, AcousticModelOutput
from utils.io.wav import save_wav
from utils.tensor import to_numpy

class AcousticModelEngine(L.LightningModule):
    def __init__(
        self,
        model: AcousticModel,
        loss: AcousticModelLoss,
        optimizer: functools.partial,
        scheduler: functools.partial,
        dsp_processor: Processor,
        exp_path: str,
        controls: Optional[List[Dict[str, Any]]] = None,
        n_valid_log_samples: int = 8,
        n_test_save_samples: int = 20,
    ):
        super().__init__()
        self.model = model
        self.loss = loss
        self.optimizer = optimizer(self.model.parameters())
        self.scheduler = scheduler(self.optimizer)
        self.dsp_processor = dsp_processor
        self.sample_rate = dsp_processor.sample_rate
        self.exp_path = exp_path
        self.controls = controls
        self.valid_metrics = {}
        self.n_valid_log_samples = n_valid_log_samples
        self.n_test_save_samples = n_test_save_samples

    def configure_optimizers(self):
        return [self.optimizer], [self.scheduler]

    # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    def training_step(self, batch: Optional[DataLoaderOutput], batch_idx):
        if batch is None:
            return
            
        model_output: AcousticModelOutput = self.model(batch)
        loss_output: LossOutput = self.loss(batch, model_output)
        loss_components = loss_output.loss_components if loss_output.loss_components else {}
        log_dict = {
            "train/epoch": self.current_epoch,
            "train/learning_rate": self.optimizer.param_groups[0]["lr"],
            "train/total_loss": loss_output.total_loss,
            **{f"train/{k}": v for k, v in loss_components.items()}
        }
        
        wandb.log(log_dict, step=self.global_step)
        self.log_dict(log_dict)

        return loss_output.total_loss

    # validation
    # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    def on_validation_epoch_start(self):
        self.valid_loss_outputs = []
        self.audio_table = wandb.Table(columns=["target"])
        self.feature_table = {}
        self.loggable_table = None

    # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    def validation_step(self, batch: Optional[DataLoaderOutput], batch_idx):
        if batch is None:
            return
        
        model_output: AcousticModelOutput = self.model(batch)
        loss_output = self.loss(batch, model_output)
        self.valid_loss_outputs += [loss_output]
        
        if not self.valid_metrics:
            keys = model_output.pred_features.keys()
            self.valid_metrics = {key: MSEMetric(dim=1) for key in keys}
        for key, pred in model_output.pred_features.items():
            self.valid_metrics[key].update(pred, batch.features[key])
            
        if batch_idx < 8:
            self._add_to_wandb_table(batch, model_output)

        return
    
    # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    def on_validation_epoch_end(self):
        total_loss, loss_components = self._aggregate_valid_loss_outputs()
        log_dict = {
            "validation/loss": total_loss,
            **{f"validation/{k}": v for k, v in loss_components.items()},
            **{f"validation/metrics/{k}": v.compute().mean for k, v in self.valid_metrics.items()},
            "audio": self.audio_table,
            **{f"validation/{k}": v for k, v in self.feature_table.items()},
            "validation/outputs": self.loggable_table,
        }
        wandb.log(log_dict, step=self.global_step)
        self.log("valid_loss", total_loss)
        
        for k in self.valid_metrics.keys():
            self.valid_metrics[k].reset()

        return

    # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    def _add_to_wandb_table(self, batch: DataLoaderOutput, model_output: AcousticModelOutput):
        self.audio_table.add_data(wandb.Audio(to_numpy(batch.wav.squeeze(), np.float32), self.sample_rate))
        
        # add columns to table ------------------------------
        if self.feature_table == {}:
            self.feature_table = {
                _k: wandb.Table(columns=["target", "predicted"])
                for _k in model_output.pred_features.keys()
            }

        # add predicted features --------------------
        for _key, _predicted in model_output.pred_features.items():
            _target = batch.features.get(_key, None)
            if _target is None:
                _target = getattr(self.dsp_processor, _key)(batch.wav).squeeze()
            self.feature_table[_key].add_data(
                tensor_to_loggable(_target, _key).to_wandb_media(),
                tensor_to_loggable(_predicted, _key).to_wandb_media(),
            )
            
        if model_output.loggable_outputs is None:
            return
        if self.loggable_table is None:
            self.loggable_table = wandb.Table(columns=list(model_output.loggable_outputs.keys()))
        self.loggable_table.add_data(*[v.to_wandb_media() for v in model_output.loggable_outputs.values()])

    # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    def _aggregate_valid_loss_outputs(self) -> Tuple[float, Dict[str, float]]:
        valid_total_losses = [l.total_loss for l in self.valid_loss_outputs]
        valid_total_loss = torch.stack(valid_total_losses).mean().item()

        component_keys = []
        if self.valid_loss_outputs[0].loss_components is not None:
            component_keys = list(self.valid_loss_outputs[0].loss_components.keys())

        valid_loss_components = {
            k: torch.stack([
                l.loss_components[k] for l in self.valid_loss_outputs
                if l.loss_components is not None
            ]).mean().item()
            for k in component_keys
        }
        return valid_total_loss, valid_loss_components

    # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    def on_test_epoch_start(self):
        date_str = datetime.now().strftime("%y%m%d_%H%M")
        hostname = socket.gethostname().split(".")[0]
        self.result_dir = os.path.join(self.exp_path, "results", f"{date_str}_{hostname}")

    def test_step(self, batch: Optional[DataLoaderOutput], batch_idx):
        if batch is None:
            return
        
        filename = batch.filename[0]
        model_output: AcousticModelOutput = self.model.inference(batch)
        
        if self.n_test_save_samples is not None and self.n_test_save_samples <= batch_idx:
            return

        # save predicted features -----------------
        for k, v in model_output.pred_features.items():
            _gt = batch.features[k]
            self._save_feature(_gt, category=os.path.join("ground_truth", k), filename=filename)
            self._save_feature(v, category=os.path.join("pred", k), filename=filename)

        wav_path = os.path.join(self.result_dir, "wav", "ground_truth", f"{filename}.wav")
        os.makedirs(os.path.dirname(wav_path), exist_ok=True)
        save_wav(batch.wav, wav_path, self.dsp_processor.sample_rate)

        if self.controls is None:
            return

        # predict controlled features and save them ------
        for c in self.controls:
            label = "-".join([f"{k}_{v}" for k, v in c.items()])
            control_output: AcousticModelOutput = self.model.inference(batch, control=c)
            for k, v in control_output.pred_features.items():
                self._save_feature(v, category=os.path.join("pred_control", label, k), filename=filename)

    def _save_feature(self, data: torch.Tensor, category: str, filename: str):
        tensor_path = os.path.join(self.result_dir, category, "tensor", f"{filename}.pt")
        img_path = os.path.join(self.result_dir, category, "img", f"{filename}.png")
        os.makedirs(os.path.dirname(tensor_path), exist_ok=True)
        os.makedirs(os.path.dirname(img_path), exist_ok=True)

        torch.save(data, tensor_path)
        tensor_to_loggable(data, filename).save(img_path)

from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from interface.data import DataLoaderOutput
from interface.feature import AcousticFeature
from interface.model import E2EModel, E2EModelOutput
from interface.model import AcousticModel, AcousticModelOutput, Generator, GeneratorOutput
from engine._common.tensor import slice_segment_by_id

def load_model_from_checkpoint(
    model: nn.Module, 
    checkpoint_path: str, 
    prefix: str,
    device: Optional[torch.device] = None
):
    checkpoint = torch.load(checkpoint_path, weights_only=False, map_location=device)
    state_dict = checkpoint['state_dict']
    
    model_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith(f"{prefix}."):
            # 'model.layer1.weight' -> 'layer1.weight'
            new_key = k.replace(f"{prefix}.", "", 1)
            model_state_dict[new_key] = v
    model.load_state_dict(model_state_dict)
    return model

class JointModel(E2EModel):
    """
    teacher_forcing_until: whether to use ground truth feature for vocoder training
        -1: teacher forcing until the end of training
        None: never use ground truth feature during training
        int: use ground truth feature for the first n steps during training
    """
    def __init__(
        self,
        acoustic_model: AcousticModel,
        generator: Generator,
        teacher_forcing_until: Union[int, Dict[AcousticFeature, int]] = 0,
        freeze_acoustic_model: bool = False,
        detach_between: bool = True,
        pretrained_acoustic_model_path: Optional[str] = None,
        pretrained_generator_path: Optional[str] = None,
    ):
        super().__init__()
        self.acoustic_model = acoustic_model
        self.generator = generator
        self.teacher_forcing_until = teacher_forcing_until
        self.register_buffer("counter", torch.tensor(0))
        if freeze_acoustic_model:
            for param in self.acoustic_model.parameters():
                param.requires_grad = False
        self.detach_between = detach_between

        device = next(self.acoustic_model.parameters()).device
        if pretrained_acoustic_model_path is not None:
            load_model_from_checkpoint(
                self.acoustic_model,
                pretrained_acoustic_model_path,
                prefix="model",
                device=device,
            )
        if pretrained_generator_path is not None:
            load_model_from_checkpoint(
                self.generator,
                pretrained_generator_path,
                prefix="generator",
                device=device,
            )

    def _build_features(
        self,
        acoustic_model_pred: Dict[AcousticFeature, torch.Tensor],
        ground_truth: Dict[AcousticFeature, torch.Tensor],
        segment_id: Optional[torch.Tensor]
    ) -> Dict[AcousticFeature, torch.Tensor]:
        
        feature_keys = acoustic_model_pred.keys()
        features = {}
        if isinstance(self.teacher_forcing_until, int):
            threshold = {k: self.teacher_forcing_until for k in feature_keys}
        else:
            threshold = {k: self.teacher_forcing_until.get(k, None) for k in feature_keys}
        
        for k in feature_keys:
            _t = threshold[k]
            if (_t is not None) and ((_t == -1) or (self.counter < _t)):
                _f = ground_truth[k]
                if segment_id is not None:
                    _f = slice_segment_by_id(_f, segment_id.unsqueeze(1).expand(-1, _f.shape[1], -1), dim=-1)
                features[k] = _f
            else:
                if self.detach_between:
                    features[k] = acoustic_model_pred[k].detach()
                else:
                    features[k] = acoustic_model_pred[k]
        return features

    def forward(self, batch: DataLoaderOutput) -> E2EModelOutput:
        acoustic_output = self.acoustic_model(batch)
        features = self._build_features(acoustic_output.pred_features, batch.features, batch.segment_id_feats)
        generator_output = self.generator(features, batch.wav)

        # -----------------------------------------
        if self.training:
            self.counter += 1

        # aggregate outputs -----------------------
        outputs = {}
        loggable_outputs = {}
        if acoustic_output.outputs is not None:
            outputs.update(acoustic_output.outputs)
        if acoustic_output.loggable_outputs is not None:
            loggable_outputs.update(acoustic_output.loggable_outputs)
        if generator_output.outputs is not None:
            outputs.update(generator_output.outputs)
        if generator_output.loggable_outputs is not None:
            loggable_outputs.update(generator_output.loggable_outputs)

        return E2EModelOutput(
            pred=generator_output.pred,
            pred_features=acoustic_output.pred_features,
            outputs=outputs if outputs else None,
            loggable_outputs=loggable_outputs if loggable_outputs else None,
        )

    def inference(self, batch: DataLoaderOutput, control: Optional[Any] = None) -> E2EModelOutput:
        acoustic_output: AcousticModelOutput = self.acoustic_model.inference(batch, control)
        features = acoustic_output.pred_features
        generator_output: GeneratorOutput = self.generator.inference(features, control)

        outputs = {}
        loggable_outputs = {}
        if acoustic_output.outputs is not None:
            outputs.update(acoustic_output.outputs)
        if acoustic_output.loggable_outputs is not None:
            loggable_outputs.update(acoustic_output.loggable_outputs)
        if generator_output.outputs is not None:
            outputs.update(generator_output.outputs)
        if generator_output.loggable_outputs is not None:
            loggable_outputs.update(generator_output.loggable_outputs)

        return E2EModelOutput(
            pred=generator_output.pred,
            pred_features=acoustic_output.pred_features,
            outputs=outputs if outputs else None,
            loggable_outputs=loggable_outputs if loggable_outputs else None,
        )

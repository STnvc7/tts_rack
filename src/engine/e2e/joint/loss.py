from typing import Dict
from interface.model import E2EModelOutput, DiscriminatorOutput, AcousticModelOutput, GeneratorOutput
from interface.loss import LossOutput, AcousticModelLoss, GeneratorLoss, E2EModelLoss
from interface.data import DataLoaderOutput

class JointLoss(E2EModelLoss):
    def __init__(
        self, 
        acoustic_loss: AcousticModelLoss, 
        generator_loss: GeneratorLoss
    ):
        super().__init__()
        self.acoustic_loss = acoustic_loss
        self.generator_loss = generator_loss

    def forward(
        self, 
        batch: DataLoaderOutput, 
        e2e_output: E2EModelOutput,
        discriminator_outputs: Dict[str, DiscriminatorOutput]
    ) -> LossOutput:
        
        pred_features = e2e_output.pred_features
        assert pred_features is not None, "pred_features should not be None"
        acoustic_outputs = AcousticModelOutput(
            pred_features=pred_features,
            outputs=e2e_output.outputs,
        )
        acoustic_loss: LossOutput = self.acoustic_loss(batch, acoustic_outputs)
        
        generator_outputs = GeneratorOutput(
            pred=e2e_output.pred,
            outputs=e2e_output.outputs
        )
        generator_loss = self.generator_loss(batch, generator_outputs, discriminator_outputs)
        
        total_loss = acoustic_loss.total_loss + generator_loss.total_loss
        acoustic_losses = acoustic_loss.loss_components if acoustic_loss.loss_components is not None else {}
        generator_losses = generator_loss.loss_components if generator_loss.loss_components is not None else {}
        
        return LossOutput(
            total_loss=total_loss,
            loss_components={
                **{f"acoustic_{k}": v for k, v in acoustic_losses.items()},
                **{f"generator_{k}": v for k, v in generator_losses.items()}
            }
        )
import hydra
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

import wandb
from utils.environment import seed_everything, register_resolvers
register_resolvers()

@hydra.main(version_base=None, config_path="../conf", config_name="config")
def train(cfg: DictConfig):
    seed_everything(cfg.general.seed)
    torch.set_float32_matmul_precision("medium")

    wandb.init(
        mode="disabled" if cfg.general.analysis is False else "online",
        project=cfg.general.project_name,
        group=cfg.general.group_name,
        name=cfg.general.exp_name,
        config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
        reinit=True,
    )
    
    print("start training!")
    print("--------------------------------")
    print(f"experiment:     {cfg.general.exp_name}")
    print(f"corpus:         {cfg.corpus.name}")
    print(f"engine:         {cfg.engine.name}")
    print(f"text processor: {cfg.text.name}")
    print(f"dsp processor:  {cfg.dsp.name}")
    print("--------------------------------")
    
    lightning_module = instantiate(cfg.engine.lightning_module)
    data_module = instantiate(cfg.engine.data_module)
    trainer = instantiate(cfg.trainer)
    trainer.logger.log_hyperparams(params=cfg)
    trainer.fit(model=lightning_module, datamodule=data_module, ckpt_path=cfg.general.restore_checkpoint)
    trainer.test(model=lightning_module, datamodule=data_module)


if __name__ == "__main__":
    train()

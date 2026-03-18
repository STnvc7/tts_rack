import hydra
import torch
from hydra import initialize, compose
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
import questionary
from argparse import ArgumentParser
import wandb
from utils.environment import seed_everything, register_resolvers
register_resolvers()

# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
def evaluate(args):
    exp_dir = Path(args.exp)
    cfg_path = exp_dir / "hparams.yaml"
    cfg = OmegaConf.load(cfg_path)
    checkpoint_dir = exp_dir / "checkpoints"
    checkpoints = list(checkpoint_dir.glob("**/*.ckpt"))
    checkpoint_path = questionary.select(
        "Select a checkpoint",
        choices=[str(path) for path in checkpoints]
    ).ask()
    
    wandb.init(mode="disabled")
    seed_everything(cfg.general.seed)
    torch.set_float32_matmul_precision('medium')

    if args.metrics is not None:
        with initialize(version_base=None, config_path="../conf"):
            _cfg = compose(config_name=args.config, overrides=[f"metrics={args.metrics}"])
            cfg.engine.lightning_module.test_metrics = _cfg.metrics
    
    cfg.engine.lightning_module.n_test_save_samples = args.n_save_samples
    lightning_module = instantiate(cfg.engine.lightning_module)
    ckpt = torch.load(checkpoint_path, map_location=torch.device(args.device), weights_only=False)
    lightning_module.load_state_dict(ckpt['state_dict'], strict=False)
    data_module = instantiate(cfg.engine.data_module)
    trainer = instantiate(cfg.trainer, devices=1, accelerator=args.device, precision=32, logger=False)
    trainer.test(model=lightning_module, datamodule=data_module)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--exp", required=True, help="Path to the experiment directory (must contain hparams.yaml and checkpoints/).")
    parser.add_argument("--metrics", type=str, default=None, help="Name of the metrics set under conf/metrics/.")
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda"], help="Device to use for evaluation (cpu or gpu).")
    parser.add_argument("--n-save-samples", default=20)
    args = parser.parse_args()

    evaluate(args)
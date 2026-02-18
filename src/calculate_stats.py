import os
import hydra
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from tqdm import tqdm

from interface.preprocess import StatsCalculator
from utils.io.yaml import save_as_yaml
from utils.environment import seed_everything, register_resolvers
register_resolvers()

@hydra.main(version_base=None, config_path="../conf", config_name="config")
def calculate(cfg: DictConfig):
    seed_everything(cfg.general.seed)
    
    data_module = instantiate(cfg.engine.data_module)
    data_module.batch_size = 1
    data_module.segment_size["train"] = None
    data_module.setup()
    train_loader = data_module.train_dataloader()
    train_loader.shuffle = False
    
    dsp = instantiate(cfg.dsp.config)
    silence = torch.zeros(dsp.sample_rate*1)
    avoid_value_map = {k: getattr(dsp, k)(silence).min().item() for k in data_module.features_to_extract}
    calculators = {k: StatsCalculator(avoid_value=avoid_value_map[k]) for k in data_module.features_to_extract}
    
    for batch in tqdm(train_loader):
        if batch is None:
            continue
        features = batch.features
        for k, v in features.items():
            calculators[k].update(v)
            
    stats = {k: v.compute() for k, v in calculators.items()}
    os.makedirs(os.path.dirname(cfg.corpus.config.feature_stats_path), exist_ok=True)
    save_as_yaml(cfg.corpus.config.feature_stats_path, stats) 
    print("Done.")
    
if __name__ == "__main__":
    calculate()

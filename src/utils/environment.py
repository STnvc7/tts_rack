import random
import os
import numpy as np
import lightning as L
from omegaconf import OmegaConf

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    L.pytorch.seed_everything(seed, workers=True)
    return
    
def register_resolvers():
    from interface.corpus import add_feat_stats_to_config
    OmegaConf.register_new_resolver("create_feature_stats", add_feat_stats_to_config)
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

from utils.environment import seed_everything, register_resolvers
register_resolvers()

@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    seed_everything(cfg.general.seed)
    corpus = instantiate(cfg.corpus.config)
    utt_list_maker = instantiate(cfg.corpus.utterance_list_maker)
    utt_list_maker.make()
    print("Done.")
    
    
if __name__ == "__main__":
    main()
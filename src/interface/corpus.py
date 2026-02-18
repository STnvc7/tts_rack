from typing import Dict, Optional
from dataclasses import dataclass
from omegaconf import DictConfig
from interface.text import AvailableLanguage
from utils.io.yaml import load_from_yaml
@dataclass
class FeatureStats:
    mean: float
    std: float
    max: float
    min: float
    
@dataclass
class Utterance:
    wav_path: str
    filename: str
    speaker_id: int
    text: str
    
@dataclass
class UtteranceListPath:
    train: str
    valid: str
    test: str
    
@dataclass
class Corpus:
    root: str
    n_speakers: int
    n_phonemes: int
    n_prosodies: Optional[int]
    lang: AvailableLanguage
    utterance_list_path: UtteranceListPath
    feature_stats_path: str
    feature_stats: Dict[str, FeatureStats]
    
# 特徴量の統計量を設定ファイルに結合するためのリゾルバ
def add_feat_stats_to_config(path: str):
    try:
        feat_stats = load_from_yaml(path, Dict[str, FeatureStats])
    except FileNotFoundError:
        feat_stats = {}
    
    return DictConfig(feat_stats)
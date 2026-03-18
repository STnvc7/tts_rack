from typing import Dict, List, Literal, Optional
import lightning as L
from torch.utils.data import DataLoader

from interface.corpus import Corpus
from interface.feature import AcousticFeature
from interface.text import TextProcessor
from dsp import DSPProcessor
from .collate_fn import TTSCollateFn
from .dataset import TTSDataset

class TTSDataModule(L.LightningDataModule):
    def __init__(
        self,
        corpus: Corpus,
        dsp_processor: DSPProcessor,
        text_processor: TextProcessor,
        features_to_extract: List[AcousticFeature],
        require_duration: bool,
        segment_size: Dict[Literal["train", "valid", "test"], Optional[int]],
        trim: Optional[Literal["forward", "backward", "both"]],
        normalize: Optional[Literal["peak", "loudness"]],
        batch_size: int,
        num_workers: int,
        pin_memory: bool,
    ):
        super().__init__()
        assert corpus.lang == text_processor.lang, f"Corpus language {corpus.lang} does not match text processor language {text_processor.lang}"
        self.corpus = corpus
        self.dsp_processor = dsp_processor
        self.text_processor = text_processor
        self.features_to_extract = features_to_extract
        self.require_duration = require_duration
        self.segment_size = segment_size
        self.trim: Optional[Literal["forward", "backward", "both"]] = trim
        self.normalize: Optional[Literal["peak", "loudness"]] = normalize

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_dataset = TTSDataset(
                corpus_root=self.corpus.root,
                utterance_list_path=self.corpus.utterance_list_path.train,
                dsp_processor=self.dsp_processor,
                text_processor=self.text_processor,
                features_to_extract=self.features_to_extract,
                require_duration=self.require_duration,
                segment_size=self.segment_size["train"],
                random_onset=True,
                trim=self.trim,
                normalize=self.normalize
            )
            self.valid_dataset = TTSDataset(
                corpus_root=self.corpus.root,
                utterance_list_path=self.corpus.utterance_list_path.valid,
                dsp_processor=self.dsp_processor,
                text_processor=self.text_processor,
                features_to_extract=self.features_to_extract,
                require_duration=self.require_duration,
                segment_size=self.segment_size["valid"],
                random_onset=False,
                trim=self.trim,
                normalize=self.normalize
            )

        if stage == "test" or stage is None:
            self.test_dataset = TTSDataset(
                corpus_root=self.corpus.root,
                utterance_list_path=self.corpus.utterance_list_path.test,
                dsp_processor=self.dsp_processor,
                text_processor=self.text_processor,
                features_to_extract=self.features_to_extract,
                require_duration=self.require_duration,
                segment_size=self.segment_size["test"],
                random_onset=False,
                trim=self.trim,
                normalize=self.normalize
            )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=TTSCollateFn(),
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.valid_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=TTSCollateFn(),
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=TTSCollateFn(),
        )

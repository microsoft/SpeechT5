# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import logging
import os
import sys
from typing import Dict, List, Optional, Tuple
from argparse import Namespace

import numpy as np

from dataclasses import dataclass, field
from fairseq.data import Dictionary, HubertDataset, encoders
from fairseq.dataclass.configs import FairseqDataclass
from fairseq.tasks import register_task
from fairseq.tasks.fairseq_task import FairseqTask
from fairseq.dataclass.constants import ChoiceEnum
from omegaconf import MISSING

logger = logging.getLogger(__name__)

TOKENIZER_CHOICES = ChoiceEnum(["sentencepiece", "hubert_letters", "none"])

def _lang_token(lang: str):
    return "<lang:{}>".format(lang)

def _lang_token_index(dic: Dictionary, lang: str):
    """Return language token index."""
    idx = dic.index(_lang_token(lang))
    assert idx != dic.unk_index, "cannot find language token for lang {}".format(lang)
    return idx

class LabelEncoder(object):
    def __init__(self, dictionary: Dictionary) -> None:
        self.dictionary = dictionary

    def __call__(self, label: str) -> List[str]:
        return self.dictionary.encode_line(
            label,
            append_eos=False,
            add_if_not_exist=False,
        )


@dataclass
class HubertPretrainingConfig(FairseqDataclass):
    data: str = field(default=MISSING, metadata={"help": "path to data directory"})
    fine_tuning: bool = field(
        default=False, metadata={"help": "set to true if fine-tuning Hubert"}
    )
    labels: List[str] = field(
        default_factory=lambda: ["ltr"],
        metadata={
            "help": (
                "extension of the label files to load, frame-level labels for"
                " pre-training, and sequence-level label for fine-tuning"
            )
        },
    )
    label_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "if set, looks for labels in this directory instead",
        },
    )
    label_rate: int = field(
        default=-1,
        metadata={"help": "label frame rate. -1 for sequence label"},
    )
    sample_rate: int = field(
        default=16_000,
        metadata={
            "help": "target sample rate. audio files will be up/down "
            "sampled to this rate"
        },
    )
    normalize: bool = field(
        default=False,
        metadata={"help": "if set, normalizes input to have 0 mean and unit variance"},
    )
    enable_padding: bool = field(
        default=False,
        metadata={"help": "pad shorter samples instead of cropping"},
    )
    max_keep_size: Optional[int] = field(
        default=None,
        metadata={"help": "exclude sample longer than this"},
    )
    max_sample_size: Optional[int] = field(
        default=None,
        metadata={"help": "max sample size to crop to for batching"},
    )
    min_sample_size: Optional[int] = field(
        default=None,
        metadata={"help": "min sample size to crop to for batching"},
    )
    single_target: Optional[bool] = field(
        default=False,
        metadata={
            "help": "if set, AddTargetDatasets outputs same keys as AddTargetDataset"
        },
    )
    random_crop: Optional[bool] = field(
        default=True,
        metadata={"help": "always crop from the beginning if false"},
    )
    pad_audio: Optional[bool] = field(
        default=False,
        metadata={"help": "pad audio to the longest one in the batch if true"},
    )
    store_labels: Optional[bool] = field(
        default=False,
        metadata={"help": "store spm labels in memory, should be true when fine-tune with bpe"},
    )
    add_decoder: bool = field(
        default=False,
        metadata={"help": "whether to add decoder for CE Loss on code"},
    )
    speech_tgt_lang: str = field(
        default="",
        metadata={"help": "prepend <tgt-id> to prev_output_tokens to replace <eos>, only used for decoder"},
    )
    hubert_tokenizer: Optional[TOKENIZER_CHOICES] = field(
        default="none",
        metadata={"help": "which tokenizer for processing text"},
    )
    sp_path: Optional[str] = field(
        default=None,
        metadata={"help": "sentencepiece model path if using bpe tokenizer"},
    )
    mbart_style_lang_id: Optional[bool] = field(
        default=False,
        metadata={"help": "Set lang id by mbart style"},
    )
    reduce_label_for_dec: Optional[bool] = field(
        default=True,
        metadata={"help": "whether to reduce label for decoder"},
    )


@register_task("hubert_pretraining", dataclass=HubertPretrainingConfig)
class HubertPretrainingTask(FairseqTask):

    cfg: HubertPretrainingConfig

    def __init__(
        self,
        cfg: HubertPretrainingConfig,
    ) -> None:
        super().__init__(cfg)

        logger.info(f"current directory is {os.getcwd()}")
        logger.info(f"HubertPretrainingTask Config {cfg}")

        self.cfg = cfg
        self.fine_tuning = cfg.fine_tuning

        self.state.add_factory("hubert_tokenizer", self.build_tokenizer)
        if cfg.fine_tuning:
            self.state.add_factory("target_dictionary", self.load_dictionaries)
        else:
            self.state.add_factory("dictionaries", self.load_dictionaries)

        self.blank_symbol = "<s>"

    @property
    def source_dictionary(self) -> Optional[Dictionary]:
        return None

    @property
    def target_dictionary(self) -> Optional[Dictionary]:
        return self.state.target_dictionary

    @property
    def dictionaries(self) -> List[Dictionary]:
        return self.state.dictionaries

    @property
    def hubert_tokenizer(self):
        return self.state.hubert_tokenizer

    @classmethod
    def setup_task(
        cls, cfg: HubertPretrainingConfig, **kwargs
    ) -> "HubertPretrainingTask":
        return cls(cfg)

    def load_dictionaries(self):
        label_dir = self.cfg.data if self.cfg.label_dir is None else self.cfg.label_dir
        dictionaries = [
            Dictionary.load(f"{label_dir}/dict.{label}.txt")
            for label in self.cfg.labels
        ]
        return dictionaries[0] if self.cfg.fine_tuning else dictionaries

    def get_label_dir(self) -> str:
        if self.cfg.label_dir is None:
            return self.cfg.data
        return self.cfg.label_dir

    def load_dataset(self, split: str, **kwargs) -> None:
        manifest = f"{self.cfg.data}/{split}.tsv"
        dicts = [self.target_dictionary] if self.cfg.fine_tuning else self.dictionaries
        pad_list = [dict.pad() for dict in dicts]
        eos_list = [dict.eos() for dict in dicts]
        procs = [LabelEncoder(dict) for dict in dicts]
        paths = [f"{self.get_label_dir()}/{split}.{l}" for l in self.cfg.labels]

        if self.cfg.speech_tgt_lang != "":
            tgt_lang_idx = _lang_token_index(dicts[0], self.cfg.speech_tgt_lang)
            logger.info(f"Will prepend <{tgt_lang_idx}> at the beginning of prev_output_tokens to replace <eos>")
        else:
            tgt_lang_idx = None

        # hubert v1: pad_audio=True, random_crop=False;
        self.datasets[split] = HubertDataset(
            manifest,
            sample_rate=self.cfg.sample_rate,
            label_paths=paths,
            label_rates=self.cfg.label_rate,
            pad_list=pad_list,
            eos_list=eos_list,
            label_processors=procs,
            max_keep_sample_size=self.cfg.max_keep_size,
            min_keep_sample_size=self.cfg.min_sample_size,
            max_sample_size=self.cfg.max_sample_size,
            pad_audio=self.cfg.pad_audio,
            normalize=self.cfg.normalize,
            store_labels=self.cfg.store_labels,
            random_crop=self.cfg.random_crop,
            single_target=self.cfg.single_target,
            tgt_dict=dicts[0],
            add_decoder=self.cfg.add_decoder,
            fine_tuning=self.cfg.fine_tuning,
            tgt_lang_idx=tgt_lang_idx,
            tokenizer=self.hubert_tokenizer,
            mbart_style_lang_id=self.cfg.mbart_style_lang_id,
            reduce_label_for_dec=self.cfg.reduce_label_for_dec,
        )

    def max_positions(self) -> Tuple[int, int]:
        return (sys.maxsize, sys.maxsize)

    def filter_indices_by_size(self, indices: np.array, *args, **kwargs) -> np.array:
        return indices

    def build_tokenizer(self, cfg=None):
        logger.info(f"tokenizer: {self.cfg.hubert_tokenizer}")
        if self.cfg.hubert_tokenizer != "none":
            return encoders.build_bpe(Namespace(**{"bpe": self.cfg.hubert_tokenizer, "sentencepiece_model": self.cfg.sp_path}))
        else:
            return None

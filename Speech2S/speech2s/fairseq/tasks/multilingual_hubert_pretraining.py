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
from fairseq.data import Dictionary, HubertDataset, ResamplingDataset, ConcatDataset, encoders
from fairseq.dataclass.configs import FairseqDataclass
from fairseq.tasks import register_task
from fairseq.tasks.fairseq_task import FairseqTask
from fairseq.dataclass.constants import ChoiceEnum
from fairseq.tasks.hubert_pretraining import LabelEncoder
from omegaconf import MISSING

logger = logging.getLogger(__name__)

TOKENIZER_CHOICES = ChoiceEnum(["sentencepiece", "hubert_letters", "none"])

def _lang_token(lang: str):
    return "__{}__".format(lang)


def _lang_token_index(dic: Dictionary, lang: str):
    """Return language token index."""
    idx = dic.index(_lang_token(lang))
    assert idx != dic.unk_index, "cannot find language token for lang {}".format(lang)
    return idx


@dataclass
class MultilingualHubertPretrainingConfig(FairseqDataclass):
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
    add_decoder: bool = field(
        default=False,
        metadata={"help": "whether to add decoder for CE Loss on code"},
    )
    decoder_langtok: Optional[bool] = field(
        default=True,
        metadata={"help": "replace beginning-of-sentence in target sentence with target language token"},
    )
    sampling_alpha: float = field(
        default=0.2,
        metadata={
            "help": "Hyper-parameter alpha = 1/T for temperature-based resampling."
            "(alpha = 1 for no resampling)"
        },
    )
    tgt_langs: str = field(
        default=MISSING,
        metadata={
            "help": "tgt language for multilingual, like es,fr,pt,it"
        },
    )
    hubert_tokenizer: Optional[TOKENIZER_CHOICES] = field(
        default="none",
        metadata={"help": "which tokenizer for processing text"},
    )
    sp_path: Optional[str] = field(
        default=None,
        metadata={"help": "sentencepiece model path if using bpe tokenizer"},
    )
    spec_tgt_lang: Optional[str] = field(
        default=None,
        metadata={
            "help": "tgt language for multilingual, like es,fr,pt,it"
        },
    )


@register_task("multilingual_hubert_pretraining", dataclass=MultilingualHubertPretrainingConfig)
class MultilingualHubertPretrainingTask(FairseqTask):

    cfg: MultilingualHubertPretrainingConfig

    def __init__(
        self,
        cfg: MultilingualHubertPretrainingConfig,
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
        
        self.tgt_langs = cfg.tgt_langs.split(',')

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
        cls, cfg: MultilingualHubertPretrainingConfig, **kwargs
    ) -> "MultilingualHubertPretrainingTask":
        return cls(cfg)

    def load_dictionaries(self):
        label_dir = self.cfg.data if self.cfg.label_dir is None else self.cfg.label_dir
        dictionaries = [Dictionary.load(f"{label_dir}/dict.{label}.txt") for label in self.cfg.labels]
        if self.cfg.decoder_langtok:
            for d in (dictionaries):
                for lang_to_add in self.tgt_langs:
                    d.add_symbol(_lang_token(lang_to_add))
        return dictionaries[0] if self.cfg.fine_tuning else dictionaries

    def get_label_dir(self) -> str:
        if self.cfg.label_dir is None:
            return self.cfg.data
        return self.cfg.label_dir

    def load_dataset(self, split: str, epoch=1, **kwargs) -> None:
        _splits = split.split(',')
        datasets = [self._load_dataset(sp) for sp in _splits]
        
        if 'train' in split and len(_splits) > 1 and self.cfg.sampling_alpha != 1.0:
            # temperature-based sampling
            size_ratios = self._get_size_ratios(
                _splits, [len(s) for s in datasets], alpha=self.cfg.sampling_alpha
            )
            datasets = [
                ResamplingDataset(
                    d, size_ratio=r, seed=0, epoch=epoch, replace=(r >= 1.0)
                )
                for d, r in zip(datasets, size_ratios)
            ]

        self.datasets[split] = ConcatDataset(datasets) if len(datasets) > 1 else datasets[0]

    def _load_dataset(self, split: str) -> HubertDataset:
        manifest = f"{self.cfg.data}/{split}.tsv"
        dicts = [self.target_dictionary] if self.cfg.fine_tuning else self.dictionaries
        pad_list = [dict.pad() for dict in dicts]
        eos_list = [dict.eos() for dict in dicts]
        procs = [LabelEncoder(dict) for dict in dicts]
        paths = [f"{self.get_label_dir()}/{split}.{l}" for l in self.cfg.labels]

        if self.cfg.spec_tgt_lang is not None:
            assert self.cfg.spec_tgt_lang in self.tgt_langs, "specific target language must be in the tgt_langs"
            tgt_lang = self.cfg.spec_tgt_lang
        else:
            tgt_lang = split.split("-")[-1] if "-" in split else split.split("_")[-1]
        # hubert v1: pad_audio=True, random_crop=False;
        return HubertDataset(
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
            store_labels=False,
            random_crop=self.cfg.random_crop,
            single_target=self.cfg.single_target,
            tgt_dict=dicts[0],
            add_decoder=self.cfg.add_decoder,
            fine_tuning=self.cfg.fine_tuning,
            tgt_lang_idx=_lang_token_index(dicts[0], tgt_lang),
            tokenizer=self.hubert_tokenizer,
        )

    def max_positions(self) -> Tuple[int, int]:
        return (sys.maxsize, sys.maxsize)

    def filter_indices_by_size(self, indices: np.array, *args, **kwargs) -> np.array:
        return indices

    @classmethod
    def _get_size_ratios(cls, ids: List[str], sizes: List[int], alpha: float = 1.0):
        """Size ratios for temperature-based sampling
        (https://arxiv.org/abs/1907.05019)"""
        _sizes = np.array(sizes)
        prob = _sizes / _sizes.sum()
        smoothed_prob = prob ** alpha
        smoothed_prob = smoothed_prob / smoothed_prob.sum()
        size_ratio = (smoothed_prob * _sizes.sum()) / _sizes

        o_str = str({_i: f"{prob[i]:.3f}" for i, _i in enumerate(ids)})
        logger.info(f"original sampling probability: {o_str}")
        p_str = str({_i: f"{smoothed_prob[i]:.3f}" for i, _i in enumerate(ids)})
        logger.info(f"balanced sampling probability: {p_str}")
        sr_str = str({_id: f"{size_ratio[i]:.3f}" for i, _id in enumerate(ids)})
        logger.info(f"balanced sampling size ratio: {sr_str}")
        return size_ratio.tolist()

    def build_tokenizer(self, cfg=None):
        logger.info(f"tokenizer: {self.cfg.hubert_tokenizer}")
        if self.cfg.hubert_tokenizer != "none":
            return encoders.build_bpe(Namespace(**{"bpe": self.cfg.hubert_tokenizer, "sentencepiece_model": self.cfg.sp_path}))
        else:
            return None

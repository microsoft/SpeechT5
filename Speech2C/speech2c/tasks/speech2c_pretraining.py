# --------------------------------------------------------
# Pre-Training Transformer Decoder for End-to-End ASR Model with Unpaired Speech Data (https://arxiv.org/abs/2203.17113)
# Github source: https://github.com/microsoft/SpeechT5/tree/main/Speech2C
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Based on fairseq code bases
# https://github.com/pytorch/fairseq
# --------------------------------------------------------

import logging

from dataclasses import dataclass, field
from fairseq.data import Dictionary
from fairseq.tasks import register_task
from fairseq.tasks.hubert_pretraining import HubertPretrainingConfig, HubertPretrainingTask, LabelEncoder
from speech2c.data.speech2c_dataset import Speech2cDataset

logger = logging.getLogger(__name__)


@dataclass
class Speech2cPretrainingConfig(HubertPretrainingConfig):
    add_decoder: bool = field(
        default=False,
        metadata={"help": "whether to add decoder for CE Loss on code"},
    )
    
    # For inference
    ctc_weight: float = field(
        default=0.0,
        metadata={"help": "ctc weight during inference"},
    )


@register_task("speech2c_pretraining", dataclass=Speech2cPretrainingConfig)
class Speech2cPretrainingTask(HubertPretrainingTask):

    cfg: Speech2cPretrainingConfig

    def load_dictionaries(self):
        label_dir = self.cfg.data if self.cfg.label_dir is None else self.cfg.label_dir
        dictionaries = [Dictionary.load(f"{label_dir}/dict.{label}.txt") for label in self.cfg.labels]
        return dictionaries[0] if self.cfg.fine_tuning else dictionaries

    def load_dataset(self, split: str, **kwargs) -> None:
        manifest = f"{self.cfg.data}/{split}.tsv"
        dicts = [self.target_dictionary] if self.cfg.fine_tuning else self.dictionaries
        pad_list = [dict.pad() for dict in dicts]
        eos_list = [dict.eos() for dict in dicts]
        procs = [LabelEncoder(dict) for dict in dicts]
        paths = [
            f"{self.get_label_dir()}/{split}.{l}" for l in self.cfg.labels
        ]

        # hubert v1: pad_audio=True, random_crop=False;
        self.datasets[split] = Speech2cDataset(
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
        )

    def build_generator(
        self,
        models,
        args,
        seq_gen_cls=None,
        extra_gen_cls_kwargs=None,
    ):
        from speech2c.squence_generator import SequenceGenerator
        extra_gen_cls_kwargs = {
            "ctc_weight": self.cfg.ctc_weight,
            **extra_gen_cls_kwargs
        }
        return super().build_generator(
            models, args, seq_gen_cls=SequenceGenerator, extra_gen_cls_kwargs=extra_gen_cls_kwargs
        )

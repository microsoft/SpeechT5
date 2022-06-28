# --------------------------------------------------------
# The YiTrans End-to-End Speech Translation System for IWSLT 2022 Offline Shared Task (https://arxiv.org/abs/2206.05777)
# Github source: https://github.com/microsoft/SpeechT5/tree/main/YiTrans
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Based on fairseq code bases
# https://github.com/facebookresearch/fairseq
# --------------------------------------------------------

"""
    Modified from https://github.com/facebookresearch/fairseq/blob/main/fairseq/data/audio/multi_modality_dataset.py
"""


from typing import Optional

import numpy as np
import torch
from fairseq.data import (
    LanguagePairDataset,
)
from fairseq.data.audio.multi_modality_dataset import LangPairMaskDataset as FairseqLangPairMaskDataset

class LangPairMaskDataset(FairseqLangPairMaskDataset):
    def __init__(
        self,
        dataset: LanguagePairDataset,
        src_eos: int,
        src_bos: Optional[int] = None,
        noise_id: Optional[int] = -1,
        mask_ratio: Optional[float] = 0,
        mask_type: Optional[str] = "random",
    ):
        super.__init__(
            dataset,
            src_eos,
            src_bos,
            noise_id,
            mask_ratio,
            mask_type,
        )
    def mask_src_tokens(self, sample):
        src_item = sample["source"]
        mask = None
        if self.mask_type == "random":
            mask = torch.rand(len(src_item)).le(self.mask_ratio)
        else:
            mask = torch.ones(len(src_item))
            mask[: int(len(src_item) * (1 - self.mask_ratio))] = 0
            mask = mask.eq(1)
        if src_item[0] == self.src_bos:
            mask[0] = False
        if src_item[-1] == self.src_eos:
            mask[-1] = False
        mask_src_item = src_item.masked_fill(mask, self.noise_id)
        smp = sample
        smp["source"] = mask_src_item
        return smp

    def collater(self, samples, pad_to_length=None):
        return self.dataset.collater(samples, pad_to_length=pad_to_length)


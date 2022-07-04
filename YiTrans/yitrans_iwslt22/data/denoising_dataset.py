# --------------------------------------------------------
# The YiTrans End-to-End Speech Translation System for IWSLT 2022 Offline Shared Task (https://arxiv.org/abs/2206.05777)
# Github source: https://github.com/microsoft/SpeechT5/tree/main/YiTrans
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Based on fairseq code bases
# https://github.com/facebookresearch/fairseq
# --------------------------------------------------------

import math

import numpy as np
import torch

from fairseq.data import FairseqDataset, data_utils, DenoisingDataset


class DenoisingDatasetLang(DenoisingDataset):
    """
    A wrapper around DenoisingDataset for BART dataset.

    """

    def __init__(
        self,
        dataset,
        sizes,
        vocab,
        mask_idx,
        mask_whole_words,
        shuffle,
        seed,
        args,
        eos=None,
        item_transform_func=None,
        tgt_lang_idx=None,
    ):
        super().__init__(
            dataset,
            sizes,
            vocab,
            mask_idx,
            mask_whole_words,
            shuffle,
            seed,
            args,
            eos,
            item_transform_func,
        )
        
        self.tgt_lang_idx=tgt_lang_idx

    def __getitem__(self, index):
        with data_utils.numpy_seed(self.seed, self.epoch, index):
            tokens = self.dataset[index]
            assert tokens[-1] == self.eos
            source, target = tokens, tokens.clone()

            if self.permute_sentence_ratio > 0.0:
                source = self.permute_sentences(source, self.permute_sentence_ratio)

            if self.mask_ratio > 0:
                source = self.add_whole_word_mask(source, self.mask_ratio)

            if self.insert_ratio > 0:
                source = self.add_insertion_noise(source, self.insert_ratio)

            if self.rotate_ratio > 0.0 and np.random.random() < self.rotate_ratio:
                source = self.add_rolling_noise(source)
        # there can additional changes to make:
        if self.item_transform_func is not None:
            source, target = self.item_transform_func(source, target)

        assert (source >= 0).all()
        assert (source[1:-1] >= 1).all()
        assert (source <= len(self.vocab)).all()
        assert source[0] == self.vocab.bos()
        assert target[0] == self.vocab.bos()
        assert source[-1] == self.eos

        if self.tgt_lang_idx is not None:
            tgt_lang_idx = torch.LongTensor([self.tgt_lang_idx])
            source = torch.cat([source[1:], tgt_lang_idx])
            target = torch.cat([target[1:], tgt_lang_idx])
        sample = {
            "id": index,
            "source": source,
            "target": target,
        }
        return sample

# ----------------------------------------------------------------------------
# VatLM: Visual-Audio-Text Pre-Training  with Unified Masked Prediction for Speech Representation Learning
# Github source: https://github.com/microsoft/SpeechT5/tree/main/VATLM
# Code based on fairseq: https://github.com/facebookresearch/fairseq and av_hubert: https://github.com/facebookresearch/av_hubert
# 
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# ----------------------------------------------------------------------------

import itertools
import logging
import os
import sys
import time
from typing import Any, List, Optional, Union

import numpy as np

import torch
import torch.nn.functional as F
from fairseq.data import data_utils
from fairseq.data.fairseq_dataset import FairseqDataset

DBG=True if len(sys.argv) == 1 else False

if DBG:
    import utils as custom_utils
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=os.environ.get("LOGLEVEL", "DEBUG").upper(),
        stream=sys.stdout,
    )
else:
    from . import utils as custom_utils

logger = logging.getLogger(__name__)


def load_text(manifest_path, max_keep, min_keep, frame_rate, label_paths, label_rates, tol=0.1):

    n_long, n_short, n_unaligned = 0, 0, 0
    names, inds, sizes = [], [], []
    dur_from_label_list = []

    with open(manifest_path) as f:
        for ind, line in enumerate(f):
            items = line.strip().split("\t")
            frames = items[0]
            sz = int(frames)
            if min_keep is not None and sz < min_keep:
                n_short += 1
            elif max_keep is not None and sz > max_keep:
                n_long += 1
            else:
                inds.append(ind)
                sizes.append(sz)

    logger.info(
        (
            f"max_keep={max_keep}, min_keep={min_keep}, "
            f"loaded {len(inds)}, skipped {n_short} short and {n_long} long"
            f"longest-loaded={max(sizes)}, shortest-loaded={min(sizes)}"
        )
    )

    return inds, sizes


def load_label(label_path, inds):
    with open(label_path) as f:
        labels = [line.rstrip() for line in f]
        labels = [labels[i] for i in inds]
    return labels

def load_phone_label(tsv, inds):
    with open(tsv) as f:
        labels = [line.rstrip() for line in f.readlines()]
        labels = [labels[i] for i in inds]
    return labels


def load_label_offset(label_path, inds):
    with open(label_path) as f:
        code_lengths = [len(line.encode("utf-8")) for line in f]
        offsets = list(itertools.accumulate([0] + code_lengths))
        offsets = [(offsets[i], offsets[i + 1]) for i in inds]
    return offsets



class TextHubertDataset(FairseqDataset):
    def __init__(
            self,
            manifest_path: str,
            sample_rate: float,
            label_paths: List[str],
            label_rates: Union[List[float], float],  # -1 for sequence labels
            pad_list: List[str],
            eos_list: List[str],
            label_processors: Optional[List[Any]] = None,
            phone_sequence_processors: Optional[List[Any]] = None,
            max_keep_sample_size: Optional[int] = None,
            min_keep_sample_size: Optional[int] = None,
            max_sample_size: Optional[int] = None,
            shuffle: bool = True,
            pad_audio: bool = False,
            normalize: bool = False,
            store_labels: bool = True,
            single_target: bool = False,
            stack_order_audio: int=1,
            skip_verify: bool=False,
            is_s2s=False,
    ):
        self.label_rates = (
            [label_rates for _ in range(len(label_paths))]
            if isinstance(label_rates, int)
            else label_rates
        )
        inds, self.sizes = load_text(manifest_path, max_keep_sample_size, min_keep_sample_size, frame_rate=sample_rate, label_paths=label_paths, label_rates=self.label_rates)
        self.sample_rate = sample_rate
        self.stack_order_audio = stack_order_audio
        self.shuffle = shuffle

        self.num_labels = len(label_paths)
        self.pad_list = pad_list
        self.eos_list = eos_list
        self.label_processors = label_processors
        self.phone_processors = phone_sequence_processors
        self.single_target = single_target
        self.store_labels = store_labels
        self.is_s2s = is_s2s


        if store_labels:
            self.label_list = [load_label(p, inds) for p in label_paths]
            self.phone_list = [load_phone_label(p, inds) for p in [manifest_path]]

        else:
            self.label_paths = label_paths
            self.label_offsets_list = [
                load_label_offset(p, inds) for p in label_paths
            ]

        self.max_sample_size = (
            max_sample_size if max_sample_size is not None else sys.maxsize
        )
        self.pad_audio = pad_audio
        self.normalize = normalize


    def get_label(self, index, label_idx):
        if self.store_labels:
            label = self.label_list[label_idx][index]
        else:
            with open(self.label_paths[label_idx]) as f:
                offset_s, offset_e = self.label_offsets_list[label_idx][index]
                f.seek(offset_s)
                label = f.read(offset_e - offset_s)

        if self.label_processors is not None:
            label = self.label_processors[label_idx](label)
        return label

    def get_labels(self, index):
        return [self.get_label(index, i) for i in range(self.num_labels)]

    def get_phone(self, index, label_idx):
        label = self.phone_list[label_idx][index]
        if self.phone_processors is not None:
            label = self.phone_processors[label_idx](label)
        return label

    def get_phones(self, index):
        return [self.get_phone(index, i) for i in range(1)]


    def __getitem__(self, index):
        labels = self.get_labels(index)
        phone_sequence_list = self.get_phones(index)

    
        return {"id": index, "label_list": labels, "phone_sequence_list": phone_sequence_list}


    def __len__(self):
        return len(self.sizes)


    def collater(self, samples):
        samples = [s for s in samples if s["id"] is not None]
        if len(samples) == 0:
            return {}

        targets_by_label = [
            [s["label_list"][i] for s in samples]
            for i in range(self.num_labels)
        ]
        targets_list, lengths_list, ntokens_list = self.collater_label(
            targets_by_label,
        )

        phone_sequence_list = [s["phone_sequence_list"] for s in samples]
        if phone_sequence_list[0] is None:
            phone_sequence_list = None
        
        targets_by_phone_label = [
            [s["phone_sequence_list"][i] for s in samples]
            for i in range(self.num_labels)
        ]
        targets_phone_list, lengths_phone_list, ntokens_phone_list = self.collater_phone_label(
            targets_by_phone_label,
        )

        net_input = {"source": None}
        batch = {
            "id": torch.LongTensor([s["id"] for s in samples]),
            "net_input": net_input,
        }

        if self.single_target:
            batch["target_lengths"] = lengths_list[0]
            batch["ntokens"] = ntokens_list[0]
            if self.is_s2s:
                batch['target'], net_input['prev_output_tokens'] = targets_list[0][0], targets_list[0][1]
            else:
                batch["target"] = targets_list[0]
        else:
            batch["target_lengths_list"] = lengths_list
            batch["ntokens_list"] = ntokens_list
            batch["target_list"] = targets_list 

        batch["extra_text_phone_list"] = targets_phone_list       

        return batch

    def collater_frm_label(
        self, targets, label_rate, pad
    ):
        lengths = torch.LongTensor([len(t) for t in targets])
        ntokens = lengths.sum().item()
        targets = data_utils.collate_tokens(
            targets, pad_idx=pad, left_pad=False
        )
        return targets, lengths, ntokens


    def collater_frm_phone_label(
        self, targets, pad
    ):

        lengths = torch.LongTensor([len(t) for t in targets])
        ntokens = lengths.sum().item()
        targets = data_utils.collate_tokens(
            targets, pad_idx=pad, left_pad=False
        )
        return targets, lengths, ntokens

    def collater_label(self, targets_by_label,):
        targets_list, lengths_list, ntokens_list = [], [], []
        itr = zip(targets_by_label, self.label_rates, self.pad_list)
        for targets, label_rate, pad in itr:
            targets, lengths, ntokens = self.collater_frm_label(
                targets, label_rate, pad
            )
            targets_list.append(targets)
            lengths_list.append(lengths)
            ntokens_list.append(ntokens)
        return targets_list, lengths_list, ntokens_list


    def collater_phone_label(self, targets_by_label):
        targets_list, lengths_list, ntokens_list = [], [], []
        itr = zip(targets_by_label, self.label_rates, self.pad_list)
        for targets, label_rate, pad in itr:
            targets, lengths, ntokens = self.collater_frm_phone_label(
                targets, pad
            )
            targets_list.append(targets)
            lengths_list.append(lengths)
            ntokens_list.append(ntokens)
        return targets_list, lengths_list, ntokens_list


    def num_tokens(self, index):
        return self.size(index)

    def size(self, index):
        if self.pad_audio:
            return self.sizes[index]
        return min(self.sizes[index], self.max_sample_size)

    def ordered_indices(self):
        if self.shuffle:
            order = [np.random.permutation(len(self))]
        else:
            order = [np.arange(len(self))]

        order.append(self.sizes)
        return np.lexsort(order)[::-1]

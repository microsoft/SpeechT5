# --------------------------------------------------------
# SpeechT5: Unified-Modal Encoder-Decoder Pre-Training for Spoken Language Processing (https://arxiv.org/abs/2110.07205)
# Github source: https://github.com/microsoft/SpeechT5/tree/main/SpeechT5
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Based on fairseq and espnet code bases
# https://github.com/pytorch/fairseq; https://github.com/espnet/espnet
# --------------------------------------------------------

import logging
import os
from typing import Any, List, Optional

import numpy as np

import torch
import torch.nn.functional as F
from fairseq.data import data_utils, Dictionary
from fairseq.data.fairseq_dataset import FairseqDataset

logger = logging.getLogger(__name__)


def load_audio(manifest_path, max_keep, min_keep):
    """manifest tsv: wav_path, wav_nframe, wav_class

    Args
        manifest_path: str
        max_keep: int
        min_keep: int
    
    Return
        root, names, inds, tot, sizes, classes
    """
    n_long, n_short = 0, 0
    names, inds, sizes, classes = [], [], [], []
    with open(manifest_path) as f:
        root = f.readline().strip()
        for ind, line in enumerate(f):
            items = line.strip().split("\t")
            assert len(items) >= 2, line
            sz = int(items[1])
            if min_keep is not None and sz < min_keep:
                n_short += 1
            elif max_keep is not None and sz > max_keep:
                n_long += 1
            else:
                names.append(items[0])
                if len(items) > 2:
                    classes.append(items[2])
                inds.append(ind)
                sizes.append(sz)
    tot = ind + 1
    logger.info(
        (
            f"max_keep={max_keep}, min_keep={min_keep}, "
            f"loaded {len(names)}, skipped {n_short} short and {n_long} long, "
            f"longest-loaded={max(sizes)}, shortest-loaded={min(sizes)}"
        )
    )
    if len(classes) == 0:
        logger.warn("no classes loaded only if inference")
    return root, names, inds, tot, sizes, classes


def sample_from_feature(x: np.ndarray, max_segment_length: int = 300):
    """Load a segment within 300-400/51200-76800 frames or the corresponding samples from a utterance.

    Args:
        x (np.ndarray): feature or waveform (frames[, features]), e.g., log mel filter bank or waveform
        max_segment_length (int, optional): maximum segment length. Defaults to 400.

    Returns:
        np.ndarray: segmented features
    """
    if len(x) <= max_segment_length:
        return x
    start = np.random.randint(0, x.shape[0] - max_segment_length)
    return x[start: start + max_segment_length]


class SpeechToClassDataset(FairseqDataset):
    def __init__(
        self,
        manifest_path: str,
        sample_rate: float,
        label_processors: Optional[List[Any]] = None,
        max_keep_sample_size: Optional[int] = None,
        min_keep_sample_size: Optional[int] = None,
        shuffle: bool = True,
        normalize: bool = False,
        tgt_dict: Optional[Dictionary] = None,
        max_length: Optional[int] = None
    ):
        self.audio_root, self.audio_names, inds, tot, self.wav_sizes, self.wav_classes = load_audio(
            manifest_path, max_keep_sample_size, min_keep_sample_size
        )
        self.sample_rate = sample_rate
        self.shuffle = shuffle

        self.label_processors = label_processors

        self.normalize = normalize
        self.tgt_dict = tgt_dict
        self.max_length = max_length
        logger.info(
            f"max_length={max_length}, normalize={normalize}"
        )

    def get_audio(self, index):
        import soundfile as sf

        wav_path = os.path.join(self.audio_root, self.audio_names[index])
        wav, cur_sample_rate = sf.read(wav_path)
        if self.max_length is not None:
            wav = sample_from_feature(wav, self.max_length)
        wav = torch.from_numpy(wav).float()
        wav = self.postprocess(wav, cur_sample_rate)
        return wav

    def get_label(self, index):
        label = self.wav_classes[index]

        if self.label_processors is not None:
            label = self.label_processors(label)
        return label

    def __getitem__(self, index):
        wav = self.get_audio(index)
        label = None
        if len(self.wav_classes) == len(self.audio_names):
            label = self.get_label(index)
        return {"id": index, "source": wav, "label": label}

    def __len__(self):
        return len(self.wav_sizes)

    def collater(self, samples):
        samples = [s for s in samples if s["source"] is not None]
        if len(samples) == 0:
            return {}

        audios = [s["source"] for s in samples]
        audio_sizes = [len(s) for s in audios]

        audio_size = max(audio_sizes)
        collated_audios, padding_mask = self.collater_audio(
            audios, audio_size
        )

        decoder_label = None
        decoder_target = None
        decoder_target_lengths = None
        if samples[0]["label"] is not None:
            targets_by_label = [
                [s["label"] for s in samples]
            ]
            targets_list, lengths_list, ntokens_list = self.collater_label(targets_by_label)

            decoder_label = [
                (targets_list[0][i, :lengths_list[0][i]]).long()
                for i in range(targets_list[0].size(0))
            ]

            decoder_target = data_utils.collate_tokens(
                decoder_label,
                self.tgt_dict.pad(),
                self.tgt_dict.eos(),
                left_pad=False,
                move_eos_to_beginning=False,
            )
            decoder_target_lengths = torch.tensor(
                [x.size(0) for x in decoder_label], dtype=torch.long
            )
        prev_output_tokens = data_utils.collate_tokens(
            [torch.LongTensor([-1]) for _ in samples],
            self.tgt_dict.pad(),
            self.tgt_dict.eos(),
            left_pad=False,
            move_eos_to_beginning=True,
        )

        net_input = {
            "source": collated_audios, 
            "padding_mask": padding_mask,
            "prev_output_tokens": prev_output_tokens,
            "task_name": "s2c",
        }
        batch = {
            "id": torch.LongTensor([s["id"] for s in samples]),
            "net_input": net_input,
            "target": decoder_target,
            "target_lengths": decoder_target_lengths,
            "task_name": "s2c",
            "ntokens": len(samples),
        }

        return batch

    def collater_audio(self, audios, audio_size):
        collated_audios = audios[0].new_zeros(len(audios), audio_size)
        padding_mask = (
            torch.BoolTensor(collated_audios.shape).fill_(False)
        )
        for i, audio in enumerate(audios):
            diff = len(audio) - audio_size
            if diff == 0:
                collated_audios[i] = audio
            elif diff < 0:
                collated_audios[i] = torch.cat([audio, audio.new_full((-diff,), 0.0)])
                padding_mask[i, diff:] = True
            else:
                raise Exception("Diff should not be larger than 0")
        return collated_audios, padding_mask

    def collater_seq_label(self, targets, pad):
        lengths = torch.LongTensor([len(t) for t in targets])
        ntokens = lengths.sum().item()
        targets = data_utils.collate_tokens(targets, pad_idx=pad, left_pad=False)
        return targets, lengths, ntokens

    def collater_label(self, targets_by_label):
        targets_list, lengths_list, ntokens_list = [], [], []
        itr = zip(targets_by_label, [self.tgt_dict.pad()])
        for targets, pad in itr:
            targets, lengths, ntokens = self.collater_seq_label(targets, pad)
            targets_list.append(targets)
            lengths_list.append(lengths)
            ntokens_list.append(ntokens)
        return targets_list, lengths_list, ntokens_list

    def num_tokens(self, index):
        return self.size(index)

    def size(self, index):
        return self.wav_sizes[index]

    @property
    def sizes(self):
        return np.array(self.wav_sizes)

    def ordered_indices(self):
        if self.shuffle:
            order = [np.random.permutation(len(self))]
        else:
            order = [np.arange(len(self))]

        order.append(self.wav_sizes)
        return np.lexsort(order)[::-1]

    def postprocess(self, wav, cur_sample_rate):
        if wav.dim() == 2:
            wav = wav.mean(-1)
        assert wav.dim() == 1, wav.dim()

        if cur_sample_rate != self.sample_rate:
            raise Exception(f"sr {cur_sample_rate} != {self.sample_rate}")

        if self.normalize:
            with torch.no_grad():
                wav = F.layer_norm(wav, wav.shape)
        return wav

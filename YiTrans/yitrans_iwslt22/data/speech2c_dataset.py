# --------------------------------------------------------
# The YiTrans End-to-End Speech Translation System for IWSLT 2022 Offline Shared Task (https://arxiv.org/abs/2206.05777)
# Github source: https://github.com/microsoft/SpeechT5/tree/main/YiTrans
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Based on fairseq code bases
# https://github.com/facebookresearch/fairseq
# --------------------------------------------------------

import itertools
import logging
import os
import sys
from typing import Any, List, Optional, Union

import numpy as np

import torch
import torch.nn.functional as F
from fairseq.data import data_utils, Dictionary
from fairseq.data.audio.hubert_dataset import HubertDataset

logger = logging.getLogger(__name__)



class Speech2cDataset(HubertDataset):
    def __init__(
        self,
        manifest_path: str,
        sample_rate: float,
        label_paths: List[str],
        label_rates: Union[List[float], float],  # -1 for sequence labels
        pad_list: List[str],
        eos_list: List[str],
        label_processors: Optional[List[Any]] = None,
        max_keep_sample_size: Optional[int] = None,
        min_keep_sample_size: Optional[int] = None,
        max_sample_size: Optional[int] = None,
        shuffle: bool = True,
        pad_audio: bool = False,
        normalize: bool = False,
        store_labels: bool = True,
        random_crop: bool = False,
        single_target: bool = False,
        tgt_dict: Optional[Dictionary] = None,
        add_decoder: bool = False,
        fine_tuning: bool = False,
        tokenizer = None,
        tgt_lang_idx: int = None,
        mbart_style_lang_id: bool = False,
        retry_times: int = 5,
        reduce_label_for_dec: bool = True,
    ):
        super().__init__(
            manifest_path,
            sample_rate,
            label_paths,
            label_rates,
            pad_list,
            eos_list,
            label_processors,
            max_keep_sample_size,
            min_keep_sample_size,
            max_sample_size,
            shuffle,
            pad_audio,
            normalize,
            store_labels,
            random_crop,
            single_target
        )
        self.tgt_dict = tgt_dict
        self.add_decoder = add_decoder
        self.fine_tuning = fine_tuning
        self.tokenizer = tokenizer
        self.tgt_lang_idx = tgt_lang_idx
        self.mbart_style_lang_id = mbart_style_lang_id
        self.retry_times = retry_times
        self.reduce_label_for_dec = reduce_label_for_dec
        logger.info(
            f"tgt_lang_idx={self.tgt_lang_idx}, reduce_label_for_dec={reduce_label_for_dec}, "
            f"mbart_style_lang_id={mbart_style_lang_id}"
        )

        self.sizes = np.array(self.sizes)
    
    def get_label(self, index, label_idx):
        if self.store_labels:
            label = self.label_list[label_idx][index]
        else:
            with open(self.label_paths[label_idx]) as f:
                offset_s, offset_e = self.label_offsets_list[label_idx][index]
                f.seek(offset_s)
                label = f.read(offset_e - offset_s)

        if self.tokenizer is not None and self.fine_tuning:
            label = self.tokenizer.encode(label)

        if self.label_processors is not None:
            label = self.label_processors[label_idx](label)
        return label

    def collater(self, samples):
        # target = max(sizes) -> random_crop not used
        # target = max_sample_size -> random_crop used for long
        samples = [s for s in samples if s["source"] is not None]
        if len(samples) == 0:
            return {}

        audios = [s["source"] for s in samples]
        audio_sizes = [len(s) for s in audios]
        if self.pad_audio:
            audio_size = min(max(audio_sizes), self.max_sample_size)
        else:
            audio_size = min(min(audio_sizes), self.max_sample_size)
        collated_audios, padding_mask, audio_starts = self.collater_audio(
            audios, audio_size
        )

        targets_by_label = [
            [s["label_list"][i] for s in samples] for i in range(self.num_labels)
        ]
        targets_list, lengths_list, ntokens_list = self.collater_label(
            targets_by_label, audio_size, audio_starts
        )

        if self.add_decoder:
            if self.fine_tuning:
                    decoder_label = [
                        torch.cat((targets_list[0][i, :lengths_list[0][i]], torch.tensor([self.tgt_dict.eos()])), 0).long()
                        for i in range(targets_list[0].size(0))
                    ]
            else:
                if self.tokenizer is not None:
                    decoder_label = [
                        # Set 48 for translate int to char and avoid \n
                        torch.cat(
                            (
                                torch.tensor(
                                    self.tokenizer.sp.Encode(
                                        "".join(
                                            [chr(j + 48) for j in (
                                                targets_list[0][i, :lengths_list[0][i]].unique_consecutive() if self.reduce_label_for_dec else targets_list[0][i, :lengths_list[0][i]]
                                            ).tolist()]
                                        ), out_type=int
                                    )
                                ), 
                                torch.tensor([self.tgt_dict.eos()])
                            ), dim=0
                        ).long()
                        for i in range(targets_list[0].size(0))
                    ]
                else:
                    decoder_label = [
                        torch.cat((targets_list[0][i, :lengths_list[0][i]].unique_consecutive() if self.reduce_label_for_dec else targets_list[0][i, :lengths_list[0][i]], torch.tensor([self.tgt_dict.eos()])), 0).long()
                        for i in range(targets_list[0].size(0))
                    ]

            if self.mbart_style_lang_id:
                decoder_label = [
                    torch.cat((decoder_label[i], torch.tensor([self.tgt_lang_idx])), 0).long()
                    for i in range(targets_list[0].size(0))
                ]

            dec_ntokens = sum(x.size(0) for x in decoder_label)
            decoder_target = data_utils.collate_tokens(
                decoder_label,
                self.tgt_dict.pad(),
                self.tgt_dict.eos() if not self.mbart_style_lang_id else self.tgt_lang_idx,
                left_pad=False,
                move_eos_to_beginning=False,
            )
            decoder_target_lengths = torch.tensor(
                [x.size(0) for x in decoder_label], dtype=torch.long
            )
            prev_output_tokens = data_utils.collate_tokens(
                decoder_label,
                self.tgt_dict.pad(),
                self.tgt_dict.eos() if not self.mbart_style_lang_id else self.tgt_lang_idx,
                left_pad=False,
                move_eos_to_beginning=True,
            )
            
            if self.tgt_lang_idx is not None and not self.mbart_style_lang_id:
                assert (prev_output_tokens[:, 0] != self.tgt_dict.eos()).sum() == 0
                prev_output_tokens[:, 0] = self.tgt_lang_idx

            net_input = {
                "source": collated_audios, 
                "padding_mask": padding_mask,
                "prev_output_tokens": prev_output_tokens,
            }
            batch = {
                "id": torch.LongTensor([s["id"] for s in samples]),
                "net_input": net_input,
                "decoder_target": decoder_target,
                "decoder_target_lengths": decoder_target_lengths,
                "dec_ntokens": dec_ntokens,
                "lang_idx": self.tgt_lang_idx,
            }
        else:
            net_input = {"source": collated_audios, "padding_mask": padding_mask}
            batch = {
                "id": torch.LongTensor([s["id"] for s in samples]),
                "net_input": net_input,
            }

        if self.single_target:
            batch["target_lengths"] = lengths_list[0]
            batch["ntokens"] = ntokens_list[0]
            batch["target"] = targets_list[0]
        else:
            batch["target_lengths_list"] = lengths_list
            batch["ntokens_list"] = ntokens_list
            batch["target_list"] = targets_list
        return batch

    # @property
    # def sizes(self):
    #     return np.array(self.sizes)


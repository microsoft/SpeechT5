# --------------------------------------------------------
# Pre-Training Transformer Decoder for End-to-End ASR Model with Unpaired Speech Data (https://arxiv.org/abs/2203.17113)
# Github source: https://github.com/microsoft/SpeechT5/tree/main/Speech2C
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Based on fairseq code bases
# https://github.com/pytorch/fairseq
# --------------------------------------------------------

import logging
from typing import Any, List, Optional, Union

import torch
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
                decoder_label = [
                    torch.cat((targets_list[0][i, :lengths_list[0][i]].unique_consecutive(), torch.tensor([self.tgt_dict.eos()])), 0).long()
                    for i in range(targets_list[0].size(0))
                ]
            dec_ntokens = sum(x.size(0) for x in decoder_label)
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
                decoder_label,
                self.tgt_dict.pad(),
                self.tgt_dict.eos(),
                left_pad=False,
                move_eos_to_beginning=True,
            )
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

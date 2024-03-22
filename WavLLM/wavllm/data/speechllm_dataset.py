# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import csv
import io
import logging
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass

import numpy as np
import torch
from fairseq.data import (
    ConcatDataset,
    Dictionary,
    FairseqDataset,
    ResamplingDataset,
    data_utils as fairseq_data_utils,
)
from fairseq.data.audio.audio_utils import (
    get_fbank,
    get_waveform,
    read_from_stored_zip,
    is_npy_data,
    is_sf_audio_data,
    parse_path,
    FEATURE_OR_SF_AUDIO_FILE_EXTENSIONS,
)
from fairseq.data.audio.feature_transforms import CompositeAudioFeatureTransform
#from fairseq.data.audio.data_cfg import S2TDataConfig as SpeechLLMDataConfig

import os
from sentencepiece import SentencePieceProcessor
from copy import deepcopy

import torchaudio

logger = logging.getLogger(__name__)


def get_features_from_npy_or_audio(path):
    ext = Path(path).suffix
    if ext not in FEATURE_OR_SF_AUDIO_FILE_EXTENSIONS:
        raise ValueError(f'Unsupported file format for "{path}"')
    return np.load(path) if ext == ".npy" else get_fbank(path)


def get_features_or_waveform_from_stored_zip(
    path,
    byte_offset,
    byte_size,
    need_waveform=False,
    use_sample_rate=None,
):
    assert path.endswith(".zip")
    data = read_from_stored_zip(path, byte_offset, byte_size)
    f = io.BytesIO(data)
    if is_npy_data(data):
        features_or_waveform = np.load(f)
    elif is_sf_audio_data(data):
        features_or_waveform = (
            get_waveform(f, always_2d=False, output_sample_rate=use_sample_rate)[0]
            if need_waveform
            else get_fbank(f)
        )
    else:
        raise ValueError(f'Unknown file format for "{path}"')
    return features_or_waveform


def get_features_or_waveform(path: str, need_waveform=False, use_sample_rate=None):
    """Get speech features from .npy file or waveform from .wav/.flac file.
    The file may be inside an uncompressed ZIP file and is accessed via byte
    offset and length.

    Args:
        path (str): File path in the format of "<.npy/.wav/.flac path>" or
        "<zip path>:<byte offset>:<byte length>".
        need_waveform (bool): return waveform instead of features.
        use_sample_rate (int): change sample rate for the input wave file

    Returns:
        features_or_waveform (numpy.ndarray): speech features or waveform.
    """
    _path, slice_ptr = parse_path(path)
    # print(_path)
    if len(slice_ptr) == 0:
        if need_waveform:
            return get_waveform(
                _path, always_2d=False, output_sample_rate=use_sample_rate
            )[0]
        return get_features_from_npy_or_audio(_path)
    elif len(slice_ptr) == 2:
        features_or_waveform = get_features_or_waveform_from_stored_zip(
            _path,
            slice_ptr[0],
            slice_ptr[1],
            need_waveform=need_waveform,
            use_sample_rate=use_sample_rate,
        )
    else:
        raise ValueError(f"Invalid path: {path}")

    return features_or_waveform


def _collate_frames(
    frames: List[torch.Tensor], is_audio_input: bool = False
) -> torch.Tensor:
    """
    Convert a list of 2D frames into a padded 3D tensor
    Args:
        frames (list): list of 2D frames of size L[i]*f_dim. Where L[i] is
            length of i-th frame and f_dim is static dimension of features
    Returns:
        3D tensor of size len(frames)*len_max*f_dim where len_max is max of L[i]
    """    
    max_len = max(frame.size(0) for frame in frames)
    if is_audio_input:
        out = frames[0].new_zeros((len(frames), max_len))
    else:
        out = frames[0].new_zeros((len(frames), max_len, frames[0].size(1)))
    for i, v in enumerate(frames):
        out[i, : v.size(0)] = v
    return out

def _collate_frames_pad_number(
    frames: List[torch.Tensor], pad_number
) -> torch.Tensor:
    max_len = max(frame.size(1) for frame in frames)
    # max_len = 2250 # all pad to 30s
    out = frames[0].new_ones((len(frames), frames[0].size(0), max_len)) * pad_number
    
    for i, v in enumerate(frames):
        out[i, :, : v.size(1)] = v
    return out


@dataclass
class SpeechLLMDatasetItem(object):
    index: int
    source: torch.Tensor
    target: Optional[torch.Tensor] = None
    left_prompt: Optional[torch.Tensor] = None
    speaker_id: Optional[int] = None
    target_mask: Optional[bool] = None
    prompt_mask: Optional[bool] = None
    left_prompt_mask: Optional[bool] = None
    speech_flag: bool = None
    speech_mask: Optional[bool] = None
    audio_codec: Optional[torch.Tensor] = None
    mid_prompt: Optional[torch.Tensor] = None
    mid_prompt_mask: Optional[bool] = None
    example_source: Optional[torch.Tensor] = None
    example_speech_mask: Optional[bool] = None
    example_audio_codec: Optional[torch.Tensor] = None
    lora_scale: Optional[torch.Tensor] = None
    orig_prompt: Optional[torch.Tensor] = None
    wavlm_sources: Optional[torch.Tensor] = None
    wavlm_speech_mask: Optional[bool] = None
    example_wavlm_sources: Optional[torch.Tensor] = None
    example_wavlm_speech_mask: Optional[bool] = None

class SpeechLLMDataset(FairseqDataset):
    LANG_TAG_TEMPLATE = "<lang:{}>"

    def __init__(
        self,
        cfg,
        data_root: str,
        split: str,
        is_train_split: bool,
        text_tokenizer=None,
        audio_processor=None,
        wavlm_processor=None,
        n_frames_per_step=1,
        append_eos=True,
    ):  
        samples = self._load_samples_from_tsv(data_root, split)
        KEY_ID, KEY_AUDIO, KEY_N_FRAMES = "id", "audio", "n_frames"
        KEY_PROMPT_TEXT, KEY_TGT_TEXT = "prompt", "tgt_text"

        START_FRAME, END_FRAME = "start_frame", "end_frame"

        WITH_SPEECH = "with_speech"

        audio_root = Path(cfg.audio_root)

        ids = [s[KEY_ID] for s in samples]
        audio_paths = [(audio_root / s[KEY_AUDIO]).as_posix() for s in samples]
        n_frames = [int(s[KEY_N_FRAMES]) for s in samples]
        prompt_texts = [s[KEY_PROMPT_TEXT] for s in samples]
        tgt_texts = [s[KEY_TGT_TEXT] for s in samples]

        if START_FRAME in samples[0]:
            start_frames = [s[START_FRAME] for s in samples]
            self.start_frames = start_frames
        else:
            self.start_frames = None
        
        if END_FRAME in samples[0]:
            end_frames = [s[END_FRAME] for s in samples]
            self.end_frames = end_frames
        else:
            self.end_frames = None

        self.split, self.is_train_split = split, is_train_split
        self.cfg = cfg
        self.audio_paths, self.n_frames = audio_paths, n_frames
        self.n_samples = len(audio_paths)
        assert len(n_frames) == self.n_samples > 0
        assert prompt_texts is None or len(prompt_texts) == self.n_samples
        assert tgt_texts is None or len(tgt_texts) == self.n_samples
        assert ids is None or len(ids) == self.n_samples

        if cfg.alpaca_text:
            speech_flag = [s[WITH_SPEECH] for s in samples]
            assert speech_flag is None or len(speech_flag) == self.n_samples
            self.speech_flag = speech_flag

        if cfg.prompt_bulid:
            self.B_INST = "[INST]"
            self.B_SYS = "<<SYS>>\n"
            self.SYSTEM = "As a helpful language and speech assistant, you are able to understand the speech content provided by the user, and assist the user with a variety of tasks using natural language."
            self.E_SYS = "\n<</SYS>>\n\n"
            self.E_INST = "[/INST]"
            self.B_SPEECH = "<SPEECH>"
            self.E_SPEECH = "</SPEECH>"
            self.B_EXAMPLE = "<EXAMPLE>"
            self.E_EXAMPLE = "</EXAMPLE>"
            self.B_TARGET = "<TARGET>"
            self.E_TARGET = "</TARGET>"

        self.tgt_texts = tgt_texts
        self.prompt_texts = prompt_texts
        self.ids = ids
        self.shuffle = cfg.shuffle if is_train_split else False

        self.feature_transforms = None

        self.tokenizer = text_tokenizer
        self.audio_processor = audio_processor
        self.wavlm_processor = wavlm_processor
        self.n_frames_per_step = n_frames_per_step
        self.speaker_to_id = None

        self.tgt_lens = self.get_tgt_lens_and_check_oov()
        self.append_eos = append_eos

        logger.info(self.__repr__())

    def get_tgt_lens_and_check_oov(self):
        if self.tgt_texts is None:
            return [0 for _ in range(self.n_samples)]
        tgt_lens = []
        n_tokens, n_oov_tokens = 0, 0
        for i in range(self.n_samples):
            tokenized = self.get_tokenized_tgt_text(i)
            # oov_tokens = [
            #     t
            #     for t in tokenized
            #     if self.tgt_dict.index(t) == self.tgt_dict.unk_index
            # ]
            # n_tokens += len(tokenized)
            # n_oov_tokens += len(oov_tokens)
            tgt_lens.append(len(tokenized))
        #logger.info(f"'{self.split}' has {n_oov_tokens / n_tokens * 100:.2f}% OOV")
        return tgt_lens

    def __repr__(self):
        return (
            self.__class__.__name__
            + f'(split="{self.split}", n_samples={self.n_samples:_}, '
            f"prepend_tgt_lang_tag={self.cfg.prepend_tgt_lang_tag}, "
            f"shuffle={self.shuffle}, transforms={self.feature_transforms}, "
            f"n_frames_per_step={self.n_frames_per_step}"
        )

    @classmethod
    def is_lang_tag(cls, token):
        pattern = cls.LANG_TAG_TEMPLATE.replace("{}", "(.*)")
        return re.match(pattern, token)

    def check_tgt_lang_tag(self):
        if self.cfg.prepend_tgt_lang_tag:
            assert self.tgt_langs is not None and self.tgt_dict is not None
            tgt_lang_tags = [
                self.LANG_TAG_TEMPLATE.format(t) for t in set(self.tgt_langs)
            ]
            assert all(t in self.tgt_dict for t in tgt_lang_tags)

    @classmethod
    def tokenize(cls, tokenizer, text: str):
        return text if tokenizer is None else tokenizer.encode(text)

    def get_tokenized_tgt_text(self, index: int):
        text = self.tgt_texts[index]
        text = torch.tensor(self.tokenizer.encode(text, bos=False, eos=True), dtype=torch.int64)
        return text

    def get_tokenized_few_shot_tgt_text(self, text):
        text = torch.tensor(self.tokenizer.encode(text, bos=False, eos=False), dtype=torch.int64)
        return text

    def get_speech_flag(self, index: int):
        speech_flag = self.speech_flag[index]
        if speech_flag == "True":
            return True
        else:
            return False

    def get_tokenized_prompt_text(self, index: int):
        text = self.prompt_texts[index]
        text = torch.tensor(self.tokenizer.encode('"' + text + '"', bos=False, eos=False), dtype=torch.int64)
        return text

    def get_tokenized_left_and_right_prompts_text(self, index, left_str, right_str):
        left_text = torch.tensor(self.tokenizer.encode(left_str, bos=True, eos=False), dtype=torch.int64)
        right_text = torch.tensor(self.tokenizer.encode(" " + self.E_SPEECH + " " + self.prompt_texts[index] + " " + right_str, bos=False, eos=False), dtype=torch.int64)
        return left_text, right_text

    def pack_frames(self, feature: torch.Tensor):
        if self.n_frames_per_step == 1:
            return feature
        n_packed_frames = feature.shape[0] // self.n_frames_per_step
        feature = feature[: self.n_frames_per_step * n_packed_frames]
        return feature.reshape(n_packed_frames, -1)

    @classmethod
    def get_lang_tag_idx(cls, lang: str, dictionary: Dictionary):
        lang_tag_idx = dictionary.index(cls.LANG_TAG_TEMPLATE.format(lang))
        assert lang_tag_idx != dictionary.unk()
        return lang_tag_idx

    def _get_source_audio(self, index: int) -> torch.Tensor:
        source = get_features_or_waveform(
            self.audio_paths[index],
            need_waveform=self.cfg.use_audio_input,
            use_sample_rate=self.cfg.use_sample_rate,
        )
        if self.feature_transforms is not None:
            assert not self.cfg.use_audio_input
            source = self.feature_transforms(source)
        source = torch.from_numpy(source).float()
        return source

    def process_audio_source(self, source):
        length_in_s = len(source) / self.cfg.use_sample_rate
        dura_flag = False if length_in_s <= 30 else True
        if dura_flag:
            source_parts = []
            sample_count = 30 * self.cfg.use_sample_rate
            # offset = 1 * self.cfg.use_sample_rate
            offset = 0
            segments = int(np.ceil((len(source) - sample_count) / (sample_count - offset)) + 1)  
            for i in range(segments):  
                start = i * (sample_count - offset)  
                end = min(start + sample_count, len(source))  
                part = source[start:end]
                source_parts.append(part)
        else:
            source_parts = [source]
        return source_parts

    def __getitem__(self, index: int) -> SpeechLLMDatasetItem:
        if self.start_frames is not None:
            source_num_frames = int(self.end_frames[index]) - int(self.start_frames[index])
            source, sr = torchaudio.load(self.audio_paths[index],
                            frame_offset=int(self.start_frames[index]),
                            num_frames=source_num_frames)
            source = source[0]
        else:
            source = self._get_source_audio(index)
        
        source = self.pack_frames(source)
        source_parts = self.process_audio_source(source)
        if self.cfg.is_whisper:
            sources = []
            speech_attention_masks = []

            for part in source_parts:
                audio_output = self.audio_processor(part, sampling_rate=16000, return_tensors="pt", return_attention_mask=True)
                sources.append(audio_output.input_features[0])
                speech_attention_masks.append(audio_output.attention_mask[0].bool())

            if self.cfg.use_wavlm:
                wavlm_output = self.wavlm_processor(source, sampling_rate=16000, return_tensors="pt", return_attention_mask=True)
                wavlm_sources = wavlm_output.input_values[0]
                
            else:
                wavlm_sources = None

        
        example_sources = None
        example_speech_attention_masks = None
        example_wavlm_sources = None

        audio_codec = None
        example_audio_codec = None

        target = None
        if self.tgt_texts is not None:
            target = self.get_tokenized_tgt_text(index) ## end with eos
            if self.cfg.prepend_tgt_lang_tag:
                lang_tag_idx = self.get_lang_tag_idx(
                    self.tgt_langs[index], self.tgt_dict
                )
                target = torch.cat((torch.LongTensor([lang_tag_idx]), target), 0)

        speaker_id = None
        if self.speaker_to_id is not None:
            speaker_id = self.speaker_to_id[self.speakers[index]]
        
        orig_prompt = self.get_tokenized_prompt_text(index) ## begin with bos

        if self.cfg.prompt_bulid:
            left_prompt_text = self.B_INST + self.B_SYS + self.SYSTEM + self.E_SYS + self.B_SPEECH
            right_prompt_text = self.E_INST
            left_prompt, right_prompt = self.get_tokenized_left_and_right_prompts_text(index, left_prompt_text, right_prompt_text)
            
            prompt_target = torch.cat((right_prompt, target), dim=0)
            left_prompt_mask = torch.ones(left_prompt.shape[0]).bool()
            
            mid_prompt = None
            mid_prompt_mask = None
            prompt_mask = torch.cat((torch.ones(right_prompt[1:].shape[0]), torch.zeros(target.shape[0])), dim=0).bool()
            target_mask = torch.cat((torch.zeros(right_prompt[1:].shape[0]), torch.ones(target.shape[0])), dim=0).bool()
        
        
        lora_scale = -1

        if self.cfg.alpaca_text:
            speech_flag = self.get_speech_flag(index)
            return SpeechLLMDatasetItem(
                index=index, source=sources, target=prompt_target, speaker_id=speaker_id, left_prompt=left_prompt,
                target_mask=target_mask, prompt_mask=prompt_mask, speech_mask=speech_attention_masks,
                audio_codec=audio_codec, speech_flag=speech_flag, left_prompt_mask=left_prompt_mask,
                mid_prompt=mid_prompt, mid_prompt_mask=mid_prompt_mask, example_source=example_sources,
                example_speech_mask=example_speech_attention_masks, example_audio_codec=example_audio_codec,
                lora_scale=lora_scale, orig_prompt=orig_prompt, wavlm_sources=wavlm_sources,
                example_wavlm_sources=example_wavlm_sources,
            )
        
        return SpeechLLMDatasetItem(
            index=index, source=sources, target=prompt_target, speaker_id=speaker_id, left_prompt=left_prompt,
            target_mask=target_mask, prompt_mask=prompt_mask, speech_mask=speech_attention_masks,
            audio_codec=audio_codec, left_prompt_mask=left_prompt_mask
        )

    def __len__(self):
        return self.n_samples

    def collater(
        self, samples: List[SpeechLLMDatasetItem], return_order: bool = False
    ) -> Dict:
        if len(samples) == 0:
            return {}
        indices = torch.tensor([x.index for x in samples], dtype=torch.long)
        lora_scales = torch.tensor([x.lora_scale for x in samples], dtype=torch.long)
        if self.cfg.is_whisper:
            frames = [x.source for x in samples]
            n_frames = torch.tensor([x.source[0].size(1) * len(x.source) for x in samples], dtype=torch.long)
            batch_size = len(frames)
            audio_decoder_input_ids = torch.ones((batch_size, self.cfg.whisper_token_len)).to(torch.long)
            # audio_decoder_input_ids = audio_decoder_input_ids.to(src_tokens.device).to(torch.long)
            if self.cfg.use_wavlm:
                wavlm_input_features = [{"input_values": x.wavlm_sources} for x in samples]
                wavlm_frames = self.wavlm_processor.pad(wavlm_input_features, padding=True, return_tensors="pt")["input_values"]
                wavlm_speech_masks = self.wavlm_processor.pad(wavlm_input_features, padding=True, return_tensors="pt")["attention_mask"]
                
                example_wavlm_frames = None
                example_wavlm_speech_masks = None
        n_frames, order = n_frames.sort(descending=True)
        indices = indices.index_select(0, order)
        lora_scales = lora_scales.index_select(0, order)
        # frames = frames.index_select(0, order)
        frames = [frames[i] for i in order]
        if wavlm_frames is not None:
            wavlm_frames = [wavlm_frames[i] for i in order]
            wavlm_speech_masks = [wavlm_speech_masks[i] for i in order]

        speech_masks = [x.speech_mask for x in samples]
        speech_masks = [speech_masks[i] for i in order]

        
        audio_codecs = None
        audio_codec_masks = None

        if self.cfg.alpaca_text:
            speech_flags = [x.speech_flag for x in samples]
            speech_flags = torch.tensor(speech_flags, dtype=torch.bool)
            speech_flags = speech_flags.index_select(0, order)
        else:
            speech_flags = None

        target, target_lengths = None, None
        prev_output_tokens = None
        ntokens = None
        
        if self.tgt_texts is not None:
            # if self.prompt_texts is not None:
            #     target_sequence = [x.prompt for x in samples] + [x.target for x in samples]
            #     print ("")
            # else:
            #     target_sequence = [x.target for x in samples]

            target_masks = [x.target_mask for x in samples]
            target_masks = fairseq_data_utils.collate_tokens(target_masks, False)
            target_masks = target_masks.index_select(0, order)

            prompt_masks = [x.prompt_mask for x in samples]
            prompt_masks = fairseq_data_utils.collate_tokens(prompt_masks, False)
            prompt_masks = prompt_masks.index_select(0, order)

            if self.cfg.prompt_bulid:
                left_prompt_masks = [x.left_prompt_mask for x in samples]
                left_prompt_masks = fairseq_data_utils.collate_tokens(left_prompt_masks, False)
                left_prompt_masks = left_prompt_masks.index_select(0, order)
                
                left_prompts = fairseq_data_utils.collate_tokens(
                    [x.left_prompt for x in samples],
                    0, #self.tokenizer.pad_id,
                )
                left_prompts = left_prompts.index_select(0, order)

                mid_prompt_masks = None
                mid_prompts = None
                
            target = fairseq_data_utils.collate_tokens(
                [x.target[1:] for x in samples],
                0, #self.tokenizer.pad_id,
            )
            target = target.index_select(0, order)
            target_lengths = torch.tensor(
                [x.target.size(0) for x in samples], dtype=torch.long
            ).index_select(0, order)
            prev_output_tokens = fairseq_data_utils.collate_tokens(
                [x.target[:-1] for x in samples],
                0, #self.tokenizer.pad_id,
            )
            prev_output_tokens = prev_output_tokens.index_select(0, order)
            ntokens = sum(x.target[1:].size(0) for x in samples)

            orig_prompts = fairseq_data_utils.collate_tokens(
                [x.orig_prompt for x in samples],
                0, #self.tokenizer.pad_id,
            )
            orig_prompts = orig_prompts.index_select(0, order)

        speaker = None
        if self.speaker_to_id is not None:
            speaker = (
                torch.tensor([s.speaker_id for s in samples], dtype=torch.long)
                .index_select(0, order)
                .view(-1, 1)
            )
    
        if self.cfg.is_whisper:
            net_input = {
                "index": indices,
                "lora_index": lora_scales,
                "speech_flag": speech_flags,
                "audio_codec": audio_codecs,
                "src_tokens": frames,
                "src_lengths": n_frames,
                "audio_decoder_input_ids": audio_decoder_input_ids,
                "prev_output_tokens": prev_output_tokens,
                "target_masks": target_masks,
                "prompt_masks": prompt_masks,
                "left_prompts": left_prompts,
                "left_prompt_masks": left_prompt_masks,
                "speech_masks": speech_masks,
                "codec_masks": audio_codec_masks,
                "orig_prompts": orig_prompts,
                "wavlm_src_tokens": wavlm_frames,
                "wavlm_speech_masks": wavlm_speech_masks,
            }
        out = {
            "id": indices,
            "net_input": net_input,
            "speaker": speaker,
            "target": target,
            "target_lengths": target_lengths,
            "ntokens": ntokens,
            "nsentences": len(samples),
        }
        if return_order:
            out["order"] = order
        return out

    def num_tokens(self, index):
        return self.n_frames[index]

    def size(self, index):
        return self.n_frames[index], self.tgt_lens[index]

    @property
    def sizes(self):
        return np.array(self.n_frames)

    @property
    def can_reuse_epoch_itr_across_epochs(self):
        return True

    def ordered_indices(self):
        if self.shuffle:
            order = [np.random.permutation(len(self))]
        else:
            order = [np.arange(len(self))]
        # first by descending order of # of frames then by original/random order
        order.append([-n for n in self.n_frames])
        return np.lexsort(order)

    def prefetch(self, indices):
        raise False

    def get_transforms(self, transform_type, split, is_train):
        """Split-specific feature transforms. Allowing train set
        wildcard `_train`, evaluation set wildcard `_eval` and general
        wildcard `*` for matching."""
        from copy import deepcopy

        cfg = deepcopy(self.config)
        _cur = cfg.get(f"{transform_type}transforms", {})
        cur = _cur.get(split)
        cur = _cur.get("_train") if cur is None and is_train else cur
        cur = _cur.get("_eval") if cur is None and not is_train else cur
        cur = _cur.get("*") if cur is None else cur
        return cur

    def get_feature_transforms(self, split, is_train):
        cfg = deepcopy(self.config)
        # TODO: deprecate transforms
        cur = self.get_transforms("", split, is_train)
        if cur is not None:
            logger.warning(
                "Auto converting transforms into feature_transforms, "
                "but transforms will be deprecated in the future. Please "
                "update this in the config."
            )
            ft_transforms = self.get_transforms("feature_", split, is_train)
            if ft_transforms:
                cur.extend(ft_transforms)
        else:
            cur = self.get_transforms("feature_", split, is_train)
        cfg["feature_transforms"] = cur
        return cfg

    def get_waveform_transforms(self, split, is_train):
        cfg = deepcopy(self.config)
        cfg["waveform_transforms"] = self.get_transforms("waveform_", split, is_train)
        return cfg

    @classmethod
    def _load_samples_from_tsv(self, root: str, split: str):
        tsv_path = Path(root) / f"{split}.tsv"
        if not tsv_path.is_file():
            raise FileNotFoundError(f"Dataset not found: {tsv_path}")
        with open(tsv_path) as f:
            reader = csv.DictReader(
                f,
                delimiter="\t",
                quotechar=None,
                doublequote=False,
                lineterminator="\n",
                quoting=csv.QUOTE_NONE,
            )
            samples = [dict(e) for e in reader]
        if len(samples) == 0:
            raise ValueError(f"Empty manifest: {tsv_path}")
        return samples



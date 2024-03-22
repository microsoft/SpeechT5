from typing import Optional, Tuple
from dataclasses import dataclass
import os

import torch
from transformers import WhisperConfig
from transformers.models.whisper.modeling_whisper import WhisperEncoder as HFWhisperEncoder
from transformers.utils import ModelOutput
from transformers.models.whisper.modeling_whisper import WhisperEncoderLayer, WhisperDecoderLayer
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Embedding, Linear

from fairseq.data.data_utils import lengths_to_padding_mask, lengths_to_mask
from fairseq.models import (
    FairseqEncoder,
    FairseqDecoder,
    FairseqEncoderDecoderModel,
    register_model,
    register_model_architecture,
)
from fairseq.models.speech_to_text.modules.convolution import Conv1dSubsampler
from typing import Optional, Tuple, List
from dataclasses import dataclass
from sentencepiece import SentencePieceProcessor
import math
import json
import os
import numpy as np
from functools import partial
import time

from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
    CheckpointImpl,
    apply_activation_checkpointing,
)

non_reentrant_wrapper = partial(
    checkpoint_wrapper,
    checkpoint_impl=CheckpointImpl.NO_REENTRANT,
)

check_fn_encoder = lambda submodule: isinstance(submodule, WhisperEncoderLayer)

def apply_fsdp_checkpointing(model):
    """apply activation checkpointing to model
    returns None as model is updated directly
    """
    print(f"--> applying fsdp activation checkpointing...")

    apply_activation_checkpointing(
        model, checkpoint_wrapper_fn=non_reentrant_wrapper, check_fn=check_fn_encoder
    )

def lengths_to_padding_mask(lens):
    bsz, max_lens = lens.size(0), torch.max(lens).item()
    mask = torch.arange(max_lens).to(lens.device).view(1, max_lens)
    mask = mask.expand(bsz, -1) >= lens.view(bsz, 1).expand(-1, max_lens)
    return mask

@dataclass
class WhisperOutput(ModelOutput):
    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    output_lengths: Optional[torch.LongTensor] = None

class WhisperAdapter(nn.Module):
    def __init__(
        self,
        input_size: int,
        down_size: int,
        activation: str
    ):
        super(WhisperAdapter, self).__init__()
        self.down_layer = nn.Linear(input_size, down_size)
        self.up_layer = nn.Linear(down_size, input_size)
        self.non_linearity = F.relu if activation == 'relu' else F.gelu
        self.layer_norm = nn.LayerNorm(input_size)

    def forward(self, src_tokens):
        return self.layer_norm(self.up_layer(self.non_linearity(self.down_layer(src_tokens)))) + src_tokens

class FairseqWhisperEncoder(FairseqEncoder):
    
    def __init__(self, args):
        super().__init__(None)
        # whisper_path = "/modelblob/users/v-shujiehu/checkpoints/whisper-large-v2"
        self.model = WhisperEncoder.from_pretrained(args.whisper_path)
        self.config = self.model.config

        for param in self.model.parameters():
            param.requires_grad = False

        
        self.adapter = WhisperAdapter(1024, 512, "gelu")
        self.projector = nn.Linear(1024, 2048)
        # self.subsample = Conv1dSubsampler(1280, 1280, 4096, [3, 3])
        self.subsample = Conv1dSubsampler(1280, 1280, 1024, [3, 3])
        apply_fsdp_checkpointing(self.model)

    def forward(
        self, src_tokens, attention_mask,
    ):
        hidden_states = src_tokens
        encoder_outputs = self.model(
            hidden_states,
            attention_mask=attention_mask,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        )
        
        speech_lengths = encoder_outputs.output_lengths
        outputs, speech_lengths = self.subsample(encoder_outputs.last_hidden_state, speech_lengths)
        outputs = outputs.transpose(0, 1).contiguous()
        speech_padding_mask = lengths_to_padding_mask(speech_lengths)
        speech_atts = ~speech_padding_mask
        outputs = self.adapter(outputs)
        outputs = self.projector(outputs)
        return {
            # "encoder_out": encoder_outputs[0],  # B T C
            "encoder_out": outputs,  # B T C
            "encoder_padding_mask": speech_atts
        }

    def split_wav_codec(self, audio_out, wav_n):
        ori_type = audio_out['encoder_out'].dtype
        split_audio_out = torch.split(audio_out['encoder_out'], wav_n, dim=0)
        split_padding_mask = torch.split(audio_out['encoder_padding_mask'], wav_n, dim=0)
        padded_audio_out = []  
        padded_padding_mask = []
        for a, p in zip(split_audio_out, split_padding_mask):  
            if a.shape[0] < max(wav_n):  
                a_size = list(a.shape)  
                a_size[0] = max(wav_n) - a.shape[0]  
                a_pad_tensor = torch.zeros(a_size).to(a.device)
                a = torch.cat((a, a_pad_tensor), dim=0) 

                p_size = list(p.shape)
                p_size[0] = max(wav_n) - p.shape[0]  
                p_pad_tensor = torch.zeros(p_size).bool().to(p.device)
                p = torch.cat((p, p_pad_tensor), dim=0) 
            padded_audio_out.append(a) 
            padded_padding_mask.append(p)

        audio_out['encoder_out'] = torch.stack([torch.cat(tuple(t[i] for i in range(max(wav_n))), dim=0) for t in padded_audio_out]).to(ori_type)
        audio_out['encoder_padding_mask'] = torch.stack([torch.cat(tuple(t[i] for i in range(max(wav_n))), dim=0) for t in padded_padding_mask])
        return audio_out

    def forward_torchscript(self, net_input):
        example_src_tokens = net_input.get("example_src_tokens")
        example_speech_masks = net_input.get("example_speech_masks")
        if example_src_tokens is not None:
            example_wav_n = [len(example_src_token) for example_src_token in example_src_tokens]
            example_stacked_input = torch.stack([tensor for lst in example_src_tokens for tensor in lst])
            example_stacked_mask = torch.stack([tensor for lst in example_speech_masks for tensor in lst])
            example_audio_out = self.forward(example_stacked_input, example_stacked_mask)
        else:
            example_audio_out = None

        src_tokens = net_input["src_tokens"]
        speech_masks = net_input["speech_masks"]

        wav_n = [len(src_token) for src_token in src_tokens]

        stacked_input = torch.stack([tensor for lst in src_tokens for tensor in lst])
        stacked_mask = torch.stack([tensor for lst in speech_masks for tensor in lst])

        audio_out = self.forward(stacked_input, stacked_mask)

        audio_out = self.split_wav_codec(audio_out, wav_n)
        if example_src_tokens is not None:
            example_audio_out = self.split_wav_codec(example_audio_out, example_wav_n)
        else:
            example_audio_out = None

        return audio_out, example_audio_out

    def reorder_encoder_out(self, encoder_out, new_order):
        new_encoder_hidden = encoder_out["encoder_out"].index_select(0, new_order)
        new_encoder_padding_mask = encoder_out["encoder_padding_mask"].to(new_order.device).index_select(0, new_order)
        return {
            "encoder_out": new_encoder_hidden,  # B T C
            "encoder_padding_mask": new_encoder_padding_mask # B T
        }

class WhisperEncoder(HFWhisperEncoder):
    """
    overwrite forward to support attention_mask
    overwrite from_pretrained to support split encoder parameters from pretrained WhisperModel
    """

    def from_pretrained(model_path):
        config = WhisperConfig.from_pretrained(model_path)

        model = WhisperEncoder(config)
        old_state_dict = torch.load(os.path.join(model_path, "pytorch_model.bin"))
        state_dict = {}
        for para_name in old_state_dict.keys():
            if "model.encoder." in para_name:
                new_name = para_name.replace("model.encoder.", "")
                state_dict[new_name] = old_state_dict[para_name]
        model.load_state_dict(state_dict)

        return model

    def forward(
        self,
        input_features,
        attention_mask=None,
        head_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        output = super().forward(
            input_features,
            attention_mask,
            head_mask,
            output_attentions,
            output_hidden_states,
            return_dict
        )
        
        last_hidden_state = output.last_hidden_state # B x T x C
        input_lengths = attention_mask.sum(-1)
        output_lengths = self._get_feat_extract_output_lengths(input_lengths)
        max_length = output_lengths.max()
        last_hidden_state = last_hidden_state[:,:max_length,:]

        return WhisperOutput(
            last_hidden_state=last_hidden_state,
            hidden_states=None,
            attentions=None,
            output_lengths=output_lengths
        )
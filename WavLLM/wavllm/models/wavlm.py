from typing import Optional, Tuple
from dataclasses import dataclass
import os

import torch
from transformers import WavLMConfig
from transformers.models.wavlm.modeling_wavlm import WavLMModel, WavLMEncoderLayerStableLayerNorm
from transformers.utils import ModelOutput

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
from ..modules.convolution import Conv1dSubsampler
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

check_fn_encoder = lambda submodule: isinstance(submodule, WavLMEncoderLayerStableLayerNorm)

def apply_fsdp_checkpointing(model):
    """apply activation checkpointing to model
    returns None as model is updated directly
    """
    print(f"--> applying fsdp activation checkpointing...")

    apply_activation_checkpointing(
        model, checkpoint_wrapper_fn=non_reentrant_wrapper, check_fn=check_fn_encoder
    )

class WavLMAdapter(nn.Module):
    def __init__(
        self,
        input_size: int,
        down_size: int,
        activation: str
    ):
        super(WavLMAdapter, self).__init__()
        self.down_layer = nn.Linear(input_size, down_size)
        self.up_layer = nn.Linear(down_size, input_size)
        self.non_linearity = F.relu if activation == 'relu' else F.gelu
        self.layer_norm = nn.LayerNorm(input_size)

    def forward(self, src_tokens):
        return self.layer_norm(self.up_layer(self.non_linearity(self.down_layer(src_tokens)))) + src_tokens

class FairseqWavLMEncoder(FairseqEncoder):
    
    def __init__(self, args):
        super().__init__(None)

        torch.set_printoptions(precision=10)

        self.model = WavLMModel.from_pretrained(args.wavlm_path)
        self.config = self.model.config
        self.wavlm_plus = args.wavlm_plus
        self.wavlm_plus_weight = args.wavlm_plus_weight
        self.wavlm_plus_1layer = args.wavlm_plus_1layer
        self.wavlm_plus_1layer_5 = args.wavlm_plus_1layer_5
        self.wavlm_plus_5layer = args.wavlm_plus_5layer

        for param in self.model.parameters():
            param.requires_grad = False
        
        self.wavlm_output_weight = args.wavlm_output_weight
        self.wavlm_output_weight_by_prompts = args.wavlm_output_weight_by_prompts
        self.wavlm_first_7_layers = args.wavlm_first_7_layers
        self.adapter = WavLMAdapter(1024, 512, "gelu")
        self.projector = nn.Linear(1024, 2048)
        self.subsample = Conv1dSubsampler(768, 512, 1024, [3, 3])
        if self.wavlm_output_weight:
            initial_weights = torch.ones(13, requires_grad=True).float()
            self.output_weights = nn.Parameter(initial_weights).to(self.projector.weight)
            self.weights_predictor = None

    def forward(
        self, src_tokens, attention_mask, prompt_embedding=None,
    ):
        extract_features = self.model.feature_extractor(src_tokens)
        extract_features = extract_features.transpose(1, 2)
        attention_mask = self.model._get_feature_vector_attention_mask(extract_features.shape[1], attention_mask)
        hidden_states, extract_features = self.model.feature_projection(extract_features)
        hidden_states = self.model._mask_hidden_states(
            hidden_states, attention_mask=attention_mask
        )

        encoder_outputs = self.model.encoder(
            hidden_states,
            attention_mask=attention_mask,
            output_attentions=False,
            output_hidden_states=True,
            return_dict=True,
        )

        src_lengths = attention_mask.sum(-1).to(torch.long)
        if self.wavlm_output_weight:
            norm_output_weights = F.softmax(self.output_weights, dim=0)
            weighted_output = [output * weight for output, weight in zip(encoder_outputs.hidden_states, norm_output_weights)]  
            wavlm_output = torch.stack(weighted_output).sum(dim=0)
        
        outputs, src_lengths = self.subsample(wavlm_output, src_lengths)
        outputs = outputs.transpose(0, 1).contiguous()
        outputs = self.adapter(outputs)
        outputs = self.projector(outputs)

        attention_mask = lengths_to_mask(src_lengths)
        return {
            "encoder_out": outputs,  # B T C
            "encoder_padding_mask": attention_mask # B T
        }


    def forward_torchscript(self, net_input, prompt_embedding=None):

        example_wavlm_src_tokens = net_input.get("example_wavlm_src_tokens")
        example_wavlm_speech_masks = net_input.get("example_wavlm_speech_masks")
        example_wavlm_audio_out = None
    
        wavlm_src_tokens = net_input["wavlm_src_tokens"]
        wavlm_speech_masks = net_input["wavlm_speech_masks"]

        wavlm_input = torch.stack(wavlm_src_tokens, dim=0)
        wavlm_speech_masks_input = torch.stack(wavlm_speech_masks, dim=0)
        wavlm_audio_out = self.forward(src_tokens=wavlm_input, attention_mask=wavlm_speech_masks_input, prompt_embedding=prompt_embedding)

        return wavlm_audio_out, example_wavlm_audio_out

    def reorder_encoder_out(self, encoder_out, new_order):
        new_encoder_hidden = encoder_out["encoder_out"].index_select(0, new_order)
        new_encoder_padding_mask = encoder_out["encoder_padding_mask"].to(new_order.device).index_select(0, new_order)
        return {
            "encoder_out": new_encoder_hidden,  # B T C
            "encoder_padding_mask": new_encoder_padding_mask # B T
        }

#!/usr/bin/env python3

import logging
import math
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from collections import OrderedDict

from fairseq import checkpoint_utils, utils
from fairseq.data.data_utils import lengths_to_padding_mask
from fairseq.models import (
    FairseqEncoderModel,
    FairseqDecoder,
    FairseqEncoderDecoderModel,
    BaseFairseqModel,
    register_model,
    register_model_architecture,
)
from fairseq.models.speech_to_text.hub_interface import S2THubInterface
from fairseq.models.transformer import Embedding, TransformerDecoder
from fairseq.modules import (
    FairseqDropout,
    LayerNorm,
    PositionalEmbedding,
    TransformerEncoderLayer,
)
from .llama import LLaMADecoder
from .whisper_encoder import FairseqWhisperEncoder, WhisperAdapter
from .wavlm import FairseqWavLMEncoder
from omegaconf import II
from fairseq.dataclass import ChoiceEnum, FairseqDataclass

logger = logging.getLogger(__name__)

class Conv1dSubsampler(nn.Module):
    """Convolutional subsampler: a stack of 1D convolution (along temporal
    dimension) followed by non-linear activation via gated linear units
    (https://arxiv.org/abs/1911.08460)

    Args:
        in_channels (int): the number of input channels
        mid_channels (int): the number of intermediate channels
        out_channels (int): the number of output channels
        kernel_sizes (List[int]): the kernel size for each convolutional layer
    """

    def __init__(
        self,
        in_channels: int,
        mid_channels: int,
        out_channels: int,
        kernel_sizes: List[int] = (3, 3),
        strides: List[int] = (2, 2),
    ):
        super(Conv1dSubsampler, self).__init__()
        self.strides = strides
        self.n_layers = len(kernel_sizes)
        self.conv_layers = nn.ModuleList(
            nn.Conv1d(
                in_channels if i == 0 else mid_channels // 2,
                mid_channels if i < self.n_layers - 1 else out_channels * 2,
                k,
                stride=s,
                padding=k // 2,
            )
            for i, (k, s) in enumerate(zip(kernel_sizes, strides))
        )

    def get_out_seq_lens_tensor(self, in_seq_lens_tensor):
        out = in_seq_lens_tensor.clone()
        for i in range(self.n_layers):
            out = ((out.float() - 1) / self.strides[i] + 1).floor().long()
        return out

    def forward(self, src_tokens, src_lengths):
        bsz, in_seq_len, _ = src_tokens.size()  # B x T x (C x D)
        x = src_tokens.transpose(1, 2).contiguous()  # -> B x (C x D) x T
        for conv in self.conv_layers:
            x = conv(x)
            x = nn.functional.glu(x, dim=1)
        _, _, out_seq_len = x.size()
        x = x.transpose(1, 2).transpose(0, 1).contiguous()  # -> T x B x (C x D)
        return x, self.get_out_seq_lens_tensor(src_lengths)

@dataclass
class DecoderConfig(FairseqDataclass):
    # Text
    vocab_size: int = -1

    # Fairscale
    checkpoint_activations: bool = False
    fsdp: bool = False
    ddp_rank: int = 0
    flash_attention: bool = False
    sope_rel_pos: bool = False
    scale_length: int = 2048


    def override(self, args):
        for hp in self.__dict__.keys():
            if getattr(args, hp, None) is not None:
                self.__dict__[hp] = getattr(args, hp, None)


@dataclass
class SpeechLLMMOdelConfig(DecoderConfig):
    llama_checkpoint: str = ""
    speechllm_checkpoint: str = ""
    vicuna_model_path: str = 'lmsys/vicuna-7b-v1.5'
    # n_xatt: int = 16
    n_xatt: int = field(
        default=16,
        metadata={"help": "the number of xatt"},
    )
    d_att: int = field(
        default=256,
        metadata={"help": "the dimension of xatt"},
    )
    d_ffn: int = field(
        default=256,
        metadata={"help": "the dimension of ffn in xatt"},
    )
    freeze_gpt: bool = field(
        default=True
    )
    freeze_audio_encoder: bool = field(
        default=True
    )
    whisper_path: str = field(
        default="openai/whisper-large-v2"
        # default="openai/whisper-small.en"
    )
    wavlm_path: str = field(
        default="microsoft/wavlm-base"
        # default="openai/whisper-small.en"
    )
    wavlm_output_weight: bool = field(
        default=False
    )
    wavlm_output_weight_by_prompts: bool = field(
        default=False
    )
    wavlm_plus: bool = field(
        default=False
    )
    wavlm_plus_weight: bool = field(
        default=False
    )
    wavlm_plus_1layer: bool = field(
        default=False
    )
    wavlm_plus_1layer_5: bool = field(
        default=False
    )
    wavlm_plus_5layer: bool = field(
        default=False
    )
    wavlm_first_7_layers: bool = field(
        default=False
    )
    load_pretrained_encoder_from: str = field(
        default=""
    )
    use_lora: bool = field(
        default=False
    )

class TextEmbedding(nn.Embedding):
    def reset_parameters(self):
        nn.init.normal_(self.weight, mean=0, std=self.embedding_dim**-0.5)
        self._fill_padding_idx_with_zero()


@register_model("speechllm_model", dataclass=SpeechLLMMOdelConfig)
class SpeechLLMModel(BaseFairseqModel):
    """Adapted Transformer model (https://arxiv.org/abs/1706.03762) for
    speech-to-text tasks. The Transformer encoder/decoder remains the same.
    A trainable input subsampler is prepended to the Transformer encoder to
    project inputs into the encoder dimension as well as downsample input
    sequence for computational efficiency."""

    def __init__(self, cfg: SpeechLLMMOdelConfig, task):
        super().__init__()
        logger.info(f"SpeechLLMModel Config: {cfg}")
        self.cfg = cfg
        self.task = task
        cfg.freeze_audio_encoder = task.cfg.freeze_audio_encoder
        if task.cfg.llama_2:
            cfg.llama_checkpoint = task.cfg.llama_2_path

        
        self.audio_encoder = self.build_audio_encoder(cfg, task)
        if self.task.cfg.use_wavlm:
            self.cfg.wavlm_output_weight = task.cfg.wavlm_output_weight
            self.cfg.wavlm_output_weight_by_prompts = task.cfg.wavlm_output_weight_by_prompts
            self.cfg.wavlm_plus = task.cfg.wavlm_plus
            self.cfg.wavlm_plus_weight = task.cfg.wavlm_plus_weight
            self.cfg.wavlm_plus_1layer = task.cfg.wavlm_plus_1layer
            self.cfg.wavlm_plus_1layer_5 = task.cfg.wavlm_plus_1layer_5
            self.cfg.wavlm_plus_5layer = task.cfg.wavlm_plus_5layer
            self.cfg.wavlm_first_7_layers = task.cfg.wavlm_first_7_layers
            self.wavlm_encoder = self.build_wavlm_encoder(cfg, task)
            self.wavlm_audio_proj = nn.Linear(4096, 4096)
        self.gpt_model = self.build_gpt_model(cfg, task)
        self.audio_proj = nn.Linear(2048, 4096)

    @classmethod
    def build_audio_encoder(cls, cfg, task):
        if task.cfg.is_whisper:
            if not task.cfg.whisper_with_decoder:
                cfg.whisper_path = "openai/whisper-large-v2"
                encoder = FairseqWhisperEncoder(cfg)

        pretraining_path = getattr(cfg, "load_pretrained_encoder_from", None)
        if pretraining_path is not None and len(pretraining_path) > 0:
            if not Path(pretraining_path).exists():
                logger.warning(
                    f"skipped pretraining because {pretraining_path} does not exist"
                )
            else:
                state = torch.load(pretraining_path, map_location="cpu")
                
                component_state_dict = OrderedDict()
                component_type = "audio_encoder"
                for key in state["model"].keys():
                    if key.startswith(component_type):
                        # encoder.input_layers.0.0.weight --> input_layers.0.0.weight
                        component_subkey = key[len(component_type) + 1 :]
                        component_state_dict[component_subkey] = state["model"][key]
                encoder.load_state_dict(component_state_dict, strict=True)
            
                logger.info(f"loaded pretrained encoder from: {pretraining_path}")
        return encoder

    @classmethod
    def build_wavlm_encoder(cls, cfg, task):
        cfg.wavlm_path = "microsoft/wavlm-base"
        encoder = FairseqWavLMEncoder(cfg)
        return encoder

    @classmethod
    def build_gpt_model(cls, cfg, task):
        gpt_model = LLaMADecoder(dictionary=task.tgt_dict,
                                llama_checkpoint=cfg.llama_checkpoint,
                                n_xatt=cfg.n_xatt,
                                d_att=cfg.d_att,
                                d_ffn=cfg.d_ffn,
                                use_lora=task.cfg.use_lora,
                                lora_only_qv=task.cfg.lora_only_qv,
                                lora_scale_train=task.cfg.lora_scale_train,
                                lora_scale_index=task.cfg.lora_scale_index,
                                lora_scale_random=task.cfg.lora_scale_random,
                                lora_task_index=task.cfg.lora_task_index,
                                lora_moe=task.cfg.lora_moe,
                                lora_moe_scaling=task.cfg.lora_moe_scaling,
                                lora_moe_n_experts=task.cfg.lora_moe_n_experts,
                                lora_r=task.cfg.lora_r,
                                lora_alpha=task.cfg.lora_alpha,
                                enable_fsdp=task.cfg.enable_fsdp,
                                use_xformers=task.cfg.use_xformers,
                                second_stage_update_scale=task.cfg.second_stage_update_scale,
                                second_stage_fix_lora=task.cfg.second_stage_fix_lora,
                                scale_only_one=task.cfg.scale_only_one,
                                scale_with_audio=task.cfg.scale_with_audio,
                                scale_0_1=task.cfg.scale_0_1,
                                scale_predict_time=task.cfg.scale_predict_time,
                                scale_predict_all_dim=task.cfg.scale_predict_all_dim,
                                scale_predict_all_dim_each_layer=task.cfg.scale_predict_all_dim_each_layer,
                                prompt_loss=task.cfg.prompt_loss,
                                use_llama_adapter=task.cfg.use_llama_adapter,)
        return gpt_model

    @classmethod
    def build_model(cls, cfg, task):
        """Build a new model instance."""
        return cls(cfg, task)

    def get_targets(self, sample, net_output):
        return sample['target'][sample['net_input']['target_masks']]

    def get_normalized_probs(
        self,
        net_output: Tuple[Tensor, Optional[Dict[str, List[Optional[Tensor]]]]],
        log_probs: bool,
        sample: Optional[Dict[str, Tensor]] = None,
    ):
        # net_output['encoder_out'] is a (B, T, D) tensor
        #print ("net_output: ", net_output)
        lprobs = self.get_normalized_probs_scriptable(net_output[0], log_probs, sample)
        lprobs.batch_first = True
        return lprobs
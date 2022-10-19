# ----------------------------------------------------------------------------
# SpeechUT: Bridging Speech and Text with Hidden-Unit for Encoder-Decoder Based Speech-Text Pre-training (https://arxiv.org/abs/2210.03730)
# Github source: https://github.com/microsoft/SpeechT5/tree/main/SpeechUT
# Code based on fairseq: https://github.com/facebookresearch/fairseq/tree/272c4c5197250997148fb12c0db6306035f166a4
# 
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# ----------------------------------------------------------------------------

import contextlib
import torch
import torch.nn as nn
from argparse import Namespace
from dataclasses import dataclass, field
from typing import Any
from fairseq import checkpoint_utils, tasks, utils
from fairseq.models import BaseFairseqModel, register_model
from fairseq.models.fairseq_decoder import FairseqDecoder
from fairseq.models.fairseq_encoder import FairseqEncoder
from fairseq.tasks import FairseqTask
from fairseq.dataclass import ChoiceEnum
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq.data.data_utils import lengths_to_padding_mask

from fairseq.models.hubert import HubertAsrConfig, HubertCtc, HubertEncoder
from speechut.modules import TransformerDecoderBaseScriptable

@dataclass
class SpeechUTASRConfig(HubertAsrConfig):
    add_decoder: bool = field(
        default=True,
        metadata={"help": "add decoder for fine-tune"},
    )
    decoder_layerdrop: float = field(
        default=0.1,
        metadata={"help": "probability of dropping a decoder layer in hubert"},
    )

@register_model("speechut_asr", dataclass=SpeechUTASRConfig)
class SpeechUTASR(BaseFairseqModel):
    """
    A encoder-ctc-decoder model if cfg.add_decoder is True, or a encoder-ctc model
    """
    def __init__(self, cfg: SpeechUTASRConfig, encoder: FairseqEncoder):
        super().__init__()
        self.cfg = cfg
        self.encoder = encoder
        if not cfg.add_decoder:
            self.encoder.w2v_model.decoder = None

    def upgrade_state_dict_named(self, state_dict, name):
        super().upgrade_state_dict_named(state_dict, name)
        return state_dict
    
    @classmethod
    def build_model(cls, cfg: SpeechUTASRConfig, task: FairseqTask):
        """Build a new model instance."""
        encoder = SpeechUTEncoder(cfg, task)
        return cls(cfg, encoder)

    def forward(self, source, padding_mask, prev_output_tokens, **kwargs):
        encoder_out = self.encoder(source, padding_mask, **kwargs)

        x = self.encoder.final_dropout(encoder_out['encoder_out'][0])  # (T, B, C)
        if self.encoder.proj:
            x = self.encoder.proj(x)
        if self.encoder.conv_ctc_proj:
            padding_mask = self.encoder.w2v_model.downsample_ctc_padding_mask(encoder_out["encoder_padding_mask"][0])
        else:
            padding_mask = encoder_out["encoder_padding_mask"]

        decoder_out = self.encoder.w2v_model.decoder(
            prev_output_tokens, encoder_out=encoder_out, **kwargs
        ) if self.cfg.add_decoder else None
        
        return {
            "encoder_out_ctc": x,           # (T, B, C), for CTC loss
            "padding_mask": padding_mask,   # (B, T), for CTC loss
            "decoder_out": decoder_out,     # for ED loss
        }
    
    def forward_decoder(self, prev_output_tokens, **kwargs):
        return self.encoder.w2v_model.decoder(prev_output_tokens, **kwargs)

    def get_logits(self, net_output):
        """For CTC decoding"""
        logits = net_output["encoder_out"]
        padding = net_output["encoder_padding_mask"]
        if padding is not None and padding.any():
            padding = padding.T
            logits[padding][..., 0] = 0
            logits[padding][..., 1:] = float("-inf")

        return logits

    def get_normalized_probs(self, net_output, log_probs, sample=None):
        """For 1) computing CTC loss, 2) decoder decoding."""

        if "encoder_out_ctc" in net_output:
            logits = net_output["encoder_out_ctc"]
        else:
            return self.encoder.w2v_model.decoder.get_normalized_probs(net_output, log_probs, sample)
        
        if isinstance(logits, list):
            logits = logits[0]

        if log_probs:
            return utils.log_softmax(logits.float(), dim=-1)
        else:
            return utils.softmax(logits.float(), dim=-1)

    @property
    def decoder(self):
        self.encoder.w2v_model.decoder


class SpeechUTEncoder(FairseqEncoder):
    """
    Modified from fairseq.models.hubert.hubert_asr.HubertEncoder
    1. make it compatible with encoder-decoder model
    """
    def __init__(self, cfg: SpeechUTASRConfig, task):
        self.apply_mask = cfg.apply_mask

        arg_overrides = {
            "dropout": cfg.dropout,
            "activation_dropout": cfg.activation_dropout,
            "dropout_input": cfg.dropout_input,
            "attention_dropout": cfg.attention_dropout,
            "mask_length": cfg.mask_length,
            "mask_prob": cfg.mask_prob,
            "mask_selection": cfg.mask_selection,
            "mask_other": cfg.mask_other,
            "no_mask_overlap": cfg.no_mask_overlap,
            "mask_channel_length": cfg.mask_channel_length,
            "mask_channel_prob": cfg.mask_channel_prob,
            "mask_channel_selection": cfg.mask_channel_selection,
            "mask_channel_other": cfg.mask_channel_other,
            "no_mask_channel_overlap": cfg.no_mask_channel_overlap,
            "encoder_layerdrop": cfg.layerdrop,
            "feature_grad_mult": cfg.feature_grad_mult,
            "add_decoder": cfg.add_decoder,
            "text_transformer": {
                "dropout": cfg.dropout,
                "activation_dropout": cfg.activation_dropout,
                "attention_dropout": cfg.attention_dropout,
                "encoder":{
                    "layerdrop": cfg.layerdrop,
                },
                "decoder":{
                    "layerdrop": cfg.decoder_layerdrop,
                },
            }
        }

        if cfg.w2v_args is None:
            state = checkpoint_utils.load_checkpoint_to_cpu(cfg.w2v_path, arg_overrides)
            w2v_args = state.get("cfg", None)
            if w2v_args is None:
                w2v_args = convert_namespace_to_omegaconf(state["args"])
            cfg.w2v_args = w2v_args
        else:
            state = None
            w2v_args = cfg.w2v_args
            if isinstance(w2v_args, Namespace):
                cfg.w2v_args = w2v_args = convert_namespace_to_omegaconf(w2v_args)

        assert cfg.normalize == w2v_args.task.normalize, (
            "Fine-tuning works best when data normalization is the same. "
            "Please check that --normalize is set or unset for "
            "both pre-training and here"
        )

        w2v_args.task.data = cfg.data
        pretrain_task = tasks.setup_task(w2v_args.task)
        if state is not None and "task_state" in state:
            # This will load the stored "dictionaries" object
            pretrain_task.load_state_dict(state["task_state"])
        else:
            pretrain_task.load_state_dict(task.state_dict())

        model = pretrain_task.build_model(w2v_args.model, from_checkpoint=True)
        if state is not None and not cfg.no_pretrained_weights:
            # set strict=False because we omit some modules
            model.load_state_dict(state["model"], strict=False)

        model.remove_pretraining_modules()

        super().__init__(pretrain_task.source_dictionary)

        d = w2v_args.model.encoder_embed_dim

        self.w2v_model = model

        self.final_dropout = nn.Dropout(cfg.final_dropout)
        self.freeze_finetune_updates = cfg.freeze_finetune_updates
        self.num_updates = 0

        self.conv_ctc_proj = False
        if task.target_dictionary is not None:
            if hasattr(self.w2v_model, "unit_encoder_ctc_head"):
                self.proj = self.w2v_model.unit_encoder_ctc_head
                self.conv_ctc_proj = True
            else:
                self.proj = Linear(d, len(task.target_dictionary))
        elif getattr(cfg, "decoder_embed_dim", d) != d:
            self.proj = Linear(d, cfg.decoder_embed_dim)
        else:
            self.proj = None
    
    def forward(self, source, padding_mask, tbc=True, **kwargs):
        w2v_args = {
            "source": source,
            "padding_mask": padding_mask,
            "mask": self.apply_mask and self.training,
        }
        ft = self.freeze_finetune_updates <= self.num_updates
        with torch.no_grad() if not ft else contextlib.ExitStack():
            x, padding_mask = self.w2v_model.extract_features(**w2v_args)
            if tbc:
                # B x T x C -> T x B x C
                x = x.transpose(0, 1)
        return {
            "encoder_out": [x],  # T x B x C
            "encoder_padding_mask": [padding_mask],  # B x T
        }
    
    def forward_torchscript(self, net_input):
        """A TorchScript-compatible version of forward.

        Forward the encoder out.
        """
        x, padding_mask = self.w2v_model.extract_features(**net_input, mask=False)
        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        encoder_out = {
            "encoder_out" : [x],
            "encoder_padding_mask" : [padding_mask],
        }
        return encoder_out

    def reorder_encoder_out(self, encoder_out, new_order):
        if encoder_out["encoder_out"] is not None:
            encoder_out["encoder_out"] = [
                x.index_select(1, new_order) for x in encoder_out["encoder_out"]
            ]
        if encoder_out["encoder_padding_mask"] is not None:
            encoder_out["encoder_padding_mask"] = [
                x.index_select(0, new_order) for x in encoder_out["encoder_padding_mask"]
            ]
        return encoder_out


def Embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim**-0.5)
    nn.init.constant_(m.weight[padding_idx], 0)
    return m

def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.0)
    return m


def Embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim**-0.5)
    nn.init.constant_(m.weight[padding_idx], 0)
    return m

def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.0)
    return m

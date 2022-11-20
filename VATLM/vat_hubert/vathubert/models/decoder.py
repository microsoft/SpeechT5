# ----------------------------------------------------------------------------
# VatLM: Visual-Audio-Text Pre-Training  with Unified Masked Prediction for Speech Representation Learning
# Github source: https://github.com/microsoft/SpeechT5/tree/main/VATLM
# Code based on fairseq: https://github.com/facebookresearch/fairseq and av_hubert: https://github.com/facebookresearch/av_hubert
# 
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# ----------------------------------------------------------------------------

from argparse import Namespace
import contextlib
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from omegaconf import MISSING, II, open_dict
from typing import Any, Optional

from fairseq import checkpoint_utils, tasks, utils
from fairseq.dataclass import FairseqDataclass
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq.tasks import FairseqTask
from fairseq.models import (
    BaseFairseqModel,
    FairseqEncoder,
    FairseqEncoderDecoderModel,
    FairseqIncrementalDecoder,
    register_model,
)
# from fairseq.models.wav2vec.wav2vec2 import MASKING_DISTRIBUTION_CHOICES
from fairseq.modules import (
    LayerNorm,
    PositionalEmbedding,
    TransformerDecoderLayer,
)


class TransformerDecoder(FairseqIncrementalDecoder):
    """
    Transformer decoder consisting of *args.decoder_layers* layers. Each layer
    is a :class:`TransformerDecoderLayer`.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): decoding dictionary
        embed_tokens (torch.nn.Embedding): output embedding
        no_encoder_attn (bool, optional): whether to attend to encoder outputs
            (default: False).
    """

    def __init__(
        self,
        cfg,
        dictionary,
        embed_tokens,
        no_encoder_attn=False,
    ):
        super().__init__(dictionary)

        self.dropout = cfg.decoder_dropout
        self.share_input_output_embed = cfg.share_decoder_input_output_embed

        input_embed_dim = embed_tokens.embedding_dim
        embed_dim = cfg.decoder_embed_dim
        self.output_embed_dim = cfg.decoder_embed_dim

        self.layerdrop = cfg.decoder_layerdrop

        padding_idx = embed_tokens.padding_idx
        self.max_target_positions = cfg.max_target_positions

        self.embed_tokens = embed_tokens
        # self.embed_scale = math.sqrt(embed_dim)  # todo: try with input_embed_dim
        self.embed_scale = 1.0 if cfg.no_scale_embedding else math.sqrt(embed_dim)

        self.project_in_dim = (
            Linear(input_embed_dim, embed_dim, bias=False)
            if embed_dim != input_embed_dim
            else None
        )

        self.embed_positions = (
            PositionalEmbedding(
                cfg.max_target_positions,
                embed_dim,
                padding_idx,
                learned=cfg.decoder_learned_pos,
            )
            if not cfg.no_token_positional_embeddings
            else None
        )

        # TODO: update this when transformer gets converted to dataclass configs
        transformer_cfg = copy.deepcopy(cfg)
        # with open_dict(transformer_cfg):
        transformer_cfg.dropout = transformer_cfg.decoder_dropout
        transformer_cfg.attention_dropout = (
            transformer_cfg.decoder_attention_dropout
        )
        transformer_cfg.activation_dropout = (
            transformer_cfg.decoder_activation_dropout
        )

        self.layers = nn.ModuleList([])
        self.layers.extend(
            [
                TransformerDecoderLayer(transformer_cfg, no_encoder_attn)
                for _ in range(transformer_cfg.decoder_layers)
            ]
        )

        if not self.share_input_output_embed:
            self.embed_out = nn.Parameter(
                torch.Tensor(len(dictionary), self.output_embed_dim)
            )
            nn.init.normal_(self.embed_out, mean=0, std=self.output_embed_dim ** -0.5)

        if transformer_cfg.decoder_normalize_before:
            self.layer_norm = LayerNorm(embed_dim)
        else:
            self.layer_norm = None

    def forward(
        self, prev_output_tokens, encoder_out=None, incremental_state=None, **unused
    ):
        """
        Args:
            prev_output_tokens (LongTensor): previous decoder outputs of shape
                `(batch, tgt_len)`, for teacher forcing
            encoder_out (Tensor, optional): output from the encoder, used for
                encoder-side attention
            incremental_state (dict): dictionary used for storing state during
                :ref:`Incremental decoding`

        Returns:
            tuple:
                - the decoder's output of shape `(batch, tgt_len, vocab)`
                - a dictionary with any model-specific outputs
        """
        prev_output_tokens = prev_output_tokens.long()
        x, extra = self.extract_features(
            prev_output_tokens, encoder_out, incremental_state
        )
        x = self.output_layer(x)
        return x, extra

    def extract_features(
        self, prev_output_tokens, encoder_out=None, incremental_state=None, **unused
    ):
        """
        Similar to *forward* but only return features.

        Returns:
            tuple:
                - the decoder's features of shape `(batch, tgt_len, embed_dim)`
                - a dictionary with any model-specific outputs
        """

        # embed positions
        positions = (
            self.embed_positions(
                prev_output_tokens, incremental_state=incremental_state
            )
            if self.embed_positions is not None
            else None
        )

        if incremental_state is not None:
            prev_output_tokens = prev_output_tokens[:, -1:]
            if positions is not None:
                positions = positions[:, -1:]

        # embed tokens and positions
        x = self.embed_scale * self.embed_tokens(prev_output_tokens)

        if self.project_in_dim is not None:
            x = self.project_in_dim(x)

        if positions is not None:
            x += positions
        x = F.dropout(x, p=self.dropout, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)
        attn = None

        inner_states = [x]

        # decoder layers
        for layer in self.layers:
            dropout_probability = np.random.random()
            if not self.training or (dropout_probability > self.layerdrop):
                x, attn, _ = layer(
                    x,
                    encoder_out["encoder_out"] if encoder_out is not None else None,
                    encoder_out["padding_mask"] if encoder_out is not None else None,
                    incremental_state,
                    self_attn_mask=self.buffered_future_mask(x)
                    if incremental_state is None
                    else None,
                )
                inner_states.append(x)

        if self.layer_norm:
            x = self.layer_norm(x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        return x, {"attn": attn, "inner_states": inner_states}

    def output_layer(self, features, **kwargs):
        """Project features to the vocabulary size."""
        # project back to size of vocabulary
        emb_mat = self.embed_tokens.weight if self.share_input_output_embed else self.embed_out
        return torch.matmul(features, emb_mat.transpose(0, 1))
        # if self.share_input_output_embed:
        #     return F.linear(features, self.embed_tokens.weight)
        # else:
        #     return F.linear(features, self.embed_out)

    def max_positions(self):
        """Maximum output length supported by the decoder."""
        if self.embed_positions is None:
            return self.max_target_positions
        return min(self.max_target_positions, self.embed_positions.max_positions)

    def buffered_future_mask(self, tensor):
        dim = tensor.size(0)
        if (
            not hasattr(self, "_future_mask")
            or self._future_mask is None
            or self._future_mask.device != tensor.device
            or self._future_mask.size(0) < dim
        ):
            self._future_mask = torch.triu(
                utils.fill_with_neg_inf(tensor.new(dim, dim)), 1
            )
        return self._future_mask[:dim, :dim]

    def upgrade_state_dict_named(self, state_dict, name):
        return state_dict


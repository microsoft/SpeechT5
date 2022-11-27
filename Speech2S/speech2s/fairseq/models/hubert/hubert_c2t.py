# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import contextlib
from argparse import Namespace
from typing import Any, Optional

import torch
import torch.nn as nn
from dataclasses import dataclass, field
from fairseq import checkpoint_utils, tasks, utils
from fairseq.dataclass import FairseqDataclass
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq.models import BaseFairseqModel, FairseqEncoder, register_model
from fairseq.models.hubert.hubert import MASKING_DISTRIBUTION_CHOICES
from fairseq.tasks import FairseqTask
from omegaconf import II, open_dict

from .hubert_asr import HubertAsrConfig
from fairseq.models.transformer import TransformerConfig
logger = logging.getLogger(__name__)


@dataclass
class HubertMTConfig(HubertAsrConfig):
    load_pretrained_mbart_from: Optional[str] = field(
        default=None,
        metadata={
            "help": "model to take text encoder decoder weights from (for initialization)"
        },
    )
    use_rel_pos_enc: bool = field(
        default=True,
        metadata={"help": "whether to use relative positional encoding"},
    )
    text_transformer_encoder_layers: int = field(
        default=12,
        metadata={"help": "reset text_transformer_encoder_layers"},
    )


@register_model("hubert_c2t", dataclass=HubertMTConfig)
class HubertMT(BaseFairseqModel):
    def __init__(self, cfg: HubertMTConfig, w2v_encoder: BaseFairseqModel):
        super().__init__()
        self.cfg = cfg
        self.w2v_encoder = w2v_encoder

    def upgrade_state_dict_named(self, state_dict, name):
        super().upgrade_state_dict_named(state_dict, name)
        return state_dict

    @classmethod
    def build_model(cls, cfg: HubertMTConfig, task: FairseqTask):
        """Build a new model instance."""
        w2v_encoder = HubertEncoder(cfg, task.target_dictionary)
        return cls(cfg, w2v_encoder)

    def get_normalized_probs(self, net_output, log_probs, sample=None):
        """Get normalized probabilities (or log probs) from a net's output."""
        if "decoder_out" in net_output:
            return self.w2v_encoder.get_normalized_probs_decoder(net_output["decoder_out"], log_probs, sample)

        assert "encoder_out" not in net_output
        if "encoder_out" not in net_output:
            return self.w2v_encoder.get_normalized_probs_decoder(net_output, log_probs, sample)


    def get_logits(self, net_output):
        logits = net_output["encoder_out"]
        padding = net_output["encoder_padding_mask"]
        if padding is not None and padding.any():
            padding = padding.T
            logits[padding][..., 0] = 0
            logits[padding][..., 1:] = float("-inf")

        return logits

    def forward(self, **kwargs):
        x = self.w2v_encoder(**kwargs)
        return x

    @property
    def encoder(self):
        return self.w2v_encoder

    def reorder_encoder_out(self, encoder_out, new_order):
        return self.encoder.reorder_encoder_out(encoder_out, new_order)

    @property
    def decoder(self):
        return self.w2v_encoder.w2v_model.decoder


class HubertEncoder(FairseqEncoder):
    def __init__(self, cfg: HubertMTConfig, tgt_dict=None):
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
            "decoder_layerdrop": cfg.decoder_layerdrop,
            "feature_grad_mult": cfg.feature_grad_mult,
            "decoder_dict_size": -1,
            "add_text_modality": True,
            "add_text_encoder": True,
            "load_pretrained_mbart_from": None,
            "load_pretrained_w2v_from": None,
            "text_transformer": {
                "encoder":{
                    "layers": cfg.text_transformer_encoder_layers,
                    "layerdrop": cfg.layerdrop,
                }, 
                'dropout': cfg.dropout,
                'attention_dropout': cfg.attention_dropout,
                'activation_dropout': cfg.activation_dropout,
                }
            }
        
        if cfg.no_pretrained_weights:
            arg_overrides["use_rel_pos_enc"] = cfg.use_rel_pos_enc

        if cfg.w2v_args is None:
            state = checkpoint_utils.load_checkpoint_to_cpu(
                cfg.w2v_path, arg_overrides
            )
            w2v_args = state.get("cfg", None)
            if w2v_args is None:
                w2v_args = convert_namespace_to_omegaconf(state["args"])
            cfg.w2v_args = w2v_args
        else:
            state = None
            w2v_args = cfg.w2v_args
            if isinstance(w2v_args, Namespace):
                cfg.w2v_args = w2v_args = convert_namespace_to_omegaconf(w2v_args)
        
        w2v_args.task.data = cfg.data
        w2v_args.task.text_cfg.text_data = cfg.data

        ### update configs that have not been override
        if "data_config" in w2v_args.task.text_cfg:
            w2v_args.task.text_cfg.data_config = None
        if not "add_text_modality" in w2v_args.model:
            with open_dict(w2v_args.model): 
                w2v_args.model.add_text_modality = True
        if not "add_text_encoder" in w2v_args.model:
            with open_dict(w2v_args.model): 
                w2v_args.model.add_text_encoder = True
        if not "encoder_dict_size" in w2v_args.model:
            with open_dict(w2v_args.model): 
                w2v_args.model.encoder_dict_size = 504

        task = tasks.setup_task(w2v_args.task)

        if state is not None and "task_state" in state:
            # This will load the stored "dictionaries" object
            task.load_state_dict(state["task_state"])

        model = task.build_model(w2v_args.model)


        ### load mbart if specificed
        if cfg.load_pretrained_mbart_from is not None and cfg.no_pretrained_weights:
            logger.info("Loading mbart....")
            mbart_model_state = model.load_checkpoint(cfg.load_pretrained_mbart_from)
            model.text_encoder = model.load_pretrained_component_from_model(
                component=model.text_encoder, state=mbart_model_state
            )
            model.decoder = model.load_pretrained_component_from_model(
                component=model.decoder, state=mbart_model_state
            )
        
        if state is not None and not cfg.no_pretrained_weights:
            logger.info("Loading pre-trained models....")
            for name in ("text_encoder.embed_tokens.weight", 
                "text_encoder.embed_tokens.weight", "decoder.embed_tokens_list.0.weight", "decoder.embed_tokens_list.1.weight", 
                "decoder.output_projection.weight", "decoder.output_projection.0.weight", "decoder.output_projection.1.weight"):
                if name in state["model"]:
                    logger.info(f"Excluding {name}..")
                    del state["model"][name]
            try:
                model.load_state_dict(state["model"], strict=True)
            except Exception as e:
                logger.warn(e)
                model.load_state_dict(state["model"], strict=False)
        
        ### remove_pretraining_modules model.remove_pretraining_modules()
        model.target_glu = None
        model.final_proj = None
        model.feature_extractor = None
        model.post_extract_proj = None
        model.encoder = None
        model.dropout_input = None
        model.dropout_features = None
        model.layer_norm = None

        dropout_keys = [ n for n in w2v_args.model.text_transformer if n.find("drop") >= 0 ]
        for key in dropout_keys:
            logger.info(f"{key}: {w2v_args.model.text_transformer[key]}")

        super().__init__(task.source_dictionary)

        d = w2v_args.model.encoder_embed_dim
        
        self.w2v_model = model

        self.final_dropout = nn.Dropout(cfg.final_dropout)
        self.freeze_finetune_updates = cfg.freeze_finetune_updates
        self.freeze_decoder_updates = cfg.freeze_decoder_updates
        self.num_updates = 0


    def set_num_updates(self, num_updates):
        """Set the number of parameters updates."""
        super().set_num_updates(num_updates)
        self.num_updates = num_updates

    def forward(self, src_tokens, src_lengths, prev_output_tokens, tbc=True, **kwargs):

        # ft = self.freeze_finetune_updates <= self.num_updates
        w2v_args = {
            "src_tokens": src_tokens,
            "src_lengths": src_lengths,
            "mask": self.apply_mask and self.training,
            "prev_output_tokens": prev_output_tokens,
            "text_modal_idx": 1,
        }

        results = self.w2v_model(**w2v_args)
        return results

    def get_normalized_probs_decoder(self, net_output, log_probs, sample=None):
        # net_output['encoder_out'] is a (B, T, D) tensor
        return self.w2v_model.get_normalized_probs(net_output, log_probs, sample)

    def reorder_encoder_out(self, encoder_out, new_order):
        if encoder_out["encoder_out"] is not None:
            if isinstance(encoder_out["encoder_out"], list):
                encoder_out["encoder_out"] = (
                    [] if len(encoder_out["encoder_out"]) == 0
                    else [x.index_select(1, new_order) for x in encoder_out["encoder_out"]]
                )
            else:
                encoder_out["encoder_out"] = encoder_out[
                    "encoder_out"
                ].index_select(1, new_order)
        if encoder_out["encoder_padding_mask"] is not None:
            if isinstance(encoder_out["encoder_padding_mask"], list):
                encoder_out["encoder_padding_mask"] = (
                    [] if len(encoder_out["encoder_padding_mask"]) == 0
                    else [x.index_select(0, new_order) for x in encoder_out["encoder_padding_mask"]]
                )
            else:
                encoder_out["encoder_padding_mask"] = encoder_out[
                    "encoder_padding_mask"
                ].index_select(0, new_order)
        if "decoder_out" in encoder_out and encoder_out["decoder_out"] is not None:
            if isinstance(encoder_out["decoder_out"], list):
                encoder_out["decoder_out"] = (
                    [] if len(encoder_out["decoder_out"]) == 0
                    else [x.index_select(0, new_order) for x in encoder_out["decoder_out"]]
                )
            else:
                encoder_out["decoder_out"] = encoder_out[
                    "decoder_out"
                ].index_select(0, new_order)
        if "encoder_out_for_ctc" in encoder_out and encoder_out["encoder_out_for_ctc"] is not None:
            if isinstance(encoder_out["encoder_out_for_ctc"], list):
                encoder_out["encoder_out_for_ctc"] = (
                    [] if len(encoder_out["encoder_out_for_ctc"]) == 0
                    else [x.index_select(1, new_order) for x in encoder_out["encoder_out_for_ctc"]]
                )
            else:
                encoder_out["encoder_out_for_ctc"] = encoder_out[
                    "encoder_out_for_ctc"
                ].index_select(1, new_order)

        return encoder_out

    def forward_torchscript(self, net_input):
        """A TorchScript-compatible version of forward.

        Encoders which use additional arguments may want to override
        this method for TorchScript compatibility.
        """
        encoder_out = self.w2v_model.forward_torchscript(net_input)
        if "encoder_out_for_ctc" in encoder_out:
            del encoder_out['encoder_out_for_ctc']
        
        return encoder_out

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        return None

    def upgrade_state_dict_named(self, state_dict, name):
        return state_dict


def Embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
    nn.init.constant_(m.weight[padding_idx], 0)
    return m


def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.0)
    return m

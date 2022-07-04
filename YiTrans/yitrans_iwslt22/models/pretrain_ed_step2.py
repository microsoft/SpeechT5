# --------------------------------------------------------
# The YiTrans End-to-End Speech Translation System for IWSLT 2022 Offline Shared Task (https://arxiv.org/abs/2206.05777)
# Github source: https://github.com/microsoft/SpeechT5/tree/main/YiTrans
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Based on fairseq code bases
# https://github.com/facebookresearch/fairseq
# --------------------------------------------------------

import logging
import contextlib
from argparse import Namespace
from typing import Any, Optional

import torch
import torch.nn as nn
import pickle
from dataclasses import dataclass, field
from fairseq import checkpoint_utils, tasks, utils
from fairseq.dataclass import FairseqDataclass
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq.models import BaseFairseqModel, FairseqEncoder, register_model
from fairseq.models.hubert.hubert import MASKING_DISTRIBUTION_CHOICES
from fairseq.models.hubert.hubert_asr import HubertAsrConfig
from fairseq.tasks import FairseqTask
from omegaconf import II, MISSING

from yitrans_iwslt22.modules.multimodal_transformer_decoder import MultimodalTransformerDecoder

logger = logging.getLogger(__name__)

@dataclass
class JointStep2Config(HubertAsrConfig):
    ## for decoder overrides
    decoder_layerdrop: float = field(
        default=0.1,
        metadata={"help": "probability of dropping a decoder layer in hubert"},
    )
    add_decoder: bool = field(
        default=False,
        metadata={"help": "whether to add decoder for CE Loss on code"},
    )
    reuse_text_emb: bool = field(
        default=False,
        metadata={"help": "reuse text token embeddings instead of initialize randomly"},
    )
    freeze_decoder_updates: int = field(
        default=0,
        metadata={"help": "dont finetune hubert for this many updates"},
    )
    # share_enc_dec_embeddings: bool = field(
    #     default=False,
    #     metadata={"help": "share embeddings of (text encoder, text decoder)"},
    # )
    share_s2t_t2t_embeddings: bool = field(
        default=False,
        metadata={"help": "share embeddings of (speech2text(code), text2text)"},
    )
    share_ctc_decoder_embed: bool = field(
        default=False,
        metadata={"help": "share ctc and decoder embedding (only when share_decoder_input_output_embed is true)"},
    )
    enc_grad_mult: float = field(
        default=1.0,
        metadata={"help": "reset feature grad mult in hubert to this (only for st2t)"},
    )
    retain_dict_path: Optional[str] = field(
        default=None,
        metadata={"help": "delete embeddings according to this path"},
    )
    load_step2_model_from: Optional[str] = field(
        default=None,
        metadata={
            "help": "load step2 model from"
        },
    )

    # for other overrides
    adaptor_stride: int = field(
        default=2,
        metadata={"help": "adaptor stride"},
    )

    # ## for reset some configs
    # load_pretrained_mbart_from: Optional[str] = field(
    #     default=None,
    #     metadata={
    #         "help": "model to take text encoder decoder weights from (for initialization)"
    #     },
    # )
    # load_pretrained_w2v_from: Optional[str] = field(
    #     default=None,
    #     metadata={
    #         "help": "model to take speech encoder weights from (for initialization)"
    #     },
    # )
    # use_rel_pos_enc: bool = field(
    #     default=True,
    #     metadata={"help": "whether to use relative positional encoding"},
    # )
    # encoder_layers: int = field(
    #     default=12,
    #     metadata={"help": "encoder_layers"},
    # )
    # add_text_modality: bool = field(
    #     default=True,
    #     metadata={"help": "add_text_modality"},
    # )
    # add_text_encoder: bool = field(
    #     default=True,
    #     metadata={"help": "add_text_encoder"},
    # )
    # share_all_embeddings: bool = field(
    #     default=True,
    #     metadata={"help": "share text_encoder, decoder_input, decoder_output embeddings"},
    # )
    # add_adaptor: bool = field(
    #     default=True,
    #     metadata={"help": "add_adaptor"},
    # )


@register_model("hubert_step2", dataclass=JointStep2Config)
class JointStep2Model(BaseFairseqModel):
    def __init__(self, cfg: JointStep2Config, w2v_encoder: BaseFairseqModel):
        super().__init__()
        self.cfg = cfg
        self.w2v_encoder = w2v_encoder

    def upgrade_state_dict_named(self, state_dict, name):
        super().upgrade_state_dict_named(state_dict, name)
        return state_dict

    @classmethod
    def build_model(cls, cfg: JointStep2Config, task: FairseqTask):
        """Build a new model instance."""
        w2v_encoder = JointED(cfg, task.target_dictionary)
        return cls(cfg, w2v_encoder)

    def get_normalized_probs(self, net_output, log_probs, sample=None):
        """Get normalized probabilities (or log probs) from a net's output."""
        if "encoder_out" not in net_output:
            return self.w2v_encoder.get_normalized_probs_decoder(net_output, log_probs, sample)

        if "encoder_out_for_ctc" in net_output:
            logits = net_output["encoder_out_for_ctc"]
        else:
            logits = net_output["encoder_out"]
        
        if isinstance(logits, list):
            logits = logits[0]

        if log_probs:
            return utils.log_softmax(logits.float(), dim=-1)
        else:
            return utils.softmax(logits.float(), dim=-1)

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

class JointED(FairseqEncoder):
    def __init__(self, cfg: JointStep2Config, tgt_dict=None):
        self.apply_mask = cfg.apply_mask
        logger.info(f"self.apply_mask: {self.apply_mask}")

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
            "decoder_dict_size": len(tgt_dict) if cfg.add_decoder else -1,
            "share_decoder_input_output_embed": cfg.share_decoder_input_output_embed,
            "share_s2t_t2t_embeddings": cfg.share_s2t_t2t_embeddings,
            "load_pretrained_w2v_from": None,
            "load_pretrained_mbart_from": None,
            "adaptor_stride": cfg.adaptor_stride,
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

        if cfg.normalize != w2v_args.task.normalize:
            logger.warn(
                "Fine-tuning works best when data normalization is the same. "
                "Please check that --normalize is set or unset for "
                "both pre-training and here"
            )

        w2v_args.task.data = cfg.data
        if hasattr(w2v_args.task, "text_cfg"):
            w2v_args.task.text_cfg.data_config = None
        w2v_args.task.add_decoder = cfg.add_decoder
        task = tasks.setup_task(w2v_args.task)
        if state is not None and "task_state" in state:
            # This will load the stored "dictionaries" object
            task.load_state_dict(state["task_state"])
        model = task.build_model(w2v_args.model)

        ### delete the embed_tokens and output_projection of decoder
        if state is not None and not cfg.no_pretrained_weights:
            if cfg.retain_dict_path is not None:
                assert model.add_text_modality, "Mustc have text modality if retain dict path"
                logger.info("Cut embedding to a smaller size according to ratin dict")
                with open(cfg.retain_dict_path, "rb") as fp:
                    overlap_idxs = pickle.load(fp)
                state['model']['decoder.output_projection.1.weight'] = state['model']['decoder.output_projection.1.weight'][overlap_idxs]
                state["model"]["decoder.embed_tokens_list.1.weight"] = state["model"]["decoder.embed_tokens_list.1.weight"][overlap_idxs]
            if cfg.reuse_text_emb:
                assert model.add_text_modality, "Mustc have text modality if reuse text embed"
                logger.info("Loading text-text pretrained token-embedding for speech-text finetuning...")
                state["model"]["decoder.embed_tokens_list.0.weight"] = state["model"]["decoder.embed_tokens_list.1.weight"]
                del state["model"]["decoder.embed_tokens_list.1.weight"]
                state["model"]["decoder.output_projection.0.weight"] = state["model"]["decoder.output_projection.1.weight"]
                del state["model"]["decoder.output_projection.1.weight"]
                try:
                    model.load_state_dict(state["model"], strict=True)
                except Exception as e:
                    logger.warn(e)
                    model.load_state_dict(state["model"], strict=False)
            else:
                for pname in list(state["model"].keys()):
                    if pname.startswith("decoder.embed_tokens") or pname.startswith("decoder.output_projection"):
                        del state["model"][pname]
                # set strict=False because we omit some modules
                model.load_state_dict(state["model"], strict=False)

        model.remove_pretraining_modules(step2=True)

        super().__init__(task.source_dictionary)

        d = w2v_args.model.encoder_embed_dim

        self.w2v_model = model

        self.final_dropout = nn.Dropout(cfg.final_dropout)
        self.freeze_finetune_updates = cfg.freeze_finetune_updates
        self.freeze_decoder_updates = cfg.freeze_decoder_updates
        self.num_updates = 0

        if cfg.share_ctc_decoder_embed:
            assert cfg.add_decoder and cfg.share_decoder_input_output_embed, "Must share decoder input and output embed before share ctc and decoder embed"
            if isinstance(self.w2v_model.decoder, MultimodalTransformerDecoder):
                self.proj = nn.Linear(
                    self.w2v_model.decoder.embed_tokens_list[0].weight.shape[1],
                    self.w2v_model.decoder.embed_tokens_list[0].weight.shape[0],
                    bias=False,
                )
                self.proj.weight = self.w2v_model.decoder.embed_tokens_list[0].weight
            else:
                self.proj = nn.Linear(
                    self.w2v_model.decoder.embed_tokens.weight.shape[1],
                    self.w2v_model.decoder.embed_tokens.weight.shape[0],
                    bias=False,
                )
                self.proj.weight = self.w2v_model.decoder.embed_tokens.weight
        elif tgt_dict is not None:
            self.proj = Linear(d, len(tgt_dict))
        elif getattr(cfg, "decoder_embed_dim", d) != d:
            self.proj = Linear(d, cfg.decoder_embed_dim)
        else:
            self.proj = None

    def set_num_updates(self, num_updates):
        """Set the number of parameters updates."""
        super().set_num_updates(num_updates)
        self.num_updates = num_updates

    def forward(self, source=None, src_tokens=None, src_lengths=None, padding_mask=None, prev_output_tokens=None, tbc=True, **kwargs):
        assert source is not None or src_tokens is not None
        if source is not None:
            ### 1. go speech cnn-encoder-decoder branch
            ft = self.freeze_finetune_updates <= self.num_updates
            w2v_args = {
                "source": source,
                "padding_mask": padding_mask,
                "mask": self.apply_mask and self.training,
                "prev_output_tokens": prev_output_tokens,
                "ft": ft,
            }

            if self.freeze_decoder_updates <= self.num_updates:
                self.w2v_model.add_decoder = True
            else:
                self.w2v_model.add_decoder = False
        
            x, padding_mask, decoder_out = self.w2v_model.extract_features(**w2v_args)

            if tbc:
                # B x T x C -> T x B x C
                x = x.transpose(0, 1)

            x = self.final_dropout(x)

            if self.proj:
                x = self.proj(x)

            return {
                "encoder_out": x,  # T x B x C
                "encoder_padding_mask": padding_mask,  # B x T
                "padding_mask": padding_mask,
                "decoder_out": decoder_out,
            }
        else:
            ### 2. go text encoder-decoder branch
            w2v_args = {
                "src_tokens": src_tokens,
                "src_lengths": src_lengths,
                "prev_output_tokens": prev_output_tokens,
            }

            return self.w2v_model(**w2v_args)

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
        
        assert self.proj is not None
        encoder_out['encoder_out_for_ctc'] = [self.proj(encoder_out['encoder_out'][0])]
        
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

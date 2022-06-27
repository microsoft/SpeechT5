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
import pickle
from argparse import Namespace
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from fairseq import checkpoint_utils, tasks, utils
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq.models import BaseFairseqModel, FairseqEncoder, register_model
from fairseq.models.hubert.hubert_asr import HubertCtcConfig, HubertAsrConfig
from fairseq.tasks import FairseqTask
from fairseq.data.data_utils import lengths_to_padding_mask
from omegaconf import II, open_dict


logger = logging.getLogger(__name__)

@dataclass
class HubertSt2tCtcConfig(HubertCtcConfig):
    load_speech_only: bool = II("task.load_speech_only")
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

@register_model("hubert_st2t", dataclass=HubertSt2tCtcConfig)
class HubertST2T(BaseFairseqModel):
    def __init__(self, cfg: HubertSt2tCtcConfig, w2v_encoder: BaseFairseqModel):
        super().__init__()
        self.cfg = cfg
        self.w2v_encoder = w2v_encoder
        self.num_updates = 0

        ### in case we need load hubert_step2 model
        if cfg.load_step2_model_from:
            logger.info(f"Loading hubert_step2 pretrained model for finetuning: {cfg.load_step2_model_from}")
            hubert_step2_states = self.w2v_encoder.w2v_model.load_checkpoint(cfg.load_step2_model_from)["model"]
            if cfg.retain_dict_path is not None:
                with open(cfg.retain_dict_path, "rb") as fp:
                    overlap_idxs = pickle.load(fp)
                if hubert_step2_states['w2v_encoder.w2v_model.decoder.output_projection.0.weight'].size(0) != len(overlap_idxs):
                    assert self.w2v_encoder.w2v_model.add_text_modality, "Mustc have text modality if retain dict path"
                    logger.info("Cut embedding to a smaller size according to retain dict")
                    hubert_step2_states['w2v_encoder.w2v_model.decoder.output_projection.0.weight'] = hubert_step2_states['w2v_encoder.w2v_model.decoder.output_projection.0.weight'][overlap_idxs]
                    hubert_step2_states["w2v_encoder.w2v_model.decoder.embed_tokens_list.0.weight"] = hubert_step2_states["w2v_encoder.w2v_model.decoder.embed_tokens_list.0.weight"][overlap_idxs]
                    if hubert_step2_states.get("w2v_encoder.w2v_model.text_encoder.embed_tokens.weight") is not None:
                        hubert_step2_states["w2v_encoder.w2v_model.text_encoder.embed_tokens.weight"] = hubert_step2_states["w2v_encoder.w2v_model.text_encoder.embed_tokens.weight"][overlap_idxs]
                else:
                    logger.info(f"cfg.load_step2_model_from matches the cut embedding dims {len(overlap_idxs)}, no cutting needs to do")
            if not self.cfg.load_speech_only and hubert_step2_states.get("w2v_encoder.w2v_model.text_encoder.embed_tokens.weight", None) is None:
                hubert_step2_states["w2v_encoder.w2v_model.text_encoder.embed_tokens.weight"] = hubert_step2_states["w2v_encoder.w2v_model.decoder.embed_tokens_list.0.weight"]
            try:
                self.load_state_dict(hubert_step2_states, strict=True)
            except Exception as e:
                logger.warn(e)
                self.load_state_dict(hubert_step2_states, strict=False)

    def upgrade_state_dict_named(self, state_dict, name):
        super().upgrade_state_dict_named(state_dict, name)
        return state_dict

    @classmethod
    def build_model(cls, cfg: HubertSt2tCtcConfig, task: FairseqTask):
        """Build a new model instance."""
        w2v_encoder = HubertEncoder(cfg, task.target_dictionary)
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
    
    def set_num_updates(self, num_updates):
        """Set the number of parameters updates."""
        super().set_num_updates(num_updates)
        self.num_updates = num_updates


class HubertEncoder(FairseqEncoder):
    def __init__(self, cfg: HubertAsrConfig, tgt_dict=None):
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
            "load_pretrained_w2v_from": cfg.load_pretrained_w2v_from,
            "load_pretrained_mbart_from": None,
            "adaptor_stride": cfg.adaptor_stride,
            "share_speech_text_embeddings": cfg.share_speech_text_embeddings,
        }

        if cfg.no_pretrained_weights:
            arg_overrides["use_rel_pos_enc"] = cfg.use_rel_pos_enc
            arg_overrides["encoder_layers"] = cfg.encoder_layers
            arg_overrides["add_text_encoder"] = cfg.add_text_encoder
            arg_overrides["share_all_embeddings"] = cfg.share_all_embeddings
            arg_overrides["add_adaptor"] = cfg.add_adaptor

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

        ## in speech_text_joint_to_text, data is loaded by soundfile, which returns without normalization
        self.need_preprocess = w2v_args.task.normalize
        logger.warn("We need normalize the input wavform from the src_tokens")

        if cfg.normalize != w2v_args.task.normalize:
            logger.warn(
            "Fine-tuning works best when data normalization is the same. "
            "Please check that --normalize is set or unset for "
            "both pre-training and here"
        )
        
        if not "share_speech_text_embeddings" in w2v_args.model:
            with open_dict(w2v_args.model): 
                w2v_args.model.share_speech_text_embedding = cfg.share_speech_text_embeddings
        logger.info(f"share_speech_text_embeddings: {(getattr(w2v_args.model, 'share_speech_text_embeddings', False))}")
        w2v_args.task.data = cfg.data
        w2v_args.task.add_decoder = cfg.add_decoder
        assert w2v_args.model._name == "hubert"

        task = tasks.setup_task(w2v_args.task)
        if state is not None and "task_state" in state:
            # This will load the stored "dictionaries" object
            task.load_state_dict(state["task_state"])
        model = task.build_model(w2v_args.model)

        ### modify the embed_tokens and output_projection of decoder
        if state is not None and not cfg.no_pretrained_weights:
            model_states = self.modify_states(state['model'], cfg.retain_dict_path, cfg.reuse_text_emb)
            try:
                model.load_state_dict(model_states, strict=True)
            except Exception as e:
                logger.warn(e)
                model.load_state_dict(model_states, strict=False)

        ### in case we need load mbart
        if cfg.no_pretrained_weights and cfg.load_pretrained_mbart_from:
            logger.info("Loading mbart ...")
            mbart_state = model.load_checkpoint(cfg.load_pretrained_mbart_from)
            mbart_state["model"] = self.modify_states(mbart_state["model"], cfg.retain_dict_path, cfg.reuse_text_emb, is_mbart=True)
            model.text_encoder = model.load_pretrained_component_from_model(
                component=model.text_encoder, state=mbart_state
            )
            model.decoder = model.load_pretrained_component_from_model(
                component=model.decoder, state=mbart_state
            )

        model.remove_pretraining_modules(step2=(not cfg.load_speech_only))
        # model.remove_pretraining_modules()

        super().__init__(task.source_dictionary)

        d = w2v_args.model.encoder_embed_dim

        self.w2v_model = model

        self.final_dropout = nn.Dropout(cfg.final_dropout)
        self.freeze_finetune_updates = cfg.freeze_finetune_updates
        self.freeze_decoder_updates = cfg.freeze_decoder_updates
        self.num_updates = 0
        self.enc_grad_mult = cfg.enc_grad_mult
    
    def modify_states(self, model_states, retain_dict_path=None, reuse_text_emb=False, is_mbart=False):
        if retain_dict_path is not None:
            logger.info("Cut embedding to a smaller size according to retain dict")
            with open(retain_dict_path, "rb") as fp:
                overlap_idxs = pickle.load(fp)
            if is_mbart:
                model_states["decoder.embed_tokens_list.1.weight"] = model_states["decoder.embed_tokens.weight"][overlap_idxs]
                model_states["decoder.output_projection.1.weight"] = model_states["decoder.output_projection.weight"][overlap_idxs]
                model_states["decoder.embed_tokens.weight"] = model_states["decoder.embed_tokens.weight"][overlap_idxs]
                model_states["decoder.output_projection.weight"] = model_states["decoder.output_projection.weight"][overlap_idxs]
                model_states["encoder.embed_tokens.weight"] = model_states["encoder.embed_tokens.weight"][overlap_idxs]
            else:
                model_states['decoder.output_projection.1.weight'] = model_states['decoder.output_projection.1.weight'][overlap_idxs]
                model_states["decoder.embed_tokens_list.1.weight"] = model_states["decoder.embed_tokens_list.1.weight"][overlap_idxs]
                model_states["text_encoder.embed_tokens.weight"] = model_states["text_encoder.embed_tokens.weight"][overlap_idxs]
        if reuse_text_emb:
            logger.info("Loading decoder.embed_tokens_list.0 <-- decoder.embed_tokens_list.1")
            model_states["decoder.embed_tokens_list.0.weight"] = model_states["decoder.embed_tokens_list.1.weight"]
            model_states["decoder.output_projection.0.weight"] = model_states["decoder.output_projection.1.weight"]
            del model_states["decoder.embed_tokens_list.1.weight"]
            del model_states["decoder.output_projection.1.weight"]
        return model_states
    
    def set_num_updates(self, num_updates):
        """Set the number of parameters updates."""
        super().set_num_updates(num_updates)
        self.num_updates = num_updates

    def forward(self, src_tokens=None, src_lengths=None, src_txt_tokens=None, src_txt_lengths=None, prev_output_tokens=None, tbc=True, **kwargs):
        padding_mask = lengths_to_padding_mask(src_lengths)
        if self.need_preprocess:
            src_tokens = torch.stack([F.layer_norm(wav, wav.shape) for wav in src_tokens])
            src_tokens[padding_mask] = 0.0

        ft = self.freeze_finetune_updates <= self.num_updates
        w2v_args = {
            "source": src_tokens,
            "padding_mask": padding_mask,
            "mask": self.apply_mask and self.training,
            "prev_output_tokens": prev_output_tokens,
            "ft": ft,
            "enc_grad_mult": self.enc_grad_mult,
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

        if src_txt_tokens is not None:
            w2v_args_text = {
                "src_tokens": src_txt_tokens,
                "src_lengths": src_txt_lengths,
                "prev_output_tokens": prev_output_tokens,
            }

            decoder_output_text = self.w2v_model(**w2v_args_text)
            decoder_out = (torch.cat([decoder_out[0], decoder_output_text['decoder_out'][0]], dim=0), {"attn_cost": None})

        return decoder_out

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

        return encoder_out

    def forward_torchscript(self, net_input):
        """A TorchScript-compatible version of forward.

        Encoders which use additional arguments may want to override
        this method for TorchScript compatibility.
        """
        padding_mask = lengths_to_padding_mask(net_input["src_lengths"])
        src_tokens = net_input["src_tokens"]
        if self.need_preprocess:
            src_tokens = torch.stack([F.layer_norm(wav, wav.shape) for wav in src_tokens])
            src_tokens[padding_mask] = 0.0

        _net_input = {
            "source": src_tokens,
            "padding_mask": padding_mask,
        }

        encoder_out = self.w2v_model.forward_torchscript(_net_input)
        
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

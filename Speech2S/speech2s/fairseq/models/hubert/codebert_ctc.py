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
import pickle
from dataclasses import dataclass, field
from fairseq import checkpoint_utils, tasks, utils
from fairseq.dataclass import FairseqDataclass
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq.models import BaseFairseqModel, FairseqEncoder, register_model
from fairseq.models.transformer import MultimodalTransformerDecoder
from fairseq.models.hubert.hubert import MASKING_DISTRIBUTION_CHOICES
from fairseq.tasks import FairseqTask
from omegaconf import II, MISSING, open_dict

logger = logging.getLogger(__name__)

@dataclass
class HubertAsrConfig(FairseqDataclass):
    w2v_path: str = field(default=MISSING, metadata={"help": "path to hubert model"})
    no_pretrained_weights: bool = field(
        default=False,
        metadata={"help": "if true, does not load pretrained weights"},
    )
    dropout_input: float = field(
        default=0.0,
        metadata={"help": "dropout to apply to the input (after feat extr)"},
    )
    final_dropout: float = field(
        default=0.0,
        metadata={"help": "dropout after transformer and before final projection"},
    )
    dropout: float = field(
        default=0.0,
        metadata={"help": "dropout probability inside hubert model"},
    )
    attention_dropout: float = field(
        default=0.0,
        metadata={
            "help": "dropout probability for attention weights " "inside hubert model"
        },
    )
    activation_dropout: float = field(
        default=0.0,
        metadata={
            "help": "dropout probability after activation in FFN " "inside hubert model"
        },
    )

    # masking
    apply_mask: bool = field(
        default=False, metadata={"help": "apply masking during fine-tuning"}
    )
    mask_length: int = field(
        default=10, metadata={"help": "repeat the mask indices multiple times"}
    )
    mask_prob: float = field(
        default=0.5,
        metadata={
            "help": "probability of replacing a token with mask "
            "(normalized by length)"
        },
    )
    mask_selection: MASKING_DISTRIBUTION_CHOICES = field(
        default="static", metadata={"help": "how to choose masks"}
    )
    mask_other: float = field(
        default=0,
        metadata={
            "help": "secondary mask argument "
            "(used for more complex distributions), "
            "see help in compute_mask_indices"
        },
    )
    no_mask_overlap: bool = field(
        default=False, metadata={"help": "whether to allow masks to overlap"}
    )

    # channel masking
    mask_channel_length: int = field(
        default=10,
        metadata={"help": "length of the mask for features (channels)"},
    )
    mask_channel_prob: float = field(
        default=0.0,
        metadata={"help": "probability of replacing a feature with 0"},
    )
    mask_channel_selection: MASKING_DISTRIBUTION_CHOICES = field(
        default="static",
        metadata={"help": "how to choose mask length for channel masking"},
    )
    mask_channel_other: float = field(
        default=0,
        metadata={
            "help": "secondary mask argument "
            "(used for more complex distributions), "
            "see help in compute_mask_indices"
        },
    )
    no_mask_channel_overlap: bool = field(
        default=False,
        metadata={"help": "whether to allow channel masks to overlap"},
    )
    freeze_finetune_updates: int = field(
        default=0,
        metadata={"help": "dont finetune hubert for this many updates"},
    )
    feature_grad_mult: float = field(
        default=0.0,
        metadata={"help": "reset feature grad mult in hubert to this"},
    )
    layerdrop: float = field(
        default=0.0,
        metadata={"help": "probability of dropping a layer in hubert"},
    )
    decoder_layerdrop: float = field(
        default=0.1,
        metadata={"help": "probability of dropping a decoder layer in hubert"},
    )
    adaptor_stride: int = field(
        default=2,
        metadata={"help": "adaptor stride"},
    )
    normalize: bool = II("task.normalize")
    data: str = II("task.data")

    # this holds the loaded hubert args
    w2v_args: Any = None
    
    # for decoder
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
    share_decoder_input_output_embed: bool = field(
        default=False,
        metadata={"help": "share decoder input and output embeddings"},
    )
    share_speech_text_embeddings: bool = field(
        default=False,
        metadata={"help": "share all embeddings (speech, text)"},
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


@dataclass
class HubertCtcConfig(HubertAsrConfig):
    # pass
    load_pretrained_mbart_from: Optional[str] = field(
        default=None,
        metadata={
            "help": "model to take text encoder decoder weights from (for initialization)"
        },
    )
    load_pretrained_w2v_from: Optional[str] = field(
        default=None,
        metadata={
            "help": "model to take speech encoder weights from (for initialization)"
        },
    )
    use_rel_pos_enc: bool = field(
        default=True,
        metadata={"help": "whether to use relative positional encoding"},
    )
    encoder_layers: int = field(
        default=12,
        metadata={"help": "encoder_layers"},
    )
    add_text_modality: bool = field(
        default=True,
        metadata={"help": "add_text_modality"},
    )
    add_text_encoder: bool = field(
        default=True,
        metadata={"help": "add_text_encoder"},
    )
    share_all_embeddings: bool = field(
        default=True,
        metadata={"help": "share text_encoder, decoder_input, decoder_output embeddings"},
    )
    add_adaptor: bool = field(
        default=True,
        metadata={"help": "add_adaptor"},
    )
    add_code_encoder: bool = field(
        default=True,
        metadata={"help": "add_code_encoder"},
    )
    seek_code_embedding: bool = field(
        default=True,
        metadata={"help": "get soft code embeddings"},
    )


@register_model("codebert_ctc", dataclass=HubertCtcConfig)
class CodeBertCTC(BaseFairseqModel):
    def __init__(self, cfg: HubertCtcConfig, w2v_encoder: BaseFairseqModel):
        super().__init__()
        self.cfg = cfg
        self.w2v_encoder = w2v_encoder

        ### in case we need load hubert_step2 model
        if cfg.load_step2_model_from:
            logger.info(f"Loading hubert_step2 pretrained model for finetuning: {cfg.load_step2_model_from}")
            hubert_step2_states = self.w2v_encoder.w2v_model.load_checkpoint(cfg.load_step2_model_from)["model"]
            if cfg.retain_dict_path is not None:
                assert self.w2v_encoder.w2v_model.add_text_modality, "Mustc have text modality if retain dict path"
                logger.info("Cut embedding to a smaller size according to retain dict")
                with open(cfg.retain_dict_path, "rb") as fp:
                    overlap_idxs = pickle.load(fp)
                hubert_step2_states['w2v_encoder.w2v_model.decoder.output_projection.0.weight'] = hubert_step2_states['w2v_encoder.w2v_model.decoder.output_projection.0.weight'][overlap_idxs]
                hubert_step2_states["w2v_encoder.w2v_model.decoder.embed_tokens_list.0.weight"] = hubert_step2_states["w2v_encoder.w2v_model.decoder.embed_tokens_list.0.weight"][overlap_idxs]
                hubert_step2_states["w2v_encoder.proj.weight"] = hubert_step2_states["w2v_encoder.proj.weight"][overlap_idxs]
            try:
                self.load_state_dict(hubert_step2_states, strict=True)
            except Exception as e:
                logger.warn(e)
                self.load_state_dict(hubert_step2_states, strict=False)

    def upgrade_state_dict_named(self, state_dict, name):
        super().upgrade_state_dict_named(state_dict, name)
        return state_dict

    @classmethod
    def build_model(cls, cfg: HubertCtcConfig, task: FairseqTask):
        """Build a new model instance."""
        w2v_encoder = HubertEncoder(cfg, task.tgt_dict)
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

@dataclass
class HubertSeq2SeqConfig(HubertAsrConfig):
    decoder_embed_dim: int = field(
        default=768, metadata={"help": "decoder embedding dimension"}
    )
    decoder_ffn_embed_dim: int = field(
        default=3072, metadata={"help": "decoder embedding dimension for FFN"}
    )
    decoder_layers: int = field(default=6, metadata={"help": "num of decoder layers"})
    decoder_layerdrop: float = field(
        default=0.0, metadata={"help": "decoder layerdrop chance"}
    )
    decoder_attention_heads: int = field(
        default=4, metadata={"help": "num decoder attention heads"}
    )
    decoder_learned_pos: bool = field(
        default=False,
        metadata={"help": "use learned positional embeddings in the decoder"},
    )
    decoder_normalize_before: bool = field(
        default=False,
        metadata={"help": "apply layernorm before each decoder block"},
    )
    no_token_positional_embeddings: bool = field(
        default=False,
        metadata={
            "help": "if set, disables positional embeddings " "(outside self attention)"
        },
    )
    decoder_dropout: float = field(
        default=0.0, metadata={"help": "dropout probability in the decoder"}
    )
    decoder_attention_dropout: float = field(
        default=0.0,
        metadata={
            "help": "dropout probability for attention weights " "inside the decoder"
        },
    )
    decoder_activation_dropout: float = field(
        default=0.0,
        metadata={
            "help": "dropout probability after activation in FFN " "inside the decoder"
        },
    )
    max_target_positions: int = field(
        default=2048, metadata={"help": "max target positions"}
    )
    share_decoder_input_output_embed: bool = field(
        default=False,
        metadata={"help": "share decoder input and output embeddings"},
    )


class HubertEncoder(FairseqEncoder):
    def __init__(self, cfg: HubertAsrConfig, tgt_dict=None):
        self.apply_mask = cfg.apply_mask
        self.enc_grad_mult = cfg.enc_grad_mult
        self.padding_idx = tgt_dict.pad()
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
            "load_pretrained_mbart_from": cfg.load_pretrained_mbart_from,
            "adaptor_stride": cfg.adaptor_stride,
            "add_code_encoder": cfg.add_code_encoder,
            "seek_code_embedding": cfg.seek_code_embedding,
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
        if cfg.normalize != w2v_args.task.normalize:
            logger.warn(
                "Fine-tuning works best when data normalization is the same. "
                "Please check that --normalize is set or unset for "
                "both pre-training and here"
            )

        w2v_args.task.data = cfg.data
        if hasattr(w2v_args.task, "text_cfg") and hasattr(w2v_args.task.text_cfg, "data_config"):
            w2v_args.task.text_cfg.data_config = None
        if not "seek_code_embedding" in w2v_args.model:
            with open_dict(w2v_args.model): 
                w2v_args.model.seek_code_embedding = cfg.seek_code_embedding
        
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
                logger.info("Loading text-text pretrained token-embedding for speech-text finetuning...")
                if "decoder.embed_tokens_list.1.weight" in state["model"]:
                    state["model"]["decoder.embed_tokens_list.0.weight"] = state["model"]["decoder.embed_tokens_list.1.weight"]
                    del state["model"]["decoder.embed_tokens_list.1.weight"]
                    state["model"]["decoder.output_projection.0.weight"] = state["model"]["decoder.output_projection.1.weight"]
                    del state["model"]["decoder.output_projection.1.weight"]
            else:
                for pname in list(state["model"].keys()):
                    if pname.startswith("decoder.embed_tokens") or pname.startswith("decoder.output_projection"):
                        del state["model"][pname]
            try:
                model.load_state_dict(state["model"], strict=True)
            except Exception as e:
                logger.warn(e)
                model.load_state_dict(state["model"], strict=False)

        ### in case we need load mbart embedding into asr embedding
        if cfg.no_pretrained_weights and cfg.load_pretrained_mbart_from and cfg.reuse_text_emb:
            logger.info("Loading mbart pretrained token-embedding for speech-text finetuning...")
            mbart_dec_states = model.decoder.state_dict()
            loading_states = {}
            if cfg.retain_dict_path is not None:
                logger.info("Cut embedding to a smaller size according to ratin dict")
                with open(cfg.retain_dict_path, "rb") as fp:
                    overlap_idxs = pickle.load(fp)
                loading_states["output_projection.0.weight"] = mbart_dec_states['output_projection.1.weight'][overlap_idxs]
                loading_states["embed_tokens_list.0.weight"] = mbart_dec_states['embed_tokens_list.1.weight'][overlap_idxs]
            else:
                loading_states["output_projection.0.weight"] = mbart_dec_states['output_projection.1.weight']
                loading_states["embed_tokens_list.0.weight"] = mbart_dec_states['embed_tokens_list.1.weight']
            model.decoder.load_state_dict(loading_states, strict=False)

        model.remove_pretraining_modules()

        super().__init__(task.source_dictionary)

        d = w2v_args.model.encoder_embed_dim

        self.w2v_model = model

        self.final_dropout = nn.Dropout(cfg.final_dropout)
        self.freeze_finetune_updates = cfg.freeze_finetune_updates
        self.freeze_decoder_updates = cfg.freeze_decoder_updates
        self.num_updates = 0
        self.proj = Linear(d, len(tgt_dict))

    def set_num_updates(self, num_updates):
        """Set the number of parameters updates."""
        super().set_num_updates(num_updates)
        self.num_updates = num_updates

    def forward(self, src_tokens, src_lengths, prev_output_tokens=None, tbc=True, **kwargs):
        padding_mask = src_tokens == self.padding_idx
        w2v_args = {
            "src_tokens": src_tokens,
            "padding_mask": padding_mask,
            "mask": self.apply_mask and self.training,
            "prev_output_tokens": prev_output_tokens,
            "features_only": True,
        }

        encoder_out = self.w2v_model(**w2v_args)
        x = encoder_out['encoder_out'][0]
        x = self.final_dropout(x)
        x = self.proj(x)

        return {
            "encoder_out": x,  # T x B x C
            "encoder_padding_mask": padding_mask,  # B x T
            "padding_mask": padding_mask,
        }

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
        encoder_out = self.w2v_model(**net_input, mask=False, features_only=True)
        
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

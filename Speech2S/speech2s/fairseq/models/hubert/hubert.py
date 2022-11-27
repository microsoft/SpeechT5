# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import contextlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union
from collections import OrderedDict

import numpy as np

import math
import copy
import torch
import torch.nn as nn
from omegaconf import II

from fairseq import utils, checkpoint_utils
from fairseq.data.data_utils import compute_mask_indices
from fairseq.data.dictionary import Dictionary
from fairseq.dataclass import ChoiceEnum, FairseqDataclass
from fairseq.models import BaseFairseqModel, register_model, FairseqDecoder, FairseqEncoder
from fairseq.models.speech_to_text.s2t_transformer import TransformerDecoderScriptable
from fairseq.models.transformer import MultimodalTransformerDecoder
from fairseq.models.speech_to_text import Conv1dAdaptor
from fairseq.models.transformer import Embedding
from fairseq.models.transformer import MoMETransformerEncoder
from fairseq.file_io import PathManager
from torch import Tensor
from fairseq.models.wav2vec.wav2vec2 import (
    ConvFeatureExtractionModel,
    TransformerEncoder,
)
from fairseq.modules import GradMultiply, LayerNorm, PositionalEmbedding
from fairseq.tasks.hubert_pretraining import (
    HubertPretrainingConfig,
    HubertPretrainingTask,
)
from fairseq.models.transformer import (
    TransformerEncoderBase,
    TransformerConfig,
)

logger = logging.getLogger(__name__)

EXTRACTOR_MODE_CHOICES = ChoiceEnum(["default", "layer_norm"])
MASKING_DISTRIBUTION_CHOICES = ChoiceEnum(["static", "uniform", "normal", "poisson"])


@dataclass
class HubertConfig(FairseqDataclass):
    label_rate: int = II("task.label_rate")

    extractor_mode: EXTRACTOR_MODE_CHOICES = field(
        default="default",
        metadata={
            "help": "mode for feature extractor. default has a single group "
            "norm with d groups in the first conv block, whereas layer_norm "
            "has layer norms in every block (meant to use with normalize=True)"
        },
    )
    encoder_layers: int = field(
        default=12, metadata={"help": "num encoder layers in the transformer"}
    )
    encoder_embed_dim: int = field(
        default=768, metadata={"help": "encoder embedding dimension"}
    )
    encoder_ffn_embed_dim: int = field(
        default=3072, metadata={"help": "encoder embedding dimension for FFN"}
    )
    encoder_attention_heads: int = field(
        default=12, metadata={"help": "num encoder attention heads"}
    )
    activation_fn: ChoiceEnum(utils.get_available_activation_fns()) = field(
        default="gelu", metadata={"help": "activation function to use"}
    )

    # dropouts
    dropout: float = field(
        default=0.1,
        metadata={"help": "dropout probability for the transformer"},
    )
    attention_dropout: float = field(
        default=0.1,
        metadata={"help": "dropout probability for attention weights"},
    )
    activation_dropout: float = field(
        default=0.0,
        metadata={"help": "dropout probability after activation in FFN"},
    )
    encoder_layerdrop: float = field(
        default=0.0,
        metadata={"help": "probability of dropping a tarnsformer layer"},
    )
    dropout_input: float = field(
        default=0.0,
        metadata={"help": "dropout to apply to the input (after feat extr)"},
    )
    dropout_features: float = field(
        default=0.0,
        metadata={"help": "dropout to apply to the features (after feat extr)"},
    )

    final_dim: int = field(
        default=0,
        metadata={
            "help": "project final representations and targets to this many "
            "dimensions. set to encoder_embed_dim is <= 0"
        },
    )
    untie_final_proj: bool = field(
        default=False,
        metadata={"help": "use separate projection for each target"},
    )
    layer_norm_first: bool = field(
        default=False,
        metadata={"help": "apply layernorm first in the transformer"},
    )
    conv_feature_layers: str = field(
        default="[(512,10,5)] + [(512,3,2)] * 4 + [(512,2,2)] * 2",
        metadata={
            "help": "string describing convolutional feature extraction "
            "layers in form of a python list that contains "
            "[(dim, kernel_size, stride), ...]"
        },
    )
    conv_bias: bool = field(
        default=False, metadata={"help": "include bias in conv encoder"}
    )
    logit_temp: float = field(
        default=0.1, metadata={"help": "temperature to divide logits by"}
    )
    target_glu: bool = field(
        default=False, metadata={"help": "adds projection + glu to targets"}
    )
    feature_grad_mult: float = field(
        default=1.0,
        metadata={"help": "multiply feature extractor var grads by this"},
    )

    # masking
    mask_length: int = field(default=10, metadata={"help": "mask length"})
    mask_prob: float = field(
        default=0.65,
        metadata={"help": "probability of replacing a token with mask"},
    )
    mask_selection: MASKING_DISTRIBUTION_CHOICES = field(
        default="static", metadata={"help": "how to choose mask length"}
    )
    mask_other: float = field(
        default=0,
        metadata={
            "help": "secondary mask argument "
            "(used for more complex distributions), "
            "see help in compute_mask_indicesh"
        },
    )
    no_mask_overlap: bool = field(
        default=False, metadata={"help": "whether to allow masks to overlap"}
    )
    mask_min_space: int = field(
        default=1,
        metadata={"help": "min space between spans (if no overlap is enabled)"},
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
            "see help in compute_mask_indicesh"
        },
    )
    no_mask_channel_overlap: bool = field(
        default=False,
        metadata={"help": "whether to allow channel masks to overlap"},
    )
    mask_channel_min_space: int = field(
        default=1,
        metadata={"help": "min space between spans (if no overlap is enabled)"},
    )

    # positional embeddings
    conv_pos: int = field(
        default=128,
        metadata={"help": "number of filters for convolutional positional embeddings"},
    )
    conv_pos_groups: int = field(
        default=16,
        metadata={"help": "number of groups for convolutional positional embedding"},
    )
    use_rel_pos_enc: bool = field(
        default=False,
        metadata={"help": "whether to use relative positional encoding"},
    )

    latent_temp: Tuple[float, float, float] = field(
        default=(2, 0.5, 0.999995),
        metadata={"help": "legacy (to be removed)"},
    )

    # loss computation
    skip_masked: bool = field(
        default=False,
        metadata={"help": "skip computing losses over masked frames"},
    )
    skip_nomask: bool = field(
        default=False,
        metadata={"help": "skip computing losses over unmasked frames"},
    )
    
    # other
    checkpoint_activations: bool = field(
        default=False, metadata={"help": "recompute activations and save memory for extra compute"}
    )
    
    # decoder
    decoder_layers: int = field(
        default=6, metadata={"help": "num decoder layers in the transformer"}
    )
    decoder_embed_dim: int = field(
        default=768, metadata={"help": "decoder embedding dimension"}
    )
    decoder_ffn_embed_dim: int = field(
        default=3072, metadata={"help": "decoder embedding dimension for FFN"}
    )
    decoder_attention_heads: int = field(
        default=12, metadata={"help": "num decoder attention heads"}
    )
    decoder_normalize_before: bool = field(
        default=False,
        metadata={"help": "apply layernorm before each decoder block"},
    )
    layernorm_embedding: bool = field(
        default=False,
        metadata={"help": "apply layernorm to embedding for decoder"},
    )
    decoder_layerdrop: float = field(
        default=0.1,
        metadata={"help": "probability of dropping a tarnsformer layer"},
    )
    share_decoder_input_output_embed: bool = field(
        default=False,
        metadata={"help": "share decoder input and output embeddings"},
    )
    share_all_embeddings: bool = field(
        default=False,
        metadata={"help": "share all embeddings (text encoder, text decoder)"},
    )
    share_speech_text_embeddings: bool = field(
        default=False,
        metadata={"help": "share all embeddings (speech, text)"},
    )
    decoder_output_dim: int = field(
        default=768, metadata={"help": "decoder output dimension"}
    )
    max_target_positions: int = field(
        default=3000, metadata={"help": "max target position"}
    )
    no_scale_embedding: bool = field(
        default=False,
        metadata={"help": "not scale embedding"},
    )
    adaptive_input: bool = field(
        default=False,
        metadata={"help": "adaptive input"},
    )
    quant_noise_pq: int = field(
        default=0, metadata={"help": "quant noise pq"}
    )
    decoder_learned_pos: bool = field(
        default=False,
        metadata={"help": "decoder learnable positional embedding"},
    )
    no_token_positional_embeddings: bool = field(
        default=False,
        metadata={"help": "no token positional embeddings"},
    )
    add_text_modality: bool = field(
        default=-False,
        metadata={"help": "add text modality, mainly used in pretrainnig"},
    )
    add_text_encoder: bool = field(
        default=False,
        metadata={"help": "add_text_encoder"},
    )
    share_text_encoder: bool = field(
        default=True,
        metadata={"help": "share text encoder so that speech branch go through it"},
    )
    split_attention: bool = field(
        default=False,
        metadata={"help": "use shared but split encoders"},
    )
    add_adaptor: bool = field(
        default=False,
        metadata={"help": "add adaptor and text encoder on the top of speech encoder"},
    )
    adaptor_n_layers: int = field(
        default=3,
        metadata={"help": "number of layers for adaptor"},
    )
    adaptor_kernel_size: int = field(
        default=3,
        metadata={"help": "kernel size for adaptor"},
    )
    adaptor_stride: int = field(
        default=2,
        metadata={"help": "adaptor stride"},
    )
    adaptor_layernorm: bool = field(
        default=False,
        metadata={"help": "adaptor layernorm"},
    )


    # text encoder related, TransformerConfig is used in bart but we only use its enconder
    text_transformer: TransformerConfig = TransformerConfig()

    checkpoint_activations: bool = field(
        default=False,
        metadata={"help": "recompute activations and save memory for extra compute"},
    )

    # FP16 optimization
    required_seq_len_multiple: int = field(
        default=1,
        metadata={
            "help": "pad the input to encoder such that the sequence length is divisible by multiple"
        },
    )
    crop_seq_to_multiple: int = field(
        default=1,
        metadata={
            "help": "crop convolutional feature extractor output such that the sequence length is divisible by multiple"
        },
    )
    
    # Load pre-train model
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
    
    # Finetune related
    decoder_dict_size: int = field(
        default=-1,
        metadata={"help": "decoder dictionary dimension"},
    )
    encoder_dict_size: int = field(
        default=-1,
        metadata={"help": "text encoder dictionary dimension"},
    )


@register_model("hubert", dataclass=HubertConfig)
class HubertModel(BaseFairseqModel):
    def __init__(
        self,
        cfg: HubertConfig,
        task_cfg: HubertPretrainingConfig,
        dictionaries: List[Dictionary],
        text_dictionary: Dictionary = None,
    ) -> None:
        super().__init__()
        logger.info(f"HubertModel Config: {cfg}")

        feature_enc_layers = eval(cfg.conv_feature_layers)  # noqa
        self.embed = feature_enc_layers[-1][0]

        self.feature_extractor = ConvFeatureExtractionModel(
            conv_layers=feature_enc_layers,
            dropout=0.0,
            mode=cfg.extractor_mode,
            conv_bias=cfg.conv_bias,
        )
        feature_ds_rate = np.prod([s for _, _, s in feature_enc_layers])
        self.feat2tar_ratio = cfg.label_rate * feature_ds_rate / task_cfg.sample_rate

        self.post_extract_proj = (
            nn.Linear(self.embed, cfg.encoder_embed_dim)
            if self.embed != cfg.encoder_embed_dim
            else None
        )

        self.mask_prob = cfg.mask_prob
        self.mask_selection = cfg.mask_selection
        self.mask_other = cfg.mask_other
        self.mask_length = cfg.mask_length
        self.no_mask_overlap = cfg.no_mask_overlap
        self.mask_min_space = cfg.mask_min_space

        self.mask_channel_prob = cfg.mask_channel_prob
        self.mask_channel_selection = cfg.mask_channel_selection
        self.mask_channel_other = cfg.mask_channel_other
        self.mask_channel_length = cfg.mask_channel_length
        self.no_mask_channel_overlap = cfg.no_mask_channel_overlap
        self.mask_channel_min_space = cfg.mask_channel_min_space

        self.dropout_input = nn.Dropout(cfg.dropout_input)
        self.dropout_features = nn.Dropout(cfg.dropout_features)

        self.feature_grad_mult = cfg.feature_grad_mult
        self.logit_temp = cfg.logit_temp
        self.skip_masked = cfg.skip_masked
        self.skip_nomask = cfg.skip_nomask

        final_dim = cfg.final_dim if cfg.final_dim > 0 else cfg.encoder_embed_dim

        self.mask_emb = nn.Parameter(
            torch.FloatTensor(cfg.encoder_embed_dim).uniform_()
        )

        self.split_attention = cfg.split_attention
        if self.split_attention:
            assert not cfg.add_text_encoder, f"In case using shared attention, add_text_encoder should be set to False"
            self.encoder = MoMETransformerEncoder(cfg)
        else:
            self.encoder = TransformerEncoder(cfg)
        self.layer_norm = LayerNorm(self.embed)

        self.target_glu = None
        if cfg.target_glu:
            self.target_glu = nn.Sequential(
                nn.Linear(final_dim, final_dim * 2), nn.GLU()
            )

        self.untie_final_proj = cfg.untie_final_proj
        if self.untie_final_proj:
            self.final_proj = nn.Linear(
                cfg.encoder_embed_dim, final_dim * len(dictionaries)
            )
        else:
            self.final_proj = nn.Linear(cfg.encoder_embed_dim, final_dim)

        # modules below are not needed during fine-tuning
        if any([d is None for d in dictionaries]):
            logger.info("cannot find dictionary. assume will be used for fine-tuning")
        else:
            self.num_classes = [len(d) for d in dictionaries]
            self.label_embs_concat = nn.Parameter(
                torch.FloatTensor(sum(self.num_classes), final_dim)
            )
            nn.init.uniform_(self.label_embs_concat)

        ### build speeech-text joint_pretrain net from:
        ### - add_text_modality is false: no text network
        ### - add_text_modality is true, add_text_encoder=False: build text embedding
        ### - add_text_modality is true, add_text_encoder=True: build text embedding and encoder
        self.add_text_modality = cfg.add_text_modality
        self.add_text_encoder = cfg.add_text_encoder
        self.share_text_encoder = cfg.share_text_encoder

        enc_dicti = self.cutting_dictionary(text_dictionary, cfg.encoder_dict_size)
        if self.add_text_modality and text_dictionary is not None:
            if not self.add_text_encoder:
                self.text_encoder_embed_tokens = self.build_embedding(
                    enc_dicti, cfg.text_transformer.encoder.embed_dim
                )
                self.text_padding_idx = self.text_encoder_embed_tokens.padding_idx
                self.text_embed_positions = PositionalEmbedding(
                    cfg.text_transformer.max_source_positions,
                    cfg.encoder_embed_dim,
                    self.text_padding_idx,
                    learned=False,
                )
                self.text_embed_scale = 1.0 if cfg.text_transformer.no_scale_embedding \
                    else math.sqrt(cfg.encoder_embed_dim)
            else:
                text_encoder_embed_tokens = self.build_embedding(
                    enc_dicti, cfg.text_transformer.encoder.embed_dim
                )
                self.text_encoder = TransformerEncoderBase(
                    cfg.text_transformer, 
                    enc_dicti, 
                    text_encoder_embed_tokens
                )

        self.add_decoder = task_cfg.add_decoder
        if task_cfg.add_decoder:
            # To make sure that the decoder dict size is the same as the fine-tuning tgt_dict size or bpe code dict size
            dec_dict = self.cutting_dictionary(dictionaries[0], cfg.decoder_dict_size)
            if text_dictionary is None:
                decoder_dict_list = [dec_dict]
            else:
                decoder_dict_list = [dec_dict, text_dictionary]

            decoder_embed_tokens = [
                self.build_embedding(dictionary, cfg.decoder_embed_dim)
                for dictionary in decoder_dict_list
            ]
            
            if cfg.share_all_embeddings and text_dictionary is not None:
                assert cfg.share_decoder_input_output_embed, "Must share decoder input and output embed before share all embed"
                if cfg.share_speech_text_embeddings:
                    logger.info("--------------------------------: share speech text embeddings")
                    assert len(dec_dict) == len(text_dictionary), "Speech embed len must be equal to text embed len"
                    decoder_embed_tokens[0] = text_encoder_embed_tokens if self.add_text_encoder else self.text_encoder_embed_tokens
                decoder_embed_tokens[-1] = text_encoder_embed_tokens if self.add_text_encoder else self.text_encoder_embed_tokens

            if len(decoder_embed_tokens) == 1:
                self.decoder = TransformerDecoderScriptable(cfg, decoder_dict_list[0], decoder_embed_tokens[0])
            else:
                self.decoder = MultimodalTransformerDecoder(cfg, decoder_dict_list, decoder_embed_tokens)

        self.add_adaptor = cfg.add_adaptor
        if self.add_adaptor:
            assert self.add_text_encoder, "Cannot shared encoder for text and speech once add adaptor"
            self.adaptor = Conv1dAdaptor(
                cfg.encoder_embed_dim,
                cfg.decoder_embed_dim,
                n_layers=cfg.adaptor_n_layers,
                kernel_size=cfg.adaptor_kernel_size,
                stride=cfg.adaptor_stride,
                add_layernorm=cfg.adaptor_layernorm,
            )

        if cfg.load_pretrained_w2v_from is not None:
            w2v_model_state = self.load_checkpoint(cfg.load_pretrained_w2v_from)
            self.feature_extractor = self.load_pretrained_component_from_model(
                component=self.feature_extractor, state=w2v_model_state
            )
            
            self.encoder = self.load_pretrained_component_from_model(
                component=self.encoder, state=w2v_model_state
            )
            
            self.post_extract_proj.weight = torch.nn.Parameter(w2v_model_state["model"]["post_extract_proj.weight"])
            self.post_extract_proj.bias = torch.nn.Parameter(w2v_model_state["model"]["post_extract_proj.bias"])

            # self.final_proj.weight = torch.nn.Parameter(w2v_model_state["model"]["final_proj.weight"])
            # self.final_proj.bias = torch.nn.Parameter(w2v_model_state["model"]["final_proj.bias"])

            self.layer_norm.weight = torch.nn.Parameter(w2v_model_state["model"]["layer_norm.weight"])
            self.layer_norm.bias = torch.nn.Parameter(w2v_model_state["model"]["layer_norm.bias"])

            # self.label_embs_concat.data = torch.nn.Parameter(w2v_model_state["model"]["label_embs_concat"])
            self.mask_emb.data = torch.nn.Parameter(w2v_model_state["model"]["mask_emb"])

        if cfg.load_pretrained_mbart_from is not None:
            mbart_model_state = self.load_checkpoint(cfg.load_pretrained_mbart_from)
            if self.add_text_modality and self.add_text_encoder:
                self.text_encoder = self.load_pretrained_component_from_model(
                    component=self.text_encoder, state=mbart_model_state
                )
            if self.add_decoder:
                self.decoder = self.load_pretrained_component_from_model(
                    component=self.decoder, state=mbart_model_state
                )
    
    def cutting_dictionary(self, dictionary, dict_size):
        if dictionary is None or dict_size <= 0:
            return dictionary
        else:
            cut_dictionary = copy.deepcopy(dictionary)
            if dict_size > len(cut_dictionary):
                for i in range(dict_size - len(cut_dictionary)):
                    cut_dictionary.symbols.append(f'_{i}_')
            else:
                cut_dictionary.symbols = cut_dictionary.symbols[:dict_size]
            return cut_dictionary

    def build_embedding(self, dictionary, embed_dim):
        num_embeddings = len(dictionary)
        padding_idx = dictionary.pad()
        return Embedding(num_embeddings, embed_dim, padding_idx)

    def upgrade_state_dict_named(self, state_dict, name):
        """Upgrade a (possibly old) state dict for new versions of fairseq."""

        super().upgrade_state_dict_named(state_dict, name)
        return state_dict

    @classmethod
    def build_model(cls, cfg: HubertConfig, task: HubertPretrainingTask):
        """Build a new model instance."""
        # Change dict size for bpe code
        if hasattr(task, "hubert_tokenizer") and task.hubert_tokenizer is not None and not task.fine_tuning and cfg.decoder_dict_size == -1:
            cfg.decoder_dict_size = len(task.hubert_tokenizer.sp)
            logger.info(f"set decoder dict size to {len(task.hubert_tokenizer.sp)}")

        text_dictionary = getattr(task, "text_dictionary", None)
        model = HubertModel(cfg, task.cfg, task.dictionaries, text_dictionary)
        return model

    def get_normalized_probs(
        self,
        net_output: Tuple[Tensor, Optional[Dict[str, List[Optional[Tensor]]]]],
        log_probs: bool,
        sample: Optional[Dict[str, Tensor]] = None,
    ):
        # net_output['encoder_out'] is a (B, T, D) tensor
        lprobs = self.get_normalized_probs_scriptable(net_output, log_probs, sample)
        lprobs.batch_first = True
        return lprobs

    def apply_mask(self, x, padding_mask, target_list):
        B, T, C = x.shape
        if self.mask_prob > 0:
            mask_indices = compute_mask_indices(
                (B, T),
                padding_mask,
                self.mask_prob,
                self.mask_length,
                self.mask_selection,
                self.mask_other,
                min_masks=2,
                no_overlap=self.no_mask_overlap,
                min_space=self.mask_min_space,
            )
            mask_indices = torch.from_numpy(mask_indices).to(x.device)
            x[mask_indices] = self.mask_emb
        else:
            mask_indices = None

        if self.mask_channel_prob > 0:
            mask_channel_indices = compute_mask_indices(
                (B, C),
                None,
                self.mask_channel_prob,
                self.mask_channel_length,
                self.mask_channel_selection,
                self.mask_channel_other,
                no_overlap=self.no_mask_channel_overlap,
                min_space=self.mask_channel_min_space,
            )
            mask_channel_indices = (
                torch.from_numpy(mask_channel_indices)
                .to(x.device)
                .unsqueeze(1)
                .expand(-1, T, -1)
            )
            x[mask_channel_indices] = 0

        return x, mask_indices

    def compute_nce(self, x, pos, negs):
        neg_is_pos = (pos == negs).all(-1)
        pos = pos.unsqueeze(0)
        targets = torch.cat([pos, negs], dim=0)

        logits = torch.cosine_similarity(x.float(), targets.float(), dim=-1).type_as(x)
        logits /= self.logit_temp
        if neg_is_pos.any():
            logits[1:][neg_is_pos] = float("-inf")
        logits = logits.transpose(0, 1)  # (num_x, num_cls+1)
        return logits

    def forward_features(self, source: torch.Tensor) -> torch.Tensor:
        if self.feature_grad_mult > 0:
            features = self.feature_extractor(source)
            if self.feature_grad_mult != 1.0:
                features = GradMultiply.apply(features, self.feature_grad_mult)
        else:
            with torch.no_grad():
                features = self.feature_extractor(source)
        return features

    def forward_targets(
        self,
        features: torch.Tensor,
        target_list: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Trim features to ensure labels exist and then get aligned labels
        feat_tsz = features.size(2)
        targ_tsz = min([t.size(1) for t in target_list])
        if self.feat2tar_ratio * feat_tsz > targ_tsz:
            feat_tsz = int(targ_tsz / self.feat2tar_ratio)
            features = features[..., :feat_tsz]
        target_inds = torch.arange(feat_tsz).float() * self.feat2tar_ratio
        target_list = [t[:, target_inds.long()] for t in target_list]
        return features, target_list

    def forward_padding_mask(
        self,
        features: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        extra = padding_mask.size(1) % features.size(1)
        if extra > 0:
            padding_mask = padding_mask[:, :-extra]
        padding_mask = padding_mask.view(padding_mask.size(0), features.size(1), -1)
        padding_mask = padding_mask.all(-1)
        return padding_mask

    def forward(
        self,
        source: torch.Tensor = None,
        src_tokens: torch.Tensor = None,
        src_lengths: torch.Tensor = None,
        target_list: Optional[List[torch.Tensor]] = None,
        padding_mask: Optional[torch.Tensor] = None,
        mask: bool = True,
        features_only: bool = False,
        output_layer: Optional[int] = None,
        prev_output_tokens: Optional[torch.Tensor] = None,
        text_modal_idx: Optional[int] = -1,
    ) -> Dict[str, torch.Tensor]:
        """output layer is 1-based"""
        assert source is not None or src_tokens is not None
        if source is not None:
            ### 1. go speech cnn-encoder-decoder branch
            features = self.forward_features(source)
            if target_list is not None:
                features, target_list = self.forward_targets(features, target_list)

            features_pen = features.float().pow(2).mean()

            features = features.transpose(1, 2)
            features = self.layer_norm(features)
            unmasked_features = features.clone()

            if padding_mask is not None:
                padding_mask = self.forward_padding_mask(features, padding_mask)

            if self.post_extract_proj is not None:
                features = self.post_extract_proj(features)

            features = self.dropout_input(features)
            unmasked_features = self.dropout_features(unmasked_features)

            if mask:
                x, mask_indices = self.apply_mask(features, padding_mask, target_list)
            else:
                x = features
                mask_indices = None

            # feature: (B, T, D), float
            # target: (B, T), long
            # x: (B, T, D), float
            # padding_mask: (B, T), bool
            # mask_indices: (B, T), bool
            kwargs={"modality": "speech"} if self.split_attention else {}
            x, _ = self.encoder(
                x,
                padding_mask=padding_mask,
                layer=None if output_layer is None else output_layer - 1,
                **kwargs,
            )

            if features_only:
                return {"x": x, "padding_mask": padding_mask, "features": features}

            def compute_pred(proj_x, target, label_embs):
                # compute logits for the i-th label set
                y = torch.index_select(label_embs, 0, target.long())
                negs = label_embs.unsqueeze(1).expand(-1, proj_x.size(0), -1)
                if self.target_glu:
                    y = self.target_glu(y)
                    negs = self.target_glu(negs)
                # proj_x: (S, D)
                # y: (S, D)
                # negs: (Neg, S, D)
                return self.compute_nce(proj_x, y, negs)

            label_embs_list = self.label_embs_concat.split(self.num_classes, 0)

            if not self.skip_masked:
                masked_indices = torch.logical_and(~padding_mask, mask_indices)
                proj_x_m = self.final_proj(x[masked_indices])
                if self.untie_final_proj:
                    proj_x_m_list = proj_x_m.chunk(len(target_list), dim=-1)
                else:
                    proj_x_m_list = [proj_x_m for _ in range(len(target_list))]
                logit_m_list = [
                    compute_pred(proj_x_m, t[masked_indices], label_embs_list[i])
                    for i, (proj_x_m, t) in enumerate(zip(proj_x_m_list, target_list))
                ]
            else:
                logit_m_list = [None for _ in target_list]

            if not self.skip_nomask:
                nomask_indices = torch.logical_and(~padding_mask, ~mask_indices)
                proj_x_u = self.final_proj(x[nomask_indices])
                if self.untie_final_proj:
                    proj_x_u_list = proj_x_u.chunk(len(target_list), dim=-1)
                else:
                    proj_x_u_list = [proj_x_u for _ in range(len(target_list))]

                logit_u_list = [
                    compute_pred(proj_x_u, t[nomask_indices], label_embs_list[i])
                    for i, (proj_x_u, t) in enumerate(zip(proj_x_u_list, target_list))
                ]
            else:
                logit_u_list = [None for _ in target_list]

            result = {
                "logit_m_list": logit_m_list,
                "logit_u_list": logit_u_list,
                "padding_mask": padding_mask,
                "features_pen": features_pen,
            }
            
            x = x.transpose(0, 1) # T x B x C
            # adaptor layers
            if self.add_adaptor:
                x, padding_mask = self.adaptor(x, padding_mask)

            # text encoder layers
            if self.add_text_encoder and self.share_text_encoder:
                for layer in self.text_encoder.layers:
                    x = layer(
                        x, encoder_padding_mask=padding_mask
                    )
                if self.text_encoder.layer_norm is not None:
                    x = self.text_encoder.layer_norm(x)

            # decoder layers
            if self.add_decoder:
                encoder_out = {
                    "encoder_out": [x],  # T x B x C
                    "encoder_padding_mask": [padding_mask],  # B x T
                }
                assert prev_output_tokens is not None
                decoder_out = self.decoder(
                    prev_output_tokens=prev_output_tokens, encoder_out=encoder_out
                )
                result['decoder_out'] = decoder_out
        else:
            ### 2. go text encoder-decoder branch
            if self.add_text_encoder:
                encoder_out = self.text_encoder(
                    src_tokens, src_lengths=src_lengths, return_all_hiddens=False
                )
            else:
                encoder_padding_mask = src_tokens.eq(self.text_padding_idx)
                has_pads = src_tokens.device.type == "xla" or encoder_padding_mask.any()
                x = self.text_embed_scale * self.text_encoder_embed_tokens(src_tokens)
                x = x + self.text_embed_positions(src_tokens)
                # x = self.dropout_input(x)
                if has_pads:
                    x = x * (1 - encoder_padding_mask.unsqueeze(-1).type_as(x))
                kwargs={"modality": "text"} if self.split_attention else {}
                x, _ = self.encoder(
                    x,
                    padding_mask=encoder_padding_mask,
                    conv_pos=False,
                    **kwargs,
                )
                encoder_out = {
                    "encoder_out": [x.transpose(0, 1)],  # T x B x C
                    "encoder_padding_mask": [encoder_padding_mask],  # B x T
                    "src_lengths": [src_lengths],
                }
            
            result = {"encoder_out": encoder_out}
            if features_only:
                return result
            assert prev_output_tokens is not None
            decoder_out = self.decoder(
                prev_output_tokens=prev_output_tokens, encoder_out=encoder_out, modal_idx=text_modal_idx,
            )
            result['decoder_out'] = decoder_out

        return result

    def forward_torchscript(self, net_input: Dict[str, Tensor]):
        """A TorchScript-compatible version of forward.

        Encoders which use additional arguments may want to override
        this method for TorchScript compatibility.
        """
        res = self.forward(
            mask=False,
            features_only=True,
            **net_input,
        )

        if "source" in net_input:
            res["x"] = res["x"].transpose(0, 1) # T x B x C

            x = res["x"] # T x B x C
            padding_mask = res["padding_mask"]
            if self.add_adaptor:
                x, padding_mask = self.adaptor(x, padding_mask)

            # text encoder layers
            if self.add_text_encoder and self.share_text_encoder:
                for layer in self.text_encoder.layers:
                    x = layer(
                        x, encoder_padding_mask=padding_mask
                    )

                if self.text_encoder.layer_norm is not None:
                    x = self.text_encoder.layer_norm(x)
                
                res["x"] = x
                res["padding_mask"] = padding_mask

            encoder_out = {
                "encoder_out": [res["x"]],  # T x B x C
                "encoder_padding_mask": [res["padding_mask"]],  # B x T
            }
        else:
            encoder_out = res["encoder_out"]
            if "encoder_states" in encoder_out:
                del encoder_out["encoder_states"]
            if "src_tokens" in encoder_out:
                del encoder_out["src_tokens"]
            if "src_tokens" in encoder_out:
                del encoder_out["src_lengths"]
        return encoder_out

    def extract_features(
        self,
        source: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        mask: bool = False,
        ret_conv: bool = False,
        output_layer: Optional[int] = None,
        prev_output_tokens: Optional[torch.Tensor] = None,
        ft: bool = True,
        enc_grad_mult: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """only for speech input"""
        with torch.no_grad() if not ft else contextlib.ExitStack():
            res = self.forward(
                source,
                padding_mask=padding_mask,
                mask=mask,
                features_only=True,
                output_layer=output_layer,
            )

        feature = res["features"] if ret_conv else res["x"]

        res["x"] = res["x"].transpose(0, 1) # T x B x C
        x = res["x"] # T x B x C
        padding_mask = res["padding_mask"]
        if self.add_adaptor:
            x, padding_mask = self.adaptor(x, padding_mask)

        # text encoder layers
        if self.add_text_encoder and self.share_text_encoder:
            for layer in self.text_encoder.layers:
                x = layer(
                    x, encoder_padding_mask=padding_mask
                )

            if self.text_encoder.layer_norm is not None:
                x = self.text_encoder.layer_norm(x)
            
            res["x"] = x
            res["padding_mask"] = padding_mask

        if self.add_decoder and prev_output_tokens is not None:
            encoder_out = {
                "encoder_out": [res["x"]],  # T x B x C
                "encoder_padding_mask": [res["padding_mask"]],  # B x T
            }
            
            if enc_grad_mult != 1.0:
                encoder_out = self.mult_rst_grad(encoder_out, enc_grad_mult)

            assert prev_output_tokens is not None
            decoder_out = self.decoder(
                prev_output_tokens=prev_output_tokens, 
                encoder_out=encoder_out,
            )
        else:
            decoder_out = None
        return feature, res["padding_mask"], decoder_out

    def mult_rst_grad(self, rst, ratio):
        assert isinstance(rst, dict)  # instead of EncoderOut
        assert len(rst["encoder_out"]) == 1
        rst["encoder_out"][0] = GradMultiply.apply(rst["encoder_out"][0], ratio)
        return rst

    def get_logits(self, net_output, is_masked=True):
        if is_masked:
            logits_list = net_output["logit_m_list"]
        else:
            logits_list = net_output["logit_u_list"]
        logits_list = [x.float() for x in logits_list if x is not None]
        return logits_list

    def get_targets(self, net_output, is_masked=True):
        logits_list = self.get_logits(net_output, is_masked)
        targets_list = [x.new_zeros(x.size(0), dtype=torch.long) for x in logits_list]
        return targets_list

    def get_extra_losses(self, net_output):
        extra_losses = []
        names = []

        if "features_pen" in net_output:
            extra_losses.append(net_output["features_pen"])
            names.append("features_pen")

        return extra_losses, names

    def remove_pretraining_modules(self, step2=False):
        self.target_glu = None
        self.final_proj = None
        if self.add_text_modality:
            # Delete text embeddings of text encoder
            if not step2:
                if self.add_text_encoder:
                    self.text_encoder.embed_tokens = None
                    if hasattr(self.text_encoder, "embed_positions"):
                        self.text_encoder.embed_tokens = None
                    if hasattr(self.text_encoder, "layernorm_embedding"):
                        self.text_encoder.layernorm_embedding = None
                else:
                    self.text_encoder_embed_tokens = None
                    self.text_embed_positions = None
            if isinstance(self.decoder, MultimodalTransformerDecoder):
                # Delete text embeddings of decoder
                self.decoder.embed_tokens_list = self.decoder.embed_tokens_list[:1]
                self.decoder.output_projection = self.decoder.output_projection[:1]

    def load_checkpoint(self, checkpoint: str):
        if not PathManager.exists(checkpoint):
            raise IOError("Model file not found: {}".format(checkpoint))
        state = checkpoint_utils.load_checkpoint_to_cpu(checkpoint)
        return state
        
    def load_pretrained_component_from_model(
        self, component: Union[TransformerEncoderBase, MoMETransformerEncoder, TransformerEncoder, FairseqDecoder, ConvFeatureExtractionModel], state
    ):
        """
        Load a pretrained FairseqEncoder or FairseqDecoder from checkpoint into the
        provided `component` object. If state_dict fails to load, there may be a
        mismatch in the architecture of the corresponding `component` found in the
        `checkpoint` file.
        """
        if isinstance(component, (TransformerEncoderBase, MoMETransformerEncoder, TransformerEncoder)):
            component_type = "encoder"
        elif isinstance(component, FairseqDecoder):
            component_type = "decoder"
            if isinstance(component, MultimodalTransformerDecoder):
                state["model"]["decoder.embed_tokens_list.1.weight"] = state["model"]["decoder.embed_tokens.weight"]
                state["model"]["decoder.output_projection.1.weight"] = state["model"]["decoder.output_projection.weight"]
        elif isinstance(component, ConvFeatureExtractionModel):
            component_type = "feature_extractor"
        else:
            raise ValueError(
                "component to load must be either a FairseqEncoder or "
                "FairseqDecoder. Loading other component types are not supported."
            )
        component_state_dict = OrderedDict()
        for key in state["model"].keys():
            if key.startswith(component_type):
                # encoder.input_layers.0.0.weight --> input_layers.0.0.weight
                component_subkey = key[len(component_type) + 1 :]
                component_state_dict[component_subkey] = state["model"][key]
        try:
            logger.info(f"Load {component_type}")
            component.load_state_dict(component_state_dict, strict=True)
        except Exception as e:
            logger.warn(e)
            component.load_state_dict(component_state_dict, strict=False)
        return component

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
import torch.nn.functional as F
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

from fairseq.models.hubert.stbert import HubertConfig

logger = logging.getLogger(__name__)

EXTRACTOR_MODE_CHOICES = ChoiceEnum(["default", "layer_norm"])
MASKING_DISTRIBUTION_CHOICES = ChoiceEnum(["static", "uniform", "normal", "poisson"])


@dataclass
class HubertConfig(FairseqDataclass):
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
    add_code_encoder: bool = field(
        default=False,
        metadata={"help": "add_code_encoder"},
    )
    seek_code_embedding: bool = field(
        default=True,
        metadata={"help": "get soft code embeddings"},
    )
    prob_code_embedding: bool = field(
        default=True,
        metadata={"help": "get soft code embeddings based on the probility of code"},
    )
    pad_with_code: bool = field(
        default=False,
        metadata={"help": "pad the masked potitions with code, as the input to the code encoder"},
    )
    ### deprecated arguments
    ############################################################################
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
    ############################################################################

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


@register_model("codebert", dataclass=HubertConfig)
class CodeBertModel(BaseFairseqModel):
    def __init__(
        self,
        cfg: HubertConfig,
        dictionaries: List[Dictionary],
        code_dictionary: Dictionary = None,
    ) -> None:
        super().__init__()
        logger.info(f"CodeBert Model Config: {cfg}")

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

        self.logit_temp = cfg.logit_temp
        self.skip_masked = cfg.skip_masked
        self.skip_nomask = cfg.skip_nomask

        final_dim = cfg.final_dim if cfg.final_dim > 0 else cfg.text_transformer.encoder.embed_dim

        self.mask_emb = nn.Parameter(
            torch.FloatTensor(cfg.text_transformer.encoder.embed_dim).uniform_()
        )

        self.target_glu = None
        if cfg.target_glu:
            self.target_glu = nn.Sequential(
                nn.Linear(final_dim, final_dim * 2), nn.GLU()
            )

        self.untie_final_proj = cfg.untie_final_proj
        if self.untie_final_proj:
            self.final_proj = nn.Linear(
                cfg.text_transformer.encoder.embed_dim, final_dim * len(dictionaries)
            )
        else:
            self.final_proj = nn.Linear(cfg.text_transformer.encoder.embed_dim, final_dim)

        # modules below are not needed during fine-tuning
        if any([d is None for d in dictionaries]):
            logger.info("cannot find dictionary. assume will be used for fine-tuning")
        else:
            self.num_classes = [len(d) for d in dictionaries]
            self.label_embs_concat = nn.Parameter(
                torch.FloatTensor(sum(self.num_classes), final_dim)
            )
            nn.init.uniform_(self.label_embs_concat)

        ### build code encoder:
        assert cfg.add_code_encoder
        assert len(code_dictionary) == sum(self.num_classes), f"{len(code_dictionary)} {sum(self.num_classes)}"

        self.code_embed_tokens = self.build_embedding(
                code_dictionary, cfg.text_transformer.encoder.embed_dim
            )
        ### change self.label_embs_concat to code_embeddings
        if final_dim == cfg.text_transformer.encoder.embed_dim:
            logger.info("change label_embs_concat to code_embeddings")
            nn.init.uniform_(self.code_embed_tokens.weight)
            self.label_embs_concat = self.code_embed_tokens.weight

        ### build code encoder
        self.encoder = TransformerEncoder(cfg)
        # self.code_encoder = TransformerEncoderBase(
        #         cfg.text_transformer, 
        #         code_dictionary, 
        #         self.code_embed_tokens
        #     )

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
        code_dictionary = task.src_dict
        model = CodeBertModel(cfg, [code_dictionary], code_dictionary)
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

    def forward(
        self,
        src_tokens: torch.Tensor = None,
        src_lengths: torch.Tensor = None,
        padding_mask: Optional[torch.Tensor] = None,
        mask: bool = True,
        features_only: bool = False,
        output_layer: Optional[int] = None,
        prev_output_tokens: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:

        features = self.code_embed_tokens(src_tokens)
        target_list = [src_tokens]
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

        x, _ = self.encoder(
            x,
            padding_mask=padding_mask,
            layer=None if output_layer is None else output_layer - 1,
        )
        encoder_out = {'encoder_out': [x.transpose(0, 1)], 'padding_mask': padding_mask}
        # encoder_out = self.code_encoder(src_tokens, token_embeddings=x)
        x = encoder_out['encoder_out'][0].transpose(0, 1)

        if features_only:
            return encoder_out

        def compute_pred(proj_x, target, label_embs):
            ## this is equivalent but at least 5x faster than the original code
            if self.target_glu:
                label_embs = self.target_glu(label_embs)

            x = F.normalize(proj_x.float(), dim=-1)                 # (S, D)
            label_embs = F.normalize(label_embs.float(), dim=-1)    # (C, D)
            logits = torch.matmul(x, label_embs.T).type_as(proj_x)  # (S, C)
            logits /= self.logit_temp
            return (logits, target)

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
        }
        return result

    def get_logits(self, net_output, is_masked=True):
        if is_masked:
            logits_list = net_output["logit_m_list"]
        else:
            logits_list = net_output["logit_u_list"]
        logits_list = [x[0].float() for x in logits_list if x is not None]
        return logits_list

    def get_targets(self, net_output, is_masked=True):
        if is_masked:
            logits_list = net_output["logit_m_list"]
        else:
            logits_list = net_output["logit_u_list"]
        targets_list = [x[1].long() for x in logits_list if x is not None]
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

# --------------------------------------------------------
# Pre-Training Transformer Decoder for End-to-End ASR Model with Unpaired Speech Data (https://arxiv.org/abs/2203.17113)
# Github source: https://github.com/microsoft/SpeechT5/tree/main/Speech2C
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Based on fairseq code bases
# https://github.com/pytorch/fairseq
# --------------------------------------------------------

import logging
import copy
import contextlib
from typing import Dict, List, Optional, Tuple

import torch
from dataclasses import dataclass, field
from fairseq.data.dictionary import Dictionary
from fairseq.models import register_model
from fairseq.models.hubert import HubertConfig, HubertModel
from fairseq.models.transformer import Embedding
from torch import Tensor
from speech2c.tasks.speech2c_pretraining import (
    Speech2cPretrainingConfig,
    Speech2cPretrainingTask,
)

from speech2c.models.modules.transformer_decoder import TransformerDecoderScriptable
from speech2c.models.modules.transformer_encoder import TransformerEncoder

logger = logging.getLogger(__name__)


@dataclass
class Speech2cConfig(HubertConfig):
    use_rel_pos_enc: bool = field(
        default=False,
        metadata={"help": "whether to use relative positional encoding"},
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
    decoder_layerdrop: float = field(
        default=0.0,
        metadata={"help": "probability of dropping a tarnsformer layer"},
    )
    share_decoder_input_output_embed: bool = field(
        default=False,
        metadata={"help": "share decoder input and output embeddings"},
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
    decoder_dict_size: int = field(
        default=-1,
        metadata={"help": "decoder dictionary dimension, only used for fine-tuning"},
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


@register_model("speech2c", dataclass=Speech2cConfig)
class Speech2cModel(HubertModel):
    def __init__(
        self,
        cfg: Speech2cConfig,
        task_cfg: Speech2cPretrainingConfig,
        dictionaries: List[Dictionary],
    ) -> None:
        super().__init__(cfg, task_cfg, dictionaries)
        logger.info(f"Speech2cModel Config: {cfg}")

        self.encoder = TransformerEncoder(cfg)

        self.add_decoder = task_cfg.add_decoder
        if task_cfg.add_decoder:
            def build_embedding(dictionary, embed_dim):
                num_embeddings = len(dictionary)
                padding_idx = dictionary.pad()
                return Embedding(num_embeddings, embed_dim, padding_idx)

            # To make sure that the decoder dict size is the same as the fine-tuning tgt_dict size
            cut_dictionary = copy.deepcopy(dictionaries[0])
            if cfg.decoder_dict_size != -1:
                cut_dictionary.symbols = cut_dictionary.symbols[:cfg.decoder_dict_size]

            decoder_embed_tokens = build_embedding(
                cut_dictionary, cfg.decoder_embed_dim
            )

            self.decoder = TransformerDecoderScriptable(cfg, cut_dictionary, decoder_embed_tokens)


    @classmethod
    def build_model(cls, cfg: Speech2cConfig, task: Speech2cPretrainingTask):
        """Build a new model instance."""

        model = Speech2cModel(cfg, task.cfg, task.dictionaries)
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

    def forward(
        self,
        source: torch.Tensor,
        target_list: Optional[List[torch.Tensor]] = None,
        padding_mask: Optional[torch.Tensor] = None,
        mask: bool = True,
        features_only: bool = False,
        output_layer: Optional[int] = None,
        prev_output_tokens: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """output layer is 1-based"""
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
        x, _ = self.encoder(
            x,
            padding_mask=padding_mask,
            layer=None if output_layer is None else output_layer - 1,
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
        if self.add_decoder:
            encoder_out = {
                "encoder_out": [x.transpose(0, 1)],  # T x B x C
                "encoder_padding_mask": [padding_mask],  # B x T
            }
            assert prev_output_tokens is not None
            decoder_out = self.decoder(
                prev_output_tokens=prev_output_tokens, encoder_out=encoder_out
            )
            result['decoder_out'] = decoder_out
        return result

    def forward_torchscript(self, net_input: Dict[str, Tensor]):
        """A TorchScript-compatible version of forward.
        Encoders which use additional arguments may want to override
        this method for TorchScript compatibility.
        """
        res = self.forward(
            net_input["source"],
            padding_mask=net_input["padding_mask"],
            mask=False,
            features_only=True
        )

        encoder_out = {
            "encoder_out": [res["x"].transpose(0, 1)],  # T x B x C
            "encoder_padding_mask": [res["padding_mask"]],  # B x T
        }
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
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad() if not ft else contextlib.ExitStack():
            res = self.forward(
                source,
                padding_mask=padding_mask,
                mask=mask,
                features_only=True,
                output_layer=output_layer,
            )
            
        feature = res["features"] if ret_conv else res["x"]
        if self.add_decoder:
            encoder_out = {
                "encoder_out": [feature.transpose(0, 1)],  # T x B x C
                "encoder_padding_mask": [res["padding_mask"]],  # B x T
            }
            assert prev_output_tokens is not None
            decoder_out = self.decoder(
                prev_output_tokens=prev_output_tokens, 
                encoder_out=encoder_out,
            )
        else:
            decoder_out = None
        return feature, res["padding_mask"], decoder_out

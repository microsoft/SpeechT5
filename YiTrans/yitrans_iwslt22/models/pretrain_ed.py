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
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union
from collections import OrderedDict

import copy
import torch
from omegaconf import II

from fairseq import checkpoint_utils
from fairseq.data.dictionary import Dictionary
from fairseq.dataclass import ChoiceEnum
from fairseq.models import register_model, FairseqDecoder
from fairseq.models.transformer import (
    TransformerEncoderBase,
    TransformerConfig,
)
from fairseq.models.speech_to_text import Conv1dAdaptor
from fairseq.models.transformer import Embedding
from fairseq.file_io import PathManager
from torch import Tensor
from fairseq.models.wav2vec.wav2vec2 import ConvFeatureExtractionModel
from fairseq.modules import GradMultiply

from fairseq.models.hubert import HubertConfig, HubertModel

from fairseq.models.wav2vec.wav2vec2 import TransformerEncoder as W2vTransformerEncoder
from yitrans_iwslt22.modules.w2v_encoder import TransformerEncoder
from yitrans_iwslt22.modules.transformer_decoder import TransformerDecoderScriptable
from yitrans_iwslt22.modules.multimodal_transformer_decoder import MultimodalTransformerDecoder
from yitrans_iwslt22.tasks.iwslt_joint_pretraining import (
    JointPretrainingConfig,
    JointPretrainingTask,
)

logger = logging.getLogger(__name__)

EXTRACTOR_MODE_CHOICES = ChoiceEnum(["default", "layer_norm"])
MASKING_DISTRIBUTION_CHOICES = ChoiceEnum(["static", "uniform", "normal", "poisson"])


@dataclass
class JointEDConfig(HubertConfig):
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
    share_enc_dec_embeddings: bool = field(
        default=False,
        metadata={"help": "share embeddings of (text encoder, text decoder)"},
    )
    share_s2t_t2t_embeddings: bool = field(
        default=False,
        metadata={"help": "share embeddings of (speech2text(code), text2text)"},
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
    # Finetune related
    decoder_dict_size: int = field(
        default=-1,
        metadata={"help": "decoder dictionary dimension"},
    )

    # text encoder related, TransformerConfig is used in bart but we only use its enconder
    text_transformer: TransformerConfig = TransformerConfig()

    # other
    checkpoint_activations: bool = field(
        default=False, metadata={"help": "recompute activations and save memory for extra compute"}
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

@register_model("joint_ed", dataclass=JointEDConfig)
class JointEDModel(HubertModel):
    def __init__(
        self,
        cfg: JointEDConfig,
        task_cfg: JointPretrainingConfig,
        dictionaries: List[Dictionary],
        text_dictionary: Dictionary = None,
    ) -> None:
        super().__init__(cfg, task_cfg, dictionaries)
        logger.info(f"JointEDModel Config: {cfg}")

        self.encoder = TransformerEncoder(cfg)

        ### build speeech-text joint_pretrain net from:
        ### - add_text_modality is false: no text network
        ### - add_text_modality is true, add_text_encoder=False: build text embedding
        ### - add_text_modality is true, add_text_encoder=True: build text embedding and encoder
        assert cfg.add_text_modality
        assert cfg.add_text_encoder
        assert cfg.share_text_encoder
        assert text_dictionary is not None
        self.add_text_modality = cfg.add_text_modality
        self.add_text_encoder = cfg.add_text_encoder
        self.share_text_encoder = cfg.share_text_encoder

        if cfg.share_s2t_t2t_embeddings:
            text_dictionary = self.cutting_dictionary(text_dictionary, cfg.decoder_dict_size)
        
        ### build text encoder
        text_encoder_embed_tokens = self.build_embedding(
                text_dictionary, cfg.text_transformer.encoder.embed_dim
            )
        self.text_encoder = TransformerEncoderBase(
            cfg.text_transformer, 
            text_dictionary, 
            text_encoder_embed_tokens
        )
        
        ### build text decoder
        self.add_decoder = task_cfg.add_decoder
        if self.add_decoder:
            # To make sure that the decoder dict size is the same as the fine-tuning tgt_dict size or bpe code dict size
            s2t_dec_dict = self.cutting_dictionary(dictionaries[0], cfg.decoder_dict_size)
            if text_dictionary is None:
                decoder_dict_list = [s2t_dec_dict]
            else:
                decoder_dict_list = [s2t_dec_dict, text_dictionary]

            decoder_embed_tokens = [
                self.build_embedding(dictionary, cfg.decoder_embed_dim)
                for dictionary in decoder_dict_list
            ]
            
            if cfg.share_enc_dec_embeddings and text_dictionary is not None:
                assert cfg.share_decoder_input_output_embed, "Must share decoder input-output embed before share encoder-decoder embed"
                logger.info("--------------------------------: share input-output embeddings")
                decoder_embed_tokens[-1] = text_encoder_embed_tokens
            
            if cfg.share_s2t_t2t_embeddings:
                logger.info("--------------------------------: share s2t-t2t embeddings")
                assert len(s2t_dec_dict) == len(text_dictionary), "s2t embed len must be equal to t2t embed len"
                decoder_embed_tokens[0] = text_encoder_embed_tokens

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

    @classmethod
    def build_model(cls, cfg: HubertConfig, task: JointPretrainingTask):
        """Build a new model instance."""
        # Change dict size for bpe code
        if hasattr(task, "hubert_tokenizer") and task.hubert_tokenizer is not None and not task.fine_tuning and cfg.decoder_dict_size == -1:
            cfg.decoder_dict_size = len(task.hubert_tokenizer.sp)
            logger.info(f"Use acoustic pieces as code, set decoder dict size to {len(task.hubert_tokenizer.sp)}")

        text_dictionary = getattr(task, "text_dictionary", None)
        model = JointEDModel(cfg, task.cfg, task.dictionaries, text_dictionary)
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
        self, component: Union[TransformerEncoderBase, TransformerEncoder, W2vTransformerEncoder, FairseqDecoder, ConvFeatureExtractionModel], state
    ):
        """
        Load a pretrained FairseqEncoder or FairseqDecoder from checkpoint into the
        provided `component` object. If state_dict fails to load, there may be a
        mismatch in the architecture of the corresponding `component` found in the
        `checkpoint` file.
        """
        if isinstance(component, (TransformerEncoderBase, TransformerEncoder, W2vTransformerEncoder)):
            component_type = "encoder"
        elif isinstance(component, FairseqDecoder):
            component_type = "decoder"
            if isinstance(component, MultimodalTransformerDecoder):
                state["model"]["decoder.embed_tokens_list.1.weight"] = state["model"]["decoder.embed_tokens.weight"]
                state["model"]["decoder.output_projection.1.weight"] = state["model"]["decoder.output_projection.weight"]
        elif isinstance(component, ConvFeatureExtractionModel):
            component_type = "feature_extractor"
        else:
            print(component)
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

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq import utils
from fairseq.dataclass import ChoiceEnum
from fairseq.modules import (
    LayerNorm,
    MultiheadAttention,
    SamePad,
    RelativePositionalEncoding,
)
from fairseq.modules.checkpoint_activations import checkpoint_wrapper
from fairseq.modules.transformer_sentence_encoder import init_bert_params
from fairseq.utils import index_put
from fairseq.distributed import fsdp_wrap


EXTRACTOR_MODE_CHOICES = ChoiceEnum(["default", "layer_norm"])
MASKING_DISTRIBUTION_CHOICES = ChoiceEnum(["static", "uniform", "normal", "poisson"])


from fairseq.modules import (
    LayerNorm,
    MultiheadAttention,
    SamePad,
    RelativePositionalEncoding,
)
import torch.nn.functional as F
from fairseq.modules.checkpoint_activations import checkpoint_wrapper
from fairseq.modules.transformer_sentence_encoder import init_bert_params
from fairseq.utils import index_put
from fairseq.distributed import fsdp_wrap

class MoMETransformerEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.dropout = args.dropout
        self.embedding_dim = args.encoder_embed_dim
        self.num_experts = args.moe_number

        if getattr(args, "moe_part", "ffn") == "attn":
            MoMETransformerSentenceEncoderLayer = MoAttTransformerSentenceEncoderLayer
        else:
            MoMETransformerSentenceEncoderLayer = MoFFnTransformerSentenceEncoderLayer

        self.pos_conv = nn.Conv1d(
            self.embedding_dim,
            self.embedding_dim,
            kernel_size=args.conv_pos,
            padding=args.conv_pos // 2,
            groups=args.conv_pos_groups,
        )
        dropout = 0
        std = math.sqrt((4 * (1.0 - dropout)) / (args.conv_pos * self.embedding_dim))
        nn.init.normal_(self.pos_conv.weight, mean=0, std=std)
        nn.init.constant_(self.pos_conv.bias, 0)

        self.pos_conv = nn.utils.weight_norm(self.pos_conv, name="weight", dim=2)
        self.pos_conv = nn.Sequential(self.pos_conv, SamePad(args.conv_pos), nn.GELU())

        layers = []
        self.use_rel_pos_enc = getattr(args, "use_rel_pos_enc", False)
        for _ in range(args.encoder_layers):
            layer = MoMETransformerSentenceEncoderLayer(
                    embedding_dim=self.embedding_dim,
                    ffn_embedding_dim=args.encoder_ffn_embed_dim,
                    num_attention_heads=args.encoder_attention_heads,
                    dropout=self.dropout,
                    attention_dropout=args.attention_dropout,
                    activation_dropout=args.activation_dropout,
                    activation_fn=args.activation_fn,
                    layer_norm_first=args.layer_norm_first,
                    has_relative_attention_bias=self.use_rel_pos_enc,
                    num_experts=args.moe_number
            )
            if args.checkpoint_activations:
                layer = fsdp_wrap(layer)
                layer = checkpoint_wrapper(layer)
            layers.append(layer)
        self.layers = nn.ModuleList(layers)

        self.layer_norm_first = args.layer_norm_first
        self.layer_norm = LayerNorm(self.embedding_dim)
        self.layerdrop = args.encoder_layerdrop

        if self.use_rel_pos_enc:
            self.pos_emb = nn.ModuleList([
                RelativePositionalEncoding(args.encoder_embed_dim // args.encoder_attention_heads, 160 // (i + 1))
                for i in range(self.num_experts)
            ])
        self.apply(init_bert_params)

    def forward(self, x, padding_mask=None, layer=None, conv_pos=True, modal_idx=None):
        x, layer_results = self.extract_features(x, padding_mask, layer, conv_pos, modal_idx)

        if self.layer_norm_first and layer is None:
            x = self.layer_norm(x)

        return x, layer_results

    def extract_features(self, x, padding_mask=None, tgt_layer=None, conv_pos=True, modal_idx=None):
        if padding_mask is not None:
            x = index_put(x, padding_mask, 0)

        if conv_pos:
            x_conv = self.pos_conv(x.transpose(1, 2))
            x_conv = x_conv.transpose(1, 2)
            x = x + x_conv

        if not self.layer_norm_first:
            x = self.layer_norm(x)

        x = F.dropout(x, p=self.dropout, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        if self.use_rel_pos_enc:
            x_len = x.shape[0]
            pos_seq = torch.arange(0, x_len).long().to(x.device)
            pos_seq = pos_seq[:, None] - pos_seq[None, :]
            pos_k, pos_v = self.pos_emb[modal_idx](pos_seq)
        else:
            pos_k = None

        layer_results = []
        r = None
        for i, layer in enumerate(self.layers):
            dropout_probability = np.random.random()
            if not self.training or (dropout_probability > self.layerdrop):
                x, z = layer(x, self_attn_padding_mask=padding_mask, need_weights=False, pos_bias=pos_k, modal_idx=modal_idx)
                if tgt_layer is not None:
                    layer_results.append((x, z))
            if i == tgt_layer:
                r = x
                break

        if r is not None:
            x = r

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        return x, layer_results

    def max_positions(self):
        """Maximum output length supported by the encoder."""
        return self.args.max_positions

    def upgrade_state_dict_named(self, state_dict, name):
        """Upgrade a (possibly old) state dict for new versions of fairseq."""
        return state_dict


class MoFFnTransformerSentenceEncoderLayer(nn.Module):
    """
    Implements a Transformer Encoder Layer used in BERT/XLM style pre-trained
    models.
    """

    def __init__(
        self,
        embedding_dim: float = 768,
        ffn_embedding_dim: float = 3072,
        num_attention_heads: float = 8,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.1,
        activation_fn: str = "relu",
        layer_norm_first: bool = False,
        has_relative_attention_bias: bool = False,
        num_experts: int = 2,
    ) -> None:

        super().__init__()
        # Initialize parameters
        self.embedding_dim = embedding_dim
        self.dropout = dropout
        self.activation_dropout = activation_dropout
        self.num_experts = num_experts

        # Initialize blocks
        self.activation_fn = utils.get_activation_fn(activation_fn)
        self.self_attn = MultiheadAttention(
            self.embedding_dim,
            num_attention_heads,
            dropout=attention_dropout,
            self_attention=True,
            has_relative_attention_bias=has_relative_attention_bias,
        )

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(self.activation_dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.layer_norm_first = layer_norm_first

        # layer norm associated with the self attention layer
        self.self_attn_layer_norm = LayerNorm(self.embedding_dim)
        self.fc1 = nn.ModuleList([
            nn.Linear(self.embedding_dim, ffn_embedding_dim)
            for _ in range(self.num_experts)
        ])
        self.fc2 = nn.ModuleList([
            nn.Linear(ffn_embedding_dim, self.embedding_dim)
            for _ in range(self.num_experts)
        ])

        # layer norm associated with the position wise feed-forward NN
        self.final_layer_norm = nn.ModuleList([
            LayerNorm(self.embedding_dim)
            for _ in range(self.num_experts)
        ])

        if has_relative_attention_bias:
            self.norm_k = LayerNorm(self.embedding_dim//num_attention_heads)

    def forward(
        self,
        x: torch.Tensor,
        self_attn_mask: torch.Tensor = None,
        self_attn_padding_mask: torch.Tensor = None,
        need_weights: bool = False,
        att_args=None,
        pos_bias=None,
        modal_idx=None,
    ):
        """
        LayerNorm is applied either before or after the self-attention/ffn
        modules similar to the original Transformer imlementation.
        """
        residual = x

        if self.layer_norm_first:
            x = self.self_attn_layer_norm(x)
            if pos_bias is not None:
                pos_bias = self.norm_k(pos_bias)
            x, attn = self.self_attn(
                query=x,
                key=x,
                value=x,
                key_padding_mask=self_attn_padding_mask,
                attn_mask=self_attn_mask,
                position_bias=pos_bias,
            )
            x = self.dropout1(x)
            x = residual + x

            residual = x
            x = self.final_layer_norm[modal_idx](x)
            x = self.activation_fn(self.fc1[modal_idx](x))
            x = self.dropout2(x)
            x = self.fc2[modal_idx](x)
            x = self.dropout3(x)
            x = residual + x
        else:
            x, attn = self.self_attn(
                query=x,
                key=x,
                value=x,
                key_padding_mask=self_attn_padding_mask,
                position_bias=pos_bias,
            )

            x = self.dropout1(x)
            x = residual + x

            x = self.self_attn_layer_norm(x)

            residual = x
            x = self.activation_fn(self.fc1[modal_idx](x))
            x = self.dropout2(x)
            x = self.fc2[modal_idx](x)
            x = self.dropout3(x)
            x = residual + x
            x = self.final_layer_norm[modal_idx](x)

        return x, attn


class MoAttTransformerSentenceEncoderLayer(nn.Module):
    """
    Implements a Transformer Encoder Layer used in BERT/XLM style pre-trained
    models.
    """

    def __init__(
        self,
        embedding_dim: float = 768,
        ffn_embedding_dim: float = 3072,
        num_attention_heads: float = 8,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.1,
        activation_fn: str = "relu",
        layer_norm_first: bool = False,
        has_relative_attention_bias: bool = False,
        num_experts: int = 2,
    ) -> None:

        super().__init__()
        # Initialize parameters
        self.embedding_dim = embedding_dim
        self.dropout = dropout
        self.activation_dropout = activation_dropout
        self.num_experts = num_experts

        # Initialize blocks
        self.activation_fn = utils.get_activation_fn(activation_fn)
        self.self_attn = nn.ModuleList([
            MultiheadAttention(
                self.embedding_dim,
                num_attention_heads,
                dropout=attention_dropout,
                self_attention=True,
                has_relative_attention_bias=has_relative_attention_bias,
            ) for _ in range(self.num_experts)
        ])


        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(self.activation_dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.layer_norm_first = layer_norm_first

        # layer norm associated with the self attention layer
        self.self_attn_layer_norm = nn.ModuleList([
            LayerNorm(self.embedding_dim)
            for _ in range(self.num_experts)
        ])
        self.fc1 = nn.Linear(self.embedding_dim, ffn_embedding_dim)
        self.fc2 = nn.Linear(ffn_embedding_dim, self.embedding_dim)

        # layer norm associated with the position wise feed-forward NN
        self.final_layer_norm = LayerNorm(self.embedding_dim)

        if has_relative_attention_bias:
            self.norm_k = nn.ModuleList([
                LayerNorm(self.embedding_dim//num_attention_heads)
                for _ in range(self.num_experts)
        ])

    def forward(
        self,
        x: torch.Tensor,
        self_attn_mask: torch.Tensor = None,
        self_attn_padding_mask: torch.Tensor = None,
        need_weights: bool = False,
        att_args=None,
        pos_bias=None,
        modal_idx=None,
    ):
        """
        LayerNorm is applied either before or after the self-attention/ffn
        modules similar to the original Transformer imlementation.
        """
        residual = x

        if self.layer_norm_first:
            x = self.self_attn_layer_norm[modal_idx](x)
            if pos_bias is not None:
                pos_bias = self.norm_k[modal_idx](pos_bias)
            x, attn = self.self_attn[modal_idx](
                query=x,
                key=x,
                value=x,
                key_padding_mask=self_attn_padding_mask,
                attn_mask=self_attn_mask,
                position_bias=pos_bias,
            )
            x = self.dropout1(x)
            x = residual + x

            residual = x
            x = self.final_layer_norm(x)
            x = self.activation_fn(self.fc1(x))
            x = self.dropout2(x)
            x = self.fc2(x)
            x = self.dropout3(x)
            x = residual + x
        else:
            x, attn = self.self_attn[modal_idx](
                query=x,
                key=x,
                value=x,
                key_padding_mask=self_attn_padding_mask,
                position_bias=pos_bias,
            )

            x = self.dropout1(x)
            x = residual + x

            x = self.self_attn_layer_norm[modal_idx](x)

            residual = x
            x = self.activation_fn(self.fc1(x))
            x = self.dropout2(x)
            x = self.fc2(x)
            x = self.dropout3(x)
            x = residual + x
            x = self.final_layer_norm(x)

        return x, attn

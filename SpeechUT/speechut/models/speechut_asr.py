# ----------------------------------------------------------------------------
# SpeechUT: Bridging Speech and Text with Hidden-Unit for Encoder-Decoder Based Speech-Text Pre-training (https://arxiv.org/abs/2210.03730)
# Github source: https://github.com/microsoft/SpeechT5/tree/main/SpeechUT
# Code based on fairseq: https://github.com/facebookresearch/fairseq/tree/272c4c5197250997148fb12c0db6306035f166a4
# 
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# ----------------------------------------------------------------------------

import contextlib
import torch
from dataclasses import dataclass, field
from fairseq import utils
from fairseq.models import BaseFairseqModel, register_model
from fairseq.models.fairseq_encoder import FairseqEncoder
from fairseq.models.hubert import HubertAsrConfig, HubertEncoder
from fairseq.tasks import FairseqTask

@dataclass
class SpeechUTASRConfig(HubertAsrConfig):
    add_decoder: bool = field(
        default=True,
        metadata={"help": "add decoder for fine-tune"},
    )

@register_model("speechut_asr", dataclass=SpeechUTASRConfig)
class SpeechUTASR(BaseFairseqModel):
    """
    A encoder-ctc-decoder model if cfg.add_decoder is True, or a encoder-ctc model
    """
    def __init__(self, cfg: SpeechUTASRConfig, encoder: FairseqEncoder):
        super().__init__()
        self.cfg = cfg
        self.encoder = encoder
        if not cfg.add_decoder:
            self.encoder.w2v_model.decoder = None

    def upgrade_state_dict_named(self, state_dict, name):
        super().upgrade_state_dict_named(state_dict, name)
        return state_dict
    
    @classmethod
    def build_model(cls, cfg: SpeechUTASRConfig, task: FairseqTask):
        """Build a new model instance."""
        encoder = SpeechUTEncoder(cfg, task)
        return cls(cfg, encoder)

    def forward(self, source, padding_mask, prev_output_tokens, **kwargs):
        encoder_out = self.encoder(source, padding_mask, **kwargs)

        x = self.encoder.final_dropout(encoder_out['encoder_out'][0])  # (T, B, C)
        if self.encoder.proj:
            x = self.encoder.proj(x)
        if self.encoder.conv_ctc_proj:
            padding_mask = self.encoder.w2v_model.downsample_ctc_padding_mask(encoder_out["encoder_padding_mask"][0])
        else:
            padding_mask = encoder_out["encoder_padding_mask"]

        decoder_out = self.decoder(
            prev_output_tokens, encoder_out=encoder_out, **kwargs
        ) if self.cfg.add_decoder else None
        
        return {
            "encoder_out_ctc": x,           # (T, B, C), for CTC loss
            "padding_mask": padding_mask,   # (B, T), for CTC loss
            "decoder_out": decoder_out,     # for ED loss
        }
    
    def forward_decoder(self, prev_output_tokens, **kwargs):
        return self.decoder(prev_output_tokens, **kwargs)

    def get_logits(self, net_output):
        """For CTC decoding"""
        logits = net_output["encoder_out"]
        padding = net_output["encoder_padding_mask"]
        if padding is not None and padding.any():
            padding = padding.T
            logits[padding][..., 0] = 0
            logits[padding][..., 1:] = float("-inf")

        return logits

    def get_normalized_probs(self, net_output, log_probs, sample=None):
        """For 1) computing CTC loss, 2) decoder decoding."""

        if "encoder_out_ctc" in net_output:
            logits = net_output["encoder_out_ctc"]
        else:
            return self.decoder.get_normalized_probs(net_output, log_probs, sample)
        
        if isinstance(logits, list):
            logits = logits[0]

        if log_probs:
            return utils.log_softmax(logits.float(), dim=-1)
        else:
            return utils.softmax(logits.float(), dim=-1)

    @property
    def decoder(self):
        return self.encoder.w2v_model.decoder


class SpeechUTEncoder(HubertEncoder):
    """
    Modified from fairseq.models.hubert.hubert_asr.HubertEncoder
    1. make it compatible with encoder-decoder model
    """
    def __init__(self, cfg: HubertAsrConfig, task):
        super().__init__(cfg, task)
        
        if (task.target_dictionary is not None) and (
            hasattr(self.w2v_model, "unit_encoder_ctc_head")
        ):
            self.proj = self.w2v_model.unit_encoder_ctc_head
            self.conv_ctc_proj = True
        else:
            self.conv_ctc_proj = False
    
    def forward(self, source, padding_mask, tbc=True, **kwargs):
        w2v_args = {
            "source": source,
            "padding_mask": padding_mask,
            "mask": self.apply_mask and self.training,
        }
        ft = self.freeze_finetune_updates <= self.num_updates
        with torch.no_grad() if not ft else contextlib.ExitStack():
            x, padding_mask = self.w2v_model.extract_features(**w2v_args)
            if tbc:
                # B x T x C -> T x B x C
                x = x.transpose(0, 1)
        return {
            "encoder_out": [x],  # T x B x C
            "encoder_padding_mask": [padding_mask],  # B x T
        }
    
    def forward_torchscript(self, net_input):
        """A TorchScript-compatible version of forward.

        Forward the encoder out.
        """
        x, padding_mask = self.w2v_model.extract_features(**net_input, mask=False)
        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        encoder_out = {
            "encoder_out" : [x],
            "encoder_padding_mask" : [padding_mask],
        }
        if self.proj:
            x = self.proj(x)
            encoder_out["encoder_out_ctc"] = x

        return encoder_out

    def reorder_encoder_out(self, encoder_out, new_order):
        if encoder_out["encoder_out"] is not None:
            encoder_out["encoder_out"] = [
                x.index_select(1, new_order) for x in encoder_out["encoder_out"]
            ]
        if encoder_out["encoder_padding_mask"] is not None:
            encoder_out["encoder_padding_mask"] = [
                x.index_select(0, new_order) for x in encoder_out["encoder_padding_mask"]
            ]
        return encoder_out

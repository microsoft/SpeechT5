# ----------------------------------------------------------------------------
# SpeechUT: Bridging Speech and Text with Hidden-Unit for Encoder-Decoder Based Speech-Text Pre-training (https://arxiv.org/abs/2210.03730)
# Github source: https://github.com/microsoft/SpeechT5/tree/main/SpeechUT
# Code based on fairseq: https://github.com/facebookresearch/fairseq/tree/272c4c5197250997148fb12c0db6306035f166a4
# 
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# ----------------------------------------------------------------------------

import math
from argparse import Namespace
from dataclasses import dataclass, field
from omegaconf import II
from typing import Optional

import torch
import torch.nn.functional as F
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.criterions.label_smoothed_cross_entropy import label_smoothed_nll_loss
from fairseq.dataclass import FairseqDataclass
from fairseq.data.data_utils import post_process
from fairseq.tasks import FairseqTask
from fairseq.logging.meters import safe_round


@dataclass
class CtcCeCriterionConfig(FairseqDataclass):
    zero_infinity: bool = field(
        default=False,
        metadata={"help": "zero inf loss when source length <= target length"},
    )
    sentence_avg: bool = II("optimization.sentence_avg")
    post_process: str = field(
        default="letter",
        metadata={
            "help": "how to post process predictions into words. can be letter, "
            "wordpiece, BPE symbols, etc. "
            "See fairseq.data.data_utils.post_process() for full list of options"
        },
    )
    wer_kenlm_model: Optional[str] = field(
        default=None,
        metadata={
            "help": "if this is provided, use kenlm to compute wer (along with other wer_* args)"
        },
    )
    wer_lexicon: Optional[str] = field(
        default=None,
        metadata={"help": "lexicon to use with wer_kenlm_model"},
    )
    wer_lm_weight: float = field(
        default=2.0,
        metadata={"help": "lm weight to use with wer_kenlm_model"},
    )
    wer_word_score: float = field(
        default=-1.0,
        metadata={"help": "lm word score to use with wer_kenlm_model"},
    )

    wer_args: Optional[str] = field(
        default=None,
        metadata={
            "help": "DEPRECATED: tuple of (wer_kenlm_model, wer_lexicon, wer_lm_weight, wer_word_score)"
        },
    )

    dec_weight: float = field(
        default=0.5,
        metadata={"help": "weights for decoder CE Loss, loss will be ((1 - dec_weight) * hubert_loss + dec_weight * CE_Loss)"},
    )
    report_accuracy: bool = field(
        default=True,
        metadata={"help": "report decoder accuracy metric"},
    )
    ignore_prefix_size: int = field(
        default=0,
        metadata={"help": "Ignore first N tokens"},
    )
    label_smoothing: float = field(
        default=0.1,
        metadata={"help": "epsilon for label smoothing, 0 means no label smoothing"},
    )


@register_criterion("ctc_ce", dataclass=CtcCeCriterionConfig)
class CtcCeCriterion(FairseqCriterion):
    def __init__(self, cfg: CtcCeCriterionConfig, task: FairseqTask):
        super().__init__(task)
        self.blank_idx = (
            task.target_dictionary.index(task.blank_symbol)
            if hasattr(task, "blank_symbol")
            else 0
        )
        self.pad_idx = task.target_dictionary.pad()
        self.eos_idx = task.target_dictionary.eos()
        self.post_process = cfg.post_process

        if cfg.wer_args is not None:
            (
                cfg.wer_kenlm_model,
                cfg.wer_lexicon,
                cfg.wer_lm_weight,
                cfg.wer_word_score,
            ) = eval(cfg.wer_args)

        if cfg.wer_kenlm_model is not None:
            from examples.speech_recognition.w2l_decoder import W2lKenLMDecoder

            dec_args = Namespace()
            dec_args.nbest = 1
            dec_args.criterion = "ctc"
            dec_args.kenlm_model = cfg.wer_kenlm_model
            dec_args.lexicon = cfg.wer_lexicon
            dec_args.beam = 50
            dec_args.beam_size_token = min(50, len(task.target_dictionary))
            dec_args.beam_threshold = min(50, len(task.target_dictionary))
            dec_args.lm_weight = cfg.wer_lm_weight
            dec_args.word_score = cfg.wer_word_score
            dec_args.unk_weight = -math.inf
            dec_args.sil_weight = 0

            self.w2l_decoder = W2lKenLMDecoder(dec_args, task.target_dictionary)
        else:
            self.w2l_decoder = None

        self.zero_infinity = cfg.zero_infinity
        self.sentence_avg = cfg.sentence_avg

        self.dec_weight = cfg.dec_weight
        self.report_accuracy = cfg.report_accuracy
        self.ignore_prefix_size = cfg.ignore_prefix_size
        self.eps = cfg.label_smoothing

    def forward(self, model, sample, reduce=True):
        net_output = model(**sample["net_input"])
        lprobs = model.get_normalized_probs(
            net_output, log_probs=True
        ).contiguous()  # (T, B, C) from the encoder

        if "src_lengths" in sample["net_input"]:
            input_lengths = sample["net_input"]["src_lengths"]
        else:
            if net_output["padding_mask"] is not None:
                non_padding_mask = ~net_output["padding_mask"]
                input_lengths = non_padding_mask.long().sum(-1)
            else:
                input_lengths = lprobs.new_full(
                    (lprobs.size(1),), lprobs.size(0), dtype=torch.long
                )

        pad_mask = (sample["target"] != self.pad_idx) & (
            sample["target"] != self.eos_idx
        )
        targets_flat = sample["target"].masked_select(pad_mask)
        if "target_lengths" in sample:
            target_lengths = sample["target_lengths"]
        else:
            target_lengths = pad_mask.sum(-1)

        with torch.backends.cudnn.flags(enabled=False):
            loss = F.ctc_loss(
                lprobs,
                targets_flat,
                input_lengths,
                target_lengths,
                blank=self.blank_idx,
                reduction="sum",
                zero_infinity=self.zero_infinity,
            )

        ntokens = (
            sample["ntokens"] if "ntokens" in sample else target_lengths.sum().item()
        )

        sample_size = sample["target"].size(0) if self.sentence_avg else ntokens

        logging_output = {}
        if "decoder_target" in sample:
            if net_output["decoder_out"] is not None:
                dec_sample_size = sample["target"].size(0) if self.sentence_avg else sample["dec_ntokens"]
                dec_loss, dec_nll_loss = self.compute_ce_loss(model, net_output["decoder_out"], sample, reduce=reduce)
                logging_output["ctc_loss"] = loss.item()
                loss = (1 - self.dec_weight) * loss + (self.dec_weight * dec_loss *  sample_size / dec_sample_size)
                logging_output["dec_loss"] = dec_loss.item()
                logging_output["dec_nll_loss"] = dec_nll_loss.item()
                logging_output["dec_sample_size"] = dec_sample_size

                if self.report_accuracy:
                    n_correct, total = self.compute_accuracy(model, net_output["decoder_out"], sample)
                    logging_output["dec_n_correct"] = utils.item(n_correct.data)
                    logging_output["total"] = utils.item(total.data)
            else:
                logging_output["ctc_loss"] = loss.item()
                loss = (1 - self.dec_weight) * loss
                logging_output["dec_loss"] = 0
                logging_output["dec_nll_loss"] = 0
                logging_output["dec_sample_size"] = 1
                if self.report_accuracy:
                    logging_output["dec_n_correct"] = 0
                    logging_output["total"] = 1
            
        logging_output = {
            "loss": utils.item(loss.data),  # * sample['ntokens'],
            "ntokens": ntokens,
            "nsentences": sample["id"].numel(),
            "sample_size": sample_size,
            **logging_output,
        }

        if not model.training and self.dec_weight < 1.0:
            import editdistance

            with torch.no_grad():
                lprobs_t = lprobs.transpose(0, 1).float().contiguous().cpu()

                c_err = 0
                c_len = 0
                w_errs = 0
                w_len = 0
                wv_errs = 0
                for lp, t, inp_l in zip(
                    lprobs_t,
                    sample["target_label"]
                    if "target_label" in sample
                    else sample["target"],
                    input_lengths,
                ):
                    lp = lp[:inp_l].unsqueeze(0)

                    decoded = None
                    if self.w2l_decoder is not None:
                        decoded = self.w2l_decoder.decode(lp)
                        if len(decoded) < 1:
                            decoded = None
                        else:
                            decoded = decoded[0]
                            if len(decoded) < 1:
                                decoded = None
                            else:
                                decoded = decoded[0]

                    p = (t != self.task.target_dictionary.pad()) & (
                        t != self.task.target_dictionary.eos()
                    )
                    targ = t[p]
                    targ_units = self.task.target_dictionary.string(targ)
                    targ_units_arr = targ.tolist()

                    toks = lp.argmax(dim=-1).unique_consecutive()
                    pred_units_arr = toks[toks != self.blank_idx].tolist()

                    c_err += editdistance.eval(pred_units_arr, targ_units_arr)
                    c_len += len(targ_units_arr)

                    targ_words = post_process(targ_units, self.post_process).split()

                    pred_units = self.task.target_dictionary.string(pred_units_arr)
                    pred_words_raw = post_process(pred_units, self.post_process).split()

                    if decoded is not None and "words" in decoded:
                        pred_words = decoded["words"]
                        w_errs += editdistance.eval(pred_words, targ_words)
                        wv_errs += editdistance.eval(pred_words_raw, targ_words)
                    else:
                        dist = editdistance.eval(pred_words_raw, targ_words)
                        w_errs += dist
                        wv_errs += dist

                    w_len += len(targ_words)

                logging_output["wv_errors"] = wv_errs
                logging_output["w_errors"] = w_errs
                logging_output["w_total"] = w_len
                logging_output["c_errors"] = c_err
                logging_output["c_total"] = c_len

        return loss, sample_size, logging_output

    def compute_ce_loss(self, model, net_output, sample, reduce=True):
        lprobs, target = self.get_lprobs_and_target(model, net_output, sample)
        loss, nll_loss = label_smoothed_nll_loss(
            lprobs,
            target,
            self.eps,
            ignore_index=self.pad_idx,
            reduce=reduce,
        )
        return loss, nll_loss

    def compute_accuracy(self, model, net_output, sample):
        lprobs, target = self.get_lprobs_and_target(model, net_output, sample)
        mask = target.ne(self.pad_idx)
        n_correct = torch.sum(
            lprobs.argmax(1).masked_select(mask).eq(target.masked_select(mask))
        )
        total = torch.sum(mask)
        return n_correct, total

    def get_lprobs_and_target(self, model, net_output, sample):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        target = sample["decoder_target"]
        if self.ignore_prefix_size > 0:
            if getattr(lprobs, "batch_first", False):
                lprobs = lprobs[:, self.ignore_prefix_size :, :].contiguous()
                target = target[:, self.ignore_prefix_size :].contiguous()
            else:
                lprobs = lprobs[self.ignore_prefix_size :, :, :].contiguous()
                target = target[self.ignore_prefix_size :, :].contiguous()
        return lprobs.view(-1, lprobs.size(-1)), target.view(-1)


    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""

        loss_sum = utils.item(sum(log.get("loss", 0) for log in logging_outputs))
        ntokens = utils.item(sum(log.get("ntokens", 0) for log in logging_outputs))
        nsentences = utils.item(
            sum(log.get("nsentences", 0) for log in logging_outputs)
        )
        sample_size = utils.item(
            sum(log.get("sample_size", 0) for log in logging_outputs)
        )

        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar("ntokens", ntokens)
        metrics.log_scalar("nsentences", nsentences)
        if sample_size != ntokens:
            metrics.log_scalar(
                "nll_loss", loss_sum / ntokens / math.log(2), ntokens, round=3
            )

        c_errors = sum(log.get("c_errors", 0) for log in logging_outputs)
        metrics.log_scalar("_c_errors", c_errors)
        c_total = sum(log.get("c_total", 0) for log in logging_outputs)
        metrics.log_scalar("_c_total", c_total)
        w_errors = sum(log.get("w_errors", 0) for log in logging_outputs)
        metrics.log_scalar("_w_errors", w_errors)
        wv_errors = sum(log.get("wv_errors", 0) for log in logging_outputs)
        metrics.log_scalar("_wv_errors", wv_errors)
        w_total = sum(log.get("w_total", 0) for log in logging_outputs)
        metrics.log_scalar("_w_total", w_total)

        if c_total > 0:
            metrics.log_derived(
                "uer",
                lambda meters: safe_round(
                    meters["_c_errors"].sum * 100.0 / meters["_c_total"].sum, 3
                )
                if meters["_c_total"].sum > 0
                else float("nan"),
            )
        if w_total > 0:
            metrics.log_derived(
                "wer",
                lambda meters: safe_round(
                    meters["_w_errors"].sum * 100.0 / meters["_w_total"].sum, 3
                )
                if meters["_w_total"].sum > 0
                else float("nan"),
            )
            metrics.log_derived(
                "raw_wer",
                lambda meters: safe_round(
                    meters["_wv_errors"].sum * 100.0 / meters["_w_total"].sum, 3
                )
                if meters["_w_total"].sum > 0
                else float("nan"),
            )

        if "dec_loss" in logging_outputs[0]:
            ctc_loss_sum = sum(log.get("ctc_loss", 0) for log in logging_outputs)
            dec_loss_sum = sum(log.get("dec_loss", 0) for log in logging_outputs)
            dec_nll_loss_sum = sum(log.get("dec_nll_loss", 0) for log in logging_outputs)
            dec_sample_size = sum(log.get("dec_sample_size", 0) for log in logging_outputs)
            metrics.log_scalar(
                "dec_loss", dec_loss_sum / dec_sample_size / math.log(2), dec_sample_size, round=3
            )
            metrics.log_scalar(
                "ctc_loss", ctc_loss_sum / sample_size / math.log(2), sample_size, round=3
            )
            metrics.log_scalar(
                "dec_nll_loss", dec_nll_loss_sum / dec_sample_size / math.log(2), dec_sample_size, round=3
            )
            metrics.log_derived(
                "dec_ppl", lambda meters: utils.get_perplexity(meters["dec_nll_loss"].avg)
            )
            total = utils.item(sum(log.get("total", 0) for log in logging_outputs))
            if total > 0:
                metrics.log_scalar("total", total)
                n_correct = utils.item(
                    sum(log.get("dec_n_correct", 0) for log in logging_outputs)
                )
                metrics.log_scalar("dec_n_correct", n_correct)
                metrics.log_derived(
                    "dec_accuracy",
                    lambda meters: round(
                        meters["dec_n_correct"].sum * 100.0 / meters["total"].sum, 3
                    )
                    if meters["total"].sum > 0
                    else float("nan"),
                )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True

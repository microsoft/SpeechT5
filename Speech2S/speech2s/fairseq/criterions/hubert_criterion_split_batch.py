# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import math
import re
from dataclasses import dataclass, field
from typing import List, Optional

import torch
import torch.nn.functional as F
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.criterions.label_smoothed_cross_entropy import label_smoothed_nll_loss
from fairseq.dataclass import FairseqDataclass

logger = logging.getLogger(__name__)
@dataclass
class HubertCriterionConfig(FairseqDataclass):
    pred_masked_weight: float = field(
        default=1.0,
        metadata={"help": "weight for predictive loss for masked frames"},
    )
    pred_nomask_weight: float = field(
        default=0.0,
        metadata={"help": "weight for predictive loss for unmasked frames"},
    )
    loss_weights: Optional[List[float]] = field(
        default=None,
        metadata={"help": "weights for additional loss terms (not first one)"},
    )
    log_keys: List[str] = field(
        default_factory=lambda: [],
        metadata={"help": "output keys to log"},
    )
    dec_weight: float = field(
        default=1.0,
        metadata={"help": "weights for decoder CE Loss, loss will be (hubert_loss + dec_weight * CE_Loss)"},
    )
    text_weight: float = field(
        default=1.0,
        metadata={"help": "weights for text ED CE Loss, loss will be (hubert_loss + dec_weight * CE_Loss + text_weight * CE_Loss)"},
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
        default=0.0,
        metadata={"help": "epsilon for label smoothing, 0 means no label smoothing"},
    )


@register_criterion("hubert_split_batch", dataclass=HubertCriterionConfig)
class HubertCriterion(FairseqCriterion):
    def __init__(
        self, 
        task, 
        pred_masked_weight, 
        pred_nomask_weight, 
        loss_weights=None, 
        log_keys=None, 
        dec_weight=1.0,
        text_weight=1.0,
        report_accuracy=False, 
        ignore_prefix_size=0, 
        label_smoothing=0.0
    ):
        super().__init__(task)
        self.pred_masked_weight = pred_masked_weight
        self.pred_nomask_weight = pred_nomask_weight
        self.loss_weights = loss_weights
        self.log_keys = [] if log_keys is None else log_keys
        self.dec_weight = dec_weight
        self.text_weight = text_weight
        self.report_accuracy = report_accuracy
        self.ignore_prefix_size = ignore_prefix_size
        self.eps = label_smoothing
        self.padding_idx = task.dictionaries[0].pad()
        self.text_dict = task.text_dictionary

    def forward(self, model, sample, reduce=True, log_pred=False):
        """Compute the loss for the given sample.
        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        text_type = [name for name in sample.keys() if name.startswith("text")]
        loss = 0.
        sample_size = 0
        logging_output = {}
        reduction = "sum" if reduce else "none"

        if "speech" in sample.keys():
            assert len(text_type) == 0
            sample = sample["speech"]
            sample["modality"] = "speech"

            ### 1. do hubert forward and loss computation
            net_output = model(target_list=sample["target_list"], **sample["net_input"])
            loss_m_list = []
            logp_m_list = model.get_logits(net_output, True)
            targ_m_list = model.get_targets(net_output, True)
            assert self.pred_masked_weight == 0 or len(logp_m_list) > 0
            for i, (logp_m, targ_m) in enumerate(zip(logp_m_list, targ_m_list)):
                loss_m = F.cross_entropy(logp_m, targ_m, reduction=reduction)
                loss_m_list.append(loss_m)
                logging_output[f"loss_m_{i}"] = loss_m.detach().item() / targ_m_list[0].numel()
            if self.pred_masked_weight > 0:
                loss += self.pred_masked_weight * sum(loss_m_list)
                sample_size += targ_m_list[0].numel()

            loss_u_list = []
            logp_u_list = model.get_logits(net_output, False)
            targ_u_list = model.get_targets(net_output, False)
            assert self.pred_nomask_weight == 0 or len(logp_u_list) > 0
            for i, (logp_u, targ_u) in enumerate(zip(logp_u_list, targ_u_list)):
                loss_u = F.cross_entropy(logp_u, targ_u, reduction=reduction)
                loss_u_list.append(loss_u)
                logging_output[f"loss_u_{i}"] = loss_u.detach().item() / targ_m_list[0].numel()
            if self.pred_nomask_weight > 0:
                loss += self.pred_nomask_weight * sum(loss_u_list)
                sample_size += targ_u_list[0].numel()

            if self.loss_weights is not None:
                assert hasattr(model, "get_extra_losses")
                extra_losses, names = model.get_extra_losses(net_output)
                if torch.is_tensor(extra_losses):
                    extra_losses = [extra_losses]
                    names = [names]
                if len(self.loss_weights) == 1 and len(extra_losses) != 1:
                    self.loss_weights = [self.loss_weights[0]] * len(extra_losses)
                assert len(extra_losses) == len(self.loss_weights), f"{len(extra_losses)}, {len(self.loss_weights)}"
                for p, n, coef in zip(extra_losses, names, self.loss_weights):
                    if coef != 0 and p is not None:
                        p = coef * p.float() * sample_size
                        loss += p
                        logging_output[f"loss_{n}"] = p.item() / sample_size

            if "decoder_target" in sample:
                dec_sample_size = sample["dec_ntokens"]
                dec_loss, dec_nll_loss = self.compute_ce_loss(model, net_output["decoder_out"], sample, reduce=reduce)
                loss = loss + (self.dec_weight * dec_loss *  sample_size / dec_sample_size)
                logging_output["dec_loss"] = dec_loss.item()
                logging_output["dec_nll_loss"] = dec_nll_loss.item()
                logging_output["dec_sample_size"] = dec_sample_size
                logging_output["hubert_sample_size"] = sample_size

                if self.report_accuracy:
                    n_correct, total = self.compute_accuracy(model, net_output["decoder_out"], sample)
                    logging_output["dec_n_correct"] = utils.item(n_correct.data)
                    logging_output["total"] = utils.item(total.data)
            
            loss = loss / sample_size

            for lk in self.log_keys:
                if lk in net_output:
                    logging_output[lk] = float((net_output[lk]))

            def compute_correct(logits):
                if logits.numel() == 0:
                    return 0, 0
                else:
                    assert logits.dim() > 1, logits.shape
                    max = logits.argmax(-1) == 0
                    min = logits.argmin(-1) == 0
                    both = max & min
                    corr = max.long().sum().item() - both.long().sum().item()
                    count = max.numel()
                    return corr, count

            with torch.no_grad():
                for i, logp_m in enumerate(logp_m_list):
                    corr_m, count_m = compute_correct(logp_m)
                    logging_output[f"correct_m_{i}"] = corr_m
                    logging_output[f"count_m_{i}"] = count_m

                for i, logp_u in enumerate(logp_u_list):
                    corr_u, count_u = compute_correct(logp_u)
                    logging_output[f"correct_u_{i}"] = corr_u
                    logging_output[f"count_u_{i}"] = count_u
            logging_output["speech_sample_size"] = sample_size

        else:
            assert len(text_type) == 1
            text_type = text_type[0]
            text_sample = sample[text_type]
            text_sample["modality"] = "text"
            ### 2. do text forward and loss computation
            text_net_output = model(**text_sample["net_input"])
            text_dec_loss, text_dec_nll_loss = self.compute_ce_loss(model, text_net_output["decoder_out"], text_sample, reduce=reduce)
            text_sample_size = text_sample["ntokens"]
            loss = loss + (self.text_weight * text_dec_loss)
            logging_output["text_dec_loss"] = text_dec_loss.item()
            logging_output["text_dec_nll_loss"] = text_dec_nll_loss.item()
            logging_output["text_sample_size"] = text_sample_size

            loss = loss / text_sample_size
            sample_size = text_sample_size
            sample = text_sample

            if self.report_accuracy:
                n_correct, total = self.compute_accuracy(model, text_net_output["decoder_out"], text_sample)
                logging_output["text_dec_n_correct"] = utils.item(n_correct.data)
                logging_output["text_total"] = utils.item(total.data)

        logging_output = {
            "loss": loss.item() if reduce else loss,
            "ntokens": sample_size,
            "nsentences": sample["id"].numel(),
            "sample_size": 1,
            **logging_output,
        }

        return loss, 1, logging_output

    def compute_ce_loss(self, model, net_output, sample, reduce=True):
        lprobs, target = self.get_lprobs_and_target(model, net_output, sample)
        loss, nll_loss = label_smoothed_nll_loss(
            lprobs,
            target,
            self.eps,
            ignore_index=self.padding_idx,
            reduce=reduce,
        )
        return loss, nll_loss

    def compute_accuracy(self, model, net_output, sample):
        lprobs, target = self.get_lprobs_and_target(model, net_output, sample)
        mask = target.ne(self.padding_idx)
        n_correct = torch.sum(
            lprobs.argmax(1).masked_select(mask).eq(target.masked_select(mask))
        )
        total = torch.sum(mask)
        return n_correct, total

    def get_lprobs_and_target(self, model, net_output, sample):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        if sample["modality"] == "speech":
            target = sample["decoder_target"]
            if self.ignore_prefix_size > 0:
                if getattr(lprobs, "batch_first", False):
                    lprobs = lprobs[:, self.ignore_prefix_size :, :].contiguous()
                    target = target[:, self.ignore_prefix_size :].contiguous()
                else:
                    lprobs = lprobs[self.ignore_prefix_size :, :, :].contiguous()
                    target = target[self.ignore_prefix_size :, :].contiguous()
        else:
            target = sample["target"]

        return lprobs.view(-1, lprobs.size(-1)), target.view(-1)

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training (copied from normal cross entropy)."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        speech_sample_size = sum(log.get("speech_sample_size", 0) for log in logging_outputs)

        metrics.log_scalar("loss", loss_sum / sample_size / math.log(2), sample_size, round=3)
        if sample_size != ntokens:
            metrics.log_scalar("nll_loss", loss_sum / ntokens / math.log(2), ntokens, round=3)
            metrics.log_derived("ppl", lambda meters: utils.get_perplexity(meters["nll_loss"].avg))
        else:
            metrics.log_derived("ppl", lambda meters: utils.get_perplexity(meters["loss"].avg))

        counts = {}
        log_keys = []
        for log in logging_outputs:
            log_keys += list(log.keys())
        log_keys = set(log_keys)

        for lk in log_keys:
            if lk.startswith("count_"):
                val = sum(log.get(lk, 0) for log in logging_outputs)
                metrics.log_scalar(lk, val)
                counts[lk] = val

        for lk in log_keys:
            if lk.startswith("loss_") and speech_sample_size > 0:
                val = sum(log.get(lk, 0) for log in logging_outputs)
                metrics.log_scalar(lk, val / speech_sample_size / math.log(2), round=3)
            elif lk.startswith("correct_"):
                val = sum(log.get(lk, 0) for log in logging_outputs)
                metrics.log_scalar(lk, val / counts[re.sub("correct", "count", lk)])

        if "dec_loss" in logging_outputs[0]:
            dec_loss_sum = sum(log.get("dec_loss", 0) for log in logging_outputs)
            dec_nll_loss_sum = sum(log.get("dec_nll_loss", 0) for log in logging_outputs)
            dec_sample_size = sum(log.get("dec_sample_size", 0) for log in logging_outputs)
            metrics.log_scalar(
                "dec_loss", dec_loss_sum / dec_sample_size / math.log(2), dec_sample_size, round=3
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
        
        # if "text_dec_loss" in logging_outputs[0]:
        if any("text_dec_loss" in logging_output for logging_output in logging_outputs):
            text_dec_loss_sum = sum(log.get("text_dec_loss", 0) for log in logging_outputs)
            text_dec_nll_loss_sum = sum(log.get("text_dec_nll_loss", 0) for log in logging_outputs)
            text_sample_size = sum(log.get("text_sample_size", 0) for log in logging_outputs)
            metrics.log_scalar(
                "text_dec_loss", text_dec_loss_sum / text_sample_size / math.log(2), text_sample_size, round=3
            )
            metrics.log_scalar(
                "text_dec_nll_loss", text_dec_nll_loss_sum / text_sample_size / math.log(2), text_sample_size, round=3
            )
            metrics.log_derived(
                "text_dec_ppl", lambda meters: utils.get_perplexity(meters["text_dec_nll_loss"].avg)
            )
            text_total = utils.item(sum(log.get("text_total", 0) for log in logging_outputs))
            if text_total > 0:
                metrics.log_scalar("text_total", text_total)
                text_n_correct = utils.item(
                    sum(log.get("text_dec_n_correct", 0) for log in logging_outputs)
                )
                metrics.log_scalar("text_dec_n_correct", text_n_correct)
                metrics.log_derived(
                    "text_dec_accuracy",
                    lambda meters: round(
                        meters["text_dec_n_correct"].sum * 100.0 / meters["text_total"].sum, 3
                    )
                    if meters["text_total"].sum > 0
                    else float("nan"),
                )

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        raise NotImplementedError()

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return False

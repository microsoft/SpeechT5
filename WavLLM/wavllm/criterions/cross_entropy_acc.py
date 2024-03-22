# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from dataclasses import dataclass

import torch.nn.functional as F
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass
from omegaconf import II
import torch


@dataclass
class CrossEntropyAccuracyCriterionConfig(FairseqDataclass):
    sentence_avg: bool = II("optimization.sentence_avg")


@register_criterion("cross_entropy_acc", dataclass=CrossEntropyAccuracyCriterionConfig)
class CrossEntropyAccuracyCriterion(FairseqCriterion):
    def __init__(self, task, sentence_avg):
        super().__init__(task)
        self.sentence_avg = sentence_avg

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        #print ("self.padding_idx: ", self.padding_idx) # pad_id is 0 (unk_id is 0 in LLaMA)
        net_output = model(**sample["net_input"])

        loss, lprobs, target = self.compute_loss(model, net_output, sample, reduce=reduce)
        ## cal acoustic accuracy
        mask = target != self.padding_idx
        correct = torch.sum(lprobs.argmax(1).masked_select(mask) == target.masked_select(mask))
        total = torch.sum(mask)


        sample_size = (
            sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        )
        logging_output = {
            "loss": loss.data,
            "correct": correct,
            "total": total,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
        }
        return loss, sample_size, logging_output

    def compute_loss(self, model, net_output, sample, reduce=True):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        lprobs = lprobs.view(-1, lprobs.size(-1))
        target = model.get_targets(sample, net_output).view(-1)
        loss = F.nll_loss(
            lprobs,
            target,
            ignore_index=self.padding_idx,
            reduction="sum" if reduce else "none",
        )
        # if net_output[0][1] is not None:
        #     cosine_loss = 1 - F.cosine_similarity(net_output[0][1], net_output[0][2], dim=-1).mean()
        #     loss = loss + cosine_loss
        return loss, lprobs, target

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        correct_sum = utils.item(sum(log.get("correct", 0) for log in logging_outputs))
        total_sum = utils.item(sum(log.get("total", 0) for log in logging_outputs))

        # we divide by log(2) to convert the loss from base e to base 2
        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        
        if sample_size != ntokens:
            metrics.log_scalar(
                "nll_loss", loss_sum / ntokens / math.log(2), ntokens, round=3
            )
            metrics.log_derived(
                "ppl", lambda meters: utils.get_perplexity(meters["nll_loss"].avg)
            )
        else:
            metrics.log_derived(
                "ppl", lambda meters: utils.get_perplexity(meters["loss"].avg)
            )

        if total_sum > 0:
            metrics.log_scalar("total_sum", total_sum)
            metrics.log_scalar("correct_sum", correct_sum)
            metrics.log_derived(
                "accuracy",
                lambda meters: round(
                    meters["correct_sum"].sum * 100.0 / meters["total_sum"].sum, 3
                )
                if meters["total_sum"].sum > 0
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
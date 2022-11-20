# ----------------------------------------------------------------------------
# VatLM: Visual-Audio-Text Pre-Training  with Unified Masked Prediction for Speech Representation Learning
# Github source: https://github.com/microsoft/SpeechT5/tree/main/VATLM
# Code based on fairseq: https://github.com/facebookresearch/fairseq and av_hubert: https://github.com/facebookresearch/av_hubert
# 
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# ----------------------------------------------------------------------------
import math
import re
from dataclasses import dataclass, field
from typing import List, Optional

import torch
import torch.nn.functional as F
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass


@dataclass
class VATHubertCriterionConfig(FairseqDataclass):
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
    banlance_loss_weights: Optional[List[float]] = field(
        default=None,
        metadata={"help": "weights for additional loss terms (not first one)"},
    )

@register_criterion("vat_hubert", dataclass=VATHubertCriterionConfig)
class VATHubertCriterion(FairseqCriterion):
    def __init__(self, task, pred_masked_weight, pred_nomask_weight, banlance_loss_weights, loss_weights=None, log_keys=None):
        super().__init__(task)
        self.pred_masked_weight = pred_masked_weight
        self.pred_nomask_weight = pred_nomask_weight
        self.loss_weights = loss_weights
        self.log_keys = [] if log_keys is None else log_keys
        self.banlance_loss_weights = banlance_loss_weights

    def forward(self, model, sample, reduce=True, log_pred=False):
        """Compute the loss for the given sample.
        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """

        videoaudio_sample = sample.get("videoaudio", None)
        audiotext_sample = sample.get("audiotext", None)
        onlytext_sample = sample.get("onlytext", None)
        onlyaudio_sample = sample.get("onlyaudio", None)


        loss = 0.
        loss1 = 0.
        loss2 = 0.
        loss3 = 0.
        loss4 = 0.
        sample_size = 0
        logging_output = {}
        reduction = "sum" if reduce else "none"

        if videoaudio_sample is not None:
            # print("videoaudio_sample")
            net_output = model(target_list=videoaudio_sample["target_list"], **videoaudio_sample["net_input"])

            loss_m_list = []
            logp_m_list, targ_m_list = net_output['logit_m_list'], net_output['target_m_list']
            for i, (logp_m, targ_m) in enumerate(zip(logp_m_list, targ_m_list)):
                loss_m = F.cross_entropy(logp_m, targ_m, reduction=reduction)
                loss_m_list.append(loss_m)
                logging_output[f"loss_m_videoaudio_{i}"] = loss_m.detach().item()
            if self.pred_masked_weight > 0:
                loss1 += self.pred_masked_weight * sum(loss_m_list)
                sample_size += targ_m_list[0].numel()

            loss_u_list = []
            logp_u_list, targ_u_list = net_output['logit_u_list'], net_output['target_u_list']
            for i, (logp_u, targ_u) in enumerate(zip(logp_u_list, targ_u_list)):
                loss_u = F.cross_entropy(logp_u, targ_u, reduction=reduction)
                loss_u_list.append(loss_u)
                logging_output[f"loss_u_videoaudio_{i}"] = loss_u.detach().item()
            if self.pred_nomask_weight > 0:
                loss1 += self.pred_nomask_weight * sum(loss_u_list)
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
                        loss1 += p
                        logging_output[f"loss_videoaudio_{n}"] = p.item()

            logging_output = {
                "loss_video_audio": loss1.item() if reduce else loss1,
                **logging_output,
            }

            for lk in self.log_keys:
                if lk in net_output:
                    logging_output[lk] = float((net_output[lk]))

            with torch.no_grad():
                for i, logp_m in enumerate(logp_m_list):
                    # corr_m, count_m = compute_correct(logp_m)
                    if logp_m.numel() == 0:
                        corr_m, count_m = 0
                    else:
                        corr_m, count_m = (logp_m.argmax(dim=-1)==targ_m_list[i]).sum().item(), len(targ_m_list[i])
                    logging_output[f"correct_m_videoaudio_{i}"] = corr_m
                    logging_output[f"count_m_videoaudio_{i}"] = count_m

                for i, logp_u in enumerate(logp_u_list):
                    if logp_u.numel() == 0:
                        corr_u, count_u = 0, 0
                    else:
                        corr_u, count_u = (logp_u.argmax(dim=-1)==targ_u_list[i]).sum().item(), len(targ_u_list[i])
                    logging_output[f"correct_u_videoaudio_{i}"] = corr_u
                    logging_output[f"count_u_videoaudio_{i}"] = count_u
            

        if audiotext_sample is not None:
            # print("audiotext_sample")
            net_output = model(target_list=audiotext_sample["target_list"], targets_phone_list=audiotext_sample["targets_phone_list"], **audiotext_sample["net_input"])

            loss_m_list = []
            logp_m_list, targ_m_list = net_output['logit_m_list'], net_output['target_m_list']
            for i, (logp_m, targ_m) in enumerate(zip(logp_m_list, targ_m_list)):
                loss_m = F.cross_entropy(logp_m, targ_m, reduction=reduction)
                loss_m_list.append(loss_m)               
                logging_output[f"loss_m_audiotext_{i}"] = loss_m.detach().item()

                
            if self.pred_masked_weight > 0:
                loss2 += self.pred_masked_weight * sum(loss_m_list)
                sample_size += targ_m_list[0].numel()

            loss_u_list = []
            logp_u_list, targ_u_list = net_output['logit_u_list'], net_output['target_u_list']
            for i, (logp_u, targ_u) in enumerate(zip(logp_u_list, targ_u_list)):
                loss_u = F.cross_entropy(logp_u, targ_u, reduction=reduction)
                loss_u_list.append(loss_u)
                logging_output[f"loss_u_audiotext_{i}"] = loss_u.detach().item()
            if self.pred_nomask_weight > 0:
                loss2 += self.pred_nomask_weight * sum(loss_u_list)
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
                        loss2 += p
                        logging_output[f"loss_audiotext_{n}"] = p.item()
            

            logging_output = {
                "loss_audiotext": loss2.item() if reduce else loss2,
                **logging_output,
            }

            for lk in self.log_keys:
                if lk in net_output:
                    logging_output[lk] = float((net_output[lk]))

            with torch.no_grad():
                for i, logp_m in enumerate(logp_m_list):
                    # corr_m, count_m = compute_correct(logp_m)
                    if logp_m.numel() == 0:
                        corr_m, count_m = 0
                    else:
                        corr_m, count_m = (logp_m.argmax(dim=-1)==targ_m_list[i]).sum().item(), len(targ_m_list[i])
                    logging_output[f"correct_m_audiotext_{i}"] = corr_m
                    logging_output[f"count_m_audiotext_{i}"] = count_m

                for i, logp_u in enumerate(logp_u_list):
                    if logp_u.numel() == 0:
                        corr_u, count_u = 0, 0
                    else:
                        corr_u, count_u = (logp_u.argmax(dim=-1)==targ_u_list[i]).sum().item(), len(targ_u_list[i])
                    logging_output[f"correct_u_audiotext_{i}"] = corr_u
                    logging_output[f"count_u_audiotext_{i}"] = count_u


        if onlytext_sample is not None:
            # print("onlytext_sample")
            net_output = model(target_list=onlytext_sample["target_list"], extra_text_phone_list=onlytext_sample["extra_text_phone_list"], **onlytext_sample["net_input"])

            loss_m_list = []
            logp_m_list, targ_m_list = net_output['logit_m_list'], net_output['target_m_list']
            for i, (logp_m, targ_m) in enumerate(zip(logp_m_list, targ_m_list)):
                loss_m = F.cross_entropy(logp_m, targ_m, reduction=reduction)
                loss_m_list.append(loss_m)               
                logging_output[f"loss_m_onlytext_{i}"] = loss_m.detach().item()

                
            if self.pred_masked_weight > 0:
                loss3 += self.pred_masked_weight * sum(loss_m_list)
                sample_size += targ_m_list[0].numel()

            loss_u_list = []
            logp_u_list, targ_u_list = net_output['logit_u_list'], net_output['target_u_list']
            for i, (logp_u, targ_u) in enumerate(zip(logp_u_list, targ_u_list)):
                loss_u = F.cross_entropy(logp_u, targ_u, reduction=reduction)
                loss_u_list.append(loss_u)
                logging_output[f"loss_u_onlytext_{i}"] = loss_u.detach().item()
            if self.pred_nomask_weight > 0:
                loss3 += self.pred_nomask_weight * sum(loss_u_list)
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
                        loss3 += p
                        logging_output[f"loss_onlytext_{n}"] = p.item()
            

            logging_output = {
                "loss_onlytext": loss3.item() if reduce else loss3,
                **logging_output,
            }

            for lk in self.log_keys:
                if lk in net_output:
                    logging_output[lk] = float((net_output[lk]))

            with torch.no_grad():
                for i, logp_m in enumerate(logp_m_list):
                    # corr_m, count_m = compute_correct(logp_m)
                    if logp_m.numel() == 0:
                        corr_m, count_m = 0
                    else:
                        corr_m, count_m = (logp_m.argmax(dim=-1)==targ_m_list[i]).sum().item(), len(targ_m_list[i])
                    logging_output[f"correct_m_onlytext_{i}"] = corr_m
                    logging_output[f"count_m_onlytext_{i}"] = count_m

                for i, logp_u in enumerate(logp_u_list):
                    if logp_u.numel() == 0:
                        corr_u, count_u = 0, 0
                    else:
                        corr_u, count_u = (logp_u.argmax(dim=-1)==targ_u_list[i]).sum().item(), len(targ_u_list[i])
                    logging_output[f"correct_u_onlytext_{i}"] = corr_u
                    logging_output[f"count_u_onlytext_{i}"] = count_u


        if onlyaudio_sample is not None:
            # print("onlytext_sample")
            net_output = model(target_list=onlyaudio_sample["target_list"], **onlyaudio_sample["net_input"])

            loss_m_list = []
            logp_m_list, targ_m_list = net_output['logit_m_list'], net_output['target_m_list']
            for i, (logp_m, targ_m) in enumerate(zip(logp_m_list, targ_m_list)):
                loss_m = F.cross_entropy(logp_m, targ_m, reduction=reduction)
                loss_m_list.append(loss_m)               
                logging_output[f"loss_m_onlyaudio_{i}"] = loss_m.detach().item()

                
            if self.pred_masked_weight > 0:
                loss4 += self.pred_masked_weight * sum(loss_m_list)
                sample_size += targ_m_list[0].numel()

            loss_u_list = []
            logp_u_list, targ_u_list = net_output['logit_u_list'], net_output['target_u_list']
            for i, (logp_u, targ_u) in enumerate(zip(logp_u_list, targ_u_list)):
                loss_u = F.cross_entropy(logp_u, targ_u, reduction=reduction)
                loss_u_list.append(loss_u)
                logging_output[f"loss_u_onlyaudio_{i}"] = loss_u.detach().item()
            if self.pred_nomask_weight > 0:
                loss4 += self.pred_nomask_weight * sum(loss_u_list)
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
                        loss4 += p
                        logging_output[f"loss_onlyaudio_{n}"] = p.item()
            

            logging_output = {
                "loss_onlyaudio": loss4.item() if reduce else loss4,
                **logging_output,
            }

            for lk in self.log_keys:
                if lk in net_output:
                    logging_output[lk] = float((net_output[lk]))

            with torch.no_grad():
                for i, logp_m in enumerate(logp_m_list):
                    # corr_m, count_m = compute_correct(logp_m)
                    if logp_m.numel() == 0:
                        corr_m, count_m = 0
                    else:
                        corr_m, count_m = (logp_m.argmax(dim=-1)==targ_m_list[i]).sum().item(), len(targ_m_list[i])
                    logging_output[f"correct_m_onlyaudio_{i}"] = corr_m
                    logging_output[f"count_m_onlyaudio_{i}"] = count_m

                for i, logp_u in enumerate(logp_u_list):
                    if logp_u.numel() == 0:
                        corr_u, count_u = 0, 0
                    else:
                        corr_u, count_u = (logp_u.argmax(dim=-1)==targ_u_list[i]).sum().item(), len(targ_u_list[i])
                    logging_output[f"correct_u_onlyaudio_{i}"] = corr_u
                    logging_output[f"count_u_onlyaudio_{i}"] = count_u



        loss = loss1 + loss2 + self.banlance_loss_weights[0] * loss3 + self.banlance_loss_weights[1] * loss4

        logging_output = {
            "loss": loss.item() if reduce else loss,
            "ntokens": sample_size,
            "nsentences": sample["videoaudio"]["id"].numel(),
            "sample_size": sample_size,
            **logging_output,
        }

        return loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training (copied from normal cross entropy)."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        metrics.log_scalar("loss", loss_sum / sample_size / math.log(2), sample_size, round=3)
        if sample_size != ntokens:
            metrics.log_scalar("nll_loss", loss_sum / ntokens / math.log(2), ntokens, round=3)
            metrics.log_derived("ppl", lambda meters: utils.get_perplexity(meters["nll_loss"].avg))
        else:
            metrics.log_derived("ppl", lambda meters: utils.get_perplexity(meters["loss"].avg))

        counts = {}
        for lk in logging_outputs[0].keys():
            if lk.startswith("count_"):
                val = sum(log[lk] for log in logging_outputs)
                metrics.log_scalar(lk, val)
                counts[lk] = val

        for lk in logging_outputs[0].keys():
            if lk.startswith("loss_"):
                val = sum(log[lk] for log in logging_outputs)
                metrics.log_scalar(lk, val / sample_size / math.log(2), round=3)
            elif lk.startswith("correct_"):
                val = sum(log[lk] for log in logging_outputs)
                metrics.log_scalar(lk, val / counts[re.sub("correct", "count", lk)])

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

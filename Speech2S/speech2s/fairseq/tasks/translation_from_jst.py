# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import torch
import logging
from dataclasses import dataclass, field
from typing import List, Optional, NamedTuple

from fairseq import utils
from fairseq.data import LanguagePairDataset, TransformEosLangPairDataset, ConcatDataset, FairseqDataset
from fairseq.data.audio.multi_modality_dataset import LangPairMaskDataset

from . import register_task
from .translation import TranslationTask, load_langpair_dataset, TranslationConfig

logger = logging.getLogger(__name__)


@dataclass
class AddTranslationConfig(TranslationConfig):
    langs: str = ""
    prepend_bos: bool = False
    normalize: bool = False
    append_source_id: bool = False
    mask_text_ratio: float = 0
    modal_idx: int = 0


@register_task("translation_from_jst", dataclass=AddTranslationConfig)
class TranslationFromJST(TranslationTask):
    """
    Translate from source language to target language with a model initialized with a multilingual pretrain.

    Args:
        src_dict (~fairseq.data.Dictionary): dictionary for the source language
        tgt_dict (~fairseq.data.Dictionary): dictionary for the target language

    .. note::

        The translation task is compatible with :mod:`fairseq-train`,
        :mod:`fairseq-generate` and :mod:`fairseq-interactive`.

    The translation task provides the following additional command-line
    arguments:

    .. argparse::
        :ref: fairseq.tasks.translation_parser
        :prog:
    """
    args: AddTranslationConfig

    def __init__(self, args: AddTranslationConfig, src_dict, tgt_dict):
        super().__init__(args, src_dict, tgt_dict)
        self.args = args
        self.langs = args.langs.split(",") if args.langs != "" else []
        for d in [src_dict, tgt_dict]:
            for l in self.langs:
                d.add_symbol("[{}]".format(l))
        src_dict.add_symbol("<mask>")


    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        paths = utils.split_paths(self.args.data)
        assert len(paths) > 0
        data_path = paths[(epoch - 1) % len(paths)]

        if getattr(self.args, "append_source_id", False):
            logger.info(f"Appending <lang-id> to the end of samples")

        # infer langcode
        src, tgt = self.args.source_lang, self.args.target_lang
        paired_datasets = []
        for sub_split in split.split(","):
            paired_dataset= load_langpair_dataset(
                data_path,
                sub_split,
                src,
                self.src_dict,
                tgt,
                self.tgt_dict,
                combine=combine,
                dataset_impl=self.args.dataset_impl,
                upsample_primary=self.args.upsample_primary,
                left_pad_source=self.args.left_pad_source,
                left_pad_target=self.args.left_pad_target,
                max_source_positions=getattr(self.args, "max_source_positions", 1024),
                max_target_positions=getattr(self.args, "max_target_positions", 1024),
                load_alignments=self.args.load_alignments,
                prepend_bos=getattr(self.args, "prepend_bos", False),
                append_source_id=getattr(self.args, "append_source_id", False),
            )
            if sub_split.find("train") >= 0 and getattr(self.args, "mask_text_ratio", 0) > 0:
                mask_text_ratio = getattr(self.args, "mask_text_ratio", 0)
                # noise_token_id = self.src_dict.index("<mask>")
                noise_token_id = 3
                logger.info(f"Masking {sub_split} at a probability: {mask_text_ratio}")
                paired_dataset = LangPairMaskDataset(
                    paired_dataset,
                    src_bos=self.src_dict.bos(),
                    src_eos=self.src_dict.eos(),
                    noise_id=noise_token_id,
                    mask_ratio=mask_text_ratio,
                )
            paired_datasets.append(paired_dataset)
        paired_dataset = paired_datasets[0] if len(paired_datasets) == 1 else ConcatDataset(paired_datasets, 1)
        self.datasets[split] = paired_dataset

    def build_generator(self, models, args, **unused):
        if getattr(args, "score_reference", False):
            from fairseq.sequence_scorer import SequenceScorer

            return SequenceScorer(
                self.target_dictionary,
                eos=self.tgt_dict.index("[{}]".format(self.args.target_lang)),
            )
        else:
            from fairseq.sequence_generator import SequenceGenerator

            return SequenceGenerator(
                models,
                self.target_dictionary,
                beam_size=getattr(args, "beam", 5),
                max_len_a=getattr(args, "max_len_a", 0),
                max_len_b=getattr(args, "max_len_b", 200),
                min_len=getattr(args, "min_len", 1),
                normalize_scores=(not getattr(args, "unnormalized", False)),
                len_penalty=getattr(args, "lenpen", 1),
                unk_penalty=getattr(args, "unkpen", 0),
                temperature=getattr(args, "temperature", 1.0),
                match_source_len=getattr(args, "match_source_len", False),
                no_repeat_ngram_size=getattr(args, "no_repeat_ngram_size", 0),
                eos=self.tgt_dict.index("[{}]".format(self.args.target_lang)) if getattr(self.args, "append_source_id", False) else None,
            )

    def build_dataset_for_inference(self, src_tokens, src_lengths, constraints=None):
        if getattr(self.args, "append_source_id", False):
            src_lang_id = self.source_dictionary.index("[{}]".format(self.args.source_lang))
            source_tokens = []
            for s_t in src_tokens:
                s_t = torch.cat([s_t, s_t.new(1).fill_(src_lang_id)])
                source_tokens.append(s_t)
        else:
            source_tokens = src_tokens
        
        dataset = LanguagePairDataset(
            source_tokens,
            src_lengths,
            self.source_dictionary,
            tgt_dict=self.target_dictionary,
            constraints=constraints,
        )
        return dataset

    def inference_step(
        self, generator, models, sample, prefix_tokens=None, constraints=None
    ):
        with torch.no_grad():
            return generator.generate(
                models, sample, prefix_tokens=prefix_tokens, constraints=constraints, modal_idx=self.args.modal_idx,
            )


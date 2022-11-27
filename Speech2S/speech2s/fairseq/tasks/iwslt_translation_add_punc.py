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
from fairseq.data.shrinking_dataset import ShrinkingDataset

from . import register_task
from .translation import TranslationTask, load_langpair_dataset, TranslationConfig

logger = logging.getLogger(__name__)



class LangPairStripDataset(FairseqDataset):
    def __init__(
        self,
        dataset: LanguagePairDataset,
        src_eos: int,
        src_bos: Optional[int] = None,
        noise_id: Optional[int] = -1,
        mask_ratio: Optional[float] = 0,
        mask_type: Optional[str] = "random",
    ):
        self.dataset = dataset
        self.src_eos = src_eos
        self.src_bos = src_bos
        self.noise_id = noise_id
        self.mask_ratio = mask_ratio
        self.mask_type = mask_type
        assert mask_type in ("random", "tail")

    @property
    def src_sizes(self):
        return self.dataset.src_sizes

    @property
    def tgt_sizes(self):
        return self.dataset.tgt_sizes

    @property
    def sizes(self):
        # dataset.sizes can be a dynamically computed sizes:
        return self.dataset.sizes

    def get_batch_shapes(self):
        return self.dataset.buckets

    def num_tokens_vec(self, indices):
        return self.dataset.num_tokens_vec(indices)

    def __len__(self):
        return len(self.dataset)

    def num_tokens(self, index):
        return self.dataset.num_tokens(index)

    def size(self, index):
        return self.dataset.size(index)

    def ordered_indices(self):
        return self.dataset.ordered_indices()

    @property
    def supports_prefetch(self):
        return getattr(self.dataset, "supports_prefetch", False)

    def prefetch(self, indices):
        return self.dataset.prefetch(indices)

    def mask_src_tokens(self, sample):
        src_item = sample["source"]
        mask = None
        if self.mask_type == "random":
            mask = torch.rand(len(src_item)).le(self.mask_ratio)
        else:
            mask = torch.ones(len(src_item))
            mask[: int(len(src_item) * (1 - self.mask_ratio))] = 0
            mask = mask.eq(1)
        mask[-1] = False
        if src_item[0] == self.src_bos:
            mask[0] = False
        if src_item[-2] == self.src_eos:
            mask[-2] = False
        no_mask = ~mask
        mask_src_item = src_item[no_mask]
        smp = sample
        smp["source"] = mask_src_item
        print(f"{len(src_item)}: {src_item}")
        print(f"{len(mask_src_item)}: {mask_src_item}")
        return smp

    def __getitem__(self, index):
        sample = self.dataset[index]
        if self.mask_ratio > 0:
            sample = self.mask_src_tokens(sample)
        return sample

    def collater(self, samples, pad_to_length=None):
        return self.dataset.collater(samples, pad_to_length=pad_to_length)


@dataclass
class AddTranslationConfig(TranslationConfig):
    langs: str = ""
    prepend_bos: bool = False
    normalize: bool = False
    append_source_id: bool = False
    mask_text_ratio: float = 0
    ### ShrinkingDataset related
    shrink_start_epoch: int = 0
    shrink_end_epoch: int = 0
    shrink_start_ratio: float = 1.0
    shrink_end_ratio: float = 1.0


@register_task("iwslt_translation_add_punc", dataclass=AddTranslationConfig)
class TranslationAddpuncTask(TranslationTask):
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
        self.langs = args.langs.split(",")
        for d in [src_dict, tgt_dict]:
            for l in self.langs:
                d.add_symbol("[{}]".format(l))
            # d.add_symbol("<mask>")


    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        paths = utils.split_paths(self.args.data)
        assert len(paths) > 0
        data_path = paths[(epoch - 1) % len(paths)]

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
            if not split.startswith("valid") and getattr(self.args, "mask_text_ratio", 0) > 0 and not sub_split.startswith("asr_"):
                mask_text_ratio = getattr(self.args, "mask_text_ratio", 0)
                noise_token_id = self.src_dict.index("<mask>")
                logger.info(f"Masking {sub_split} at a probability: {mask_text_ratio}")
                paired_dataset = LangPairStripDataset(
                    paired_dataset,
                    src_bos=self.src_dict.bos(),
                    src_eos=self.src_dict.eos(),
                    noise_id=noise_token_id,
                    mask_ratio=mask_text_ratio,
                )
            paired_datasets.append(paired_dataset)
        paired_dataset = paired_datasets[0] if len(paired_datasets) == 1 else ConcatDataset(paired_datasets, 1)
        
        if not split.startswith("valid") and self.args.shrink_end_epoch > 0 and self.args.shrink_end_ratio < 1.0:
            logger.info(f"Shrinking dataset from {self.args.shrink_start_epoch} to {self.args.shrink_end_epoch} epoch, from {self.args.shrink_start_ratio} to {self.args.shrink_end_ratio}")
            paired_dataset = ShrinkingDataset(
                paired_dataset, 
                self.args.shrink_start_epoch, 
                self.args.shrink_end_epoch,
                self.args.shrink_start_ratio,
                self.args.shrink_end_ratio,
                )

        if getattr(self.args, "append_source_id", False):
            logger.info(f"Appending <lang-id> to the end of samples")
            self.datasets[split] = paired_dataset
        else:
            logger.info(f"Replacing <eos> with <lang-id> for prev_output_tokens")
            self.datasets[split] = TransformEosLangPairDataset(
                paired_dataset,
                src_eos=self.src_dict.eos(),
                tgt_bos=self.tgt_dict.eos(),  # 'prev_output_tokens' starts with eos
                new_tgt_bos=self.tgt_dict.index("[{}]".format(tgt)),
            )

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
                bos=None if getattr(self.args, "append_source_id", False) else self.tgt_dict.index("[{}]".format(self.args.target_lang))
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

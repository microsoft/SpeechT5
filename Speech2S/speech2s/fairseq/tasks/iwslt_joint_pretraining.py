# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import logging
import os
import sys
from typing import Dict, List, Optional, Tuple
from pathlib import Path

import numpy as np
from argparse import Namespace
from collections import OrderedDict

from dataclasses import dataclass, field
from fairseq.data import Dictionary, HubertDataset, encoders
from fairseq.data import (
    Dictionary,
    data_utils,
    StripTokenDataset,
    PrependTokenDataset,
    AppendTokenDataset,
    DenoisingDataset,
    ConcatDataset,
    TransformEosLangPairDataset,
    FairseqDataset,
    iterators,
    ResamplingDataset,
)
from fairseq.data.audio.multi_modality_dataset import LangPairMaskDataset
from fairseq.data.audio.speech_to_text_joint_dataset import S2TJointDataConfig
from fairseq.data.audio.multi_corpus_dataset_audio import MultiCorpusDataset
from fairseq.data.shorten_dataset import maybe_shorten_dataset
from fairseq.tasks.translation import load_langpair_dataset
from fairseq.data.encoders.utils import get_whole_word_mask
from fairseq.dataclass.configs import FairseqDataclass
from fairseq.tasks import register_task
from fairseq.tasks.fairseq_task import FairseqTask
from fairseq.dataclass.constants import ChoiceEnum
from omegaconf import MISSING

logger = logging.getLogger(__name__)

TOKENIZER_CHOICES = ChoiceEnum(["sentencepiece", "hubert_letters", "none"])

def _lang_token(lang: str):
    return "<lang:{}>".format(lang)

def _lang_token_index(dic: Dictionary, lang: str):
    """Return language token index."""
    idx = dic.index(_lang_token(lang))
    assert idx != dic.unk_index, "cannot find language token for lang {}".format(lang)
    return idx


class LabelEncoder(object):
    def __init__(self, dictionary: Dictionary) -> None:
        self.dictionary = dictionary

    def __call__(self, label: str) -> List[str]:
        return self.dictionary.encode_line(
            label, append_eos=False, add_if_not_exist=False,
        )


@dataclass
class TextPretrainingConfig(FairseqDataclass):    
    ### added for joint pretraining
    text_data: Optional[str] = field(
        default=None,
        metadata={
            "help": "if set, path to text data directory",
        },
    )
    seed: Optional[int] = field(
        default=1,
        metadata={
            "help": "for ordered_indices in MulticorpusDataset",
        },
    )
    tokens_per_sample: Optional[int] = field(
        default=512,
        metadata={
            "help": "max number of total tokens over all segments per sample for dataset",
        },
    )
    sample_break_mode: Optional[str] = field(
        default="eos",
        metadata={
            "help": "mode for breaking sentence",
        },
    )
    mask: Optional[float] = field(
        default=0.3,
        metadata={
            "help": "fraction of words/subwords that will be masked",
        },
    )
    leave_unmasked_prob: float = field(
        default=0.1,
        metadata={"help": "probability that a masked token is unmasked"},
    )
    mask_random: Optional[float] = field(
        default=0.0,
        metadata={
            "help": "instead of using [MASK], use random token this often",
        },
    )
    freq_weighted_replacement: bool = field(
        default=False,
        metadata={"help": "sample random replacement words based on word frequencies"},
    )
    mask_whole_words: bool = field(
        default=False,
        metadata={"help": "mask whole words; you may also want to set --bpe"},
    )
    mask_multiple_length: int = field(
        default=1,
        metadata={"help": "repeat the mask indices multiple times"},
    )
    mask_stdev: float = field(
        default=0.0,
        metadata={"help": "stdev of the mask length"},
    )
    shorten_method: Optional[str] = field(
        default="none",
        metadata={
            "help": "if not none, shorten sequences that exceed tokens_per_sample",
            "choices": "none/truncate/random_crop"
        },
    )
    shorten_data_split_list: Optional[str] = field(
        default="",
        metadata={
            "help": "comma_separated list of dataset splits to apply shortening to, e.g., train,valid (default: all dataset splits)",
        },
    )

    ### below hypra-parameters is used in bart
    insert: Optional[float] = field(
        default=0.0,
        metadata={
            "help": "insert this percentage of additional random tokens",
        },
    )
    permute: Optional[float] = field(
        default=0.0,
        metadata={
            "help": "take this proportion of subwords and permute them",
        },
    )
    rotate: Optional[float] = field(
        default=0.0,
        metadata={
            "help": "rotate this proportion of inputs",
        },
    )
    poisson_lambda: Optional[float] = field(
        default=3,
        metadata={
            "help": "randomly shuffle sentences for this proportion of inputs",
        },
    )
    permute_sentences: Optional[float] = field(
        default=0.0,
        metadata={
            "help": "shuffle this proportion of sentences in all inputs",
        },
    )
    mask_length: Optional[str] = field(
        default="span-poisson",
        metadata={
            "help": "mask length to choose",
            "choice": "subword/word/span-poisson"
        },
    )
    replace_length: Optional[int] = field(
        default=1,
        metadata={
            "help": "when masking N tokens, replace with 0, 1, or N tokens (use -1 for N)",
        },
    )
    shuffle_instance: Optional[bool] = field(
        default=False,
        metadata={"help": "shuffle instance"},
    )
    max_source_positions: Optional[int] = field(
        default=1024,
        metadata={"help": "max number of tokens in the source sequence"},
    )
    max_target_positions: Optional[int] = field(
        default=1024,
        metadata={"help": "max number of tokens in the target sequence"},
    )
    bpe: Optional[str] = field(
        default="sentencepiece",
        metadata={
            "help": "will wrapped by the text_data_config yaml",
        },
    )
    data_config: Optional[str] = field(
        default=None,
        metadata={
            "help": "a config yaml specify the bpe model of text data",
        },
    )
    text_maxtokens_ratio: Optional[float] = field(
        default=1.0,
        metadata={
            "help": "for text, max_tokens = max_tokens * text_maxtokens_ratio / 320 ",
        },
    )
    prepend_tgt_lang_tag: bool = field(
        default=True,
        metadata={"help": "prepend tgt_lang_tag to replace <eos>"},
    )
    mask_text_ratio: Optional[float] = field(
        default=0.0,
        metadata={
            "help": "mask_text_ratio, for paired data",
        },
    )


@dataclass
class STbertPretrainingConfig(FairseqDataclass):
    data: str = field(
        default=MISSING, metadata={"help": "path to speech data directory"}
    )
    fine_tuning: bool = field(
        default=False, metadata={"help": "set to true if fine-tuning Hubert"}
    )
    labels: List[str] = field(
        default_factory=lambda: ["ltr"],
        metadata={
            "help": (
                "extension of the label files to load, frame-level labels for"
                " pre-training, and sequence-level label for fine-tuning"
            )
        },
    )
    label_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "if set, looks for labels in this directory instead",
        },
    )
    label_rate: int = field(
        default=-1,
        metadata={"help": "label frame rate. -1 for sequence label"},
    )
    sample_rate: int = field(
        default=16_000,
        metadata={
            "help": "target sample rate. audio files will be up/down "
            "sampled to this rate"
        },
    )
    normalize: bool = field(
        default=False,
        metadata={
            "help": "if set, normalizes input to have 0 mean and unit variance"
        },
    )
    enable_padding: bool = field(
        default=False,
        metadata={"help": "pad shorter samples instead of cropping"},
    )
    max_keep_size: Optional[int] = field(
        default=None,
        metadata={"help": "exclude sample longer than this"},
    )
    max_sample_size: Optional[int] = field(
        default=None,
        metadata={"help": "max sample size to crop to for batching"},
    )
    min_sample_size: Optional[int] = field(
        default=None,
        metadata={"help": "min sample size to crop to for batching"},
    )
    single_target: Optional[bool] = field(
        default=False,
        metadata={
            "help": "if set, AddTargetDatasets outputs same keys "
            "as AddTargetDataset"
        },
    )
    random_crop: Optional[bool] = field(
        default=True,
        metadata={"help": "always crop from the beginning if false"},
    )
    pad_audio: Optional[bool] = field(
        default=False,
        metadata={"help": "pad audio to the longest one in the batch if true"},
    )
    store_labels: Optional[bool] = field(
        default=False,
        metadata={"help": "store spm labels in memory, should be true when fine-tune with bpe"},
    )
    add_decoder: bool = field(
        default=False,
        metadata={"help": "whether to add decoder for CE Loss on code"},
    )
    split_modality_batch: bool = field(
        default=False,
        metadata={"help": "whether create all samples of different modalities in a batch"},
    )
    speech_tgt_lang: str = field(
        default="",
        metadata={"help": "prepend <tgt-id> to prev_output_tokens to replace <eos>, only used for decoder"},
    )
    speech_sampling_alpha: float = field(
        default=0.2,
        metadata={
            "help": "Hyper-parameter alpha = 1/T for temperature-based speech resampling."
            "(alpha = 1 for no resampling)"
        },
    )
    text_sampling_alpha: float = field(
        default=0.2,
        metadata={
            "help": "Hyper-parameter alpha = 1/T for temperature-based text resampling."
            "(alpha = 1 for no resampling)"
        },
    )
    hubert_tokenizer: Optional[TOKENIZER_CHOICES] = field(
        default="none",
        metadata={"help": "which tokenizer for processing text"},
    )
    sp_path: Optional[str] = field(
        default=None,
        metadata={"help": "sentencepiece model path if using bpe tokenizer"},
    )

    text_cfg: TextPretrainingConfig = TextPretrainingConfig()


@register_task("iwslt_joint_pretraining", dataclass=STbertPretrainingConfig)
class JointPretrainingTask(FairseqTask):

    cfg: STbertPretrainingConfig

    def __init__(
        self,
        cfg: STbertPretrainingConfig,
    ) -> None:
        super().__init__(cfg)

        logger.info(f"current directory is {os.getcwd()}")
        logger.info(f"JointPretrainingTask Config {cfg}")

        self.cfg = cfg
        self.fine_tuning = cfg.fine_tuning
        self.blank_symbol = "<s>"

        self.state.add_factory("hubert_tokenizer", self.build_tokenizer)
        self.state.add_factory("text_dictionary", self.load_text_dictionary)
        self.state.add_factory("text_src_dictionary", self.load_text_src_dictionary)
        if cfg.fine_tuning:
            self.state.add_factory("target_dictionary", self.load_dictionaries)
        else:
            self.state.add_factory("dictionaries", self.load_dictionaries)

        if cfg.text_cfg.data_config is not None:
            self.text_data_cfg = S2TJointDataConfig(Path(f"{cfg.text_cfg.text_data}/{cfg.text_cfg.data_config}"))
            self.cfg.text_cfg.bpe = self.text_data_cfg.bpe_tokenizer["bpe"]

    @property
    def source_dictionary(self) -> Optional[Dictionary]:
        return None

    @property
    def target_dictionary(self) -> Optional[Dictionary]:
        return self.state.target_dictionary

    @property
    def dictionaries(self) -> List[Dictionary]:
        return self.state.dictionaries

    @property
    def text_dictionary(self) -> Optional[Dictionary]:
        return self.state.text_dictionary

    @property
    def text_src_dictionary(self) -> Optional[Dictionary]:
        return self.state.text_src_dictionary

    @property
    def hubert_tokenizer(self):
        return self.state.hubert_tokenizer

    def load_dictionaries(self):
        label_dir = self.cfg.data if self.cfg.label_dir is None else self.cfg.label_dir
        dictionaries = [Dictionary.load(f"{label_dir}/dict.{label}.txt") for label in self.cfg.labels]
        return dictionaries[0] if self.cfg.fine_tuning else dictionaries
    
    def load_text_dictionary(self):
        tgt_dict_path = f"{self.cfg.text_cfg.text_data}/{self.text_data_cfg.vocab_filename}"
        if not os.path.isfile(tgt_dict_path):
            raise FileNotFoundError(f"Dict not found: {tgt_dict_path}")
        text_dictionary = Dictionary.load(tgt_dict_path)
        self.mask_idx = text_dictionary.add_symbol("<mask>")
        return text_dictionary
    
    def load_text_src_dictionary(self):
        return self.load_text_dictionary()

    @classmethod
    def setup_task(
        cls, cfg: STbertPretrainingConfig, **kwargs
    ) -> "JointPretrainingTask":
        return cls(cfg)

    def get_label_dir(self) -> str:
        if self.cfg.label_dir is None:
            return self.cfg.data
        return self.cfg.label_dir

    def load_dataset(self, split: str, epoch=1, **kwargs) -> None:
        """
            Create Wav dataset for audio, and Index dataset for phonemized text, 
            then concatenate them to by fairseq.data.multi_corpus_dataset.MultiCorpusDataset.
        """
        if len(split.split("+")) == 1:
            speech_splits = split.split(",")
            has_text = False
        else:
            has_text = True
            speech_splits, text_splits = split.split("+")
            speech_splits = speech_splits.split(",")
            speech_splits = [item for item in speech_splits if item != '']
            text_splits = text_splits.split(",")
            text_splits = [item for item in text_splits if item != '']
            logging.info(f"text_splits: {text_splits}")
        logging.info(f"speech_splits: {speech_splits}")

        ### 1, create a speech dataset using STSpeechDataset (modified from HubertDataset)
        dicts = [self.target_dictionary] if self.cfg.fine_tuning else self.dictionaries
        pad_list = [dict.pad() for dict in dicts]
        eos_list = [dict.eos() for dict in dicts]
        procs = [LabelEncoder(dict) for dict in dicts]
        if self.cfg.speech_tgt_lang != "":
            tgt_lang_idx = _lang_token_index(dicts[0], self.cfg.speech_tgt_lang)
            logger.info(f"Will prepend <{tgt_lang_idx}> at the beginning of prev_output_tokens to replace <eos>")
        else:
            tgt_lang_idx = None

        speech_dataset = None
        mono_dataset = None
        paired_dataset = None

        speech_datasets = []
        for speech_split in speech_splits:
            # hubert v1: pad_audio=True, random_crop=False;
            paths = [f"{self.get_label_dir()}/{speech_split}.{l}" for l in self.cfg.labels]
            speech_datasets.append( 
                HubertDataset(
                    f"{self.cfg.data}/{speech_split}.tsv",
                    sample_rate=self.cfg.sample_rate,
                    label_paths=paths,
                    label_rates=self.cfg.label_rate,
                    pad_list=pad_list,
                    eos_list=eos_list,
                    label_processors=procs,
                    max_keep_sample_size=self.cfg.max_keep_size,
                    min_keep_sample_size=self.cfg.min_sample_size,
                    max_sample_size=self.cfg.max_sample_size,
                    pad_audio=self.cfg.pad_audio,
                    normalize=self.cfg.normalize,
                    store_labels=self.cfg.store_labels,
                    random_crop=self.cfg.random_crop,
                    single_target=self.cfg.single_target,
                    tgt_dict=dicts[0],
                    add_decoder=self.cfg.add_decoder,
                    fine_tuning=self.cfg.fine_tuning,
                    tgt_lang_idx=tgt_lang_idx,
                    tokenizer=self.hubert_tokenizer,
                )
            )

        if len(speech_datasets) > 1:
            if 'train' in speech_splits[0] and self.cfg.speech_sampling_alpha != 1.0:
                size_ratios = self._get_size_ratios(
                    speech_splits, [len(s) for s in speech_datasets], alpha=self.cfg.speech_sampling_alpha
                )
                speech_datasets = [
                    ResamplingDataset(
                        d, size_ratio=r, seed=0, epoch=epoch, replace=(r >= 1.0)
                    )
                    for d, r in zip(speech_datasets, size_ratios)
                ]
            speech_dataset = ConcatDataset(speech_datasets)
        elif len(speech_datasets) == 1:
            speech_dataset = speech_datasets[0]

        ### 2, create a text dataset using StripTokenDataset
        logger.info(f"split {split} has unpaired text? {has_text}")
        if not has_text:
            assert speech_dataset is not None
            self.datasets[split] = speech_dataset
            return

        text_pairs = [ item for item in text_splits if len(item.split(".")[-1].split("-")) > 1 ]
        text_monos = [ item for item in text_splits if len(item.split(".")[-1].split("-")) == 1 ]
        logging.info(f"text_monos: {text_monos}")
        logging.info(f"text_pairs: {text_pairs}")
        ### 2.1, create unpaired text datasets using DenoisingDataset
        mono_datasets = []
        if len(text_monos) > 0:
            for text_split in text_monos:
                lang = text_split.split('.')[-2]    # mono_deduped_filt_sort.de_DE.de_DE
                mask_whole_words = (
                    get_whole_word_mask(Namespace(**self.text_data_cfg.bpe_tokenizer), self.text_dictionary)
                    if self.cfg.text_cfg.mask_whole_words and lang in ("en_XX", "de_DE")
                    else None
                )

                mono_dataset = data_utils.load_indexed_dataset(
                    f"{self.cfg.text_cfg.text_data}/{text_split}",
                    self.text_dictionary,
                    combine=True,
                )
                mono_dataset = StripTokenDataset(mono_dataset, self.text_dictionary.eos())
                mono_dataset = maybe_shorten_dataset(
                    mono_dataset,
                    "xxxxx",
                    self.cfg.text_cfg.shorten_data_split_list,
                    self.cfg.text_cfg.shorten_method,
                    self.cfg.text_cfg.tokens_per_sample - 2,
                    self.cfg.text_cfg.seed,
                )
                logger.info("loaded {} samples from: {}".format(len(mono_dataset), text_split))
                ### prepend bos and eos to dataset
                mono_dataset = PrependTokenDataset(mono_dataset, self.text_dictionary.bos())
                mono_dataset = AppendTokenDataset(mono_dataset, self.text_dictionary.eos())
                mono_dataset = DenoisingDataset(
                    mono_dataset,
                    mono_dataset.sizes,
                    self.text_dictionary,
                    self.mask_idx,
                    mask_whole_words,
                    shuffle=self.cfg.text_cfg.shuffle_instance,
                    seed=self.cfg.text_cfg.seed,
                    args=self.cfg.text_cfg,
                    tgt_lang_idx=_lang_token_index(self.text_dictionary, lang) if self.cfg.text_cfg.prepend_tgt_lang_tag else None,
                )
                mono_datasets.append(mono_dataset)

        ### 2.2, create paired text datasets using load_langpair_dataset
        paired_datasets = []
        if len(text_pairs) > 0:
            for text_pair in text_pairs:
                text_split, lp = text_pair.rsplit('.', 1)       # e.g. "mt8corpus.de-en"
                src, tgt = lp.split("-")
                paired_dataset = load_langpair_dataset(
                    self.cfg.text_cfg.text_data,
                    text_split,
                    src,
                    self.text_src_dictionary,
                    tgt,
                    self.text_dictionary,
                    combine=True,
                    dataset_impl=None,
                    upsample_primary=1,
                    left_pad_source=False,
                    left_pad_target=False,
                    max_source_positions=self.cfg.text_cfg.tokens_per_sample,
                    max_target_positions=self.cfg.text_cfg.tokens_per_sample,
                    prepend_bos=False,
                    load_alignments=False,
                    append_source_id=True if self.cfg.text_cfg.prepend_tgt_lang_tag else False,
                    lang_format="<lang:{}>" if self.cfg.text_cfg.prepend_tgt_lang_tag else "[{}]",
                )
                if self.cfg.text_cfg.mask_text_ratio > 0:
                    # add mask
                    noise_token_id = self.text_src_dictionary.index("<mask>")
                    paired_dataset = LangPairMaskDataset(
                        paired_dataset,
                        src_bos=self.text_src_dictionary.bos(),
                        src_eos=self.text_src_dictionary.eos(),
                        noise_id=noise_token_id,
                        mask_ratio=self.cfg.text_cfg.mask_text_ratio,
                    )
                paired_datasets.append(paired_dataset)


        ### 3rd, compose a MultiCorpusDataset
        dataset_dict, max_positions_dict, distributions, max_tokens_ratios = self.resample_multi_modality_dataset(
            speech_dataset, mono_datasets, paired_datasets, text_monos, text_pairs, epoch=epoch,
        )
        self.datasets[split] = MultiCorpusDataset(
            dataset_dict,
            max_positions=max_positions_dict,
            distribution=distributions,
            max_tokens_ratio=max_tokens_ratios,
            seed=self.cfg.text_cfg.seed,
            sort_indices=True,
        )

    def max_positions(self) -> Tuple[int, int]:
        return (sys.maxsize, sys.maxsize)

    def filter_indices_by_size(
        self, indices: np.array, *args, **kwargs
    ) -> np.array:
        return indices

    def get_batch_iterator(
        self,
        dataset,
        max_tokens=None,
        max_sentences=None,
        max_positions=None,
        ignore_invalid_inputs=False,
        required_batch_size_multiple=1,
        seed=1,
        num_shards=1,
        shard_id=0,
        num_workers=0,
        epoch=1,
        data_buffer_size=0,
        disable_iterator_cache=False,
        skip_remainder_batch=False,
        grouped_shuffling=False,
        update_epoch_batch_itr=False,
    ):
        """
        Get an iterator that yields batches of data from the given dataset.
        Args:
            dataset (~fairseq.data.FairseqDataset): dataset to batch
            max_tokens (int, optional): max number of tokens in each batch
                (default: None).
            max_sentences (int, optional): max number of sentences in each
                batch (default: None).
            max_positions (optional): max sentence length supported by the
                model (default: None).
            ignore_invalid_inputs (bool, optional): don't raise Exception for
                sentences that are too long (default: False).
            required_batch_size_multiple (int, optional): require batch size to
                be a multiple of N (default: 1).
            seed (int, optional): seed for random number generator for
                reproducibility (default: 1).
            num_shards (int, optional): shard the data iterator into N
                shards (default: 1).
            shard_id (int, optional): which shard of the data iterator to
                return (default: 0).
            num_workers (int, optional): how many subprocesses to use for data
                loading. 0 means the data will be loaded in the main process
                (default: 0).
            epoch (int, optional): the epoch to start the iterator from
                (default: 1).
            data_buffer_size (int, optional): number of batches to
                preload (default: 0).
            disable_iterator_cache (bool, optional): don't cache the
                EpochBatchIterator (ignores `FairseqTask::can_reuse_epoch_itr`)
                (default: False).
            skip_remainder_batch (bool, optional): if set, discard the last
                batch in each training epoch, as the last batch is often smaller than
                    local_batch_size * distributed_word_size (default: ``True``).
            grouped_shuffling (bool, optional): group batches with each groups
                containing num_shards batches and shuffle groups. Reduces difference
                between sequence lengths among workers for batches sorted by length.
            update_epoch_batch_itr (bool optional): if true then donot use the cached
                batch iterator for the epoch
            
        Returns:
            ~fairseq.iterators.EpochBatchIterator: a batched iterator over the
                given dataset split
        """
        if not isinstance(dataset, MultiCorpusDataset):
            return super().get_batch_iterator(
                dataset,
                max_tokens=max_tokens,
                max_sentences=max_sentences,
                max_positions=max_positions,
                ignore_invalid_inputs=ignore_invalid_inputs,
                required_batch_size_multiple=required_batch_size_multiple,
                seed=seed,
                num_shards=num_shards,
                shard_id=shard_id,
                num_workers=num_workers,
                epoch=epoch,
                data_buffer_size=data_buffer_size,
                disable_iterator_cache=disable_iterator_cache,
                skip_remainder_batch=skip_remainder_batch,
                grouped_shuffling=grouped_shuffling,
                update_epoch_batch_itr=update_epoch_batch_itr,
            )

        can_reuse_epoch_itr = (
            not disable_iterator_cache
            and not update_epoch_batch_itr
            and self.can_reuse_epoch_itr(dataset)
        )
        if can_reuse_epoch_itr and dataset in self.dataset_to_epoch_iter:
            logger.debug("reusing EpochBatchIterator for epoch {}".format(epoch))
            return self.dataset_to_epoch_iter[dataset]

        assert isinstance(dataset, FairseqDataset)

        # initialize the dataset with the correct starting epoch
        dataset.set_epoch(epoch)

        # get indices ordered by example size
        with data_utils.numpy_seed(seed):
            indices = dataset.ordered_indices()

        # # filter examples that are too large
        # # we have already done it in multi_corpus_dataset
        # if max_positions is not None:
        #     indices = self.filter_indices_by_size(
        #         indices, dataset, max_positions, ignore_invalid_inputs
        #     )

        # create mini-batches with given size constraints
        batch_sampler = dataset.get_batch_sampler(
            indices,
            num_shards,
            seed,
            max_tokens=max_tokens,
            max_sentences=max_sentences,
            required_batch_size_multiple=required_batch_size_multiple,
            split_modality_batch=self.cfg.split_modality_batch,
        )

        # return a reusable, sharded iterator
        epoch_iter = iterators.EpochBatchIterator(
            dataset=dataset,
            collate_fn=dataset.collater,
            batch_sampler=batch_sampler,
            seed=seed,
            num_shards=num_shards,
            shard_id=shard_id,
            num_workers=num_workers,
            epoch=epoch,
            buffer_size=data_buffer_size,
            skip_remainder_batch=skip_remainder_batch,
            disable_shuffling=True,
            grouped_shuffling=grouped_shuffling,
        )

        if can_reuse_epoch_itr:
            self.dataset_to_epoch_iter[dataset] = epoch_iter

        return epoch_iter

    @classmethod
    def _get_size_ratios(cls, ids: List[str], sizes: List[int], alpha: float = 1.0):
        """Size ratios for temperature-based sampling
        (https://arxiv.org/abs/1907.05019)"""
        _sizes = np.array(sizes)
        prob = _sizes / _sizes.sum()
        smoothed_prob = prob ** alpha
        smoothed_prob = smoothed_prob / smoothed_prob.sum()
        size_ratio = (smoothed_prob * _sizes.sum()) / _sizes

        o_str = str({_i: f"{prob[i]:.3f}" for i, _i in enumerate(ids)})
        logger.info(f"original sampling probability: {o_str}")
        p_str = str({_i: f"{smoothed_prob[i]:.3f}" for i, _i in enumerate(ids)})
        logger.info(f"balanced sampling probability: {p_str}")
        sr_str = str({_id: f"{size_ratio[i]:.3f}" for i, _id in enumerate(ids)})
        logger.info(f"balanced sampling size ratio: {sr_str}")
        return size_ratio.tolist()

    def resample_multi_modality_dataset(self, speech_dataset, mono_datasets, paired_datasets, mono_splits, paired_splits, epoch=1, train=True):
        assert len(mono_datasets+paired_datasets) > 0, f"No text data loaded!"

        text_datasets = mono_datasets+paired_datasets
        if len(text_datasets) > 1 and self.cfg.text_sampling_alpha != 1.0:
            size_ratios = self._get_size_ratios(
                mono_splits + paired_splits, [len(s) for s in mono_datasets + paired_datasets], alpha=self.cfg.text_sampling_alpha
            )
            text_datasets = [
                ResamplingDataset(
                    d, size_ratio=r, seed=0, epoch=epoch, replace=(r >= 1.0)
                )
                for d, r in zip(text_datasets, size_ratios)
            ]

        mono_datasets = text_datasets[:len(mono_datasets)]
        paired_datasets = text_datasets[len(mono_datasets):]
        dataset_list = [speech_dataset]
        for datasets in [mono_datasets, paired_datasets]:
            if len(datasets) > 0:
                dataset_list.append(ConcatDataset(datasets))
            else:
                dataset_list.append(None)

        ### match speech/text datasets according to modality
        dataset_dict = OrderedDict((name, d) for name, d in zip(["speech", "text_mono", "text_paired"], dataset_list) if d is not None)
        max_positions_dict = OrderedDict((name, None) for name in dataset_dict.keys())
        if "text_paired" in dataset_dict:
            max_positions_dict["text_paired"] = (self.cfg.text_cfg.tokens_per_sample, self.cfg.text_cfg.tokens_per_sample)
        dataset_lens = np.array([len(dataset) for dataset in dataset_dict.values()])
        dataset_avg_sample_lens = np.array([
            sum([dataset.num_tokens(i) for i in np.random.randint(low=0, high=len(dataset), size=10000)]) / 10000.0 
            for dataset in dataset_dict.values()
        ])
        max_tokens_ratios = [1.0 / 320 / self.cfg.text_cfg.text_maxtokens_ratio] * len(dataset_dict)

        if not "speech" in dataset_dict:
            distributions = [l / sum(dataset_lens) for l in dataset_lens]
        else:
            ## we just keep the batches of speech and non-speech the same
            first_ratio = dataset_lens[0] / sum(dataset_lens)
            distributions = [max_tokens_ratios[0] * dataset_avg_sample_lens[0] / l for l in dataset_avg_sample_lens]
            text_total = sum(dataset_lens[1:])
            distributions = [1.2 * d * n / text_total for d, n in zip(distributions, dataset_lens)]
            max_tokens_ratios[0] = 1.0
            distributions[0] = 1.0
            distributions = [first_ratio * d for d in distributions]

        logging.info(f"Number samples of datasets is {dataset_lens}")
        logging.info(f"Avg sample length of datasets is {dataset_avg_sample_lens}")
        logging.info(f"Sampling distributions is {distributions}")
        logging.info(f"Maxtokens ratio is {max_tokens_ratios}")
        return dataset_dict, max_positions_dict, distributions, max_tokens_ratios

    def build_tokenizer(self, cfg=None):
        logger.info(f"tokenizer: {self.cfg.hubert_tokenizer}")
        if self.cfg.hubert_tokenizer != "none":
            return encoders.build_bpe(Namespace(**{"bpe": self.cfg.hubert_tokenizer, "sentencepiece_model": self.cfg.sp_path}))
        else:
            return None

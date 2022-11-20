# ----------------------------------------------------------------------------
# VatLM: Visual-Audio-Text Pre-Training  with Unified Masked Prediction for Speech Representation Learning
# Github source: https://github.com/microsoft/SpeechT5/tree/main/VATLM
# Code based on fairseq: https://github.com/facebookresearch/fairseq and av_hubert: https://github.com/facebookresearch/av_hubert
# 
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# ----------------------------------------------------------------------------

import logging
import os, glob
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np

from dataclasses import dataclass, field
from fairseq import metrics, search
from fairseq.data import Dictionary, encoders
from fairseq.dataclass.configs import FairseqDataclass
from fairseq.tasks import register_task
from fairseq.tasks.fairseq_task import FairseqTask
from omegaconf import MISSING, II
import numpy as np
from argparse import Namespace

DBG=True if len(sys.argv) == 1 else False

if DBG:
    from vathubert.data.vathubert_dataset import VATHubertDataset
    from vathubert.sequence_generator import SequenceGenerator
else:

    from vathubert.data.vathubert_dataset import VATHubertDataset
    from vathubert.sequence_generator import SequenceGenerator
    from vathubert.data.audiohubert_dataset import AudioHubertDataset
    from vathubert.data.texthubert_dataset import TextHubertDataset
    from vathubert.data.onlyaudiohubert_dataset import OnlyAudioHubertDataset

from fairseq.data.audio.multi_corpus_dataset_audio import MultiCorpusDataset
from collections import OrderedDict
from fairseq.data import FairseqDataset
from fairseq.data import data_utils
from fairseq.data import iterators


logger = logging.getLogger(__name__)


class LabelEncoder(object):
    def __init__(self, dictionary: Dictionary) -> None:
        self.dictionary = dictionary

    def __call__(self, label: str) -> List[str]:
        return self.dictionary.encode_line(
            label, append_eos=False, add_if_not_exist=False,
        )

class LabelEncoderS2SToken(object):
    def __init__(self, dictionary: Dictionary, bpe_tokenizer) -> None:
        self.bpe_tokenizer = bpe_tokenizer
        self.dictionary = dictionary

    def __call__(self, label: str) -> List[str]:
        label = self.bpe_tokenizer.encode(label.lower())
        return self.dictionary.encode_line(
            label, append_eos=True, add_if_not_exist=False,
        ).long()

    def decode(self, tok, symbols_ignore=None):
        tok = self.dictionary.string(tok, extra_symbols_to_ignore=symbols_ignore)
        if self.bpe_tokenizer:
            tok = self.bpe_tokenizer.decode(tok)
        return tok

@dataclass
class VATHubertPretrainingConfig(FairseqDataclass):
    data: str = field(
        default=MISSING, metadata={"help": "path to data directory"}
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
    max_sample_size: Optional[int] = field(
        default=None,
        metadata={"help": "max sample size to keep in training"},
    )
    min_sample_size: Optional[int] = field(
        default=None,
        metadata={"help": "min sample size to keep in training"},
    )
    max_trim_sample_size: Optional[int] = field(
        default=II("task.max_sample_size"),
        metadata={"help": "max sample size to trim to for batching"},
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
    pdb: Optional[bool] = field(
        default=False,
        metadata={"help": "pdb"},
    )
    stack_order_audio: int = field(
        default=1,
        metadata={"help": "concatenate n consecutive audio frames for one step"},
    )
    skip_verify: Optional[bool] = field(
        default=False,
        metadata={"help": "skip verifying label-audio alignment"},
    )

    text_sampling_alpha: float = field(
        default=0.2,
        metadata={
            "help": "Hyper-parameter alpha = 1/T for temperature-based text resampling."
            "(alpha = 1 for no resampling)"
        },
    )
    split_modality_batch: bool = field(
        default=False,
        metadata={"help": "whether create all samples of different modalities in a batch"},
    )    
    image_aug: bool = field(default=False, metadata={'help': 'image data augmentation'})
    image_crop_size: int = field(
        default=88, metadata={"help": "image ROI size"})
    image_mean: float = field(
        default=0.421, metadata={"help": "image mean"})
    image_std: float = field(
        default=0.165, metadata={"help": "image std"})
    modalities: Optional[List[str]] = field(default_factory=lambda: ["audio", "video"], metadata={'help': 'modalities to load'})
    is_s2s: bool=field(default=False, metadata={'help': 'seq2seq fine-tuning only'})
    tokenizer_bpe_name: Optional[str] = field(default=None, metadata={'help': 'tokenizer model name'})
    tokenizer_bpe_model: Optional[str] = field(default=None, metadata={'help': 'tokenizer model path'})
    noise_wav: Optional[str] = field(default=None, metadata={'help': 'manifest of noise wav files (one wav file path per line)'})
    noise_prob: float = field(default=0, metadata={'help': 'noise probability'})
    noise_snr: Optional[str] = field(default='0', metadata={'help': 'noise SNR in audio'})
    noise_num: int = field(default=1, metadata={'help': 'number of noise wav files to mix'})
    fine_tuning: bool = field(default=False, metadata={"help": "set to true if fine-tuning AV-Hubert"})
    use_supervised_data: bool = field(default=True, metadata={"help": "use paired speech-text data"})
    sup_data_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "supervised dataset path",
        },
    )
    sup_manifest: Optional[str] = field(
        default=None,
        metadata={
            "help": "supervised dataset manifest",
        },
    )
    sample_distributions: Optional[str] = field(default='0', metadata={'help': 'sample distribution'})
    ###########
    use_extra_textdata: bool = field(default=True, metadata={"help": "use extra text data"})
    onlytext_manifest: Optional[str] = field(
        default=None,
        metadata={
            "help": "text-only dataset manifest",
        },
    )
    use_extra_audiodata: bool = field(default=True, metadata={"help": "use extra audio data"})
    onlyaudio_manifest: Optional[str] = field(
        default=None,
        metadata={
            "help": "audio-only dataset manifest",
        },
    )

@register_task("vat_hubert_pretraining", dataclass=VATHubertPretrainingConfig)
class VATHubertPretrainingTask(FairseqTask):

    cfg: VATHubertPretrainingConfig

    def __init__(
        self,
        cfg: VATHubertPretrainingConfig,
    ) -> None:
        super().__init__(cfg)

        logger.info(f"current directory is {os.getcwd()}")
        logger.info(f"VATHubertPretrainingTask Config {cfg}")

        self.state.add_factory("phone_dictionary", self.load_phone_dictionaries)
        # self.state.add_factory("s2s_tokenizer", self.load_tokenizer)

        self.fine_tuning = cfg.fine_tuning
        if cfg.fine_tuning:
            self.state.add_factory("target_dictionary", self.load_dictionaries)
            if cfg.is_s2s:
                self.state.add_factory("s2s_tokenizer", self.load_tokenizer)
        else:
            self.state.add_factory("dictionaries", self.load_dictionaries)



        self.blank_symbol = "<s>"

    @property
    def source_dictionary(self) -> Optional[Dictionary]:
        return None # self._source_dictionary

    @property
    def target_dictionary(self) -> Optional[Dictionary]:
        return self.state.target_dictionary # self._target_dictionary

    @property
    def dictionaries(self) -> List[Dictionary]:
        return self.state.dictionaries

    @property
    def phone_dictionary(self) -> List[Dictionary]:
        return self.state.phone_dictionary


    def load_dictionaries(self):
        label_dir = self.cfg.data if self.cfg.label_dir is None else self.cfg.label_dir
        dictionaries = [
            Dictionary.load(f"{label_dir}/dict.{label}.txt")
            for label in self.cfg.labels
        ]
        return dictionaries[0] if self.cfg.fine_tuning else dictionaries

    def load_tokenizer(self):
        logger.info(f"Using tokenizer")
        bpe_args = Namespace(**{'bpe': self.cfg.tokenizer_bpe_name, f"{self.cfg.tokenizer_bpe_name}_model": self.cfg.tokenizer_bpe_model})
        bpe_tokenizer = encoders.build_bpe(bpe_args)
        return bpe_tokenizer

    def load_phone_dictionaries(self):
        dictionaries = [
            Dictionary.load(f"{self.cfg.sup_manifest}/dict.phn.txt")
        ]
        return dictionaries


    @property
    def s2s_tokenizer(self):
        return self.state.s2s_tokenizer

    @classmethod
    def setup_task(
        cls, cfg: VATHubertPretrainingConfig, **kwargs
    ) -> "VATHubertPretrainingTask":
        if cfg.pdb:
            import pdb
            pdb.set_trace()
        return cls(cfg)

    def get_label_dir(self) -> str:
        if self.cfg.label_dir is None:
            return self.cfg.data
        return self.cfg.label_dir

    def load_dataset(self, split: str, epoch=1, **kwargs) -> None:
        manifest = f"{self.cfg.data}/{split}.tsv"
        dictionaries = [self.target_dictionary] if self.fine_tuning else self.dictionaries
        pad_list = [dictionary.pad() for dictionary in dictionaries]   # [1], blank应该是[0]
        eos_list = [dictionary.eos() for dictionary in dictionaries]   # [2]
        if not self.cfg.is_s2s:
            procs = [LabelEncoder(dictionary) for dictionary in dictionaries]
        else:
            logger.info(f"Using tokenizer")
            bpe_tokenizer = self.s2s_tokenizer
            procs = [LabelEncoderS2SToken(dictionary, bpe_tokenizer) for dictionary in dictionaries]
        paths = [
            f"{self.get_label_dir()}/{split}.{l}" for l in self.cfg.labels
        ]
        image_aug = self.cfg.image_aug if split == 'train' else False
        noise_fn, noise_snr = f"{self.cfg.noise_wav}/{split}.tsv" if self.cfg.noise_wav is not None else None, eval(self.cfg.noise_snr)
        noise_num = self.cfg.noise_num # 
        
        all_datasets = []
        avdatasets = VATHubertDataset(
            manifest,
            sample_rate=self.cfg.sample_rate,
            label_paths=paths,
            label_rates=self.cfg.label_rate,
            pad_list=pad_list,
            eos_list=eos_list,
            label_processors=procs,
            max_keep_sample_size=self.cfg.max_sample_size,
            min_keep_sample_size=self.cfg.min_sample_size,
            max_sample_size=self.cfg.max_trim_sample_size,
            pad_audio=self.cfg.pad_audio,
            normalize=self.cfg.normalize,
            store_labels=False,
            random_crop=self.cfg.random_crop,
            single_target=self.cfg.single_target,
            stack_order_audio=self.cfg.stack_order_audio,
            skip_verify=self.cfg.skip_verify,
            image_mean=self.cfg.image_mean,
            image_std=self.cfg.image_std,
            image_crop_size=self.cfg.image_crop_size,
            image_aug=image_aug,
            modalities=self.cfg.modalities,
            is_s2s=self.cfg.is_s2s,
            noise_fn=noise_fn,
            noise_prob=self.cfg.noise_prob,
            noise_snr=noise_snr,
            noise_num=noise_num
        )
        all_datasets.append(avdatasets)

        # import pdb
        # pdb.set_trace()

        if self.cfg.use_supervised_data:
            sup_manifest = f"{self.cfg.sup_manifest}/{split}.tsv"

            sup_paths = [
                f"{self.cfg.sup_data_path}/{split}.{l}" for l in self.cfg.labels
            ]
            
            phone_dictionaries = self.phone_dictionary
            phone_procs = [LabelEncoder(dictionary) for dictionary in phone_dictionaries]
            
            atdatasets = AudioHubertDataset(
                sup_manifest,
                sample_rate=self.cfg.sample_rate,
                label_paths=sup_paths,
                label_rates=self.cfg.label_rate,
                pad_list=pad_list,
                eos_list=eos_list,
                label_processors=procs,
                phone_sequence_processors=phone_procs,
                max_keep_sample_size=self.cfg.max_sample_size,
                min_keep_sample_size=self.cfg.min_sample_size,
                max_sample_size=self.cfg.max_trim_sample_size,
                pad_audio=self.cfg.pad_audio,
                normalize=self.cfg.normalize,
                store_labels=True,
                single_target=self.cfg.single_target,
                stack_order_audio=self.cfg.stack_order_audio,
                skip_verify=self.cfg.skip_verify,
                is_s2s=self.cfg.is_s2s,
            )
            all_datasets.append(atdatasets)
  
        if self.cfg.use_extra_textdata:
            extra_text_manifest = f"{self.cfg.onlytext_manifest}/{split}.tsv"
            extra_text_paths = [
                f"{self.cfg.onlytext_manifest}/{split}.{l}" for l in self.cfg.labels
            ]
            
            # import pdb
            # pdb.set_trace()

            textdatasets = TextHubertDataset(
                extra_text_manifest,
                sample_rate=self.cfg.sample_rate,
                label_paths=extra_text_paths,
                label_rates=self.cfg.label_rate,
                pad_list=pad_list,
                eos_list=eos_list,
                label_processors=procs,
                phone_sequence_processors=phone_procs,
                max_keep_sample_size=self.cfg.max_sample_size,
                min_keep_sample_size=self.cfg.min_sample_size,
                max_sample_size=self.cfg.max_trim_sample_size,
                pad_audio=self.cfg.pad_audio,
                normalize=self.cfg.normalize,
                store_labels=True,
                single_target=self.cfg.single_target,
                stack_order_audio=self.cfg.stack_order_audio,
                skip_verify=self.cfg.skip_verify,
                is_s2s=self.cfg.is_s2s,
            )
            all_datasets.append(textdatasets)
        
        if self.cfg.use_extra_audiodata:
            extra_audio_manifest = f"{self.cfg.onlyaudio_manifest}/{split}.tsv"
            extra_audio_paths = [
                f"{self.cfg.onlyaudio_manifest}/{split}.{l}" for l in self.cfg.labels
            ]

            audiodatasets = OnlyAudioHubertDataset(
                extra_audio_manifest,
                sample_rate=self.cfg.sample_rate,
                label_paths=extra_audio_paths,
                label_rates=self.cfg.label_rate,
                pad_list=pad_list,
                eos_list=eos_list,
                label_processors=procs,
                max_keep_sample_size=self.cfg.max_sample_size,
                min_keep_sample_size=self.cfg.min_sample_size,
                max_sample_size=self.cfg.max_trim_sample_size,
                pad_audio=self.cfg.pad_audio,
                normalize=self.cfg.normalize,
                store_labels=False,
                single_target=self.cfg.single_target,
                stack_order_audio=self.cfg.stack_order_audio,
                skip_verify=self.cfg.skip_verify,
                is_s2s=self.cfg.is_s2s,
            )
            all_datasets.append(audiodatasets)        




        dataset_list = all_datasets
        dataset_dict = OrderedDict((name, d) for name, d in zip(["videoaudio", "audiotext", "onlytext", "onlyaudio"], dataset_list) if d is not None)
        if not self.fine_tuning:
            max_positions_dict = {
                "videoaudio": 1024,
                "audiotext": 1024,
                "onlytext": 1024,
                "onlyaudio": 1024,
            }
            max_positions_dict = OrderedDict((name, max_positions_dict[name]) for name in dataset_dict.keys())

            max_tokens_ratios_dict = {
                "videoaudio": 1.0,
                "audiotext": 1.0,
                "onlytext": 1.0,
                "onlyaudio": 1.0,
            }
            max_tokens_ratios = [max_tokens_ratios_dict[name] for name in dataset_dict.keys()]
            dataset_lens = np.array([len(dataset) for dataset in dataset_dict.values()])
            dataset_avg_sample_lens = np.array([
                sum([dataset.num_tokens(i) for i in np.random.randint(low=0, high=len(dataset), size=10000)]) / 10000.0 
                for dataset in dataset_dict.values()
            ])
            distributions = [eval(self.cfg.sample_distributions)[0], eval(self.cfg.sample_distributions)[1], eval(self.cfg.sample_distributions)[2], eval(self.cfg.sample_distributions)[3]]

        

            logging.info(f"Number samples of datasets is {dataset_lens}")
            logging.info(f"Avg sample length of datasets is {dataset_avg_sample_lens}")
            logging.info(f"Sampling distributions is {distributions}")
            logging.info(f"Maxtokens ratio is {max_tokens_ratios}")
            logging.info(f"split_modality_batch is {self.cfg.split_modality_batch}")


            self.datasets[split] = MultiCorpusDataset(
                dataset_dict,
                max_positions=max_positions_dict,
                distribution=distributions,
                max_tokens_ratio=max_tokens_ratios,
                seed=1234,
                sort_indices=True,
            )

        if self.fine_tuning:
            self.datasets[split] = VATHubertDataset(
                manifest,
                sample_rate=self.cfg.sample_rate,
                label_paths=paths,
                label_rates=self.cfg.label_rate,
                pad_list=pad_list,
                eos_list=eos_list,
                label_processors=procs,
                max_keep_sample_size=self.cfg.max_sample_size,
                min_keep_sample_size=self.cfg.min_sample_size,
                max_sample_size=self.cfg.max_trim_sample_size,
                pad_audio=self.cfg.pad_audio,
                normalize=self.cfg.normalize,
                store_labels=False,
                random_crop=self.cfg.random_crop,
                single_target=self.cfg.single_target,
                stack_order_audio=self.cfg.stack_order_audio,
                skip_verify=self.cfg.skip_verify,
                image_mean=self.cfg.image_mean,
                image_std=self.cfg.image_std,
                image_crop_size=self.cfg.image_crop_size,
                image_aug=image_aug,
                modalities=self.cfg.modalities,
                is_s2s=self.cfg.is_s2s,
                noise_fn=noise_fn,
                noise_prob=self.cfg.noise_prob,
                noise_snr=noise_snr,
                noise_num=noise_num
            )

    # @classmethod
    # def _get_size_ratios(cls, ids: List[str], sizes: List[int], alpha: float = 1.0):
    #     """Size ratios for temperature-based sampling
    #     (https://arxiv.org/abs/1907.05019)"""
    #     _sizes = np.array(sizes)
    #     prob = _sizes / _sizes.sum()
    #     smoothed_prob = prob ** alpha
    #     smoothed_prob = smoothed_prob / smoothed_prob.sum()
    #     size_ratio = (smoothed_prob * _sizes.sum()) / _sizes

    #     o_str = str({_i: f"{prob[i]:.3f}" for i, _i in enumerate(ids)})
    #     logger.info(f"original sampling probability: {o_str}")
    #     p_str = str({_i: f"{smoothed_prob[i]:.3f}" for i, _i in enumerate(ids)})
    #     logger.info(f"balanced sampling probability: {p_str}")
    #     sr_str = str({_id: f"{size_ratio[i]:.3f}" for i, _id in enumerate(ids)})
    #     logger.info(f"balanced sampling size ratio: {sr_str}")
    #     return size_ratio.tolist()


    # def resample_multi_modality_dataset(self, speech_dataset, paired_datasets, epoch=1, train=True):
           
    #     if len(paired_datasets) > 1 and self.cfg.text_sampling_alpha != 1.0:
    #         size_ratios = self._get_size_ratios(
    #             paired_splits, [len(s) for s in paired_datasets], alpha=self.cfg.text_sampling_alpha
    #         )
    #         paired_datasets = [
    #             ResamplingDataset(
    #                 d, size_ratio=r, seed=0, epoch=epoch, replace=(r >= 1.0)
    #             ) for d, r in zip(paired_datasets, size_ratios)
    #         ]

    #     dataset_list = [speech_dataset]
    #     for datasets in [paired_datasets]:
    #         if len(datasets) > 1:
    #             dataset_list.append(ConcatDataset(datasets))
    #         elif len(datasets) == 1:
    #             dataset_list.append(datasets[0])
    #         else:
    #             dataset_list.append(None)

    #     ### match speech/text datasets according to modality
    #     dataset_dict = OrderedDict((name, d) for name, d in zip(["speech", "speech_sup", "text_mono", "text_paired"], dataset_list) if d is not None)
    #     max_positions_dict = {
    #         "speech": None,
    #         "speech_sup": None,
    #         "text_mono": (1024, 1024),
    #         "text_paired": (1024, 1024),
    #     }
    #     max_positions_dict = OrderedDict((name, max_positions_dict[name]) for name in dataset_dict.keys())
    #     max_tokens_ratios_dict = {
    #         "speech": 1.0,
    #         "speech_sup": 1.0,
    #         "text_mono": 1.0 / 320 / 1.0,
    #         "text_paired": 1.0 / 320 / 1.0,
    #     }
    #     max_tokens_ratios = [max_tokens_ratios_dict[name] for name in dataset_dict.keys()]
    #     dataset_lens = np.array([len(dataset) for dataset in dataset_dict.values()])
    #     dataset_avg_sample_lens = np.array([
    #         sum([dataset.num_tokens(i) for i in np.random.randint(low=0, high=len(dataset), size=10000)]) / 10000.0 
    #         for dataset in dataset_dict.values()
    #     ])

    #     if not "speech" in dataset_dict:
    #         distributions = [l / sum(dataset_lens) for l in dataset_lens]
    #     else:
    #         ## we just keep the batches of speech and non-speech the same, expand_coef is to ensure speech batches is less than others
    #         first_ratio = dataset_lens[0] / sum(dataset_lens)
    #         expand_coef = 1.8 if sup_dataset is None else 1.1 * sum(dataset_lens[0:2]) / dataset_lens[0]
    #         distributions = [expand_coef * max_tokens_ratios[i] * dataset_avg_sample_lens[0] / l for (i, l) in enumerate(dataset_avg_sample_lens)]
    #         distributions[0] = 1.0
    #         if sup_dataset is not None:
    #             distributions[1] = dataset_lens[1] / dataset_lens[0]
    #         distributions = [first_ratio * d for d in distributions]

    #     logging.info(f"Number samples of datasets is {dataset_lens}")
    #     logging.info(f"Avg sample length of datasets is {dataset_avg_sample_lens}")
    #     logging.info(f"Sampling distributions is {distributions}")
    #     logging.info(f"Maxtokens ratio is {max_tokens_ratios}")
    #     return dataset_dict, max_positions_dict, distributions, max_tokens_ratios


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
        
        if self.fine_tuning or not isinstance(dataset, MultiCorpusDataset):
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
            )
        logging.info(f"num_workers is {num_workers}")
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

        # filter examples that are too large
        if max_positions is not None:
            indices = self.filter_indices_by_size(
                indices, dataset, max_positions, ignore_invalid_inputs
            )

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
            disable_shuffling=True,
        )

        if can_reuse_epoch_itr:
            self.dataset_to_epoch_iter[dataset] = epoch_iter

        return epoch_iter


    def build_generator(
        self, models, args, seq_gen_cls=None, extra_gen_cls_kwargs=None, prefix_allowed_tokens_fn=None,
    ):
        """
        Build a :class:`~fairseq.SequenceGenerator` instance for this
        task.
        Args:
            models (List[~fairseq.models.FairseqModel]): ensemble of models
            args (fairseq.dataclass.configs.GenerationConfig):
                configuration object (dataclass) for generation
            extra_gen_cls_kwargs (Dict[str, Any]): extra options to pass
                through to SequenceGenerator
            prefix_allowed_tokens_fn (Callable[[int, torch.Tensor], List[int]]):
                If provided, this function constrains the beam search to
                allowed tokens only at each step. The provided function
                should take 2 arguments: the batch ID (`batch_id: int`)
                and a unidimensional tensor of token ids (`inputs_ids:
                torch.Tensor`). It has to return a `List[int]` with the
                allowed tokens for the next generation step conditioned
                on the previously generated tokens (`inputs_ids`) and
                the batch ID (`batch_id`). This argument is useful for
                constrained generation conditioned on the prefix, as
                described in "Autoregressive Entity Retrieval"
                (https://arxiv.org/abs/2010.00904) and
                https://github.com/facebookresearch/GENRE.
        """
        if getattr(args, "score_reference", False):
            from fairseq.sequence_scorer import SequenceScorer

            return SequenceScorer(
                self.target_dictionary,
                compute_alignment=getattr(args, "print_alignment", False),
            )

        # Choose search strategy. Defaults to Beam Search.
        sampling = getattr(args, "sampling", False)
        sampling_topk = getattr(args, "sampling_topk", -1)
        sampling_topp = getattr(args, "sampling_topp", -1.0)
        diverse_beam_groups = getattr(args, "diverse_beam_groups", -1)
        diverse_beam_strength = getattr(args, "diverse_beam_strength", 0.5)
        match_source_len = getattr(args, "match_source_len", False)
        diversity_rate = getattr(args, "diversity_rate", -1)
        constrained = getattr(args, "constraints", False)
        if prefix_allowed_tokens_fn is None:
            prefix_allowed_tokens_fn = getattr(args, "prefix_allowed_tokens_fn", None)
        if (
            sum(
                int(cond)
                for cond in [
                    sampling,
                    diverse_beam_groups > 0,
                    match_source_len,
                    diversity_rate > 0,
                ]
            )
            > 1
        ):
            raise ValueError("Provided Search parameters are mutually exclusive.")
        assert sampling_topk < 0 or sampling, "--sampling-topk requires --sampling"
        assert sampling_topp < 0 or sampling, "--sampling-topp requires --sampling"

        if sampling:
            search_strategy = search.Sampling(
                self.target_dictionary, sampling_topk, sampling_topp
            )
        elif diverse_beam_groups > 0:
            search_strategy = search.DiverseBeamSearch(
                self.target_dictionary, diverse_beam_groups, diverse_beam_strength
            )
        elif match_source_len:
            # this is useful for tagging applications where the output
            # length should match the input length, so we hardcode the
            # length constraints for simplicity
            search_strategy = search.LengthConstrainedBeamSearch(
                self.target_dictionary,
                min_len_a=1,
                min_len_b=0,
                max_len_a=1,
                max_len_b=0,
            )
        elif diversity_rate > -1:
            search_strategy = search.DiverseSiblingsSearch(
                self.target_dictionary, diversity_rate
            )
        elif constrained:
            search_strategy = search.LexicallyConstrainedBeamSearch(
                self.target_dictionary, args.constraints
            )
        elif prefix_allowed_tokens_fn:
            search_strategy = search.PrefixConstrainedBeamSearch(
                self.target_dictionary, prefix_allowed_tokens_fn
            )
        else:
            search_strategy = search.BeamSearch(self.target_dictionary)

        extra_gen_cls_kwargs = extra_gen_cls_kwargs or {}
        if seq_gen_cls is None:
            if getattr(args, "print_alignment", False):
                seq_gen_cls = SequenceGeneratorWithAlignment
                extra_gen_cls_kwargs["print_alignment"] = args.print_alignment
            else:
                seq_gen_cls = SequenceGenerator

        return seq_gen_cls(
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
            search_strategy=search_strategy,
            **extra_gen_cls_kwargs,
        )

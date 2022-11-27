# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
from os import replace
import time
from collections import OrderedDict
from typing import Any, Dict, List, Optional

import numpy as np
from fairseq.data import data_utils

from fairseq.data import FairseqDataset

logger = logging.getLogger(__name__)


class MultiCorpusDataset(FairseqDataset):
    """
    Stores multiple instances of FairseqDataset together.
    Unless batch_sample=True, requires each instance
    to be the same dataset, as the collate method needs to work on batches with
    samples from each dataset.

    Allows specifying a distribution over the datasets to use. Note that unlike
    MultiCorpusSampledDataset, this distribution allows sampling for each item,
    rather than on a batch level. Note that datasets with sampling probabilty
    of 0 will be skipped.

    Each time ordered_indices() is called, a new sample is generated with
    the specified distribution.

    Args:
        datasets: a OrderedDict of FairseqDataset instances.
        distribution: a List containing the probability of getting an utterance from
                        corresponding dataset
        seed: random seed for sampling the datsets
        sort_indices: if true, will sort the ordered indices by size
        batch_sample: if true, will ensure each batch is from a single dataset
    """

    def __init__(
        self,
        datasets: Dict[str, FairseqDataset],
        max_positions: Dict,
        distribution: List[float],
        max_tokens_ratio: List[float],
        seed: int,
        sort_indices: bool = False,
        batch_sample: bool = False,
        distributed_rank: Optional[int] = None,
    ):
        super().__init__()
        assert isinstance(datasets, OrderedDict)
        assert len(datasets) == len(distribution)
        # assert sum(distribution) == 1
        self.datasets = datasets
        self.distribution = distribution
        self.max_tokens_ratio = max_tokens_ratio
        self.seed = seed
        self.sort_indices = sort_indices
        self.batch_sample = batch_sample
        self.distributed_rank = distributed_rank
        self.max_positions = max_positions

        # Avoid repeated conversions to list later
        self.dataset_list = list(datasets.values())
        self.total_num_instances = 0

        # first_dataset = self.dataset_list[0]

        self.num_instances_per_dataset = []
        self.dataset_offsets = []
        for i, dataset in enumerate(self.dataset_list):
            assert isinstance(dataset, FairseqDataset)
            # assert type(dataset) is type(first_dataset)
            self.num_instances_per_dataset.append(
                0 if self.distribution[i] == 0 else len(dataset)
            )
            self.dataset_offsets.append(self.total_num_instances)
            self.total_num_instances += self.num_instances_per_dataset[i]

    def ordered_indices(self):
        start = time.time()
        with data_utils.numpy_seed(self.seed, self.epoch):
            logger.info(f"sampling new dataset with seed {self.seed} epoch {self.epoch}")
            sampled_indices = []

            # For each dataset i, sample self.distribution[i] * self.total_num_instances
            for i, key in enumerate(self.datasets):
                if self.distribution[i] == 0:
                    # skip dataset if sampling probability is 0
                    continue

                if i < len(self.datasets) - 1:
                    num_instances = int(self.distribution[i] * self.total_num_instances)
                    high = self.dataset_offsets[i + 1]
                else:
                    num_instances = int(self.distribution[i] * self.total_num_instances)
                    high = self.total_num_instances

                logger.info(f"sampling {num_instances} from {key} dataset")

                # First, add k copies of the dataset where k = num_instances // len(dataset).
                # This ensures an equal distribution of the data points as much as possible.
                # For the remaining entries randomly sample them
                dataset_size = len(self.datasets[key])
                num_copies = num_instances // dataset_size
                dataset_indices = (
                    np.random.permutation(high - self.dataset_offsets[i])
                    + self.dataset_offsets[i]
                )[: num_instances - num_copies * dataset_size]
                if num_copies > 0:
                    dataset_indices = np.concatenate(
                            (
                                np.repeat(
                                    np.arange(self.dataset_offsets[i], high), num_copies
                                ),
                                dataset_indices,
                            )
                        )
                ### filter by size
                if self.max_positions[key] is not None:
                    dataset_indices -= self.dataset_offsets[i]
                    dataset_indices, ignored = self.datasets[key].filter_indices_by_size(
                        dataset_indices,
                        self.max_positions[key],
                    )
                    if len(ignored) > 0:
                        logger.warning(
                            (
                                "{:,} samples have invalid sizes and will be skipped, "
                                "max_positions={}, first few sample ids={}"
                            ).format(len(ignored), self.max_positions[key], ignored[:10])
                        )
                    dataset_indices += self.dataset_offsets[i]
                

                sampled_indices += list(dataset_indices)
            
            np.random.shuffle(sampled_indices)
            if self.sort_indices:
                sampled_indices.sort(key=lambda i: self.num_tokens(i))

            logger.info(
                "multi_corpus_dataset ordered_indices took {}s".format(
                    time.time() - start
                )
            )
            return np.array(sampled_indices, dtype=np.int64)

    def _map_index(self, index: int):
        """
        If dataset A has length N and dataset B has length M
        then index 1 maps to index 1 of dataset A, and index N + 1
        maps to index 1 of B.
        """
        counter = 0
        for num_instances, key in zip(self.num_instances_per_dataset, self.datasets):
            if index < counter + num_instances:
                return index - counter, key
            counter += num_instances
        raise ValueError(
            "Invalid index: {}, max: {}".format(index, self.total_num_instances)
        )

    def __len__(self):
        """
        Length of this dataset is the sum of individual datasets
        """
        return self.total_num_instances

    def __getitem__(self, index):
        new_index, key = self._map_index(index)
        try:
            item = self.datasets[key][new_index]
            item["full_id"] = index
            return item
        except Exception as e:
            e.args = (f"Error from {key} dataset", *e.args)
            raise

    def collater(self, samples):
        """
        If we are doing batch sampling, then pick the right collater to use.

        Otherwise we assume all collaters are the same.
        """
        if len(samples) == 0:
            return None
        
        samples_dict = {key: [] for key in self.datasets}
        for s in samples:
            _, key = self._map_index(s["full_id"])
            samples_dict[key].append(s)
        
        batch = {}
        for key in samples_dict:
            if len(samples_dict[key]) == 0:
                continue
            batch[key] = self.datasets[key].collater(samples_dict[key])

        return batch


    def num_tokens(self, index: int):
        index, key = self._map_index(index)
        return self.datasets[key].num_tokens(index)

    def size(self, index: int):
        index, key = self._map_index(index)
        return self.datasets[key].size(index)

    @property
    def can_reuse_epoch_itr_across_epochs(self):
        return False

    def set_epoch(self, epoch, **unused):
        super().set_epoch(epoch)
        logger.info(f"setting epoch of multi_corpus_dataset to {epoch}")
        for ds in self.dataset_list:
            if hasattr(ds, "set_epoch"):
                ds.set_epoch(epoch)
        self.epoch = epoch

    @property
    def supports_prefetch(self):
        return False

    @property
    def supports_fetch_outside_dataloader(self):
        return all(
            self.datasets[key].supports_fetch_outside_dataloader
            for key in self.datasets
        )
        



    def batch_by_size(
        self,
        indices,
        max_tokens=None,
        max_sentences=None,
        required_batch_size_multiple=1,
    ):    
        if not self.batch_sample:
            return super().batch_by_size(
                indices, max_tokens, max_sentences, required_batch_size_multiple
            )

        dataset_indices = {key: [] for key in self.datasets}
        for i in indices:
            _, key = self._map_index(i)
            dataset_indices[key].append(i)

        batches_dict = {}
        for n, key in enumerate(dataset_indices):
            max_tokens_ratio = self.max_tokens_ratio[n]
            cur_batches = super().batch_by_size(
                np.array(dataset_indices[key], dtype=np.int64),
                round(max_tokens * max_tokens_ratio),
                max_sentences,
                required_batch_size_multiple,
            )
            logger.info(f"Created {len(cur_batches)} batches for dataset {key}")
            batches_dict[key] = cur_batches

        return batches_dict





    # def get_batch_sampler(
    #     self,
    #     indices,
    #     num_shards,
    #     seed,
    #     max_tokens=None,
    #     max_sentences=None,
    #     required_batch_size_multiple=1,
    # ):

    #     def batch_sampler(dataset, epoch):
    #         start = time.time()
    #         batches_dict = dataset.batch_by_size(
    #             indices,
    #             max_tokens=max_tokens,
    #             max_sentences=max_sentences,
    #             required_batch_size_multiple=required_batch_size_multiple,
    #         )
    #         new_batches = []
    #         speech_shuf_batches, text_shuf_batches = [], []
    #         for name, batches in batches_dict.items():
    #             if name.startswith("speech"):
    #                 speech_shuf_batches += shuffle_buckets(batches, seed=seed+epoch)
    #             else:
    #                 batches = inner_bucket_shuffle(batches, seed+epoch, num_shards*10)
    #                 text_shuf_batches += batches[: (len(batches) // num_shards) * num_shards]
    #         text_shuf_batches = shuffle_buckets(text_shuf_batches, seed=seed+epoch, inner_shuf=False)
    #         if len(speech_shuf_batches) > 0:
    #             shuffled_batches_sets = [speech_shuf_batches, text_shuf_batches]
    #         else:
    #             logger.warning(f"shuffled_batches_sets have 0 items, please ensure that no speech data loaded")
    #             return text_shuf_batches

    #         num_batch = min(len(batches) for batches in shuffled_batches_sets)
    #         for i in range(num_batch):
    #             batch = np.concatenate([shuffled_batches[i] for shuffled_batches in shuffled_batches_sets])
    #             new_batches.append(batch)
            
    #         logger.info(f"multi_corpus_dataset sample {len(new_batches)} batches, took {time.time() - start}s")
    #         # for i, batch in enumerate(new_batches[::-1000]): print(f"{i*200}th batch: len {len(batch)}: [{dataset.num_tokens(batch[0])},...,{dataset.num_tokens(batch[-1])}]")
    #         return new_batches
        
    def get_batch_sampler(
        self,
        indices,
        num_shards,
        seed,
        max_tokens=None,
        max_sentences=None,
        required_batch_size_multiple=1,
        split_modality_batch=False,
    ):

        def batch_sampler(dataset, epoch):
            start = time.time()
            batches_dict = dataset.batch_by_size(
                indices,
                max_tokens=max_tokens,
                max_sentences=max_sentences,
                required_batch_size_multiple=required_batch_size_multiple,
            )
            new_batches = []

            ### shuffle inner group size, split into speech/text batches
            speech_batches, text_batches = [], []
            for name, batches in batches_dict.items():
                batches = inner_bucket_shuffle(batches, seed+epoch, num_shards*10)
                batches = batches[: (len(batches) // num_shards) * num_shards]
                if name.startswith("speech"):
                    speech_batches += batches
                else:
                    text_batches += batches
            if len(speech_batches) == 0:
                logger.warning(f"Sample 0 speech batch, please ensure that no speech data loaded.")
            if len(text_batches) == 0:
                logger.warning(f"Sample 0 text batch, please ensure that no text data loaded.")
            
            ### shuffle groups
            if split_modality_batch or len(speech_batches) == 0 or len(text_batches) == 0:
                if split_modality_batch and len(speech_batches) < len(text_batches):
                    text_batches = shuffle_buckets(text_batches, seed=seed+epoch, inner_shuf=False)
                    text_batches = text_batches[0: len(speech_batches)]
                new_batches = speech_batches + text_batches
                new_batches = shuffle_buckets(new_batches, seed=seed+epoch, inner_shuf=False)
            else:
                speech_batches = shuffle_buckets(speech_batches, seed=seed+epoch, inner_shuf=False)
                text_batches = shuffle_buckets(text_batches, seed=seed+epoch, inner_shuf=False)
                modality_batche_sets = [speech_batches, text_batches]
                num_batch = min(len(batches) for batches in modality_batche_sets)
                for i in range(num_batch):
                    batch = np.concatenate([modality_batches[i] for modality_batches in modality_batche_sets])
                    new_batches.append(batch)

            logger.info(f"multi_corpus_dataset sample {len(new_batches)} batches, took {time.time() - start}s")
            return new_batches
        
        def inner_bucket_shuffle(batches, seed, bucket_size=10, thr=0):
            """we assert batches is sorted form long to short.
                shuffle samples in a buctet(e.g. 10 batches).
                batches: a list of numpy array"""
            num_batch = len(batches)
            new_batches = []
            num_buckets = len(batches) // bucket_size
            i = 0
            while i < num_batch:
                if (i < bucket_size * thr or 
                    i >= bucket_size * (num_buckets - thr)
                ):
                    new_batches.append(batches[i])
                    i += 1
                else:
                    group = np.concatenate(batches[i: i+bucket_size])
                    with data_utils.numpy_seed(seed):
                        np.random.shuffle(group)
                    new_batches += np.array_split(group, bucket_size)
                    i += bucket_size
            assert all([len(batch) > 0 for batch in new_batches])
            return new_batches
        
        def shuffle_buckets(batches, seed, inner_shuf=True):
            if inner_shuf:
                batches = inner_bucket_shuffle(batches, seed, num_shards*10)
            batches = [batches[i: i + num_shards] for i in range(0, len(batches)-num_shards+1, num_shards)]
            assert len(batches[-1]) == num_shards
            new_batches = []
            with data_utils.numpy_seed(seed):
                np.random.shuffle(batches)
                for group in batches:
                    new_batches += group
            return new_batches
        
        return batch_sampler

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging

import numpy as np
from fairseq.data import BaseWrapperDataset, plasma_utils


logger = logging.getLogger(__name__)


class ShrinkingDataset(BaseWrapperDataset):
    """Linearly shrinking samples from a given dataset at each epoch.

    Sampling is done with or without replacement, depending on the "replace"
    parameter.


    Args:
        dataset (~torch.utils.data.Dataset): dataset on which to sample.
        start_epoch (int): epoch starting to shink, default 0.
        end_epoch (int): epoch stopping to shink, default 10.
        start_ratio (float): start with that shinking ratio, defaut 1.0.
        end_ratio (float): end with taht shinking ratio, default 0.1.
        epoch (int): starting epoch number (default: 1).
    """

    def __init__(
        self,
        dataset,
        start_epoch=0,
        end_epoch=10,
        start_ratio=1.0,
        end_ratio=0.1,
        shuffle=True,
        epoch=1,
    ):
        super().__init__(dataset)

        self.start_epoch = start_epoch
        self.end_epoch = end_epoch
        self.start_ratio = start_ratio
        self.end_ratio = end_ratio
        self.shuffle = shuffle

        self._cur_epoch = None
        self._cur_size = None

        self.set_epoch(epoch)

    def __getitem__(self, index):
        return self.dataset[index]

    def __len__(self):
        return self._cur_size

    @property
    def sizes(self):
        if isinstance(self.dataset.sizes, list):
            return [s[:self._cur_size] for s in self.dataset.sizes]
        return self.dataset.sizes[:self._cur_size]

    def num_tokens(self, index):
        return self.dataset.num_tokens(index)

    def size(self, index):
        return self.dataset.size(index)

    @property
    def sizes(self):
        return self.dataset.sizes[:self._cur_size]
    
    def ordered_indices(self):
        """
        Returns indices sorted by length. So less padding is needed.
        """
        if isinstance(self.sizes, np.ndarray) and len(self.sizes.shape) > 1:
            # special handling for concatenating lang_pair_datasets
            if self.shuffle:
                indices = np.random.permutation(len(self)).astype(np.int64)
            else:
                indices = np.arange(len(self), dtype=np.int64)
            sizes = self.sizes
            tgt_sizes = (
                sizes[:, 1] if len(sizes.shape) > 0 and sizes.shape[1] > 1 else None
            )
            src_sizes = (
                sizes[:, 0] if len(sizes.shape) > 0 and sizes.shape[1] > 1 else sizes
            )
            # sort by target length, then source length
            if tgt_sizes is not None:
                indices = indices[np.argsort(tgt_sizes[indices], kind="mergesort")]
            return indices[np.argsort(src_sizes[indices], kind="mergesort")]
        else:
            return np.argsort(self.sizes)

    @property
    def can_reuse_epoch_itr_across_epochs(self):
        return False

    def set_epoch(self, epoch):
        super().set_epoch(epoch)
        if hasattr(self.dataset, "set_epoch"):
            self.dataset.set_epoch(epoch)
        
        logger.info("ShinkingDataset.set_epoch: {}".format(epoch))
        super().set_epoch(epoch)

        if epoch == self._cur_epoch:
            return

        self._cur_epoch = epoch
        
        if epoch >= self.start_epoch:
            _cur_ratio = self.end_ratio + (self.start_ratio - self.end_ratio) * max(self.end_epoch - epoch, 0) / (self.end_epoch - self.start_epoch)
        else:
            _cur_ratio = self.start_ratio
        self._cur_size = int(_cur_ratio * len(self.dataset))
        logger.info("ShinkingDataset._cur_size: {}".format(self._cur_size))

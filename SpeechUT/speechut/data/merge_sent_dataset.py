# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import torch
import numpy as np

from fairseq.data import FairseqDataset
from fairseq.data import data_utils, Dictionary

logger = logging.getLogger(__name__)

def comput_merge_information(dataset, tokens_per_sample):
    logger.info("Computing merge information ...")
    raw_len = len(dataset)
    raw_sizes = dataset.sizes
    sizes = []
    merges = []
    index = 0
    while index < raw_len - 1:
        merge_size = raw_sizes[index]
        offset = 1
        next_size = raw_sizes[index + offset]
        while merge_size + next_size < tokens_per_sample:
            merge_size += next_size - 1
            offset += 1
            if index + offset >= raw_len:
                break
            next_size = raw_sizes[index + offset]
        sizes.append(merge_size)
        merges.append(np.arange(index, index+offset))
        index += offset
    return np.array(sizes), merges

class MergeForwardDataset(FairseqDataset):
    def __init__(self, dataset, sizes, merges, recompute_size=False):
        super(MergeForwardDataset, self).__init__()
        self.sizes = sizes
        self.merges = merges
        self.dataset = dataset
        self.recompute_size = recompute_size
    
    def __len__(self):
        return len(self.sizes)
    
    def size(self, idx: int):
        """
        Return an example's size as a float or tuple.
        """
        if not self.recompute_size:
            raw_size = self.sizes[idx]
        else:
            raw_size = np.sum(self.dataset.size(i) for i in self.merges[idx])
        if self.dataset.pad_audio:
            return raw_size
        else:
            return min(raw_size, self.dataset.max_sample_size)
    
    def num_tokens(self, index):
        return self.size(index)

    def ordered_indices(self):
        """Return an ordered list of indices. Batches will be constructed based
        on this order."""

        if self.dataset.shuffle:
            if len(self.dataset.chunk_names) > 0:
                raise NotImplementedError
            else:
                order = [np.random.permutation(len(self))]
                order.append(
                    np.minimum(
                        np.array(self.sizes),
                        self.dataset.max_sample_size,
                    )
                )
                return np.lexsort(order)[::-1]
        else:
            return np.arange(len(self))
    
    def set_epoch(self, epoch):
        self.dataset.epoch = epoch

    def batch_by_size(self, indices, max_tokens=None, max_sentences=None, required_batch_size_multiple=1):
        self.max_tokens = max_tokens
        self.max_sentences = max_sentences
        self.required_batch_size_multiple = required_batch_size_multiple
        if isinstance(indices[0], np.ndarray):
            batch_list = []
            for indice in indices:
                batch = super(MergeForwardDataset, self).batch_by_size(indice, max_tokens, max_sentences, required_batch_size_multiple)
                batch_list.append(batch)
            return batch_list
        else:
            return super(MergeForwardDataset, self).batch_by_size(indices, max_tokens, max_sentences, required_batch_size_multiple)

    def __getitem__(self, index):
        wavs = []
        labels = [[] for _ in range(self.dataset.num_labels)]
        for idx in self.merges[index]:
            wav = self.dataset.get_audio(idx)   # (Ts, )
            label_list = self.dataset.get_labels(idx)  # [(Ti, )]
            wavs.append(wav)
            for i in range(self.dataset.num_labels):
                labels[i].append(label_list[i])
        wav = torch.cat(wavs, dim=0)
        for i in range(self.dataset.num_labels):
            labels[i] = torch.cat(labels[i])
        
        dur_wav = wav.size(0) / self.dataset.sample_rate
        dur_label = labels[0].size(0) / self.dataset.label_rates[0]
        if abs(dur_wav - dur_label) > 0.1:
            logger.warning(f"audio and label duration differ too much: (|{dur_wav} - {dur_label}| > 0.1) in sample index {self.merges[index]} wirh sizes {self.sizes[index]}")
        
        return {"id": index, "source": wav, "label_list": labels}

    def collater(self, samples):
        return self.dataset.collater(samples)

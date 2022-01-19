# coding=utf-8
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""AFQMC dataset."""

import json

from megatron import print_rank_0
from tasks.data_utils import clean_text
from .data import GLUEAbstractDataset


class AFQMCDataset(GLUEAbstractDataset):

    def __init__(self, name, datapaths, tokenizer, max_seq_length,
                 test_label=0):
        self.test_label = test_label
        super().__init__('AFQMC', name, datapaths,
                         tokenizer, max_seq_length)

    def process_samples_from_single_path(self, filename):
        """"Implement abstract method."""
        print_rank_0(' > Processing {} ...'.format(filename))

        samples = []
        total = 0
        with open(filename, 'r') as f:
            for line in f:
                # {"sentence1": "花呗咋扫不上", "sentence2": "花呗扫一扫不能付钱", "label": "0"}
                sample = json.loads(line)
                uid = total
                text_a = sample["sentence1"]
                text_b = sample["sentence2"]
                label = int(sample["label"])

                assert ((label == 0) or (label == 1))
                assert uid >= 0

                sample = {'uid': uid,
                          'text_a': text_a,
                          'text_b': text_b,
                          'label': label}
                total += 1
                samples.append(sample)

                if total % 10000 == 0:
                    print_rank_0('  > processed {} so far ...'.format(total))

        print_rank_0(' >> processed {} samples.'.format(len(samples)))
        return samples

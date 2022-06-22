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

"""OCNLI dataset."""

import json

from megatron import print_rank_0
from .data import CLUEAbstractDataset

LABELS = { "entailment": 0,
           "neutral": 1,
           "contradiction": 2 }

class OCNLIDataset(CLUEAbstractDataset):

    def __init__(self, name, datapaths, tokenizer, max_seq_length,
                 test_label=0):
        self.test_label = test_label
        super().__init__('OCNLI', name, datapaths,
                         tokenizer, max_seq_length)

    def process_samples_from_single_path(self, filename):
        """"Implement abstract method."""
        print_rank_0(' > Processing {} ...'.format(filename))

        samples = []
        total = 0
        with open(filename, 'r', encoding='utf-8-sig') as f:
            for line in f:
                """Data example:
                  {"level":"easy",
                   "sentence1":"要狠抓造林质量不放松",
                   "sentence2":"种的树越多越好,至于成活率和质量并不需要考虑",
                   "label":"contradiction",
                   "label0":"contradiction",
                   "label1":"contradiction",
                   "label2":"contradiction",
                   "label3":"contradiction",
                   "label4":"contradiction",
                   "genre":"news","prem_id":"news_1355","id":12}
                """
                sample = json.loads(line)
                label_desc = sample["label"]
                if not label_desc in LABELS:   # some label may be '-'
                  continue

                text_a = sample["sentence1"]
                text_b = sample["sentence2"]
                label = LABELS[label_desc]

                uid = sample["id"]
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

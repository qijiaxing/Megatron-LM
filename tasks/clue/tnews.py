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

"""TNEWS dataset."""

import json

from megatron import print_rank_0
from tasks.data_utils import clean_text
from .data import GLUEAbstractDataset


# Label info in ~/work/CLUE/baselines/CLUEdataset/tnews/labels.json
LABELS = {
"news_story": 0,
"news_culture": 1,
"news_entertainment": 2,
"news_sports": 3, 
"news_finance": 4, 
"news_house": 5,
"news_car": 6,
"news_edu": 7,
"news_tech": 8,
"news_military": 9,
"news_travel": 10,
"news_world": 11,
"news_stock": 12,
"news_agriculture": 13,
"news_game": 14 }


class TNEWSDataset(GLUEAbstractDataset):

    def __init__(self, name, datapaths, tokenizer, max_seq_length,
                 test_label=0):
        self.test_label = test_label
        super().__init__('TNEWS', name, datapaths,
                         tokenizer, max_seq_length)

    def process_samples_from_single_path(self, filename):
        """"Implement abstract method."""
        print_rank_0(' > Processing {} ...'.format(filename))

        samples = []
        total = 0
        first = True
        is_test = False
        with open(filename, 'r') as f:
            for line in f:
                # {"label": "102", "label_desc": "news_entertainment", "sentence": "江疏影甜甜圈自拍，迷之角度竟这么好看，美吸引一切事物", "keywords": "江疏影,美少女,经纪人,甜甜圈"}
                sample = json.loads(line)
                uid = total
                text_a = sample["sentence"]
                text_b = None
                label_desc = sample["label_desc"]
                assert label_desc in LABELS
                label = LABELS[label_desc]

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

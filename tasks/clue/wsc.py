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

"""WSC dataset."""

import json

from megatron import print_rank_0
from .data import CLUEAbstractDataset


class WSCDataset(CLUEAbstractDataset):

    def __init__(self, name, datapaths, tokenizer, max_seq_length,
                 test_label=0):
        self.test_label = test_label
        super().__init__('WSC', name, datapaths,
                         tokenizer, max_seq_length)

    def process_samples_from_single_path(self, filename):
        """"Implement abstract method."""
        print_rank_0(' > Processing {} ...'.format(filename))

        samples = []
        total = 0
        with open(filename, 'r', encoding='utf-8-sig') as f:
            for line in f:
                ''' Example:
                {"target": {"span2_index": 47,
                            "span1_index": 26,
                            "span1_text": "洋女生",
                            "span2_text": "他们" },
                  "idx": 12,
                  "label": "false",
                  "text": "这种年龄的男青年，又刚刚有了一点文化，"}
                '''
                sample = json.loads(line)

                text_a = sample['text']
                text_a_list = list(text_a)
                target = sample['target']
                query = target['span1_text']
                query_idx = target['span1_index']
                pronoun = target['span2_text']
                pronoun_idx = target['span2_index']
                label = 1 if sample['label'] == 'true' else 0

                assert text_a[pronoun_idx: (pronoun_idx + len(pronoun))
                              ] == pronoun, "pronoun: {}".format(pronoun)
                assert text_a[query_idx: (query_idx + len(query))] == query, \
                        "query: {}".format(query)

                # Replace Q by _Q_ and P by [P]
                if pronoun_idx > query_idx:
                  text_a_list.insert(query_idx, "_")
                  text_a_list.insert(query_idx + len(query) + 1, "_")
                  text_a_list.insert(pronoun_idx + 2, "[")
                  text_a_list.insert(pronoun_idx + len(pronoun) + 2 + 1, "]")
                else:
                  text_a_list.insert(pronoun_idx, "[")
                  text_a_list.insert(pronoun_idx + len(pronoun) + 1, "]")
                  text_a_list.insert(query_idx + 2, "_")
                  text_a_list.insert(query_idx + len(query) + 2 + 1, "_")

                text_a = "".join(text_a_list)
                text_b = None
                uid = total

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

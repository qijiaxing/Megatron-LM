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

"""IFLYTEK dataset."""

import json

from megatron import print_rank_0
from tasks.data_utils import clean_text
from .data import GLUEAbstractDataset


# Label info in ~/work/CLUE/baselines/CLUEdataset/iflytek/labels.json

class IFLYTEKDataset(GLUEAbstractDataset):

    def __init__(self, name, datapaths, tokenizer, max_seq_length,
                 test_label=0):
        self.test_label = test_label
        super().__init__('IFLYTEK', name, datapaths,
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
                # {"label": "66", "label_des": "酒店", "sentence": "去哪儿商家，是去哪儿网为合作的酒店商户开发的专属移动应用，帮助商家摆脱职守电脑，避免漏单的发生，随时查看与去哪儿网的直销订单。同时为商户打通了团购业务，一个APP所有业务轻松搞定去哪儿商家QQ交流群428822369订单管理1、新单同步推送。2、随时管理订单。审核管理1、待审核订单同步推送。2、随时随地方便快捷的审核订单。房态管理1、房态同步显示。2、支持开关房型与数量调整。房价管理1、可以随时修改房价。财务管理1、可以随时查看自己的账单。团购管理1、团购产品验券与结算。2、支持扫码验证，免除手动输入的麻烦。点评管理1、差评及时推送，第一时间了解。2、随时随地管理酒店的点评。更新内容1.解决了图片管理中拍照后无法提交的问题。2.解决了图片管理中查看图片无法向右滑动的问题。"}
                sample = json.loads(line)
                uid = total
                text_a = sample["sentence"]
                text_b = None
                label = int(sample["label"])

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

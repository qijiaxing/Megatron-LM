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

"""GLUE finetuning/evaluation."""

from megatron import get_args
from megatron import print_rank_0
from megatron import get_tokenizer
from megatron import mpu
from megatron.model.classification import Classification
from tasks.eval_utils import accuracy_func_provider
from tasks.finetune_utils import finetune


def clue_classification(num_classes, Dataset,
                        name_from_datapath_func):

    def train_valid_datasets_provider():
        """Build train and validation dataset."""
        args = get_args()
        tokenizer = get_tokenizer()

        # JQ: for tnews, the three files are train dev test
        train_dataset = Dataset('train', args.train_data,
                                tokenizer, args.seq_length)
        valid_dataset = Dataset('dev', args.valid_data,
                                tokenizer, args.seq_length)

        return train_dataset, valid_dataset

    def model_provider(pre_process=True, post_process=True):
        """Build the model."""
        args = get_args()

        print_rank_0('building classification model for {} ...'.format(
            args.task))
        # TODO JQ: How model is modified for classification?
        model = Classification(num_classes=num_classes, num_tokentypes=2,
                               pre_process=pre_process, post_process=post_process)

        return model

    # TODO JQ: What is this?
    def metrics_func_provider():
        """Privde metrics callback function."""
        def single_dataset_provider(datapath):
            args = get_args()
            tokenizer = get_tokenizer()

            name = name_from_datapath_func(datapath)
            return Dataset(name, [datapath], tokenizer, args.seq_length)
        return accuracy_func_provider(single_dataset_provider)

    """Finetune/evaluate."""
    finetune(train_valid_datasets_provider, model_provider,
             end_of_epoch_callback_provider=metrics_func_provider)


def main():
    args = get_args()

    args.task = args.task.upper()
    if args.task == 'TNEWS':

        num_classes = 15
        from tasks.clue.tnews import TNEWSDataset as Dataset

        def name_from_datapath(datapath):
            return datapath.split('tnews')[-1].strip(
                '.json').strip('/').replace('_', '-')

    elif args.task == 'AFQMC':

        num_classes = 2
        from tasks.clue.afqmc import AFQMCDataset as Dataset

        def name_from_datapath(datapath):
            return datapath.split('afqmc')[-1].strip(
                '.json').strip('/').replace('_', '-')

    elif args.task == 'OCNLI':

        num_classes = 3
        from tasks.clue.ocnli import OCNLIDataset as Dataset

        def name_from_datapath(datapath):
            return datapath.split('ocnli')[-1].strip(
                '.json').strip('/').replace('_', '-')

    else:
        raise NotImplementedError('CLUE task {} is not implemented.'.format(
            args.task))

    clue_classification(num_classes, Dataset, name_from_datapath)

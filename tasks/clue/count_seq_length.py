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
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             os.path.pardir,
                                             os.path.pardir)))

from megatron import get_args
from megatron.initialize import initialize_megatron
from megatron import print_rank_0
from megatron import get_tokenizer

def get_tasks_args(parser):
    """Provide extra arguments required for tasks."""
    group = parser.add_argument_group(title='tasks')

    group.add_argument('--task', type=str, required=True,
                       help='Task name.')
    group.add_argument('--train-data', nargs='+', default=None,
                       help='Whitespace separated paths or corpora names '
                       'for training.')
    group.add_argument('--valid-data', nargs='*', default=None,
                       help='path(s) to the validation data.')

    return parser

def clue_classification(Dataset, task):

  args = get_args()
  tokenizer = get_tokenizer()

  # JQ: for tnews, the three files are train dev test

  print("Count sequence length for task: {}".format(task), flush=True)

  for name in ("train", "dev"):
    if name == "train":
      ds = Dataset('train', args.train_data, tokenizer, args.seq_length)
    if name == "dev":
      ds = Dataset('dev', args.valid_data, tokenizer, args.seq_length)

    count = [0] * (1024 + 1)
    for sample in ds.samples:
      text_a = sample["text_a"]
      text_a_ids = tokenizer.tokenize(text_a)

      text_b = sample["text_b"]
      text_b_ids = None
      if text_b is not None:
          text_b_ids = tokenizer.tokenize(text_b)
      total_length = len(text_a_ids) + (len(text_b_ids) if text_b else 0)
      if total_length >= 1024:
        count[-1] += 1
      else:
        count[total_length] += 1

    start = 0
    for end in (64, 128, 256, 512, 1024):
      end = end - 2
      print("Count seq length {:4d} - {:4d}: {}".format(start, end, sum(count[start:end])))
      start = end
    print("Extra long seq: {}".format(count[-1]))
   #print("Text a: {}".format(text_a))
   #print("Text b: {}".format(text_b))
   #print("Total length of a and b: {}".format(total_length))


if __name__ == '__main__':
    initialize_megatron(extra_args_provider=get_tasks_args)
    args = get_args()

    args.task = args.task.upper()
    if args.task == 'TNEWS':

        num_classes = 15
        from tasks.clue.tnews import TNEWSDataset as Dataset

    elif args.task == 'AFQMC':

        num_classes = 2
        from tasks.clue.afqmc import AFQMCDataset as Dataset

    elif args.task == 'OCNLI':

        num_classes = 3
        from tasks.clue.ocnli import OCNLIDataset as Dataset

    else:
        raise NotImplementedError('CLUE task {} is not implemented.'.format(
            args.task))

    clue_classification(Dataset, args.task)

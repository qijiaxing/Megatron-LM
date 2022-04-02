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

"""Processing data for pretraining."""

import itertools
import json
import multiprocessing
import os
import sys
import glob
import re
import random
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             os.path.pardir,
                                             os.path.pardir)))
import time
import torch
from megatron.tokenizer import build_tokenizer
from megatron.tokenizer.zh_tools import to_zh_cn, zng, has_chinese
from megatron.data import indexed_dataset

from arguments import get_args

# TODO use logging to replace print

def print_if(target, token_ids):
  decoded = Encoder.tokenizer.decode(token_ids)
  decoded_sent = ''.join(decoded)
  wanted = True
  wanted = re.search(target, decoded_sent) is not None
  if wanted:  #& doc_is_good:
   #print(f"Sentence: {sentence}")
    print(f"Decode: {decoded_sent}")

class IdentitySplitter(object):
    def tokenize(self, *text):
        return text

class Encoder(object):
    def __init__(self, args):
        self.args = args
        self.bad_titles = [ '列表', '中華民國總統選舉', '\d+月\d+日']
        self.bad_text = '（\W*）' + '|formula_\d+' + '|codice_\d+' \
          + '|&lt;.*&gt;'

    def initializer(self):
        # Use Encoder class as a container for global data
        Encoder.tokenizer = build_tokenizer(self.args)
        if self.args.split_sentences:
            Encoder.splitter = zng
        else:
            Encoder.splitter = IdentitySplitter()

    def skip_by_title(self, title):
        for bad in self.bad_titles:
          if re.search(bad, title):
           #print("Skip title: {}".format(title))
            return True
        return False

    def process_text(self, text):
        text = re.sub(self.bad_text, '', text)
        text = to_zh_cn(text)
        return text

    def encode(self, line):
        data = json.loads(line)
        num_tokens = 0  # tokens processed in this doc
        ids = {}
        for key in self.args.json_keys:
            title = data["title"]
            text = data[key] #;print(f"Text: {text}")

            doc_is_good = not self.skip_by_title(title)

            text  = self.process_text(text)
            if not has_chinese(text): doc_is_good = False

            doc_size = 0   # number of tokens in the doc
            doc_ids = []   # a list of list
            if doc_is_good:
              for sentence in Encoder.splitter(text):
                  sentence_ids = Encoder.tokenizer.tokenize(sentence)
                  s_length = len(sentence_ids)

                  # JQ: exclude doc which has a long sentence
                  if s_length >= self.args.max_sent_length:
                      doc_is_good = False  # ;print("Long Sent: {}".format(sentence))

                  # Add sentence into doc
                  if s_length > 0:
                      doc_size += s_length
                      doc_ids.append(sentence_ids)

                  if self.args.debug > 0:
                    print_if('UNK', sentence_ids)

              # append EOD to the end of doc
              if len(doc_ids) > 0 and self.args.append_eod:
                  doc_ids[-1].append(Encoder.tokenizer.eod)

            doc_is_good = (doc_size > self.args.min_doc_length) & doc_is_good
            if doc_is_good:
              ids[key] = doc_ids
              num_tokens += doc_size
            else:
             continue
             #print("Skip short doc [{}]: {}".format(title, text))

        return ids, num_tokens


def main():
    args = get_args()
    random.seed(args.seed)

    startup_start = time.time()

    encoder = Encoder(args)
    tokenizer = build_tokenizer(args)
    pool = multiprocessing.Pool(args.workers, initializer=encoder.initializer)

    lines = list()
    for filename in glob.glob(args.input):
      print("Opening file: ", filename, flush=True)
      with open(filename, 'r', encoding='utf-8') as fin:
        lines.extend(fin.readlines())

    if args.debug > 0:
      print("Select {} samples for debug!".format(args.debug))
      first = random.randrange(0, len(all_samples))
      selected = lines[first:first+args.debug]
      encoded_docs = pool.map(encoder.encode, selected, 4)
      exit()
    else:
      encoded_docs = pool.imap(encoder.encode, lines, 128)
     #encoded_docs = pool.map(encoder.encode, lines, 128)
     #exit()

    level = "document"
    if args.split_sentences:
        level = "sentence"

    print(f"Vocab size: {tokenizer.vocab_size}")
    print(f"Output prefix: {args.output_prefix}")
    output_bin_files = {}
    output_idx_files = {}
    output_info_files = {}
    builders = {}
    for key in args.json_keys:
        output_bin_files[key] = "{}_{}_{}.bin".format(args.output_prefix, key, level)
        output_idx_files[key] = "{}_{}_{}.idx".format(args.output_prefix, key, level)
        output_info_files[key]= "{}_{}_{}.info".format(args.output_prefix, key, level)
        # Create a class MMapIndexedDatasetBuilder(object):
        builders[key] = indexed_dataset.make_builder(output_bin_files[key],
                                               impl=args.dataset_impl,
                                               vocab_size=tokenizer.vocab_size)

    startup_end = time.time()
    proc_start = time.time()
    total_tokens = 0
    total_docs = 0
    print("Time to startup: {:.2f}".format(startup_end - startup_start))

    for i, (doc, doc_size) in enumerate(encoded_docs, start=1):
        total_tokens += doc_size
        for key, sentences in doc.items():
            if len(sentences) == 0:
                continue
            for sentence in sentences:
                # Convert tensor to numpy bytes, save to file
                builders[key].add_item(torch.IntTensor(sentence))
            # Add a index mark for doc
            builders[key].end_document()
            total_docs += 1

        if total_docs % args.log_interval == 0:
            current = time.time()
            elapsed = current - proc_start
            mbs = total_tokens / elapsed /1024/1024
            print(f"Processed {i} documents",
                  f"({i/elapsed:.2f} docs/s, tokens: {mbs:.2f} Million/s).",
                  file=sys.stderr)

    for key in args.json_keys:
        builders[key].finalize(output_idx_files[key])
        print("Save output to {}\nand {}".format(
            output_bin_files[key], output_idx_files[key]))
        # JQ: Save info files
        with open(output_info_files[key], 'w') as fout:
          fout.write("Input file: {}\n".format(args.input))
          fout.write("Vocab file: {}\n".format(args.vocab_file))
          fout.write("Total number of docs: {}\n".format(total_docs))
          fout.write("Total number of tokens: {}\n".format(total_tokens))


if __name__ == '__main__':
    main()

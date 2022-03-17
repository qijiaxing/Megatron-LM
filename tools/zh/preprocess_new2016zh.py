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

import argparse
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
from megatron.data import indexed_dataset

from utils import zng, has_chinese, stringQ2B, translate_punct, to_zh_cn

LONG_SEQ_LENGTH = 128

class IdentitySplitter(object):
    def tokenize(self, *text):
        return text

class Encoder(object):
    def __init__(self, args):
        self.args = args

    def initializer(self):
        # Use Encoder class as a container for global data
        Encoder.tokenizer = build_tokenizer(self.args)
        if self.args.split_sentences:
            Encoder.splitter = zng
        else:
            Encoder.splitter = IdentitySplitter()

    def encode(self, data):
        num_tokens = 0  # tokens processed in this doc
        ids = {}
        for key in self.args.json_keys:
            ids[key] = list()
            text = data[key]
           #print(f"Text: {text}")

            # JQ: Skip non-chinese text
            if not has_chinese(text):
             #print("Non zh: {}".format(text))
              continue

            # JQ: Replace "---" or "===" by a single "-"
            text = re.sub("[-=]{3,}", "-", text)
            text = translate_punct(text)
            text = to_zh_cn(text)

            doc_ids = []   # a list of list
            doc_size = 0   # number of tokens in the doc
            doc_is_good = True
            for sentence in Encoder.splitter(text):
                sentence = stringQ2B(sentence)
                sentence_ids = Encoder.tokenizer.tokenize(sentence)
                s_length = len(sentence_ids)

                # JQ: find LONG sentence
                if s_length >= LONG_SEQ_LENGTH:
                    doc_is_good = False
                   #print("Long seq: {}".format(sentence))

                if s_length > 0:
                    doc_size += len(sentence_ids)
                    doc_ids.append(sentence_ids)

                if self.args.debug > 0:
                  decoded = Encoder.tokenizer.decode(sentence_ids)
                  decoded_sent = ''.join(decoded)
                  if re.search('UNK', decoded_sent):
                    print(f"Sentence: {sentence}")
                    print(f"GOOD: {doc_is_good}, Decode: {decoded_sent}")

            # append EOD to the end of doc
            if len(doc_ids) > 0 and self.args.append_eod:
                doc_ids[-1].append(Encoder.tokenizer.eod)

            if doc_is_good:
              ids[key] = doc_ids
              num_tokens += doc_size

        return ids, num_tokens


def get_args():
    parser = argparse.ArgumentParser()
    group = parser.add_argument_group(title='input data')
    group.add_argument('--input', type=str, required=True,
                       help='Path to input JSON')
    group.add_argument('--json-keys', nargs='+', default=['text'],
                       help='space separate listed of keys to extract from json')
    group.add_argument('--split-sentences', action='store_true',
                       help='Split documents into sentences.')
    group.add_argument('--keep-newlines', action='store_true',
                       help='Keep newlines between sentences when splitting.')

    # JQ: select file format
    group.add_argument('--json-by-line', action='store_true',
                       help='Each line is a json object')
    # JQ: debug output
    group.add_argument('--debug', type=int, default=0,
                       help='Each line is a json object')
    group.add_argument('--seed', type=int, default=76,
                       help='Random seed')

    group.add_argument('--min-doc-length', type=int, default=127,
                       help='Minimum doc length')

    group = parser.add_argument_group(title='tokenizer')
    group.add_argument('--tokenizer-type', type=str, required=True,
                       choices=['BertWordPieceLowerCase','BertWordPieceCase',
                                'GPT2BPETokenizer'],
                       help='What type of tokenizer to use.')
    group.add_argument('--vocab-file', type=str, default=None,
                       help='Path to the vocab file')
    group.add_argument('--merge-file', type=str, default=None,
                       help='Path to the BPE merge file (if necessary).')
    group.add_argument('--append-eod', action='store_true',
                       help='Append an <eod> token to the end of a document.')


    group = parser.add_argument_group(title='output data')
    group.add_argument('--output-prefix', type=str, required=True,
                       help='Path to binary output file without suffix')
    group.add_argument('--dataset-impl', type=str, default='mmap',
                       choices=['lazy', 'cached', 'mmap'])

    group = parser.add_argument_group(title='runtime')
    group.add_argument('--workers', type=int, default=1,
                       help='Number of worker processes to launch')
    group.add_argument('--log-interval', type=int, default=10000,
                       help='Interval between progress updates')
    args = parser.parse_args()
    args.keep_empty = False

    if args.tokenizer_type.lower().startswith('bert'):
        if not args.split_sentences:
            print("Bert tokenizer detected, are you sure you don't want to split sentences?")

    # some default/dummy values for the tokenizer
    args.rank = 0
    args.make_vocab_size_divisible_by = 128
    args.tensor_model_parallel_size = 1
    args.vocab_extra_ids = 0

    return args

def main():
    args = get_args()
    random.seed(args.seed)

    startup_start = time.time()

    encoder = Encoder(args)
    tokenizer = build_tokenizer(args)
    pool = multiprocessing.Pool(args.workers, initializer=encoder.initializer)

    all_samples = []
    for filename in glob.glob(args.input):
      print("Opening file: ", filename, flush=True)
      with open(filename, 'r', encoding='utf-8') as fin:
        samples = json.load(fin)
      all_samples.extend(samples)
    if args.debug > 0:
      print("Select {} samples for debug!".format(args.debug))
      first = random.randrange(0, len(all_samples))
      selected = all_samples[first:first+args.debug]
      encoded_docs = pool.map(encoder.encode, selected, 4)
      exit()
    else:
      encoded_docs = pool.imap(encoder.encode, all_samples, 128)

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
        output_bin_files[key] = "{}_{}_{}.bin".format(args.output_prefix,
                                                      key, level)
        output_idx_files[key] = "{}_{}_{}.idx".format(args.output_prefix,
                                                      key, level)
        output_info_files[key]= "{}_{}_{}.info".format(args.output_prefix,
                                                      key, level)
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
        # doc is a dict: ["text"] = a list of list
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
        print("Save output to {} and {}".format(
            output_bin_files[key], output_idx_files[key]))
        # JQ: Save info files
        with open(output_info_files[key], 'w') as fout:
          fout.write("Input file: {}\n".format(args.input))
          fout.write("Vocab file: {}\n".format(args.vocab_file))
          fout.write("Total number of docs: {}\n".format(total_docs))
          fout.write("Total number of tokens: {}\n".format(total_tokens))


if __name__ == '__main__':
    main()

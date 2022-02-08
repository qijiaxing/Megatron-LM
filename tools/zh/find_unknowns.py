"""Analyze the corpus to find out which tokens has shown or not.
   Results are saved to files:
      token_shown.txt
      token_absent.txt

   Example:
     INPUT=/raid/data/wiki_zh/wiki_zh_json/wiki_zh_2019_full.json
     VOCAB=vocab/bert-chinese-expanded-vocab.txt
     python tools/vocab_stat.py --input ${INPUT} --vocab ${VOCAB}
"""

import argparse
import json
import multiprocessing
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             os.path.pardir)))
import time
import glob

from tools.preprocess_data_zh import zng
from megatron.tokenizer import build_tokenizer
from collections import Counter


class IdentitySplitter(object):
    def tokenize(self, *text):
        return text

class Encoder(object):
    def __init__(self, args):
        self.args = args
        Encoder.tokenizer = build_tokenizer(self.args)

    def initializer(self):
        # Use Encoder class as a container for global data
        # JQ: Use Chinese splitter, by default
        Encoder.splitter = zng

    def encode(self, json_line, book):
        """
          Encode one json line, i.e. one doc
        Returns:
          ids: a dict, key is "text", value is a list of token(ID) lists
        """
        data = json.loads(json_line)
       #book = Counter()
        for key in self.args.json_keys:
            text = data[key]
            for sentence in Encoder.splitter(text):
                sentence = sentence.replace("<br>", '')  # remove <br>
                sentence = sentence.replace("（）", '')  # remove ()
                tokens = Encoder.tokenizer.tokenize(sentence)

                """
                tID = 8166
                if tID in tokens:
                  print(f"Sentence: {sentence}")
                  _decoded = Encoder.tokenizer.decode(tokens); print(f"Decode  : {_decoded.replace(' ', '')}")
                  exit()
                """

                if len(tokens) > 0:
                    book.update(tokens)

        return book

def get_args():
    parser = argparse.ArgumentParser()
    group = parser.add_argument_group(title='input data')
    group.add_argument('--input', type=str, required=True,
                       help='Path to input JSON')
    group.add_argument('--json-keys', nargs='+', default=['text'], help='space separate listed of keys to extract from json')
#   group.add_argument('--split-sentences', action='store_true', help='Split documents into sentences.')
#   group.add_argument('--keep-newlines', action='store_true', help='Keep newlines between sentences when splitting.')

    group = parser.add_argument_group(title='tokenizer')
    group.add_argument('--tokenizer-type', type=str, # required=True,
                       default='BertWordPieceLowerCase',
                       choices=['BertWordPieceLowerCase','BertWordPieceCase',
                                'GPT2BPETokenizer'],
                       help='What type of tokenizer to use.')
    group.add_argument('--vocab-file', type=str,
                       default="vocab/bert-chinese-expanded-vocab.txt",
                       help='Path to the vocab file')
    group.add_argument('--merge-file', type=str, default=None,
                       help='Path to the BPE merge file (if necessary).')
    group.add_argument('--append-eod', action='store_true',
                       help='Append an <eod> token to the end of a document.')

    """
    group = parser.add_argument_group(title='output data')
    group.add_argument('--output-prefix', type=str, required=True,
                       help='Path to binary output file without suffix')
    group.add_argument('--dataset-impl', type=str, default='mmap',
                       choices=['lazy', 'cached', 'mmap'])
    """

    group = parser.add_argument_group(title='runtime')
    group.add_argument('--workers', type=int, default=16,
                       help='Number of worker processes to launch')
    group.add_argument('--log-interval', type=int, default=100,
                       help='Interval between progress updates')
    args = parser.parse_args()
    args.keep_empty = False

    # some default/dummy values for the tokenizer
    args.rank = 0
    args.make_vocab_size_divisible_by = 128
    args.tensor_model_parallel_size = 1
    args.vocab_extra_ids = 0

    return args


def main():
    args = get_args()
    print("Running find unknowns")
    print("Vocab file: {}".format(args.vocab_file))
    startup_start = time.time()

    encoder = Encoder(args)
    encoder.initializer()

    occured = Counter()
    files = glob.glob(args.input)
    for filename in files:
      print("Opening input file: ", filename)
      start_t = time.time()
      fin = open(filename, 'r', encoding='utf-8')
      for line in fin:
        encoder.encode(line, occured)
      fin.close()
      print("Done processing, time: {}".format(time.time() - start_t))


    tokenizer = encoder.tokenizer
    tokenizer.print_unknowns("unknown_tokens.txt")

    print("Save shown tokens and absent tokens.", flush=True)
    f_shown = open("token_shown.txt", "w")
    f_absent = open("token_absent.txt", "w")
    for token in range(tokenizer.vocab_size):
      token_str = tokenizer.decode((token,))
      if token in occured:
        f_shown.write("ID: {:5d}, Token: {}\n".format(token, token_str))
      else:
        f_absent.write("ID: {:5d}, Token: {}\n".format(token, token_str))
    f_shown.close()
    f_absent.close()
       #print("ID: {}, Token: {}".format(token, tokenizer.decode_token_ids((token,))))


if __name__ == '__main__':
    main()

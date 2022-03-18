import re


def get_ch(filename):
  tokens = list()
  with open(filename, 'r', encoding='utf-8') as fin:
    for line in fin:
      line = line.strip()
      if not line:
        continue
      _code, token = line.split("\t")
      tokens.append(token)
  return tokens

def get_en(filename, max_len=32):
  tokens = list()
  with open(filename, 'r', encoding='utf-8') as fin:
    for line in fin:
      token = line.strip()
      if not token:
        continue

      # exclude entries ending with numbers
      # TODO exclude non-word token, using nltk words?
      if re.match(u'(##)?[a-z]*[a-z]$', token):
          if len(token) < max_len:
            tokens.append(token)
          else:
            print("Drop very long token: [{}]".format(token))

  def length(word):
    return len(word.replace("##", ''))

  tokens.sort(key=length)
  return tokens

def get_tokens(filename):
  tokens = list()
  with open(filename, 'r', encoding='utf-8-sig') as fin:
    for line in fin:
      token = line.strip()
      if not token:
        continue
      tokens.append(token)
  return tokens

# out_file = "en_list.txt"
def save(filename, tokens):
  with open(filename, 'w', encoding='utf-8') as fout:
    for token in tokens:
      fout.write(token + "\n")

def main():
  vocab = list()
  special_tokens = ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]']
  vocab.extend(special_tokens)

  cn_filename = "dict/cn_chars.txt"
  cn_tokens = get_ch(cn_filename)
  vocab.extend(cn_tokens)

  cn_punct_filename = "dict/cn_other_chars.txt"
  cn_punct_tokens = get_tokens(cn_punct_filename)
  vocab.extend(cn_punct_tokens)

  cn_punct_filename = "dict/cn_punct_fullwidth.txt"
  cn_punct_tokens = get_tokens(cn_punct_filename)
  vocab.extend(cn_punct_tokens)

  google_vocab = "/home/jqi/work/megatron/vocab/bert-base-chinese-vocab.txt"
  max_token_length = 10
  en_tokens = get_en(google_vocab, max_token_length)
  vocab.extend(en_tokens)

  ascii_symbol_filename = "dict/ascii_symbols.txt"
  ascii_symbols = get_tokens(ascii_symbol_filename)
  vocab.extend(ascii_symbols)

  greek_filename = 'dict/greek.txt'
  greek_tokens = get_tokens(greek_filename)
  vocab.extend(greek_tokens)

  # Numbers
  years = ['2016', '2017', '2018', '2019', '2020', '2021', '2022']
  num_tokens = list()
  for num in range(101):
    num_tokens.append(str(num))
  for num in years:
    num_tokens.append(str(num))
  for num in range(21):
    num_tokens.append('##' + str(num))
  vocab.extend(num_tokens)

  print("Size of vocab: {}".format(len(vocab)))
  out_file = 'myzh.vocab'
  save(out_file, vocab)
  print("Saved vocab to file: {}".format(out_file))

def test():
  google_vocab = "/home/jqi/work/megatron/vocab/bert-base-chinese-vocab.txt"
  en_tokens = get_en(google_vocab)
  out_file = "en_list.txt"
  save(out_file, en_tokens)

# test()
main()

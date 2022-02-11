#!/bin/bash

IMAGE=pytorch2108
stage=0

# INPUT="/raid/data/wiki_zh/AB/wiki_00"
INPUT=/raid/data/wiki_zh/wiki_zh_json/wiki_zh_2019_full.json
# VOCAB=vocab/bert-chinese-expanded-vocab.txt   # added a few chinese symbols
# VOCAB=vocab/jq/jq-tokens.txt.0.vocab
VOCAB=vocab/jq/jq-tokens.txt.1.vocab

if [ ${stage} -eq 0 ]; then
 #EXE=tools/analyze_vocab.py
  EXE=tools/find_unknowns.py
  docker exec ${IMAGE} bash -c "cd `pwd`; \
  set -f;
  python ${EXE} \
         --input ${INPUT} \
         --vocab ${VOCAB} \
         --workers 1 \
         --tokenizer-type BertWordPieceLowerCase | tee last_run.log"
  exit 0
fi

#!/bin/bash

IMAGE=pytorch2108
stage=0

# INPUT=/raid/data/wiki_zh/AA/wiki_00
# INPUT=data/wiki_zh_10.json
# INPUT=data/wiki_zh_single.json
INPUT=/raid/data/wiki_zh/wiki_zh_json/wiki_zh_2019_full.json
VOCAB=vocab/jq/jq-tokens.txt.2.vocab
DATA_PREFIX=data/wiki_zh/jq_2_vocab/bert_wiki
# DATA_PREFIX=data/wiki_zh/test/bert
WORKERS=16

if [ ${stage} -eq 0 ]; then
  EXE=tools/preprocess_data_zh.py   # For Chinese
# EXE=tools/preprocess_wiki_zh.py   # create a tranditional ch version for each sentence
  docker exec ${IMAGE} bash -c "cd `pwd`; \
  python ${EXE} \
         --input ${INPUT} \
         --output-prefix ${DATA_PREFIX} \
         --vocab ${VOCAB} \
         --dataset-impl mmap \
         --split-sentences \
         --workers ${WORKERS} \
         --tokenizer-type BertWordPieceLowerCase"
  exit 0
fi



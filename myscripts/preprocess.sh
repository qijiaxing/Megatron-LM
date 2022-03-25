#!/bin/bash

IMAGE=pytorch2108
MEGATRON=.

INPUT=/raid/data/nlp-zh/nlp-chinese-corpus/new2016zh/files/new2016zh_data_1.json
# INPUT='/raid/data/nlp-zh/nlp-chinese-corpus/new2016zh/new2016zh_data_*.json'
# INPUT=data/tiny/new2016zh_10.json

# VOCAB=vocab/jq/jq-tokens.txt.2.vocab
VOCAB=vocab/jq.zh.vocab
KEYS=content
DATA_PREFIX=data/debug

SEED=13
WORKERS=1
DEBUG=256

# EXE=tools/zh/preprocess_data_zh.py   # For Chinese
EXE=tools/zh/preprocess_new2016zh.py   # For Chinese
docker exec ${IMAGE} bash -c "cd `pwd`; \
cd ${MEGATRON}; \
python ${EXE} \
       --input "${INPUT}" \
       --output-prefix ${DATA_PREFIX} \
       --vocab ${VOCAB} \
       --json-keys ${KEYS} \
       --debug ${DEBUG} \
       --seed ${SEED} \
       --dataset-impl mmap \
       --split-sentences \
       --workers ${WORKERS} \
       --tokenizer-type BertWordPieceLowerCase"

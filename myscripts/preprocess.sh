#!/bin/bash

IMAGE=pytorch2108
MEGATRON=/home/jqi/work/megatron

# INPUT=/raid/data/nlp-zh/nlp-chinese-corpus/new2016zh/files/new2016zh_data_1.json
INPUT=/raid/data/nlp-zh/nlp-chinese-corpus/new2016zh/files/new2016zh_data_*.json

VOCAB=${MEGATRON}/vocab/jq/jq-tokens.txt.2.vocab
# VOCAB=vocab/jq.zh.vocab
KEYS=content
DATA_PREFIX=${MEGATRON}/data/new2016/jq2-vocab/full_less128

MAX_LEN=128
SEED=13
WORKERS=16
DEBUG=0

# EXE=tools/zh/preprocess_data_zh.py   # For Chinese
EXE=${MEGATRON}/tools/zh/preprocess_new2016zh.py   # For Chinese
docker exec ${IMAGE} bash -c "cd `pwd`; \
cd ${MEGATRON}; \
python ${EXE} \
       --input '${INPUT}' \
       --output-prefix ${DATA_PREFIX} \
       --vocab ${VOCAB} \
       --json-keys ${KEYS} \
       --max-sent-length ${MAX_LEN} \
       --debug ${DEBUG} \
       --seed ${SEED} \
       --dataset-impl mmap \
       --split-sentences \
       --workers ${WORKERS} \
       --tokenizer-type BertWordPieceLowerCase"

#!/bin/bash

TASK=TNEWS
DATADIR=/home/jqi/work/CLUE/baselines/CLUEdataset/tnews
TRAIN_DATA=${DATADIR}/train.json
VALID_DATA=${DATADIR}/dev.json
# PRETRAINED_CHECKPOINT=exp/wiki_zh_b128_s64   # <--- Set Checkpoint!!
PRETRAINED_CHECKPOINT=exp/wiki_zh_st_b256_s128
VOCAB_FILE=vocab/bert-chinese-expanded-vocab.txt
# CHECKPOINT_PATH=exp/wiki_zh_b128_s128_tnews
CHECKPOINT_PATH=${PRETRAINED_CHECKPOINT}_tnews
MAX_SEQ_LEN=128
BATCH=32
EPOCH=3
LOG=wiki_zh_st_b256_s128_finetune_tnews.log

GPU=2
IMAGE=pytorch2108
WORLD_SIZE=1
DISTRIBUTED_ARGS="--nproc_per_node $WORLD_SIZE \
                  --nnodes 1 --node_rank 0 \
                  --master_addr localhost --master_port 600${GPU}"

BERT_ARGS="--num-layers 12 --hidden-size 768 --num-attention-heads 12 \
           --seq-length ${MAX_SEQ_LEN} --max-position-embeddings ${MAX_SEQ_LEN} \
           --lr 2.0e-5 \
           --min-lr 1.0e-6 \
           --weight-decay 1.0e-2 \
           --lr-decay-style linear \
           --lr-warmup-fraction 0.1 \
           --micro-batch-size ${BATCH} \
           --fp16"

OUTPUT_ARGS="--save-interval 50000 \
             --epochs ${EPOCH} \
             --save $CHECKPOINT_PATH \
             --log-interval 10 \
             --eval-interval 50000 \
             --eval-iters 50"

docker exec ${IMAGE} bash -c "cd `pwd`; \
CUDA_VISIBLE_DEVICES=${GPU} \
python -m torch.distributed.launch $DISTRIBUTED_ARGS ./tasks/main.py \
       ${BERT_ARGS} \
       ${OUTPUT_ARGS} \
       --task ${TASK} \
       --seed 1234 \
       --train-data $TRAIN_DATA \
       --valid-data $VALID_DATA \
       --tokenizer-type BertWordPieceLowerCase \
       --vocab-file $VOCAB_FILE \
       --pretrained-checkpoint $PRETRAINED_CHECKPOINT \
       --activations-checkpoint-method uniform | tee ${LOG}"

#!/bin/bash

IMAGE=pytorch2108

# VOCAB=vocab/bert-chinese-expanded-vocab.txt   # added a few chinese symbols
VOCAB=vocab/jq/jq-tokens.txt.2.vocab
DATA_PREFIX=data/wiki_zh/jq_2_vocab/bert_wiki

DATA_PATH=${DATA_PREFIX}_text_sentence
MICRO_BATCH=32
GPUS_PER_NODE=4
GLOBAL_BATCH=$((${MICRO_BATCH}*${GPUS_PER_NODE}))
MAX_SEQ_LEN=64
MAX_POS_EMB=64
ITERS=1000000
SAVE_INTERVAL=50000
TRAIN_VAL_TEST="949,50,1"
CHECKPOINT_PATH=exp/wiki_jq_2_b${GLOBAL_BATCH}_s${MAX_SEQ_LEN}
LOG_PATH=${CHECKPOINT_PATH}/log

# DDP settings:
# GPU=0
# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=7000
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))
DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE \
                  --nnodes $NNODES \
                  --node_rank $NODE_RANK \
                  --master_addr $MASTER_ADDR \
                  --master_port $MASTER_PORT"

BERT_ARGS="--num-layers 12 \
           --hidden-size 768 \
           --num-attention-heads 12 \
           --seq-length ${MAX_SEQ_LEN} \
           --max-position-embeddings ${MAX_POS_EMB} \
           --micro-batch-size ${MICRO_BATCH} \
           --lr 0.0001 \
           --lr-decay-iters 990000 \
           --train-iters ${ITERS} \
           --min-lr 0.00001 \
           --lr-warmup-fraction 0.01 \
           --override-lr-scheduler \
           --vocab-file ${VOCAB} \
           --split ${TRAIN_VAL_TEST} \
           --fp16"

OUTPUT_ARGS="--log-interval 100 \
             --tensorboard-dir ${LOG_PATH} \
             --tensorboard-log-interval 50 \
             --no-log-loss-scale-to-tensorboard \
             --save-interval ${SAVE_INTERVAL} \
             --eval-interval ${SAVE_INTERVAL} \
             --eval-iters 10 \
             --activations-checkpoint-method uniform"

# CUDA_VISIBLE_DEVICES=${GPU} \
mkdir -p ${CHECKPOINT_PATH}
docker exec ${IMAGE} bash -c "cd `pwd`; \
python -m torch.distributed.launch $DISTRIBUTED_ARGS \
       pretrain_bert.py \
       ${BERT_ARGS} \
       ${OUTPUT_ARGS} \
       --save ${CHECKPOINT_PATH} \
       --load ${CHECKPOINT_PATH} \
       --data-path ${DATA_PATH} | tee wiki_jq2_b${GLOBAL_BATCH}_s${MAX_SEQ_LEN}_train.log"

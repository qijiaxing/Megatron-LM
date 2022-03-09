#!/bin/bash

IMAGE=pytorch2108
MEGATRON=../../..

# VOCAB=vocab/bert-chinese-expanded-vocab.txt   # added a few chinese symbols
VOCAB=${MEGATRON}/vocab/jq/jq-tokens.txt.2.vocab
# DATA_PREFIX=data/wiki_zh/jq_2_vocab/bert_wiki
DATA_PREFIX=${MEGATRON}/data/new2016/small
KEYS=content

DATA_PATH=${DATA_PREFIX}_${KEYS}_sentence
MICRO_BATCH=64
GPUS_PER_NODE=4
GLOBAL_BATCH=$((${MICRO_BATCH}*${GPUS_PER_NODE}))
# NO_SOP="--bert-no-binary-head"
MASKING=random  # or bert
MAX_SEQ_LEN=128
MAX_POS_EMB=512
LR=1e-3
ITERS=400000
SAVE_INTERVAL=50000
TRAIN_VAL_TEST="949,50,1"
CHECKPOINT_PATH=/raid/jqi/nlp/bert/new2016/full_less128/pretrain-checkpoints
mkdir -p ${CHECKPOINT_PATH}
LOG_PATH=${CHECKPOINT_PATH}/log
LOGFILE=train.log.0

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
           --num-attention-heads 12 ${NO_SOP} \
           --seq-length ${MAX_SEQ_LEN} \
           --max-position-embeddings ${MAX_POS_EMB} \
           --micro-batch-size ${MICRO_BATCH} \
           --lr ${LR} \
           --lr-decay-iters 390000 \
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
# mkdir -p ${CHECKPOINT_PATH}
docker exec ${IMAGE} bash -c "cd `pwd`; \
MASKING_STYLE=${MASKING} \
python -m torch.distributed.launch $DISTRIBUTED_ARGS \
       ${MEGATRON}/pretrain_bert.py \
       ${BERT_ARGS} \
       ${OUTPUT_ARGS} \
       --save ${CHECKPOINT_PATH} \
       --load ${CHECKPOINT_PATH} \
       --data-path ${DATA_PATH} | tee ${LOGFILE}"

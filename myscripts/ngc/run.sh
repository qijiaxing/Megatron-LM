#!/bin/bash

# NAME=ml-model.notamodel-wiki-zh-process.exempt-tc-gpu
NAME=ml-model.bert-new2016-random
# INSTANCE=cpu.x86.tiny
# INSTANCE=dgx1v.16g.4.norm
INSTANCE=dgxa100.40g.4.norm
IMAGE=nvidia/pytorch:21.08-py3

DATASETID=96766  # new2016
DATASETID=97829  # new2016 jqi vocab
DATASETID=97861  # new2016 clue vocab
DATASET=/mount/new2016
WORKSPACE=/mount/megatron
WORKDIR=${WORKSPACE}/Megatron-LM
DATADIR=${WORKDIR}/data/bert/new2016
DATA_PREFIX=${DATADIR}/full_less128
DATAKEY=content
DATA_PATH=${DATA_PREFIX}_${DATAKEY}_sentence
# VOCAB=${WORKDIR}/vocab/bert-chinese-expanded-vocab.txt   # added a few chinese symbols
# VOCAB=${WORKDIR}/vocab/jq/jq-tokens.txt.2.vocab
VOCAB=${WORKDIR}/vocab/clue.vocab

BATCH_PER_GPU=64
BATCH=256
MASKING=random  # or bert
MAX_SEQ_LEN=128
ITERS=800000
CHECKPOINT_PATH=${WORKDIR}/exp/bert/new2016/full_less128_random
LOG_PATH=${CHECKPOINT_PATH}/log

GPUS_PER_NODE=4
MASTER_ADDR=localhost
MASTER_PORT=7000
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))
DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

#          --global-batch-size ${BATCH} \
BERT_ARGS="--num-layers 12 \
           --hidden-size 768 \
           --num-attention-heads 12 \
           --seq-length ${MAX_SEQ_LEN} \
           --max-position-embeddings 512 \
           --lr 0.001 \
           --lr-decay-iters 1000000 \
           --train-iters ${ITERS} \
           --min-lr 0.00001 \
           --lr-warmup-fraction 0.01 \
           --override-lr-scheduler \
           --micro-batch-size ${BATCH_PER_GPU} \
           --vocab-file ${VOCAB} \
           --split "949,50,1" \
           --fp16"

OUTPUT_ARGS="--log-interval 100 \
             --tensorboard-dir ${LOG_PATH} \
             --tensorboard-log-interval 50 \
             --no-log-loss-scale-to-tensorboard \
             --save-interval 50000 \
             --eval-interval 50000 \
             --eval-iters 10 \
             --activations-checkpoint-method uniform"

COMMAND="cd ${WORKDIR}; pip install zhconv; export MASKING_STYLE=${MASKING}; \
       python -m torch.distributed.launch $DISTRIBUTED_ARGS pretrain_bert.py \
       ${BERT_ARGS} ${OUTPUT_ARGS} \
       --save ${CHECKPOINT_PATH} \
       --load ${CHECKPOINT_PATH} \
       --data-path ${DATA_PATH}"

ngc batch run --name ${NAME} --image ${IMAGE} --instance ${INSTANCE} --commandline "${COMMAND}" --result /results \
    --preempt RUNONCE --ace nv-us-west-2 --org nvidian --team sae \
    --datasetid ${DATASETID}:${DATASET} \
    --workspace qG7q67EMSma_OzVg98v7-A:${WORKSPACE}:RW

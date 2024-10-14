#!/bin/bash

NAME=ml-model.bert-new2016-random
INSTANCE=dgxa100.40g.4.norm
# INSTANCE=dgx1v.16g.4.norm
# INSTANCE=dgx1v.32g.4.norm
IMAGE=nvidia/pytorch:22.03-py3

DATASETID=96766  # new2016
DATASETID=97829  # new2016 jqi vocab
DATASETID=97861  # new2016 clue vocab
DATASET=/mount/new2016

WORKSPACE=/mount/megatron
# TODO Specify Checkpoint Path:
CHECKPOINT_PATH=${WORKSPACE}/exp/bert/new2016/full_less128_random
case "${INSTANCE}" in
    *a100*)
        WORKDIR=${WORKSPACE}/Megatron-LM;;
    *)
        WORKDIR=${WORKSPACE}/Megatron-LM-v100;;
esac
# TODO move DATADIR under workspace
DATADIR=${WORKSPACE}/Megatron-LM/data/bert/new2016
DATA_PATH=${DATADIR}/full_less128_content_sentence
VOCAB=${WORKDIR}/vocab/jq/jq-tokens.txt.2.vocab

BATCH_PER_GPU=64
BATCH=256
MASKING=random  # or bert
MAX_SEQ_LEN=128
ITERS=800000
LOG_PATH=${CHECKPOINT_PATH}/log

GPUS_PER_NODE=4
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))
DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes 1 --node_rank 0 --master_addr localhost --master_port 7000"

BERT_ARGS="--num-layers 12 --hidden-size 768 --num-attention-heads 12 \
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
       torchrun $DISTRIBUTED_ARGS pretrain_bert.py \
       ${BERT_ARGS} ${OUTPUT_ARGS} \
       --save ${CHECKPOINT_PATH} \
       --load ${CHECKPOINT_PATH} \
       --data-path ${DATA_PATH}"

#   --datasetid ${DATASETID}:${DATASET} \
ngc batch run --name ${NAME} --image ${IMAGE} --instance ${INSTANCE} --commandline "${COMMAND}" --result /results \
    --preempt RUNONCE --ace nv-us-west-2 --org nvidian --team sae \
    --workspace qG7q67EMSma_OzVg98v7-A:${WORKSPACE}:RW

#!/bin/bash

IMAGE=pytorch2108
GPU=3
MEGATRON=../../../..
PRETRAINED_CHECKPOINT=./
FROM=100000
TO=100000
for (( II=${FROM}; II<=${TO}; II=II+50000 ))
do
  TASK=tnews   # afqmc, ocnli
  LR=2e-4
  DATADIR=/home/jqi/work/CLUE/baselines/CLUEdataset/${TASK}
  TRAIN_DATA=${DATADIR}/train.json
  VALID_DATA=${DATADIR}/dev.json
  VOCAB_FILE=${MEGATRON}/vocab/jq/jq-tokens.txt.2.vocab
  CHECKPOINT_PATH=${PRETRAINED_CHECKPOINT}/${TASK}
  MAX_SEQ_LEN=128
  MAX_POS_EMB=128
  BATCH=32
  EPOCH=3
  LOG=${TASK}_preiters${II}.log
  TBDIR=tb-logs/${TASK}-preiters${II}
  RES=${TASK}_results.log

  WORLD_SIZE=1
  DISTRIBUTED_ARGS="--nproc_per_node ${WORLD_SIZE} \
                    --nnodes 1 --node_rank 0 \
                    --master_addr localhost --master_port 600${GPU}"

  BERT_ARGS="--num-layers 12 --hidden-size 768 --num-attention-heads 12 \
             --seq-length ${MAX_SEQ_LEN} --max-position-embeddings ${MAX_POS_EMB} \
             --lr ${LR} \
             --min-lr 1.0e-8 \
             --weight-decay 1.0e-2 \
             --lr-decay-style linear \
             --lr-warmup-fraction 0.1 \
             --micro-batch-size ${BATCH} \
             --fp16"
  
  OUTPUT_ARGS="--save-interval 50000 \
               --tensorboard-dir ${TBDIR} \
               --tensorboard-log-interval 10 \
               --epochs ${EPOCH} \
               --save $CHECKPOINT_PATH \
               --log-interval 10 \
               --eval-interval 50000 \
               --eval-iters 50"

  docker exec ${IMAGE} bash -c "cd `pwd`; \
  rm -r ${CHECKPOINT_PATH}; \
  echo ${II} > latest_checkpointed_iteration.txt; \
  CUDA_VISIBLE_DEVICES=${GPU} \
  python -m torch.distributed.launch $DISTRIBUTED_ARGS ${MEGATRON}/tasks/main.py \
         ${BERT_ARGS} ${OUTPUT_ARGS} \
         --task ${TASK} \
         --seed 1234 \
         --train-data $TRAIN_DATA \
         --valid-data $VALID_DATA \
         --tokenizer-type BertWordPieceLowerCase \
         --vocab-file $VOCAB_FILE \
         --pretrained-checkpoint $PRETRAINED_CHECKPOINT \
         --activations-checkpoint-method uniform | tee ${LOG}"
  echo Task: ${TASK} >> ${RES}
  echo Log: ${LOG} >> ${RES}
  echo LR: ${LR} >> ${RES}
  echo BATCH: ${BATCH} >> ${RES}
  echo Pretrained iterations: ${II} >> ${RES}
  grep "metrics for dev" ${LOG} >> ${RES}
  echo "\n" >> ${RES}
done

#!/bin/bash

INSTANCE=dgxa100.40g.1.norm
IMAGE=nvidia/pytorch:21.08-py3
WORKSPACE=/mount/megatron
WORKDIR=${WORKSPACE}/Megatron-LM
LOG=/results/joblog.log

for TASK in tnews afqmc ocnli; do
JOBNAME=ml-model.bert-new2016-finetune-${TASK}
DATADIR=${WORKDIR}/data/CLUEdataset/${TASK}
TRAIN_DATA=${DATADIR}/train.json
VALID_DATA=${DATADIR}/dev.json
# VOCAB_FILE=${WORKDIR}/vocab/bert-chinese-expanded-vocab.txt
VOCAB_FILE=${WORKDIR}/vocab/jq/jq-tokens.txt.2.vocab
PRETRAINED_CHECKPOINT=${WORKDIR}/exp/bert/new2016/full_less128_random
FINETUNE_DIR=${PRETRAINED_CHECKPOINT}/finetune
CHECKPOINT_PATH=${FINETUNE_DIR}/${TASK}
RES=${PRETRAINED_CHECKLPOINT}/${TASK}_results.log
MAX_SEQ_LEN=128
LR=2e-4
if [ ${TASK} == tnews ]; then BATCH=32; else BATCH=16; fi

DISTRIBUTED_ARGS="--nproc_per_node 1 --nnodes 1 --node_rank 0 --master_addr localhost --master_port 7009"
BERT_ARGS="--num-layers 12 --hidden-size 768 --num-attention-heads 12 \
           --seq-length ${MAX_SEQ_LEN} \
           --max-position-embeddings 512 \
           --lr ${LR} \
           --min-lr 1.0e-6 \
           --weight-decay 1.0e-2 \
           --lr-decay-style linear \
           --lr-warmup-fraction 0.1 \
           --micro-batch-size ${BATCH} \
           --fp16"

OUTPUT_ARGS="--save-interval 50000 --epochs 3 --save ${CHECKPOINT_PATH} \
             --log-interval 50 --eval-interval 50000 --eval-iters 50"

COMMAND="pip install zhconv; cd ${WORKDIR}; rm -r ${CHECKPOINT_PATH}; \
python -m torch.distributed.launch $DISTRIBUTED_ARGS ${WORKDIR}/tasks/main.py \
       ${BERT_ARGS} ${OUTPUT_ARGS} --task ${TASK} \
       --seed 1234 \
       --train-data $TRAIN_DATA --valid-data $VALID_DATA \
       --vocab-file $VOCAB_FILE \
       --tokenizer-type BertWordPieceLowerCase \
       --pretrained-checkpoint $PRETRAINED_CHECKPOINT \
       --activations-checkpoint-method uniform; \
  echo Task: ${TASK}, LR: ${LR}, BATCH: ${BATCH} | tee -a ${RES}; \
  grep 'successfully loaded checkpoint' ${LOG} | tee -a ${RES}; \
  grep 'metrics for dev' ${LOG} | tee -a ${RES}; \
  echo '' >> ${RES}"

ngc batch run --name ${JOBNAME} --image ${IMAGE} --instance ${INSTANCE} --commandline "${COMMAND}" --result /results \
    --preempt RUNONCE --ace nv-us-west-2 --org nvidian --team sae \
    --workspace qG7q67EMSma_OzVg98v7-A:${WORKSPACE}:RW
done

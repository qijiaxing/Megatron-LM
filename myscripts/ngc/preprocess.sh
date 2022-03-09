#!/bin/bash

IMAGE=nvidia/pytorch:21.08-py3
NAME=ml-model.notamodel-new2016-process.exempt-tc-gpu
INSTANCE=cpu.x86.tiny

DATASETID=96432
DATASET=/mount/new2016
INPUT=${DATASET}/wiki_zh_2019_full.json

WORKSPACE=/mount/megatron
WORKDIR=${WORKSPACE}/Megatron-LM
# DATA_PREFIX=${WORKDIR}/data/wiki_zh/bert
DATA_PREFIX=${WORKDIR}/data/bert/new2016/full/bert
VOCAB=${WORKDIR}/vocab/bert-chinese-expanded-vocab.txt   # added a few chinese symbols

EXE=tools/preprocess_data_zh.py   # For Chinese
COMMAND="pwd; cd ${WORKDIR}; pwd; \
python ${EXE} \
       --input ${INPUT} \
       --output-prefix ${DATA_PREFIX} \
       --vocab ${VOCAB} \
       --dataset-impl mmap \
       --split-sentences \
       --workers 8 \
       --tokenizer-type BertWordPieceLowerCase"
COMMAND="pwd; ls ${DATASET}; lscpu"

ngc batch run --name ${NAME} --image ${IMAGE} --instance ${INSTANCE} --commandline "${COMMAND}" --result /results \
    --preempt RUNONCE \
    --total-runtime 600s \
    --ace nv-us-west-2 --org nvidian --team sae \
    --datasetid ${DATASETID}:${DATASET} \
    --workspace qG7q67EMSma_OzVg98v7-A:${WORKSPACE}:RW

#!/bin/bash

IMAGE=nvidia/pytorch:21.08-py3
NAME=ml-model.notamodel-new2016-process.exempt-tc-gpu
INSTANCE=cpu.x86.tiny

DATASETID=96766  # new2016
DATASETID=97829  # new2016 jqi vocab
DATASETID=97861  # new2016 clue vocab
DATASET=/mount/new2016
WORKSPACE=/mount/megatron
WORKDIR=${WORKSPACE}/Megatron-LM
DATADIR=${WORKDIR}/data/bert/new2016/jqi-vocab

COMMAND="pwd; ls -lh ${DATASET}; mkdir ${DATADIR}; cp ${DATASET}/* ${DATADIR}/; echo copy finished!; ls -hl ${DATADIR}"

ngc batch run --name ${NAME} --image ${IMAGE} --instance ${INSTANCE} --commandline "${COMMAND}" --result /results \
    --preempt RUNONCE \
    --total-runtime 600s \
    --ace nv-us-west-2 --org nvidian --team sae \
    --datasetid ${DATASETID}:${DATASET} \
    --workspace qG7q67EMSma_OzVg98v7-A:${WORKSPACE}:RW

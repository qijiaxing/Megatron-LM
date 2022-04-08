#!/bin/bash

IMAGE=pytorch2203
MEGATRON=/home/jqi/work/megatron

CKPT_DIR=/raid/jqi/nlp/bert/new2016/full_less128/pretrain-checkpoints/iter_0800000/mp_rank_00/
NEMO_FILE=megatron.bert-base.nemo

# Install NeMo from github source
if [ "yes" == "yes" ]; then
INSTALL_NEMO="apt-get update && DEBIAN_FRONTEND='noninteractive' TZ='Asia/Hong_Kong' apt-get install -y tzdata; \
  apt-get install -y libsndfile1 ffmpeg; \
  pip install Cython; \
  python -m pip install git+https://github.com/NVIDIA/NeMo.git@main#egg=nemo_toolkit[all]"
fi

# Convert MegatronLM checkpoint to NeMo model file
if [ "yes" == "yes" ]; then
EXE=${MEGATRON}/tools/to-nemo/megatron_lm_ckpt_to_nemo.py
HPARAMS=${MEGATRON}/tools/to-nemo/hparams.yaml
CONVERT_MODEL="python -m torch.distributed.launch --nproc_per_node=1 ${EXE} \
  --checkpoint_folder ${CKPT_DIR} \
  --checkpoint_name model_optim_rng.pt \
  --nemo_file_path ${NEMO_FILE} \
  --model_type bert \
  --hparams_file ${HPARAMS}"
fi

# Finetune CMRC
EXE=${MEGATRON}/tasks/cmrc/finetune.py
CONFIG=${MEGATRON}/tasks/cmrc/conf/cmrc.yaml
docker exec ${IMAGE} bash -c "cd `pwd`; \
  ${INSTALL_NEMO};  ${CONVERT_MODEL}; \
  CUDA_VISIBLE_DEVICES=3 \
  python ${EXE} model.language_model.lm_checkpoint=${NEMO_FILE} \
  --config-name=${CONFIG}"

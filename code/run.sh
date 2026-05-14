#!/usr/bin/env bash

#GPU1=${1}
#GPU2=${2}
#
#mkdir -p result
##CUDA_VISIBLE_DEVICES=${GPU1} python3 train.py | tee ./result/train.log
##CUDA_VISIBLE_DEVICES=${GPU2} python3 eval.py | tee ./result/eval.log
#CUDA_VISIBLE_DEVICES=${GPU1} python3 train_all.py | tee ./result/train.log
#CUDA_VISIBLE_DEVICES=${GPU2} python3 eval_all.py | tee ./result/eval.log
#python3 accuracy.py | tee ./result/accuracy.log

set -euo pipefail

usage() {
  echo "Usage: $0 {train|eval|all} GPU_1 [GPU_2]"
  echo "  train:  $0 train GPU_1"
  echo "  eval:   $0 eval GPU_1"
  echo "  all:    $0 all GPU_1 GPU_2"
}

MODE=${1}

case "${MODE}" in
  train)
    DO_TRAIN=1; DO_EVAL=0
    GPU_TRAIN=${2}
    ;;
  eval)
    DO_TRAIN=0; DO_EVAL=1
    GPU_EVAL=${2}
    ;;
  all)
    DO_TRAIN=1; DO_EVAL=1
    GPU_TRAIN=${2}
    GPU_EVAL=${3}
    ;;
  -h|--help)
    usage
    exit 0
    ;;
  *)
    echo "Unknown mode: ${MODE}"
    usage
    exit 1
    ;;
esac

mkdir -p result
if [[ ${DO_TRAIN} -eq 1 ]]; then
  echo "Starting training on GPU ${GPU_TRAIN}..."
  if [[ -z "${GPU_TRAIN}" ]]; then
    python3 train_all.py | tee ./result/train.log
  else
    CUDA_VISIBLE_DEVICES=${GPU_TRAIN} python3 train_all.py | tee ./result/train.log
  fi
fi
if [[ ${DO_EVAL} -eq 1 ]]; then
  echo "Starting evaluation on GPU ${GPU_EVAL}..."
  if [[ -z "${GPU_EVAL}" ]]; then
    python3 eval_all.py | tee ./result/eval.log
    python3 accuracy.py | tee ./result/accuracy.log
  else
    CUDA_VISIBLE_DEVICES=${GPU_EVAL} python3 eval_all.py | tee ./result/eval.log
    python3 accuracy.py | tee ./result/accuracy.log
  fi
fi
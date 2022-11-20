#!/bin/bash

decode_path=/path/to/finetuned_model
finetuned_model=checkpoint_best.pt
beam=50
data=$1
[ -z $data ] && data="test"

python -B infer_s2s.py --config-dir /path/to/vat_hubert/vathubert/conf/ --config-name s2s_decode.yaml \
  dataset.gen_subset=${data} common_eval.path=${decode_path}/checkpoints/${finetuned_model} \
  common_eval.results_path=${decode_path}/${finetuned_model}_${data}_video_beam${beam} \
  override.modalities=["video"] \
  common.user_dir=/path/to/vat_hubert/vathubert \
  override.data=/path/to/data \
  override.label_dir=/path/to/data \
  generation.beam=${beam}


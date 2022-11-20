#!/bin/bash

ngpu=$1
updatefreq=$2
max_tokens=$3
pretrained_model_path=$4
save_path=$5

python /path/to/fairseq/fairseq_cli/hydra_train.py \
       --config-dir /path/to/vat_hubert/vathubert/conf/finetune --config-name base_lrs3_30h_v.yaml \
       task.data=/path/to/30h_data_tsv \
       task.label_dir=/path/to/30h_data_tsv \
       task.tokenizer_bpe_model=/path/to/sentencepiece/model \
       task.modalities=["video"] \
       model.w2v_path=${pretrained_model_path} \
       hydra.run.dir=${save_path} \
       common.user_dir=/path/to/vat_hubert/vathubert  \
       distributed_training.distributed_world_size=${ngpu} \
       distributed_training.ddp_backend="no_c10d" \
       optimization.update_freq=[${updatefreq}] \
       dataset.max_tokens=${max_tokens} \
       +task.use_supervised_data=False \
       +task.use_extra_textdata=False \
       +task.use_extra_audiodata=False \
       



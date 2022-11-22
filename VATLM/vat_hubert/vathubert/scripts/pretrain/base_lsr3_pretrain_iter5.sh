#!/bin/bash
ngpu=$1
updatefreq=$2
datapath=/LocalData/vatlm_related/fbankdata
save_path=$3

python /path/to/fairseq/fairseq_cli/hydra_train.py \
       --config-dir /path/to/vat_hubert/vathubert/conf/pretrain --config-name base_lrs3_iter5.yaml \
       task.data=${datapath}/433pre_lrs3_433h_tsv \
       task.label_dir=${datapath}/433pre_lrs3_433h_tsv \
       +task.sup_data_path=${datapath}/433pre_tedv3_phone_concat_tsv2 \
       +task.sup_manifest=${datapath}/433pre_tedv3_phone_concat_tsv2 \
       +task.onlytext_manifest=${datapath}/433pre_cantab_tsv \
       +task.onlyaudio_manifest=${datapath}/433pre_giga_tsv_km \
       hydra.run.dir=${save_path} \
       common.user_dir=/path/to/vat_hubert/vathubert \
       distributed_training.distributed_world_size=${ngpu} \
       optimization.update_freq=[${updatefreq}] \
       dataset.max_tokens=3000  \
       model.label_rate=25  \
       common.log_interval=200 \
       checkpoint.save_interval=5 \
       +task.sample_distributions=\"0.08,0.1,0.15,0.15\" \
       +criterion.banlance_loss_weights=[1.0,1.0] \
       dataset.data_buffer_size=40 \
       +task.use_supervised_data=True \
       +task.use_extra_textdata=True \
       +task.use_extra_audiodata=True \


       
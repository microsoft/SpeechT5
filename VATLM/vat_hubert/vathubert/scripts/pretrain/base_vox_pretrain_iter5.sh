#!/bin/bash
ngpu=$1
updatefreq=$2
datapath=/LocalData/vatlm_related/fbankdata
save_path=$3


python /path/to/fairseq/fairseq_cli/hydra_train.py \
       --config-dir /path/to/vat_hubert/vathubert/conf/pretrain --config-name base_vox_iter5.yaml \
       task.data=${datapath}/fbank_lrs3_vox_tsv \
       task.label_dir=${datapath}/fbank_lrs3_vox_tsv \
       +task.sup_data_path=${datapath}/fbank_tedv3_phone_concat_vox_tsv \
       +task.sup_manifest=${datapath}/fbank_tedv3_phone_concat_vox_tsv \
       +task.onlytext_manifest=${datapath}/cantab2_vox_tsv \
       +task.onlyaudio_manifest=${datapath}/fbank_giga_vox_tsv_km \
       hydra.run.dir=${save_path} \
       common.user_dir=/path/to/vat_hubert/vathubert \
       distributed_training.distributed_world_size=${ngpu} \
       optimization.update_freq=[${updatefreq}] \
       dataset.max_tokens=3000  \
       model.label_rate=25  \
       common.log_interval=200 \
       checkpoint.save_interval=5 \
       +task.sample_distributions=\"0.13,0.15,0.32,0.3\" \
       +criterion.banlance_loss_weights=[1.0,1.0] \
       dataset.data_buffer_size=40 \
       +task.use_supervised_data=True \
       +task.use_extra_textdata=True \
       +task.use_extra_audiodata=True \       


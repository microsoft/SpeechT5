
# ####################################
# Hubert SCT2T ED model #
# ####################################

world_size=$1
update_freq=$2
exp_name=$3
[ -z $world_size ] && world_size=24
[ -z $update_freq ] && update_freq=3
[ -z $exp_name ] && exp_name=sc2t_base_esen_${world_size}gpu_${update_freq}accum1


FAIRSEQ_ROOT=/mnt/output/users/v-kunwei/code/fairseq_mlstku
CONFIG_DIR=/mnt/output/users/v-kunwei/code/stpretrain_scripts/config
DATA_DIR="/mnt/output/users/v-kunwei/data/s2s_data/speech_esen"
TEXT_DATA_DIR="/mnt/output/users/v-kunwei/data/s2s_data/text_esen"
MODEL_DIR="/mnt/output/v-kunwei/data/s2s_data/exp/S2S_esen/$exp_name"

[ -d $MODEL_DIR ] || mkdir -p $MODEL_DIR


python $FAIRSEQ_ROOT/fairseq_cli/hydra_train.py \
  --config-dir $CONFIG_DIR/pretrain \
  --config-name sc2t_base_librispeech \
  \
  +task.store_labels=true \
  task.labels='["km"]' \
  model.label_rate=50 \
  task.data=$DATA_DIR \
  task.label_dir=$DATA_DIR \
  task.text_cfg.text_data=$TEXT_DATA_DIR \
  +task.text_cfg.data_config=config.yaml \
  task.text_cfg.text_maxtokens_ratio=3.0 \
  \
  +criterion.dec_loss_type="ce" \
  \
  criterion.text_weight=1.0 \
  \
  model.use_rel_pos_enc=true \
  +model.code_use_rel_pos_enc=true \
  +model.pad_with_code=true \
  model.text_transformer.no_scale_embedding=true \
  model.text_transformer.layernorm_embedding=true \
  +model.share_decoder_input_output_embed=true \
  \
  dataset.train_subset=\"train+en.kmu-spm\" \
  dataset.valid_subset=\"valid+en_valid.kmu-spm\" \
  dataset.num_workers=0 \
  dataset.max_tokens=1000000 \
  optimization.update_freq=[${update_freq}] \
  optimization.max_update=400000 \
  \
  distributed_training.distributed_world_size=${world_size} \
  \
  common.tensorboard_logdir=$MODEL_DIR \
  checkpoint.save_dir=$MODEL_DIR \
  hydra.run.dir=$MODEL_DIR \
  hydra.job.name=${exp_name}


sleep 5m
echo "All finished"


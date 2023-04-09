#####################################
# Hubert mt model #
#####################################
[ $# -gt 3 ] && echo "Usage: $0 <world_size> <seeds>" && exit 0
world_size=$1
update_freq=$2
w2v_path=$3
Mount=""

[ -z $world_size ] && world_size=8
[ -z $update_freq ] && update_freq=1
[ -z $w2v_path ] && w2v_path="/mnt/output/users/v-kunwei/data/s2s_data/model_es_emb_90_1004.pt"


langs="ltr,kmu"
FAIRSEQ_ROOT=/mnt/output/users/v-kunwei/code/fairseq_mlstku
CONFIG_ROOT=/mnt/output/users/v-kunwei/code/stpretrain_scripts/config/translation
DATA_DIR=/mnt/output/users/v-kunwei/data/s2s_data/es_no_data/

### set save-dir
MODEL_DIR="/mnt/output/users/v-kunwei/data/s2s_data/exp/text2unicode_es"
exp_name="base_pt400k_releaseiter2_${world_size}gpu_${update_freq}accum_lr1e-4_no"
MODEL_DIR=$MODEL_DIR/$exp_name
[ -d $MODEL_DIR ] || mkdir -p $MODEL_DIR


python $FAIRSEQ_ROOT/fairseq_cli/hydra_train.py \
  --config-dir $CONFIG_ROOT \
  --config-name text2code \
  +task.data=$DATA_DIR \
  dataset.dataset_impl="raw" \
  +task.source_lang="ltr" +task.target_lang="kmu" \
  +task.normalize=false \
  \
  +criterion.label_smoothing=0.1 \
  +criterion.report_accuracy=true \
  optimizer.weight_decay=0.00001 \
  +lr_scheduler.lr="[0.0001]" \
  optimization.max_update=500000 \
  \
  +model.dropout=0.1 \
  +model.attention_dropout=0.1 \
  model.activation_dropout=0.1 \
  model.decoder_layerdrop=0 \
  model.layerdrop=0 \
  model.w2v_path=$w2v_path \
  +model.text_transformer_encoder_layers=6 \
  \
  dataset.train_subset="es_train" \
  dataset.valid_subset="es_dev" \
  optimization.update_freq=[${update_freq}] \
  optimization.clip_norm=5 \
  \
  common.seed=222 \
  common.log_interval=100 \
  common.log_format="json" \
  \
  distributed_training.distributed_world_size=${world_size} \
  distributed_training.nprocs_per_node=8 \
  distributed_training.ddp_backend="legacy_ddp" \
  \
  common.tensorboard_logdir=$MODEL_DIR \
  checkpoint.save_dir=$MODEL_DIR \
  hydra.run.dir=$MODEL_DIR \
  hydra.job.name=${exp_name} \

sleep 10s
  # sleep infinity



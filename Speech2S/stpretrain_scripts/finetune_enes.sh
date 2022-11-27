# ####################################
# Hubert ED model #
# ####################################
#source /mnt/default/v-ziqzhang/.bashrc_sing

[ $# -lt 4 ] && echo "Usage: $0 <world_size> <update_freq> <w2v_path> <cpt>" && exit 0
world_size=$1
update_freq=$2
w2v_path=$3
cpt=$4
Mount=$5

[ -z $world_size ] && world_size=8
[ -z $update_freq ] && update_freq=3
[ -z $w2v_path ] && echo "you must specify a wav_path !" && exit 1
[ -z $cpt ] && cpt=030.pt
[ -z $Mount ] && Mount=/mnt/default


FAIRSEQ_ROOT=/mnt/output/users/v-kunwei/code/fairseq_mlstku
CONFIG_DIR=/mnt/output/users/v-kunwei/code/stpretrain_scripts/config
DATA_DIR="/mnt/output/users/v-kunwei/data/s2s_data/fin_enes100"

exp_name=${w2v_path%/*}
exp_name=${exp_name##*/}
MODEL_DIR="/mnt/output/users/v-kunwei/data/s2s_data/finetune/tune_ST_from_eneshu"
exp_name="tune_enes_lr5e-5_from_$cpt"
MODEL_DIR=$MODEL_DIR/$exp_name
[ -d $MODEL_DIR ] || mkdir -p $MODEL_DIR

max_tokens=490000

python $FAIRSEQ_ROOT/fairseq_cli/hydra_train.py \
  --config-dir $CONFIG_DIR/finetune_asr \
  --config-name base_100h \
  \
  +task.store_labels=true \
  task.labels='["spm"]' \
  task.data=$DATA_DIR \
  task.label_dir=$DATA_DIR \
  task.add_decoder=true \
  +task.max_keep_size=490000 \
  \
  +model.reuse_text_emb=true \
  model._name="stbert_st" \
  model.w2v_path=${w2v_path} \
  model.add_decoder=true \
  \
  criterion._name="label_smoothed_cross_entropy" \
  +criterion.label_smoothing=0.2 \
  +criterion.report_accuracy=true \
  \
  lr_scheduler._name="polynomial_decay" \
  +lr_scheduler.warmup_updates=20000 \
  \
  optimization.lr=[0.0003] \
  optimization.max_update=100000 \
  checkpoint.best_checkpoint_metric="accuracy" \
  checkpoint.maximize_best_checkpoint_metric=true \
  checkpoint.save_interval=1 \
  \
  dataset.train_subset="train" \
  dataset.valid_subset="valid" \
  dataset.max_tokens=$max_tokens \
  optimization.update_freq=[${update_freq}] \
  \
  distributed_training.distributed_world_size=${world_size} \
  distributed_training.distributed_port=-1 \
  \
  common.log_interval=100 \
  common.tensorboard_logdir=$MODEL_DIR \
  checkpoint.save_dir=$MODEL_DIR \
  hydra.run.dir=$MODEL_DIR \
  hydra.job.name=${exp_name}



sleep 20s

  # \
  # lr_scheduler._name="polynomial_decay" \
  # +lr_scheduler.warmup_updates=5000 \


# /mnt/default/v-ziqzhang/data/stbert-ed/exp/ST_enes/sc2t_base_ende_32gpu_1accum/checkpoint_204_400000.pt

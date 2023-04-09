# ####################################
# SpeechUT Base model #
# ####################################
[ $# -lt 3 ] && echo "Usage: $0 <model_path> <data_dir> <cpt_tag> [mount=${PWD}] [world_size=8] [update_freq=2]" && exit 1
[ ${PWD##*/} != SpeechUT ] && echo "Error: dir not match! Switch to SpeechUT/ and run it again!" && exit 1

w2v_path=$1
DATA_DIR=$2
cpt=$3
mount=$4
world_size=$5
update_freq=$6
[ -z $mount ] && mount=${PWD}
[ -z $world_size ] && world_size=8
[ -z $update_freq ] && update_freq=2

CODE_ROOT=${PWD}

exp_name=${w2v_path%/*}
exp_name=${exp_name##*/}
MODEL_DIR="${mount}/exp/finetune_asr/$exp_name/edctc40k_from_${cpt}_bz2.6m_lr1e-5"
[ -d $MODEL_DIR ] || mkdir -p $MODEL_DIR

python $CODE_ROOT/fairseq/fairseq_cli/hydra_train.py \
  --config-dir $CODE_ROOT/speechut/config/finetune_asr \
  --config-name speechut_base_100h \
  common.user_dir=$CODE_ROOT/speechut \
  \
  task.data=$DATA_DIR \
  task.label_dir=$DATA_DIR \
  model.w2v_path=${w2v_path} \
  \
  optimization.lr=[0.00001] \
  optimization.max_update=40000 \
  dataset.max_tokens=1300000 \
  optimization.update_freq=[${update_freq}] \
  distributed_training.distributed_world_size=${world_size} \
  \
  dataset.train_subset="train_clean_100" \
  dataset.valid_subset="dev_other" \
  \
  common.tensorboard_logdir=$MODEL_DIR \
  checkpoint.save_dir=$MODEL_DIR \
  hydra.run.dir=$MODEL_DIR \
  hydra.job.name=edctc40k_from_${cpt}_bz2.6m_lr1e-5

# ####################################
# SpeechUT Base model #
# ####################################
[ $# -lt 3 ] && echo "Usage: $0 <data_dir> <text_data_dir> <lang=de/es> [mount=${PWD}] [world_size=32] [update_freq=1]" && exit 1
[ ${PWD##*/} != SpeechUT ] && echo "Error: dir not match! Switch to SpeechUT/ and run it again!" && exit 1
DATA_DIR=$1
TEXT_DATA_DIR=$2
lang=$3
mount=$4
world_size=$5
update_freq=$6
[ -z $mount ] && mount=${PWD}
[ -z $world_size ] && world_size=32
[ -z $update_freq ] && update_freq=1

CODE_ROOT=${PWD}
MODEL_DIR="${mount}/exp/pretrain/base_speechut4en${lang}_${world_size}gpu_${update_freq}accum"
[ -d $MODEL_DIR ] || mkdir -p $MODEL_DIR

python $CODE_ROOT/fairseq/fairseq_cli/hydra_train.py \
  --config-dir $CODE_ROOT/speechut/config/pretrain \
  --config-name speechut_base_librispeech \
  common.user_dir=$CODE_ROOT/speechut \
  \
  task.labels='["km"]' \
  model.label_rate=50 \
  task.data=$DATA_DIR \
  task.label_dir=$DATA_DIR \
  task.text_cfg.text_data=$TEXT_DATA_DIR \
  \
  model.add_text_ctc=false \
  model.text_transformer.share_decoder_input_output_embed=true \
  criterion.u2t_ed_weight=1.0 \
  criterion.u2t_ctc_weight=0 \
  \
  dataset.train_subset=\"train_960,mustcuns_${lang}+pseudo_wmt_en${lang}.kmu-spm+train_960.kmu-none,mustcuns_${lang}.kmu-none\" \
  dataset.valid_subset=\"dev_clean+pseudo_valid.kmu-spm+dev.kmu-none\" \
  dataset.num_workers=0 \
  dataset.max_tokens=1400000 \
  distributed_training.distributed_world_size=${world_size} \
  optimization.update_freq=[${update_freq}] \
  \
  common.tensorboard_logdir=$MODEL_DIR \
  checkpoint.save_dir=$MODEL_DIR \
  hydra.run.dir=$MODEL_DIR \
  hydra.job.name=base_speechut4en${lang}_${world_size}gpu_${update_freq}accum


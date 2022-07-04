export HYDRA_FULL_ERROR=1
YiTrans=/home/v-ziqzhang/Code/SpeechT5/YiTrans
DATA_DIR=/mnt/default/lozhou/speechdata/hubert_data
LABEL_DIR=${DATA_DIR}/layer9_k500_label
SP_PATH=${LABEL_DIR}/spm_unigram8000.model
TEXT_DATA_DIR=/mnt/default/lozhou/speechdata/text_data/v3/bin_idx_step1
EXP_NAME=pretrain_pt36_addadaptor_bpecode_large_step1
SAVE_DIR=${HOME}/data/speechexp/${EXP_NAME}
W2V_PATH=${HOME}/data/speechexp/hubert_large_librivox_released/checkpoint_last.pt
MBART_PATH=${HOME}/data/speechexp/mbart50.pretrained/model.pt

python ${YiTrans}/fairseq/fairseq_cli/hydra_train.py \
  --config-dir ${YiTrans}/yitrans_iwslt22/config/pretrain \
  --config-name joint_large \
  common.user_dir=${YiTrans}/yitrans_iwslt22 \
  \
  task.data=$DATA_DIR \
  task.labels='["km"]' \
  task.label_dir=$LABEL_DIR \
  task.text_cfg.text_data=$TEXT_DATA_DIR \
  +task.hubert_tokenizer="sentencepiece" \
  +task.sp_path=${SP_PATH} \
  \
  model.label_rate=50 \
  model.encoder_layers=12 \
  +model.load_pretrained_w2v_from=${W2V_PATH} \
  +model.load_pretrained_mbart_from=${MBART_PATH} \
  \
  dataset.train_subset=\"train_LS,train_MUSTC+mono_deduped_filt_sort.en_XX.en_XX,mt8corpus_filt_slct.en_XX-de_DE\" \
  dataset.valid_subset=\"dev_MUSTC+valid.en_XX-de_DE,dev_MUSTC+valid.en_XX-ja_XX,dev_MUSTC+valid.en_XX-zh_CN,dev_MUSTC+dev4x.en_XX.en_XX\" \
  dataset.max_tokens=300000 \
  \
  distributed_training.distributed_world_size=8 \
  distributed_training.nprocs_per_node=8 \
  optimization.update_freq=[2] \
  \
  common.tensorboard_logdir=$SAVE_DIR \
  checkpoint.save_dir=$SAVE_DIR \
  hydra.run.dir=$SAVE_DIR \
  hydra.job.name=$EXP_NAME \
  checkpoint.reset_optimizer=true \
  checkpoint.reset_dataloader=true



  # dataset.train_subset=\"train_CV,train_EUR,train_LS,train_MUSTC,train_TEDLIUM,train_VP+mono_deduped_filt_sort.en_XX.en_XX,mt8corpus_filt_slct.en_XX-de_DE,mt8corpus_filt_slct.en_XX-ja_XX,mt8corpus_filt_slct.en_XX-zh_CN\" \

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
  \
  common.user_dir=${YiTrans}/yitrans_iwslt22 \
  \
  task.add_decoder=true \
  criterion._name="joint_step1_split_batch" \
  +task.split_modality_batch=true \
  +task.hubert_tokenizer="sentencepiece" \
  +task.sp_path=${SP_PATH} \
  \
  model.encoder_layers=12 \
  model.text_transformer.max_source_positions=1024 \
  +model.add_text_modality=true \
  +model.add_text_encoder=true \
  +model.add_adaptor=true \
  +model.load_pretrained_w2v_from=${W2V_PATH} \
  +model.load_pretrained_mbart_from=${MBART_PATH} \
  \
  task.text_cfg.text_maxtokens_ratio=1.0 \
  task.text_cfg.tokens_per_sample=512 \
  +task.store_labels=true \
  +task.text_cfg.mask=0.3 \
  +task.text_cfg.mask_whole_words=true \
  \
  dataset.train_subset=\"train_LS,train_MUSTC+mono_deduped_filt_sort.en_XX.en_XX,mt8corpus_filt_slct.en_XX-de_DE\" \
  dataset.valid_subset=\"dev_MUSTC+valid.en_XX-de_DE,dev_MUSTC+valid.en_XX-ja_XX,dev_MUSTC+valid.en_XX-zh_CN,dev_MUSTC+dev4x.en_XX.en_XX\" \
  dataset.num_workers=4 \
  dataset.max_tokens=800000 \
  dataset.max_tokens_valid=320000 \
  optimization.max_update=400000 \
  optimization.lr=[0.00003] \
  checkpoint.save_interval_updates=10000 \
  checkpoint.keep_last_epochs=10 \
  \
  common.log_format='json' \
  common.log_interval=200 \
  distributed_training.distributed_world_size=8 \
  distributed_training.nprocs_per_node=8 \
  optimization.update_freq=[2] \
  \
  model.label_rate=50 \
  task.labels='["km"]' \
  task.data=$DATA_DIR \
  task.text_cfg.text_data=$TEXT_DATA_DIR \
  task.label_dir=$LABEL_DIR \
  common.tensorboard_logdir=$SAVE_DIR \
  checkpoint.save_dir=$SAVE_DIR \
  hydra.run.dir=$SAVE_DIR \
  hydra.job.name=$EXP_NAME


  # dataset.train_subset=\"train_CV,train_EUR,train_LS,train_MUSTC,train_TEDLIUM,train_VP+mono_deduped_filt_sort.en_XX.en_XX,mt8corpus_filt_slct.en_XX-de_DE,mt8corpus_filt_slct.en_XX-ja_XX,mt8corpus_filt_slct.en_XX-zh_CN\" \

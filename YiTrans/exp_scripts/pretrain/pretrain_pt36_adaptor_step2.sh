EXP_NAME=train_speech_text_joint_adaptor_large_step2_300k
SAVE_DIR=/datablob/users/v-junyiao/speechexp/fairseq_mlst/${EXP_NAME}
DATA_ROOT=/datablob/users/v-junyiao/speechdata/hubert_mlst
LABEL_DIR=${DATA_ROOT}/fine-tune_en_bpe250k_full
W2V_PATH=/mnt/default/v-junyiao/speechexp/train_speech_text_joint_addadaptor_bpecode_large_step1_mbartpt_400k/checkpoint_last_up.pt
TEXT_DATA_DIR=/datablob/users/v-junyiao/speechdata/text_data/v4/bin-idx
SP_PATH=${LABEL_DIR}/sentence.bpe.model
# export CUDA_VISIBLE_DEVICES=1
python fairseq_cli/hydra_train.py \
  --config-dir examples/hubert/config/pretrain \
  --config-name pretrain_step2 \
  distributed_training.distributed_world_size=64 \
  distributed_training.nprocs_per_node=8 \
  \
  dataset.train_subset=\"train_COVOST,train_asr_VP,train_punc_TEDLIUM,train_asr_MUSTC,train_punc_LS,train_asr_EUR+covost2.en_XX-ja_XX,covost2.en_XX-zh_CN,covost_eurST.en_XX-de_DE,mt8corpus_domain45.en_XX-ja_XX,mt8corpus_filt_slct80_domain44.en_XX-de_DE,mt8corpus_filt_slct80_domain40.en_XX-zh_CN,train.en_XX-de_DE,train.en_XX-ja_XX,train.en_XX-zh_CN\" \
  dataset.valid_subset=\"dev_asr_MUSTC+valid.en_XX-de_DE,dev_asr_MUSTC+valid.en_XX-ja_XX,dev_asr_MUSTC+valid.en_XX-zh_CN\" \
  dataset.max_tokens=480001 \
  dataset.num_workers=0 \
  optimization.update_freq=[1] \
  optimization.max_update=300000 \
  \
  task.hubert_tokenizer="sentencepiece" \
  task.sp_path=${SP_PATH} \
  task.max_keep_size=480000 \
  +task.split_modality_batch=true \
  +task.speech_tgt_lang="en_XX" \
  +task.mbart_style_lang_id=true \
  +task.text_sampling_alpha=1.0 \
  +task.store_labels=true \
  model.freeze_finetune_updates=15000 \
  criterion.dec_weight=0.5 \
  +model.reuse_text_emb=true \
  +model.share_ctc_decoder_embed=true \
  +model.share_speech_text_embeddings=true \
  \
  task.data=${DATA_ROOT} \
  task.label_dir=${LABEL_DIR} \
  task.text_cfg.text_data=${TEXT_DATA_DIR} \
  model.w2v_path=${W2V_PATH} \
  checkpoint.save_dir=${SAVE_DIR} \
  common.tensorboard_logdir=${SAVE_DIR} \
  hydra.run.dir=${SAVE_DIR} \
  hydra.job.name=${EXP_NAME}

sleep infinity

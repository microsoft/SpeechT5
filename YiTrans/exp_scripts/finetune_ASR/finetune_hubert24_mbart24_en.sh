world_size=$1
update_freq=$2
[ -z $world_size ] && world_size=8
[ -z $update_freq ] && update_freq=8

EXP_NAME=train_iwslt_asr_hubert24_mbart24_norel
SAVE_DIR=${HOME}/data/iwslt/asr_v3/${EXP_NAME}

DATA_ROOT=${HOME}/dataset/iwslt_mustc
LABEL_DIR=${DATA_ROOT}/fine-tune_en_bpe250k
SP_PATH=${LABEL_DIR}/sentence.bpe.model
retain_dict=${LABEL_DIR}/index_en_onlyMUSTC
W2V_PATH=${HOME}/dataset/iwslt_mustc/pretrain_ed_model_cfg.pt

TRAIN_SUBSET=train_asr_MUSTC
VALID_SUBSET=dev_asr_MUSTC


mbart_path="/mnt/default/v-junyiao/released_exsp/mbart50.pretrained/model.pt"
hubert_path="/mnt/default/v-junyiao/speechexp/fairseq_mlst/hubert_large_librivox_released/checkpoint_last.pt"

CODE_ROOT=${HOME}/code/SpeechT5/YiTrans

python $CODE_ROOT/fairseq/fairseq_cli/hydra_train.py \
  --config-dir $CODE_ROOT/yitrans_iwslt22/config/finetune_asr \
  --config-name large_mustc \
  common.user_dir=$CODE_ROOT/yitrans_iwslt22 \
  distributed_training.distributed_world_size=$world_size \
  optimization.update_freq=[$update_freq] \
  \
  dataset.max_tokens=400001 \
  dataset.num_workers=0 \
  optimization.max_update=120000 \
  \
  task._name="iwslt_joint_pretraining" \
  task.data=${DATA_ROOT} \
  task.label_dir=${LABEL_DIR} \
  +task.store_labels=True \
  task.hubert_tokenizer="sentencepiece" \
  task.sp_path=${SP_PATH} \
  task.max_keep_size=400000 \
  criterion.dec_weight=0.5 \
  \
  model._name="yitrans_asr" \
  model.w2v_path=${W2V_PATH} \
  +model.reuse_text_emb=true \
  +model.share_ctc_decoder_embed=true \
  +model.retain_dict_path=${retain_dict} \
  model.freeze_finetune_updates=15000 \
  \
  +model.no_pretrained_weights=true \
  +model.use_rel_pos_enc=false \
  +model.encoder_layers=24 \
  +model.add_text_encoder=true \
  +model.share_s2t_t2t_embeddings=false \
  +model.share_enc_dec_embeddings=false \
  +model.add_adaptor=false \
  +model.load_pretrained_w2v_from=$hubert_path \
  +model.load_pretrained_mbart_from=$mbart_path \
  \
  dataset.train_subset=${TRAIN_SUBSET} \
  dataset.valid_subset=${VALID_SUBSET} \
  checkpoint.save_dir=${SAVE_DIR} \
  common.tensorboard_logdir=${SAVE_DIR} \
  hydra.run.dir=${SAVE_DIR} \
  hydra.job.name=${EXP_NAME}


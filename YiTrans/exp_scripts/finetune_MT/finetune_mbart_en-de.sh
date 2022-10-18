#####################################
# Hubert ED model #
#####################################
[ $# -gt 2 ] && echo "Usage: $0 <world_size> <update_freq> [w2v_path] [mbart_path]" && exit 0
world_size=$1
update_freq=$2
w2v_path=$3
mbart_path=$4

[ -z $world_size ] && world_size=8
[ -z $update_freq ] && update_freq=2
[ -z $w2v_path ] && w2v_path=${HOME}/dataset/iwslt_mustc/pretrain_ed_model_cfg.pt
[ -z $mbart_path ] && mbart_path="/mnt/default/v-junyiao/released_exsp/mbart50.pretrained/model.pt"
langs=ar_AR,cs_CZ,de_DE,en_XX,es_XX,et_EE,fi_FI,fr_XX,gu_IN,hi_IN,it_IT,ja_XX,kk_KZ,ko_KR,lt_LT,lv_LV,my_MM,ne_NP,nl_XX,ro_RO,ru_RU,si_LK,tr_TR,vi_VN,zh_CN,af_ZA,az_AZ,bn_IN,fa_IR,he_IL,hr_HR,id_ID,ka_GE,km_KH,mk_MK,ml_IN,mn_MN,mr_IN,pl_PL,ps_AF,pt_XX,sv_SE,sw_KE,ta_IN,te_IN,th_TH,tl_XX,uk_UA,ur_PK,xh_ZA,gl_ES,sl_SI

DATA_DIR=/mnt/default/lozhou/speechdata/mt_data/en-de/com-filter-ende/bin-idx
exp_name=tune_mbart_com_filter_le-4
SAVE_DIR="${HOME}/data/iwslt/mt_stage1_en-de/$exp_name"
[ -d $SAVE_DIR ] || mkdir -p $SAVE_DIR

CODE_ROOT=${HOME}/code/SpeechT5/YiTrans

python $CODE_ROOT/fairseq/fairseq_cli/hydra_train.py \
  --config-dir $CODE_ROOT/yitrans_iwslt22/config/finetune_mt \
  --config-name mt_translation \
  common.user_dir=$CODE_ROOT/yitrans_iwslt22 \
  distributed_training.distributed_world_size=${world_size} \
  optimization.update_freq=[$update_freq] \
  \
  +task.data=$DATA_DIR \
  +task.source_lang="en_XX" +task.target_lang="de_DE" \
  +task.langs=\"$langs\" \
  +task.normalize=false \
  +task.append_source_id=true \
  \
  +model.dropout=0.2 \
  +model.attention_dropout=0.1 \
  model.activation_dropout=0.1 \
  model.decoder_layerdrop=0 \
  model.layerdrop=0 \
  model.freeze_finetune_updates=0 \
  \
  model.w2v_path=$w2v_path \
  +model.no_pretrained_weights=true \
  +model.load_pretrained_mbart_from=$mbart_path \
  +model.share_enc_dec_embeddings=true \
  +model.share_s2t_t2t_embeddings=false \
  +model.use_rel_pos_enc=false \
  \
  dataset.train_subset="train" \
  dataset.valid_subset="valid" \
  dataset.num_workers=4 \
  dataset.max_tokens=2000 \
  \
  optimization.max_epoch=50 \
  optimization.clip_norm=5 \
  optimization.max_update=200000 \
  lr_scheduler.total_num_update=200000 \
  \
  checkpoint.save_interval=1 \
  checkpoint.save_interval_updates=5000 \
  checkpoint.keep_last_epochs=5 \
  checkpoint.keep_best_checkpoints=5 \
  \
  common.seed=222 \
  common.log_interval=100 \
  common.log_format="json" \
  \
  checkpoint.best_checkpoint_metric="accuracy" \
  checkpoint.maximize_best_checkpoint_metric=true \
  common.tensorboard_logdir=$SAVE_DIR \
  checkpoint.save_dir=$SAVE_DIR \
  hydra.run.dir=$SAVE_DIR \
  hydra.job.name=$exp_name


world_size=$1
update_freq=$2
[ -z $world_size ] && world_size=8
[ -z $update_freq ] && update_freq=4

DATA_DIR=/mnt/default/lozhou/speechdata/st_data/en-de/com2-ende-newmt
EXP_NAME="jt_st_mustc_large_stage2_300k_11sets"
SAVE_DIR=/mnt/default/v-ziqzhang/data/iwslt/st_en-de_v4/${EXP_NAME}
retain_dict=/mnt/default/v-junyiao/dataset/iwslt/en-de/released/analyse/index_asr_st_onlyMUSTC
W2V_PATH1=/mnt/default/v-junyiao/speechexp/train_speech_text_joint_addadaptor_bpecode_large_step1_mbartpt_400k/checkpoint_last.pt
W2V_PATH2=/mnt/default/v-junyiao/speechexp/fairseq_mlst/train_speech_text_joint_adaptor_large_step2_300k/checkpoint_last.pt
mkdir -p ${SAVE_DIR}

FAIRSEQ_ROOT=/mnt/default/v-ziqzhang/code/fairseq_mlst

python $FAIRSEQ_ROOT/fairseq_cli/train.py ${DATA_DIR} \
    --save-dir ${SAVE_DIR} \
    --user-dir examples/speech_text_joint_to_text \
    --task speech_text_joint_to_text \
    --config-yaml config_step1_39k.yaml \
    --train-subset "train_11set_st_addsrc" \
    --valid-subset "dev_mustc2_en_de_addsrc_st" \
    --fp16 \
    --seed 1 \
    \
    --ddp-backend no_c10d \
    --distributed-world-size ${world_size} \
    --tensorboard-logdir ${SAVE_DIR} \
    \
    --criterion guided_label_smoothed_cross_entropy_with_accuracy \
    --label-smoothing 0.3 \
    --guide-alpha 0.8 \
    --disable-text-guide-update-num 5000 \
    --attentive-cost-regularization 0.02 \
    \
    --optimizer adam \
    --clip-norm 1.0 \
    --lr 5e-05 \
    --lr-scheduler polynomial_decay --warmup-updates 5000 \
    --warmup-updates 5000 \
    --max-update 200000 \
    --total-num-update 200000 \
    --update-freq ${update_freq} \
    \
    --max-tokens 450000 \
    --max-sentences 3 \
    --max-tokens-valid 500000 \
    --max-source-positions 450000 \
    --skip-invalid-size-inputs-valid-test \
    --num-workers 0 \
    --save-interval 1 \
    --log-format json \
    --log-interval 100 \
    --best-checkpoint-metric "acc" \
    --maximize-best-checkpoint-metric \
    \
    --arch "hubert_st2t" \
    --w2v-path ${W2V_PATH1} \
    --load-step2-model-from ${W2V_PATH2} \
    --no-pretrained-weights \
    --add-decoder \
    --reuse-text-emb \
    --layerdrop 0.1 \
    --activation-dropout 0.1 \
    --decoder-layerdrop 0.1 \
    --freeze-finetune-updates 0 \
    --feature-grad-mult 1.0 \
    --retain-dict-path ${retain_dict} \
    --share-decoder-input-output-embed \
    --share-speech-text-embeddings \
    \
    --save-interval-updates 2000 \
    --keep-interval-updates 5 \
    --keep-interval-updates-pattern 10000 \
    --keep-last-epochs 5 \
    \
    2>&1 | tee ${SAVE_DIR}/train.log

sleep 5s

    # --lr-scheduler inverse_sqrt \
    # --load-step2-model-from ${W2V_PATH2} \
    # --no-pretrained-weights \

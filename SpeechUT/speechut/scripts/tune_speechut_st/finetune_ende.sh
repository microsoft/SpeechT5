# ####################################
# Hubert ED model #
# ####################################
source /mnt/default/v-ziqzhang/.bashrc_sing

[ $# -lt 4 ] && echo "Usage: $0 <world_size> <update_freq> <w2v_path> <cpt>" && exit 0
world_size=$1
update_freq=$2
w2v_path=$3
cpt=$4
Mount=$5

[ -z $world_size ] && world_size=8
[ -z $update_freq ] && update_freq=4
[ -z $w2v_path ] && echo "you must specify a wav_path !" && exit 1
[ -z $cpt ] && cpt=400k
[ -z $Mount ] && Mount=/mnt/default


FAIRSEQ_ROOT=/mnt/default/v-ziqzhang/code/fairseq_mlst
CONFIG_DIR=/mnt/default/v-ziqzhang/code/stpretrain_scripts/config
DATA_DIR="$Mount/v-ziqzhang/dataset/mustc/en_de/legacy"

exp_name=${w2v_path%/*}
exp_name=${exp_name##*/}
MODEL_DIR="$Mount/v-ziqzhang/data/stbert-ed/finetune/tune_ST_from_ende_$exp_name"
exp_name="legacy_ende_lr3e-5_from_$cpt"
MODEL_DIR=$MODEL_DIR/$exp_name
[ -d $MODEL_DIR ] || mkdir -p $MODEL_DIR

max_tokens=800000

python $FAIRSEQ_ROOT/fairseq_cli/train.py ${DATA_DIR} \
    --save-dir ${MODEL_DIR} \
    --user-dir examples/speech_text_joint_to_text \
    --task speech_text_joint_to_text \
    --config-yaml config_ende.yaml \
    --train-subset "train_st" \
    --valid-subset "dev_st" \
    --load-speech-only \
    --fp16 \
    --seed 1 \
    \
    --ddp-backend no_c10d \
    --distributed-world-size ${world_size} \
    --tensorboard-logdir ${MODEL_DIR} \
    \
    --criterion guided_label_smoothed_cross_entropy_with_accuracy \
    --label-smoothing 0.3 \
    \
    --optimizer adam \
    --clip-norm 1.0 \
    --lr 3e-05 \
    --lr-scheduler polynomial_decay --warmup-updates 5000 \
    --max-update 50000 \
    --total-num-update 50000 \
    --update-freq ${update_freq} \
    \
    --max-tokens ${max_tokens} \
    --max-sentences 16 \
    --max-tokens-valid ${max_tokens} \
    --grouped-shuffling \
    --max-source-positions ${max_tokens} \
    --skip-invalid-size-inputs-valid-test \
    --num-workers 0 \
    --best-checkpoint-metric "acc" \
    --maximize-best-checkpoint-metric \
    \
    --arch "stbert_st_legacy" \
    --w2v-path ${w2v_path} \
    --add-decoder \
    --reuse-text-emb \
    --layerdrop 0.1 \
    --decoder-layerdrop 0.1 \
    --activation-dropout 0.0 \
    --attention-dropout 0.1 \
    --feature-grad-mult 1.0 \
    \
    --apply-mask --mask-prob 0.5 \
    \
    --log-format json \
    --log-interval 100 \
    --save-interval 1 \
    --keep-last-epochs 10 \
    --keep-best-checkpoints 5 \
    \
    2>&1 | tee ${MODEL_DIR}/train.log

sleep 5s

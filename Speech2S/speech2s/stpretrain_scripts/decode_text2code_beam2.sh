
#####################################
# Hubert ED model #
#####################################
[ $# -lt 1 ] && echo "Usage: $0 <init-model> <gen-set> <src> <tgt> <max_tokens> <world_size> <rank>" && exit 0
#source /mnt/default/v-ziqzhang/.bashrc_sing

model_path=$1
gen_set=$2
tgt=$3
src="ltr"
max_tokens=$4
word_size=$5
rank=$6
outdir=$7

[ -z $tgt ] && tgt="kmu"
[ -z $gen_set ] && gen_set="dev_clean"
[ -z $word_size ] && word_size=1
[ -z $rank ] && rank=0
[ -z $max_tokens ] && max_tokens=16000

FAIRSEQ_ROOT=/mnt/output/users/v-kunwei/code/fairseq_mlstku
DATA_DIR=/home/v-kunwei/
[ $gen_set == "test" ] && DATA_DIR=/mnt/output/users/v-kunwei/code/fairseq_mlstku
[ -z $outdir ] && outdir=$DATA_DIR


results_path=$outdir/pseudo_${gen_set}_${rank}
[ ! -d $results_path ] && mkdir -p $results_path

for subset in $gen_set; do
    python $FAIRSEQ_ROOT/fairseq_cli/generate_mt_label.py $DATA_DIR \
    --path ${model_path} \
    --task "translation_from_jst" \
    --max-target-positions 18000 \
    --gen-subset $subset \
    -t $tgt -s "ltr" \
    --dataset-impl "raw" \
    --max-tokens ${max_tokens} \
    --beam 2 \
    --max-len-a 3 --max-len-b 100 \
    --results-path $results_path \
    --distributed-world-size $word_size --distributed-rank $rank \
    
    echo "$model" > $results_path/model.record
    sleep 1s
done | tee $results_path/decode.log

sleep 2s

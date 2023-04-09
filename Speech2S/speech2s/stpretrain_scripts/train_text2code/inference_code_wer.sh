
#####################################
# Hubert ED model #
#####################################
[ $# -lt 1 ] && echo "Usage: $0 <init-model> <gen-set>" && exit 0

model_path=$1
src_dir=${model_path%/*}
cpt=${model_path##*/}
cpt=${cpt%.*}

gen_set=$2
tgt=$3
outdir=$4
src="ltr"
[ -z $tgt ] && tgt="kmu"
[ -z $gen_set ] && gen_set="en_dev"
[ -z $outdir ] && outdir=$src_dir/decode_${cpt}

# DATA_DIR=/mnt/default/v-ziqzhang/data/stbert/data/librispeech/hubert_release_iter2_layer9_kmeans/ltr-$tgt
# DATA_DIR=/mnt/default/v-ziqzhang/data/stbert/data/librispeech/speech2c_joint_splitenc_400k/ltr-$tgt
#DATA_DIR=/mnt/default/v-ziqzhang/data/stbert/data/librispeech/speech2c_400k/ltr-$tgt
DATA_DIR=/mnt/output/users/v-kunwei/data/s2s_data/es_asr_data/
FAIRSEQ_ROOT=/mnt/output/users/v-kunwei/code/fairseq_mlst

langs="ltr,$tgt"

for subset in $gen_set; do
    results_path=$outdir/${subset}
    [ ! -d $results_path ] && mkdir -p $results_path

    python $FAIRSEQ_ROOT/fairseq_cli/generate.py $DATA_DIR \
    --path ${model_path} \
    --task "translation_from_jst" \
    --max-target-positions 3000 \
    --gen-subset $subset \
    -t $tgt -s "ltr" --dataset-impl "raw" \
    --batch-size 16 \
    --max-len-a 2 --max-len-b 400 \
    --results-path $results_path \
    --scoring wer

    echo $results_path
    tail -n 1 $results_path/generate-*.txt
    sleep 1s
done

# --distributed-world-size 1000 --distributed-rank 0 \

sleep 2s

# cat generate-newstest2020_enja.txt | grep "^D-" | cut -d'-' -f 2- | sort -n -k1 | cut -f3 > decode-newstest2020_enja.txt
# sacrebleu -t wmt20 -l en-ja -i decode-newstest2020_enja.txt --tokenize char

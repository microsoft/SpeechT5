#####################################
# Hubert base model #
#####################################
[ $# -lt 1 ] && echo "Usage: $0 <init-model> <gen-set>" && exit 0

model_path=$1
src_dir=${model_path%/*}
cpt=${model_path##*/}
cpt=${cpt%.*}

#beam_size=$2
gen_set=$2
#lang=$4
[ -z $gen_set ] && gen_set="test_et"
[ -z $beam_size ] && beam_size=2
[ -z $lang ] && lang="fr"


#DATA_DIR=/mnt/output/users/v-kunwei/data/s2s_data/fin_enes
DATA_DIR=/home/v-kunwei
FAIRSEQ_ROOT=/mnt/output/users/v-kunwei/code/fairseq_mlstku

for subset in $gen_set; do
    results_path=$src_dir/decode_${cpt}_beam${beam_size}/${subset}
    [ ! -d $results_path ] && mkdir -p $results_path

    python $FAIRSEQ_ROOT/fairseq_cli/generate.py \
	    $DATA_DIR  --label-dir ${DATA_DIR} \
	    --labels '["spm"]' --gen-subset ${subset} \
            --max-tokens 9000000 --task hubert_pretraining \
	    --add-decoder --fine-tuning --random-crop \
	    --path ${model_path}  --results-path /home/v-kunwei --scoring sacrebleu  \
	    --max-len-a 0 --max-len-b 900 \
	    --beam 10 --single-target 
    
    tail -n 1 /home/v-kunwei/generate-*.txt
    sleep 1s
done

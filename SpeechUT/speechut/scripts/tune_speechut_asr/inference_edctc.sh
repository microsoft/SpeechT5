#####################################
# SpeechUT ASR model #
#####################################
[ $# -lt 2 ] && echo "Usage: $0 <model_path> <data_dir> [gen-set=dev_other] [beam_size=10] [ctc_weight=0.2]" && exit 1
[ ${PWD##*/} != SpeechUT ] && echo "Error: dir not match! Switch to SpeechUT/ and run it again!" && exit 1

model_path=$1
DATA_DIR=$2
gen_set=$3
beam_size=$4
ctc_weight=$5
world_size=$6
rank=$7
[ -z $gen_set ] && gen_set="dev_other"
[ -z $beam_size ] && beam_size=10
[ -z $ctc_weight ] && ctc_weight=0.2
[ -z $world_size ] && world_size=1
[ -z $rank ] && rank=0
[ $ctc_weight == 0 ] && [ $beam_size != 1 ] && echo "Change beam size to 1 as no ctc-decoding used..." && beam_size=1
[ $ctc_weight != 0 ] && extra="--batch-size 1"

src_dir=${model_path%/*}
cpt=${model_path##*/}
cpt=${cpt%.*}

CODE_ROOT=${PWD}

for subset in ${gen_set//,/ }; do
    results_path=$src_dir/decode_${cpt}/beam${beam_size}_ctc${ctc_weight}/${subset}_${world_size}_${rank}
    [ ! -d $results_path ] && mkdir -p $results_path

    python $CODE_ROOT/fairseq/fairseq_cli/generate.py $DATA_DIR \
    --user-dir $CODE_ROOT/speechut \
    --label-dir ${DATA_DIR} \
    --labels '["ltr"]' \
    --single-target \
    --post-process letter \
    --gen-subset ${subset} \
    --max-tokens 2000000 \
    \
    --task joint_sc2t_pretraining \
    --add-decoder-target \
    --fine-tuning \
    --pad-audio \
    --random-crop \
    \
    --ctc-weight ${ctc_weight} $extra \
    --beam ${beam_size} \
    \
    --path ${model_path} \
    --results-path $results_path \
    \
    --scoring wer --max-len-a 0.00078125 --max-len-b 200 \
    --distributed-world-size ${world_size} --distributed-rank ${rank} \
    &
done
wait


for subset in ${gen_set//,/ }; do
    results_path=$src_dir/decode_${cpt}/beam${beam_size}_ctc${ctc_weight}/${subset}_${world_size}_${rank}
    echo $results_path
    tail -n 1 $results_path/generate-*.txt
done

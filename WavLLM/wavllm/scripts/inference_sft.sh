export CUDA_VISIBLE_DEVICES=0
export HYDRA_FULL_ERROR=1
export PYTHONPATH=$$PYTHONPATH:${PWD}

model_path=$1
[ -z $model_path ] && model_path="?"

src_dir=${model_path%/*}
cpt=${model_path##*/}
cpt=${cpt%.*}

beam_size=$2
gen_set=$3
[ -z $gen_set ] && gen_set="?"
[ -z $beam_size ] && beam_size=1


FAIRSEQ_ROOT=${PWD}
DATA_DIR=$FAIRSEQ_ROOT/../wavllm/test_data

for subset in $gen_set; do
    results_path=$src_dir/decode_${cpt}_beam${beam_size}/${subset}
    [ ! -d $results_path ] && mkdir -p $results_path

    python $FAIRSEQ_ROOT/../wavllm/inference/generate.py $DATA_DIR \
    --user-dir ../wavllm \
    --tokenizer-path $FAIRSEQ_ROOT/../wavllm/tokenizer/tokenizer.model \
    --gen-subset ${subset} \
    \
    --task speechllm_task \
    \
    --path ${model_path} \
    --results-path $results_path \
    \
    --scoring wer \
    --skip-invalid-size-inputs-valid-test \
    --max-tokens 1600000 \
    --sampling --beam 1 --nbest 1 --temperature 0.5 \
    --max-len-a 0 --max-len-b 512
    # --beam ${beam_size} \
    #
    

    # echo $results_path
    # tail -n 1 $results_path/generate-*.txt
    # sleep 1s
done

    #--max-sample-size 2000000 \
    # --max-len-a 0 --max-len-b 512 \
    # --skip-invalid-size-inputs-valid-test \
    # --max-sentences 20 \

export CUDA_VISIBLE_DEVICES=0
export HYDRA_FULL_ERROR=1
export PYTHONPATH=$$PYTHONPATH:${PWD}

model_path=$1
[ -z $model_path ] && model_path="?"

src_dir=${model_path%/*}
cpt=${model_path##*/}
cpt=${cpt%.*}

gen_set=$2
[ -z $gen_set ] && gen_set="?"
[ -z $beam_size ] && beam_size=1


FAIRSEQ_ROOT=${PWD}
DATA_DIR=$FAIRSEQ_ROOT/examples/wavllm/test_data

for subset in $gen_set; do
    results_path=$src_dir/decode_${cpt}_beam${beam_size}/${subset}
    [ ! -d $results_path ] && mkdir -p $results_path

    python $FAIRSEQ_ROOT/examples/wavllm/inference/generate.py $DATA_DIR \
    --user-dir examples/wavllm \
    --tokenizer-path $FAIRSEQ_ROOT/examples/wavllm/tokenizer/tokenizer.model \
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
done
# WavLLM
## Setup

```bash
git submodule update --init WavLLM/fairseq
cd WavLLM/
conda create -n wavllm python=3.10.0
conda activate wavllm
pip install --editable fairseq/
pip install sentencepiece
pip install transformers==4.32.1
pip install numpy==1.23.5
pip install editdistance
pip install soundfile
```

## Inference
```bash
cp -r wavllm fairseq/examples
cd fairseq
bash examples/wavllm/scripts/inference_sft.sh $model_path $data_name
```

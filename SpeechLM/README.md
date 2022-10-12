# SpeechLM

<!--**Pre-trained models for speech related tasks**-->

 [**SpeechLM**](https://arxiv.org/abs/2209.15329): **Enhanced Speech Pre-Training with Unpaired Textual Data**


- The code and checkpoints will be released here.
- Oct 2022: release preprint in [arXiv](https://arxiv.org/abs/2209.15329)
- (In progress) Oct 2022: release the code and models

## Pre-Trained and Fine-tuned Models

|  Model            |               Pre-training Dataset                                                                            | Fine-tuning Dataset                                               | Model |
| :------:          | :----------------------------------------------:                                                              | :-----------------:                                               | :-----: |
| SpeechLM-P Base   | [960 hrs LibriSpeech](http://www.openslr.org/12) + [40M Text](http://www.openslr.org/11)                      |                      -                                            | [Google drive](https://drive.google.com/file/d/1iJvhSGghNrMT-wAY1nwVu2YaYuTy1pxx/view?usp=sharing)  |
| SpeechLM-P Base   | [960 hrs LibriSpeech](http://www.openslr.org/12) + [40M Text](http://www.openslr.org/11)                      | [100 hrs LibriSpeech](http://www.openslr.org/12)                  | [Google drive](https://drive.google.com/file/d/1mH3N7iKMWYk3rSBJErQPYf3x5ugqDq5x/view?usp=sharing)  |
| SpeechLM-H Base   | [960 hrs LibriSpeech](http://www.openslr.org/12) + [40M Text](http://www.openslr.org/11)                      |                      -                                            | [Coming]()  |
| SpeechLM-H Base   | [960 hrs LibriSpeech](http://www.openslr.org/12) + [40M Text](http://www.openslr.org/11)                      | [100 hrs LibriSpeech](http://www.openslr.org/12)                  | [Coming]()  |
| SpeechLM-P Large  | [60k hrs LibriLight](https://github.com/facebookresearch/libri-light) + [40M Text](http://www.openslr.org/11) |                      -                                            | [Google drive](https://drive.google.com/file/d/1QjLIgTJKIylVIp5hUkfSjGPtz8Xo7Lky/view?usp=sharing)  |
| SpeechLM-P Large  | [60k hrs LibriLight](https://github.com/facebookresearch/libri-light) + [40M Text](http://www.openslr.org/11) | [960 hrs LibriSpeech](http://www.openslr.org/12)                  | [Google drive](https://drive.google.com/file/d/1YZQDVv096o8Opt0RBnkRiZXYPRDqKZnP/view?usp=sharing)  |
| SpeechLM-P Large  | [60k hrs LibriLight](https://github.com/facebookresearch/libri-light) + [40M Text](http://www.openslr.org/11) | [En-De CoVoST-2](https://github.com/facebookresearch/covost)      | [Google drive](https://drive.google.com/file/d/1qYygNWSc11TQbBI1OzC4ChlR-dNh8t9S/view?usp=sharing)  |
| SpeechLM-P Large  | [60k hrs LibriLight](https://github.com/facebookresearch/libri-light) + [40M Text](http://www.openslr.org/11) | [En-Ca CoVoST-2](https://github.com/facebookresearch/covost)      | [Google drive](https://drive.google.com/file/d/162U88mwso2aVfzzPkEM2nP_vwTpcb57T/view?usp=sharing)  |
| SpeechLM-P Large  | [60k hrs LibriLight](https://github.com/facebookresearch/libri-light) + [40M Text](http://www.openslr.org/11) | [En-Ar CoVoST-2](https://github.com/facebookresearch/covost)      | [Google drive](https://drive.google.com/file/d/1lbTSRXewEeb2t45URunD6EiJcbniyjWW/view?usp=sharing)  |
| SpeechLM-P Large  | [60k hrs LibriLight](https://github.com/facebookresearch/libri-light) + [40M Text](http://www.openslr.org/11) | [En-Tr CoVoST-2](https://github.com/facebookresearch/covost)      | [Google drive](https://drive.google.com/file/d/1Er4I_jHS175pQQph223yKtiiLQ378VvH/view?usp=sharing)  |

<!-- 
| SpeechLM-P Base   | [960 hrs LibriSpeech](http://www.openslr.org/12)                      | [En-De CoVoST-2](https://github.com/facebookresearch/covost)      | [Google drive]()  |
| SpeechLM-P Base   | [960 hrs LibriSpeech](http://www.openslr.org/12)                      | [En-Ca CoVoST-2](https://github.com/facebookresearch/covost)      | [Google drive]()  |
| SpeechLM-P Base   | [960 hrs LibriSpeech](http://www.openslr.org/12)                      | [En-Ar CoVoST-2](https://github.com/facebookresearch/covost)      | [Google drive]()  |
| SpeechLM-P Base   | [960 hrs LibriSpeech](http://www.openslr.org/12)                      | [En-Tr CoVoST-2](https://github.com/facebookresearch/covost)      | [Google drive]()  | 
-->
## Setup
```
git submodule update --init SpeechLM/fairseq
cd SpeechLM/
pip install --editable fairseq/
pip install sacrebleu==1.5.1
```

## ASR on LibriSpeech
### Data preparation
Please follow the steps of wav2vec 2.0 manifest [here](https://github.com/pytorch/fairseq/tree/main/examples/wav2vec#prepare-training-data-manifest) to prepare `train.tsv` and `train.ltr`. We also provided exmples [here](https://github.com/microsoft/SpeechT5/tree/main/SpeechLM/dataset/LibriSpeech/asr/). You should make sure the vocabulary [`dict.ltr.txt`](https://github.com/microsoft/SpeechT5/tree/main/SpeechLM/dataset/LibriSpeech/asr/dict.ltr.txt) is the same as that used for the pre-trained model.
### Fine-tuning a CTC model
- Fine-tuning the base model
```
# Usage: speechlm/scripts/tune_speechlm_asr/finetune_base_ctc.sh <model_path> <data_dir> <cpt_tag> [mount=$PWD] [world_size=8] [update_freq=1]
model_path=path/to/your/pre-trained/model
data_dir=dataset/LibriSpeech/asr
bash speechlm/scripts/tune_speechlm_asr/finetune_base_ctc.sh $model_path $data_dir 'tag400k'
```
- Fine-tuning the large model
```
# Usage: speechlm/scripts/tune_speechlm_asr/finetune_large_ctc.sh <model_path> <data_dir> <cpt_tag> [mount=$PWD] [world_size=8] [update_freq=4]
model_path=path/to/your/pre-trained/model
data_dir=dataset/LibriSpeech/asr
bash speechlm/scripts/tune_speechlm_asr/finetune_large_ctc.sh $model_path $data_dir 'tag400k'
```
### Decoding:
- Directly decode a CTC model.
```
# Usage: speechlm/scripts/tune_speechlm_asr/inference_ctc.sh <model_path> <data_dir> [gen-set=dev_clean,dev_other,test_clean,test_other]
model_path=path/to/your/fine-tuned/model
data_dir=dataset/LibriSpeech/asr
bash speechlm/scripts/tune_speechlm_asr/inference_ctc.sh $model_path $data_dir
# for large models
# bash speechlm/scripts/tune_speechlm_asr/inference_ctc_large.sh $model_path $data_dir
```
- Decoding with 4-gram language model using [flashlight](https://github.com/flashlight/flashlight/tree/main/bindings/python) and [kenlm](https://github.com/kpu/kenlm).
> please put [4-gram.arpa](https://www.openslr.org/resources/11/4-gram.arpa.gz) and the word-to-letter lexicon [librispeech_lexicon.lst](https://drive.google.com/file/d/1q7IbNGqtwXnctjvuvpviQ4ZmepFHQmTO/view?usp=sharing) into `$data_dir`.
```
# Usage: speechlm/scripts/tune_speechlm_asr/inference_ctc_kenlm.sh <model_path> <data_dir> [gen-set=dev_clean,dev_other,test_clean,test_other]
model_path=path/to/your/fine-tuned/model
data_dir=dataset/LibriSpeech/asr
bash speechlm/scripts/tune_speechlm_asr/inference_ctc_kenlm.sh $model_path $data_dir
```
- Decoding large models with fairseq-lm using [flashlight](https://github.com/flashlight/flashlight/tree/main/bindings/python).
> please put [lm_librispeech_word_transformer.pt](https://dl.fbaipublicfiles.com/wav2letter/sota/2019/lm/lm_librispeech_word_transformer.pt) and its vocabulary [`dict.txt`](https://dl.fbaipublicfiles.com/wav2letter/sota/2019/lm/lm_librispeech_word_transformer.dict) into `$data_dir/fairseq_word_lm`, and the word-to-letter lexicon [librispeech_lexicon.lst](https://drive.google.com/file/d/1q7IbNGqtwXnctjvuvpviQ4ZmepFHQmTO/view?usp=sharing) into `$data_dir`.
Capitalize the `dict.txt` to amke it compatible with the word-to-letter lexicon.
```
# Usage: speechlm/scripts/tune_speechlm_asr/inference_ctc_large_fsqlm.sh <model_path> <data_dir> [gen-set=dev_clean,dev_other,test_clean,test_other]
model_path=path/to/your/fine-tuned/model
data_dir=dataset/LibriSpeech/asr
bash speechlm/scripts/tune_speechlm_asr/inference_ctc_large_fsqlm.sh $model_path $data_dir dev_other
```

## ST on CoVoST-2
### Data Preparation
### Fine-tuning a encoder-decoder model
- Fine-tuning the base model
```
```
- Fine-tuning the large model
```
```
### Decoding
- Decoding the base model
```
```
- Decoding the large model
```
```
## Pre-training
### Data preparation
We put examples in [dataset](https://github.com/microsoft/SpeechT5/tree/main/SpeechLM/dataset).
- **Speech:** please follow the steps of wav2vec 2.0 manifest [here](https://github.com/pytorch/fairseq/tree/main/examples/wav2vec#prepare-training-data-manifest) to prepare [`train.tsv`](https://github.com/microsoft/SpeechT5/tree/main/SpeechLM/dataset/LibriSpeech/phone_unit/train_sample100.tsv)
- **Phoneme units for speech:** use phoneme-unit tokenizer to process the speech to prepare [`train.phn`](https://github.com/microsoft/SpeechT5/tree/main/SpeechLM/dataset/LibriSpeech/phone_unit/train_sample100.phn)
- **Hidden units for speech:** use hidden-unit tokenizer to process the speech to prepare [`train.km`](https://github.com/microsoft/SpeechT5/tree/main/SpeechLM/dataset/LibriSpeech/hidden_unit/train_sample100.km)
> The hidden-unit tokenizer used in this word is [a K-means model on the top of the Hubert Base model](https://github.com/facebookresearch/fairseq/tree/main/examples/hubert/simple_kmeans).
- Create dict for the target units [`dict.phn.txt`](https://github.com/microsoft/SpeechT5/tree/main/SpeechLM/dataset/LibriSpeech/phone_unit/dict.phn.txt) or [`dict.km.txt`](https://github.com/microsoft/SpeechT5/tree/main/SpeechLM/dataset/LibriSpeech/hidden_unit/dict.km.txt)

- **Text (phoneme-unit):** the following scripts will convert the unpaired text from LibriSpeech LM corpus to paired (phonemes, letters) data `train_text.phn-ltr.{phn,ltr}.{bin,idx}`. The dictionaries are provided [`here`](https://github.com/microsoft/SpeechT5/tree/main/SpeechLM/dataset/LibriLM/bin-idx) and the kaldi-processed lexicon is provided [here](https://drive.google.com/file/d/1QVeyCpLXLnujBUAickpo-jaSVY-vKLnT/view?usp=sharing).
```
cd SpeechT5/SpeechLM
bash speechlm/data_process/prepare_phn2ltr_librilm.sh
```
### Pre-training SpeechLM-P Base model
```
```
### Pre-training SpeechLM-H Base model
```
```
### Pre-training SpeechLM-P Large model
```
```


## Tokenizers

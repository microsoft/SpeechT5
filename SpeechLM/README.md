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
Please follow the steps of wav2vec 2.0 manifest [here](https://github.com/pytorch/fairseq/tree/main/examples/wav2vec#prepare-training-data-manifest) to prepare `train.tsv` and `train.ltr`.
### Fine-tuning a CTC model
- Fine-tuning the base model
```
```
- Fine-tuning the large model
```
```
### Decoding:
- Directly decode a CTC model
```
```
- Decoding with 4-gram language model using flashlight and kenlm
```
```
- Decoding with fairseq-lm with large model using flashlight
```
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

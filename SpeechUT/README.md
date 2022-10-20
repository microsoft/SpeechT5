# SpeechUT
SpeechUT
<!--**Pre-trained models for speech related tasks**-->

 [**SpeechUT: Bridging Speech and Text with Hidden-Unit for Encoder-Decoder Based Speech-Text Pre-training**](https://arxiv.org/abs/2210.03730)


- (Updating) Oct 2022: release the code and models
- Oct 2022: release preprint in [arXiv](https://arxiv.org/abs/2210.03730)

## Pre-Trained and Fine-tuned Models
|  Model                |               Pre-training Dataset (unlabeled)                                                                                                    | Fine-tuning Dataset (labeled)                     | Model |
| :------:              | :----------------------------------------------:                                                                                                  | :-----------------:                               | :-----: |
| SpeechUT Base (ASR)   | [960 hrs LibriSpeech](http://www.openslr.org/12) + [40M Text](http://www.openslr.org/11)                                                          |                      -                            | [Preparing]|
| SpeechUT Base (ASR)   | [960 hrs LibriSpeech](http://www.openslr.org/12) + [40M Text](http://www.openslr.org/11)                                                          | [100 hrs LibriSpeech](http://www.openslr.org/12)  | [Preparing]|
| SpeechUT Large (ASR)  | [60k hrs LibriSpeech](http://www.openslr.org/12) + [40M Text](http://www.openslr.org/11)                                                          |                      -                            | [Preparing]|
| SpeechUT Large (ASR)  | [60k hrs LibriSpeech](http://www.openslr.org/12) + [40M Text](http://www.openslr.org/11)                                                          | [100 hrs LibriSpeech](http://www.openslr.org/12)  | [Preparing]|
| SpeechUT Large (ASR)  | [60k hrs LibriSpeech](http://www.openslr.org/12) + [40M Text](http://www.openslr.org/11)                                                          | [960 hrs LibriSpeech](http://www.openslr.org/12)  | [Preparing]|
| SpeechUT Base (En-De) | [960 hrs LibriSpeech](http://www.openslr.org/12) + [408 hrs MuST-C v1](https://ict.fbk.eu/must-c/) + [4.6M Text](https://www.statmt.org/wmt16/)   |                      -                            | [Preparing]|
| SpeechUT Base (En-De) | [960 hrs LibriSpeech](http://www.openslr.org/12) + [408 hrs MuST-C v1](https://ict.fbk.eu/must-c/) + [4.6M Text](https://www.statmt.org/wmt16/)   | [En-De MuST-C v1](https://ict.fbk.eu/must-c/)     | [Preparing]|
| SpeechUT Base (En-Es) | [960 hrs LibriSpeech](http://www.openslr.org/12) + [504 hrs MuST-C v1](https://ict.fbk.eu/must-c/) + [15M Text](https://www.statmt.org/wmt13/)    |                      -                            | [Preparing]|
| SpeechUT Base (En-Es) | [960 hrs LibriSpeech](http://www.openslr.org/12) + [504 hrs MuST-C v1](https://ict.fbk.eu/must-c/) + [15M Text](https://www.statmt.org/wmt13/)    | [En-Es MuST-C v1](https://ict.fbk.eu/must-c/)     | [Preparing]|
| SpeechUT Base (En-Fr) | [960 hrs LibriSpeech](http://www.openslr.org/12) + [492 hrs MuST-C v1](https://ict.fbk.eu/must-c/) + [40M Text](https://www.statmt.org/wmt14/)    |                      -                            | [Preparing]|
| SpeechUT Base (En-Fr) | [960 hrs LibriSpeech](http://www.openslr.org/12) + [492 hrs MuST-C v1](https://ict.fbk.eu/must-c/) + [40M Text](https://www.statmt.org/wmt14/)    | [En-Fr MuST-C v1](https://ict.fbk.eu/must-c/)     | [Preparing]|


## Setup
To fine-tune or pre-train more models, please follow the instructions below.

```bash
git submodule update --init SpeechUT/fairseq
cd SpeechUT/
pip install --editable fairseq/
pip install sacrebleu==1.5.1
```


## Fine-tune ASR on LibriSpeech
### Data preparation
Please follow the steps of wav2vec 2.0 manifest [here](https://github.com/pytorch/fairseq/tree/main/examples/wav2vec#prepare-training-data-manifest) to prepare `train.tsv` and `train.ltr`. You should make sure the vocabulary [`dict.ltr.txt`](dataset/LibriSpeech/dict.ltr.txt) is the same as that used for the pre-trained model. Put yout prepared data into `$data_dir`.

### Fine-tune a hybrid CTC-ED model
- Fine-tune the base model
    ```bash
    # Usage: speechut/scripts/tune_speechut_asr/finetune_base_edctc.sh <model_path> <data_dir> <cpt_tag> [mount=$PWD] [world_size=8] [update_freq=2]
    model_path=path/to/your/pre-trained/model
    data_dir=dataset/LibriSpeech/asr
    bash speechut/scripts/tune_speechut_asr/finetune_base_edctc.sh $model_path $data_dir 'tag400k'
    ```
### Decode
- CTC-ED joint decoding
    ```bash
    # Usage: speechut/scripts/tune_speechut_asr/inference_edctc.sh <model_path> <data_dir> [gen-set=dev_other] [beam_size=10] [ctc_weight=0.2]
    model_path=path/to/your/fine-tuned/model
    data_dir=dataset/LibriSpeech/asr
    bash speechlm/scripts/tune_speechlm_asr/inference_edctc.sh $model_path $data_dir test_clean
    ```
    We use the [espnet](https://github.com/espnet/espnet)-style joint decoding algorithm, currently only supporting batch_size=1. If you find it too slow, please check [`inference_nj.sh`](speechut/scripts/tune_speechut_asr/inference_nj.sh) for a multi-thread version.

- CTC-ED joint decoding with LM
    
    Updating


## Fine-tune ST on MuST-C

    Updating


## Pre-train

    Updating


## T2U Generator

    Updating

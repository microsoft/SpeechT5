# SpeechLM

<!--**Pre-trained models for speech related tasks**-->

 [**SpeechLM: Enhanced Speech Pre-Training with Unpaired Textual Data**](https://arxiv.org/abs/2209.15329)


- (Updating) Oct 2022: release the code and models
- Oct 2022: release preprint in [arXiv](https://arxiv.org/abs/2209.15329)

## Pre-Trained and Fine-tuned Models

|  Model            |               Pre-training Dataset                                                                            | Fine-tuning Dataset                                               | Model |
| :------:          | :----------------------------------------------:                                                              | :-----------------:                                               | :-----: |
| SpeechLM-P Base   | [960 hrs LibriSpeech](http://www.openslr.org/12) + [40M Text](http://www.openslr.org/11)                      |                      -                                            | [Google drive](https://drive.google.com/file/d/1iJvhSGghNrMT-wAY1nwVu2YaYuTy1pxx/view?usp=sharing)  |
| SpeechLM-P Base   | [960 hrs LibriSpeech](http://www.openslr.org/12) + [40M Text](http://www.openslr.org/11)                      | [100 hrs LibriSpeech](http://www.openslr.org/12)                  | [Google drive](https://drive.google.com/file/d/1mH3N7iKMWYk3rSBJErQPYf3x5ugqDq5x/view?usp=sharing)  |
| SpeechLM-H Base   | [960 hrs LibriSpeech](http://www.openslr.org/12) + [40M Text](http://www.openslr.org/11)                      |                      -                                            | [Google drive](https://drive.google.com/file/d/1eblW8U8f9t-NTuCNRrNHwr-8BeLAUAmQ/view?usp=sharing)  |
| SpeechLM-H Base   | [960 hrs LibriSpeech](http://www.openslr.org/12) + [40M Text](http://www.openslr.org/11)                      | [100 hrs LibriSpeech](http://www.openslr.org/12)                  | [Google drive](https://drive.google.com/file/d/1vXyO5DolbiWiTYZ6pkkKQsu2wJetaPlv/view?usp=sharing)  |
| SpeechLM-P Base   | [960 hrs LibriSpeech](http://www.openslr.org/12) + [40M Text](http://www.openslr.org/11)                      | [En-De CoVoST-2](https://github.com/facebookresearch/covost)      | [Azure Storage](https://msranlcmtteamdrive.blob.core.windows.net/share/speechlm/finetune_covost/checkpoint_ende.pt?sv=2020-10-02&st=2022-10-13T06%3A15%3A13Z&se=2023-10-14T06%3A15%3A00Z&sr=c&sp=rl&sig=b0RFMEsG5ch3OA2%2FTTsoG5DFnLTE9c%2BeHKFJKcHI3Xg%3D)  |
| SpeechLM-P Base   | [960 hrs LibriSpeech](http://www.openslr.org/12) + [40M Text](http://www.openslr.org/11)                      | [En-Ca CoVoST-2](https://github.com/facebookresearch/covost)      | [Azure Storage](https://msranlcmtteamdrive.blob.core.windows.net/share/speechlm/finetune_covost/checkpoint_enca.pt?sv=2020-10-02&st=2022-10-13T06%3A15%3A13Z&se=2023-10-14T06%3A15%3A00Z&sr=c&sp=rl&sig=b0RFMEsG5ch3OA2%2FTTsoG5DFnLTE9c%2BeHKFJKcHI3Xg%3D)  |
| SpeechLM-P Base   | [960 hrs LibriSpeech](http://www.openslr.org/12) + [40M Text](http://www.openslr.org/11)                      | [En-Ar CoVoST-2](https://github.com/facebookresearch/covost)      | [Azure Storage](https://msranlcmtteamdrive.blob.core.windows.net/share/speechlm/finetune_covost/checkpoint_enar.pt?sv=2020-10-02&st=2022-10-13T06%3A15%3A13Z&se=2023-10-14T06%3A15%3A00Z&sr=c&sp=rl&sig=b0RFMEsG5ch3OA2%2FTTsoG5DFnLTE9c%2BeHKFJKcHI3Xg%3D)  |
| SpeechLM-P Base   | [960 hrs LibriSpeech](http://www.openslr.org/12) + [40M Text](http://www.openslr.org/11)                      | [En-Tr CoVoST-2](https://github.com/facebookresearch/covost)      | [Azure Storage](https://msranlcmtteamdrive.blob.core.windows.net/share/speechlm/finetune_covost/checkpoint_entr.pt?sv=2020-10-02&st=2022-10-13T06%3A15%3A13Z&se=2023-10-14T06%3A15%3A00Z&sr=c&sp=rl&sig=b0RFMEsG5ch3OA2%2FTTsoG5DFnLTE9c%2BeHKFJKcHI3Xg%3D)  |
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
### Decoding
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
    > Please put [4-gram.arpa](https://www.openslr.org/resources/11/4-gram.arpa.gz) and the word-to-letter lexicon [librispeech_lexicon.lst](https://drive.google.com/file/d/1q7IbNGqtwXnctjvuvpviQ4ZmepFHQmTO/view?usp=sharing) into `$data_dir`.
    ```
    # Usage: speechlm/scripts/tune_speechlm_asr/inference_ctc_kenlm.sh <model_path> <data_dir> [gen-set=dev_clean,dev_other,test_clean,test_other]
    model_path=path/to/your/fine-tuned/model
    data_dir=dataset/LibriSpeech/asr
    bash speechlm/scripts/tune_speechlm_asr/inference_ctc_kenlm.sh $model_path $data_dir
    ```
- Decoding large models with fairseq-lm using [flashlight](https://github.com/flashlight/flashlight/tree/main/bindings/python).
    > Please put [lm_librispeech_word_transformer.pt](https://dl.fbaipublicfiles.com/wav2letter/sota/2019/lm/lm_librispeech_word_transformer.pt) and its vocabulary [`dict.txt`](https://dl.fbaipublicfiles.com/wav2letter/sota/2019/lm/lm_librispeech_word_transformer.dict) into `$data_dir/fairseq_word_lm`, and the word-to-letter lexicon [librispeech_lexicon.lst](https://drive.google.com/file/d/1q7IbNGqtwXnctjvuvpviQ4ZmepFHQmTO/view?usp=sharing) into `$data_dir`. Capitalize the `dict.txt` to amke it compatible with the word-to-letter lexicon.
    ```
    # Usage: speechlm/scripts/tune_speechlm_asr/inference_ctc_large_fsqlm.sh <model_path> <data_dir> [gen-set=dev_clean,dev_other,test_clean,test_other]
    model_path=path/to/your/fine-tuned/model
    data_dir=dataset/LibriSpeech/asr
    bash speechlm/scripts/tune_speechlm_asr/inference_ctc_large_fsqlm.sh $model_path $data_dir dev_other
    ```

## ST on CoVoST-2
### Data Preparation
1. Download [Common Voice audio clips](https://commonvoice.mozilla.org/en/datasets) (version 4) for English into `$cv_root/en`.
2. Get data manifest. The following script will convert mp3 files to waveform, create tsv file containing speech/translation paires, create data config files. We provide examples [here](https://github.com/microsoft/SpeechT5/tree/main/SpeechLM/dataset/CommonVoice/v4/en/en-de).
    ```
    lang=de # ca,ar,tr
    cv_root=dataset/CommonVoice/v4
    bash speechlm/data_process/prepare_covost2_enxx.sh $lang $cv_root
    ```
### Fine-tuning a encoder-decoder model
- Fine-tuning the Base model (fine-tuned models will be stored in `$mount/exp/finetune_covost`).

    ```
    model_path=path/to/your/pre-trained/model
    lang=de # ca,ar,tr
    data_dir=dataset/CommonVoice/v4/en/en-${lang}
    # Usage (Base model): speechlm/scripts/tune_speechlm_st/ft_base_covost_enxx.sh <model_path> <data_dir> <lang> <cpt-tag> [mount=$PWD] [world_size=8] [update_freq=2]
    bash speechlm/scripts/tune_speechlm_st/ft_base_covost_enxx.sh $model_path $data_dir $lang 'tag400k'
    ```
- Fine-tuning the Large model (fine-tuned models will be stored in `$mount/exp/finetune_covost`).
    ```
    # Usage (Large model): speechlm/scripts/tune_speechlm_st/ft_large_covost_enxx.sh <model_path> <data_dir> <lang> <cpt-tag> [mount=$PWD] [world_size=8] [update_freq=4]
    bash speechlm/scripts/tune_speechlm_st/ft_large_covost_enxx.sh $model_path $data_dir $lang 'tag400k'
    ```

### Decoding
- Decoding the base model
    ```
    # Usage: speechlm/scripts/tune_speechlm_st/inference_base.sh <model_path> <data_dir> <lang> [gen-set=dev] [beam_size=5]
    model_path=path/to/your/fine-tuned/model
    lang=de # ca,ar,tr
    data_dir=dataset/CommonVoice/v4/en/en-${lang}
    bash speechlm/scripts/tune_speechlm_st/inference_base.sh $model_path $data_dir $lang dev
    ```
- Decoding the large model
    ```
    # Usage: speechlm/scripts/tune_speechlm_st/inference_large.sh <model_path> <data_dir> <lang> [gen-set=dev] [beam_size=5]
    bash speechlm/scripts/tune_speechlm_st/inference_large.sh $model_path $data_dir $lang dev
    ```

<!-- ## Pre-training
### Data preparation
We put examples in [dataset](https://github.com/microsoft/SpeechT5/tree/main/SpeechLM/dataset).
- **Speech:** please follow the steps of wav2vec 2.0 manifest [here](https://github.com/pytorch/fairseq/tree/main/examples/wav2vec#prepare-training-data-manifest) to prepare [`train.tsv`](https://github.com/microsoft/SpeechT5/tree/main/SpeechLM/dataset/LibriSpeech/phone_unit/train_sample100.tsv)
- **Phoneme units for speech:** use phoneme-unit tokenizer to process the speech to prepare [`train.phn`](https://github.com/microsoft/SpeechT5/tree/main/SpeechLM/dataset/LibriSpeech/phone_unit/train_sample100.phn)
- **Hidden units for speech:** use hidden-unit tokenizer to process the speech to prepare [`train.km`](https://github.com/microsoft/SpeechT5/tree/main/SpeechLM/dataset/LibriSpeech/hidden_unit/train_sample100.km)
> The hidden-unit tokenizer used in this word is [a K-means model on the top of the Hubert Base model](https://github.com/facebookresearch/fairseq/tree/main/examples/hubert/simple_kmeans).
- Create dict for the target units [`dict.phn.txt`](https://github.com/microsoft/SpeechT5/tree/main/SpeechLM/dataset/LibriSpeech/phone_unit/dict.phn.txt) or [`dict.km.txt`](https://github.com/microsoft/SpeechT5/tree/main/SpeechLM/dataset/LibriSpeech/hidden_unit/dict.km.txt)

- **Text (phoneme-unit):**  -->
## Pre-training
- SpeechLM-P Base model

    Models will be stored in `$mount/pretrain`.
    ```
    data_dir=dataset/LibriSpeech/phone_unit   # should contain train_960.{tsv,phn}
    text_data_dir=dataset/LibriLM/phone_unit/bin-idx     # should contain train_text.phn-ltr.{phn,ltr}.{bin,idx}
    # Usage: speechlm/scripts/pretrain_speechlm/base_speechlmp.sh <data_dir> <text_data_dir> [mount=$PWD] [world_size=32] [update_freq=1]
    bash speechlm/scripts/pretrain_speechlm/base_speechlmp.sh $data_dir $text_data_dir
    ```
- SpeechLM-H Base model
    ```
    data_dir=dataset/LibriSpeech/hidden_unit  # should contain train_960.{tsv,phn}
    text_data_dir=dataset/LibriLM/km-ltr/bin-idx     # should contain train_text.km-ltr.{km,ltr}.{bin,idx}
    # Usage: speechlm/scripts/pretrain_speechlm/base_speechlmh.sh <data_dir> <text_data_dir> [mount=$PWD] [world_size=32] [update_freq=1]
    bash speechlm/scripts/pretrain_speechlm/base_speechlmp.sh $data_dir $text_data_dir
    ```
- SpeechLM-P Large model
    ```
    data_dir=dataset/LibriSpeech/phone_unit   # should contain train_960.{tsv,phn}
    text_data_dir=dataset/LibriLM/phone_unit/bin-idx     # should contain train_text.phn-ltr.{phn,ltr}.{bin,idx}
    # Usage: speechlm/scripts/pretrain_speechlm/base_speechlmp.sh <data_dir> <text_data_dir> [mount=$PWD] [world_size=32] [update_freq=1]
    bash speechlm/scripts/pretrain_speechlm/large_speechlmp.sh $data_dir $text_data_dir
    ```


## Tokenizers
### Phoneme-unit Tokenizer for Speech
This tokenizer is used to produce the frame-laigned phonemes for unlabeled speech, which is actually a hybrid HMM ASR model.

In the Base setting, we use 100h LibriSpeech labeled data to train the HMM model under Kaldi recipe, then decode the unpaired speech and get the aligned phonemes from the lattice.

We provide the processed phonemes of 960h speech here: [`train_960.tsv`](), [`train_960.phn`](), [`dev_clean.tsv`](), [`dev_clean.phn`](). Note that the label-rate is 100 (10ms).

### Phoneme-unit Tokenizer for Text
This tokenizer is used to phonemize the unpaired text data to (phonemes, letters) paired data, following a `words -> phonemes -> upsampled phones` pipeline.

The following script will download LibriSpeech LM corpus and produce the required data: `train_text.phn-ltr.phn.{idx,bin}` and `train_text.phn-ltr.ltr.{idx,bin}`. 
> Before runing it, make sure you have our provided [`dcit.phn.txt`](https://github.com/microsoft/SpeechT5/tree/main/SpeechLM/dataset/LibriLM/phone_unit/bin-idx/dcit.phn.txt) and [`dcit.ltr.txt`](https://github.com/microsoft/SpeechT5/tree/main/SpeechLM/dataset/LibriLM/phone_unit/bin-idx/dcit.ltr.txt) in the output dir `dataset/LibriLM/phone_unit/bin-idx/`.
```
# data will be in dataset/LibriLM/phone_unit/
bash speechlm/data_process/prepare_phn2ltr_librilm.sh
```
### Hidden-unit Tokenizer for Speech
Please follow the steps of wav2vec 2.0 manifest [here](https://github.com/pytorch/fairseq/tree/main/examples/wav2vec#prepare-training-data-manifest) to prepare 1) wav recordings [`train.tsv`](https://github.com/microsoft/SpeechT5/tree/main/SpeechLM/dataset/LibriSpeech/hidden_unit/train_sample100.tsv) and 2) corresponding hidden-units [`train.km`](https://github.com/microsoft/SpeechT5/tree/main/SpeechLM/dataset/LibriSpeech/hidden_unit/train_sample100.km), and 3) unit vocabulary [`dict.km.txt`](https://github.com/microsoft/SpeechT5/tree/main/SpeechLM/dataset/LibriSpeech/hidden_unit/dict.km.txt).

<!-- ### Hidden-unit Tokenizer for Text -->

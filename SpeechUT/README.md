# SpeechUT
<!--**Pre-trained models for speech related tasks**-->

 [**SpeechUT: Bridging Speech and Text with Hidden-Unit for Encoder-Decoder Based Speech-Text Pre-training**](https://arxiv.org/abs/2210.03730)


- (Done) Oct 2022: release the code and models
- Oct 2022: release preprint in [arXiv](https://arxiv.org/abs/2210.03730)

## Pre-Trained and Fine-tuned Models
|  Model                |               Pre-training Dataset (unlabeled)                                                                                                    | Fine-tuning Dataset (labeled)                     | Model |
| :------:              | :----------------------------------------------:                                                                                                  | :-----------------:                               | :-----: |
| SpeechUT Base (ASR)   | [960 hrs LibriSpeech](http://www.openslr.org/12) + [40M Text](http://www.openslr.org/11)                                                          |                      -                            | [Azure Storage](https://msranlcmtteamdrive.blob.core.windows.net/share/speechut/base_speechut4asr_32gpu_1accum/checkpoint_298_400000.pt?sv=2020-10-02&st=2022-10-13T06%3A15%3A13Z&se=2023-10-14T06%3A15%3A00Z&sr=c&sp=rl&sig=b0RFMEsG5ch3OA2%2FTTsoG5DFnLTE9c%2BeHKFJKcHI3Xg%3D)|
| SpeechUT Base (ASR)   | [960 hrs LibriSpeech](http://www.openslr.org/12) + [40M Text](http://www.openslr.org/11)                                                          | [100 hrs LibriSpeech](http://www.openslr.org/12)  | [Azure Storage](https://msranlcmtteamdrive.blob.core.windows.net/share/speechut/speechut_base_asr100h_checkpoint_best.pt?sv=2020-10-02&st=2022-10-13T06%3A15%3A13Z&se=2023-10-14T06%3A15%3A00Z&sr=c&sp=rl&sig=b0RFMEsG5ch3OA2%2FTTsoG5DFnLTE9c%2BeHKFJKcHI3Xg%3D)|
| SpeechUT Large (ASR)  | [60k hrs LibriSpeech](http://www.openslr.org/12) + [40M Text](http://www.openslr.org/11)                                                          |                      -                            | [Azure Storage](https://msranlcmtteamdrive.blob.core.windows.net/share/speechut/large_speechut4asr_32gpu_4accum/checkpoint_22_400k.pt?sv=2020-10-02&st=2022-10-13T06%3A15%3A13Z&se=2023-10-14T06%3A15%3A00Z&sr=c&sp=rl&sig=b0RFMEsG5ch3OA2%2FTTsoG5DFnLTE9c%2BeHKFJKcHI3Xg%3D)|
| SpeechUT Large (ASR)  | [60k hrs LibriSpeech](http://www.openslr.org/12) + [40M Text](http://www.openslr.org/11)                                                          | [960 hrs LibriSpeech](http://www.openslr.org/12)  | [Azure Storage](https://msranlcmtteamdrive.blob.core.windows.net/share/speechut/speechut_large_asr960h_checkpoint_best.pt?sv=2020-10-02&st=2022-10-13T06%3A15%3A13Z&se=2023-10-14T06%3A15%3A00Z&sr=c&sp=rl&sig=b0RFMEsG5ch3OA2%2FTTsoG5DFnLTE9c%2BeHKFJKcHI3Xg%3D)|
| SpeechUT Base (En-De) | [960 hrs LibriSpeech](http://www.openslr.org/12) + [408 hrs MuST-C v1](https://ict.fbk.eu/must-c/) + [4.6M Text](https://www.statmt.org/wmt16/)   |                      -                            | [Azure Storage](https://msranlcmtteamdrive.blob.core.windows.net/share/speechut/base_speechut4ende_32gpu_1accum/checkpoint_217_400000.pt?sv=2020-10-02&st=2022-10-13T06%3A15%3A13Z&se=2023-10-14T06%3A15%3A00Z&sr=c&sp=rl&sig=b0RFMEsG5ch3OA2%2FTTsoG5DFnLTE9c%2BeHKFJKcHI3Xg%3D)|
| SpeechUT Base (En-De) | [960 hrs LibriSpeech](http://www.openslr.org/12) + [408 hrs MuST-C v1](https://ict.fbk.eu/must-c/) + [4.6M Text](https://www.statmt.org/wmt16/)   | [En-De MuST-C v1](https://ict.fbk.eu/must-c/)     | [Azure Storage](https://msranlcmtteamdrive.blob.core.windows.net/share/speechut/base_speechut4ende_32gpu_1accum/fineutne_ende_checkpoint_avg.pt?sv=2020-10-02&st=2022-10-13T06%3A15%3A13Z&se=2023-10-14T06%3A15%3A00Z&sr=c&sp=rl&sig=b0RFMEsG5ch3OA2%2FTTsoG5DFnLTE9c%2BeHKFJKcHI3Xg%3D)|
| SpeechUT Base (En-Es) | [960 hrs LibriSpeech](http://www.openslr.org/12) + [504 hrs MuST-C v1](https://ict.fbk.eu/must-c/) + [15M Text](https://www.statmt.org/wmt13/)    |                      -                            | [Azure Storage](https://msranlcmtteamdrive.blob.core.windows.net/share/speechut/base_speechut4enes_32gpu_1accum/checkpoint_204_400000.pt?sv=2020-10-02&st=2022-10-13T06%3A15%3A13Z&se=2023-10-14T06%3A15%3A00Z&sr=c&sp=rl&sig=b0RFMEsG5ch3OA2%2FTTsoG5DFnLTE9c%2BeHKFJKcHI3Xg%3D)|
| SpeechUT Base (En-Es) | [960 hrs LibriSpeech](http://www.openslr.org/12) + [504 hrs MuST-C v1](https://ict.fbk.eu/must-c/) + [15M Text](https://www.statmt.org/wmt13/)    | [En-Es MuST-C v1](https://ict.fbk.eu/must-c/)     | [Azure Storage](https://msranlcmtteamdrive.blob.core.windows.net/share/speechut/base_speechut4enes_32gpu_1accum/fineutne_enes_checkpoint_avg.pt?sv=2020-10-02&st=2022-10-13T06%3A15%3A13Z&se=2023-10-14T06%3A15%3A00Z&sr=c&sp=rl&sig=b0RFMEsG5ch3OA2%2FTTsoG5DFnLTE9c%2BeHKFJKcHI3Xg%3D)|
| SpeechUT Base (En-Fr) | [960 hrs LibriSpeech](http://www.openslr.org/12) + [492 hrs MuST-C v1](https://ict.fbk.eu/must-c/) + [40M Text](https://www.statmt.org/wmt14/)    |                      -                            | [Azure Storage](https://msranlcmtteamdrive.blob.core.windows.net/share/speechut/base_speechut4enfr_32gpu_1accum/checkpoint_297_600000.pt?sv=2020-10-02&st=2022-10-13T06%3A15%3A13Z&se=2023-10-14T06%3A15%3A00Z&sr=c&sp=rl&sig=b0RFMEsG5ch3OA2%2FTTsoG5DFnLTE9c%2BeHKFJKcHI3Xg%3D)|
| SpeechUT Base (En-Fr) | [960 hrs LibriSpeech](http://www.openslr.org/12) + [492 hrs MuST-C v1](https://ict.fbk.eu/must-c/) + [40M Text](https://www.statmt.org/wmt14/)    | [En-Fr MuST-C v1](https://ict.fbk.eu/must-c/)     | [Azure Storage](https://msranlcmtteamdrive.blob.core.windows.net/share/speechut/base_speechut4enfr_32gpu_1accum/fineutne_enfr_checkpoint.pt?sv=2020-10-02&st=2022-10-13T06%3A15%3A13Z&se=2023-10-14T06%3A15%3A00Z&sr=c&sp=rl&sig=b0RFMEsG5ch3OA2%2FTTsoG5DFnLTE9c%2BeHKFJKcHI3Xg%3D)|


## Language Model
See [here](https://github.com/microsoft/SpeechT5/tree/main/Speech2C#language-model-and-vocabulary).


## Setup

```bash
git submodule update --init SpeechUT/fairseq
cd SpeechUT/
pip install --editable fairseq/
pip install sacrebleu==1.5.1
```


## ASR on LibriSpeech
### Data preparation
Please follow the steps of wav2vec 2.0 manifest [here](https://github.com/pytorch/fairseq/tree/main/examples/wav2vec#prepare-training-data-manifest) to prepare `train.tsv` and `train.ltr`. You should make sure the vocabulary [`dict.ltr.txt`](dataset/LibriSpeech/dict.ltr.txt) is the same as that used for the pre-trained model. Put yout prepared data into `$data_dir`.

### Fine-tune a hybrid CTC-ED model
- Fine-tune the base model on 100h subset
    ```bash
    # Usage: speechut/scripts/tune_speechut_asr/finetune_base_edctc.sh <model_path> <data_dir> <cpt_tag> [mount=$PWD] [world_size=8] [update_freq=2]
    model_path=path/to/your/pre-trained/model
    data_dir=dataset/LibriSpeech/asr
    bash speechut/scripts/tune_speechut_asr/finetune_base_edctc.sh $model_path $data_dir 'tag400k'
    ```

- Fine-tune the large model on 960h subset
    ```bash
    # Usage: speechut/scripts/tune_speechut_asr/finetune960h_large_edctc.sh <model_path> <data_dir> <cpt_tag> [mount=$PWD] [world_size=8] [update_freq=3]
    model_path=path/to/your/pre-trained/model
    data_dir=dataset/LibriSpeech/asr
    bash speechut/scripts/tune_speechut_asr/finetune960h_large_edctc.sh $model_path $data_dir 'tag400k'
    ```

### Decode
- CTC-ED joint decoding
    ```bash
    # Usage: speechut/scripts/tune_speechut_asr/inference_edctc.sh <model_path> <data_dir> [gen-set=dev_other] [beam_size=10] [ctc_weight=0.2] [--normalize]
    model_path=path/to/your/fine-tuned/model
    data_dir=dataset/LibriSpeech/asr
    # for base model
    bash speechut/scripts/tune_speechut_asr/inference_edctc.sh $model_path $data_dir test_clean 10 0.2
    # for large model, you should set --normalize at the end
    bash speechut/scripts/tune_speechut_asr/inference_edctc.sh $model_path $data_dir test_clean 10 0.2 --normalize
    ```
    > We use the [espnet](https://github.com/espnet/espnet)-style joint decoding algorithm, currently only supporting batch_size=1. If you find it too slow, please check [`inference_nj.sh`](speechut/scripts/tune_speechut_asr/inference_nj.sh) for a multi-thread version.

- CTC-ED joint decoding with LM
    ```bash
    # Usage: speechut/scripts/tune_speechut_asr/inference_edctclm.sh <model_path> <data_dir> [gen-set=dev_other] [beam_size=30] [ctc_weight=0.3] [lm_weight=0.7] [lm_path] [--normalize]
    model_path=path/to/your/fine-tuned/model
    data_dir=dataset/LibriSpeech/asr
    lm_path=path/to/char_lm/model
    # for base model
    bash speechut/scripts/tune_speechut_asr/inference_edctclm.sh $model_path $data_dir test_clean 30 0.3 0.7 $lm_path
    # for large model, you should set --normalize at the end
    bash speechut/scripts/tune_speechut_asr/inference_edctclm.sh $model_path $data_dir test_clean 30 0.3 0.7 $lm_path --normalize
    ```

    > We currently only support batch_size=1. If you find it too slow, please check [`inference_lm_nj.sh`](speechut/scripts/tune_speechut_asr/inference_lm_nj.sh) for a multi-thread version.

    > The released language model uses a different vocaburary [`dict.txt`](dataset/LibriSpeech/dict.txt), put it into `$data_dir` and the script will access it.


## ST on MuST-C
### Data preparation

ST models are fine-tuned with [fairseq speech-to-text](https://github.com/facebookresearch/fairseq/tree/main/examples/speech_to_text) task, so just follow the data preparation instructions [here](https://github.com/facebookresearch/fairseq/tree/main/examples/speech_to_text#data-preparation).
To fine-tune our released models, you should use the same sentecepiece models and dictionaries as ours:

- En-De: [sentencepiece_model](dataset/MuSTC/en_de/spm_unigram10000.model), [dict](dataset/MuSTC/en_de/dict.spm.txt)
- En-Es: [sentencepiece_model](dataset/MuSTC/en_es/spm_unigram10000.model), [dict](dataset/MuSTC/en_es/dict.spm.txt)
- En-Fr: [sentencepiece_model](dataset/MuSTC/en_fr/spm_unigram10000.model), [dict](dataset/MuSTC/en_fr/dict.spm.txt)

We provided examples in [`dataset`](dataset/MuSTC).

### Fine-tune an encoder-decoder model

```bash
# Usage: speechut/scripts/tune_speechut_st/finetune_base_mustc_enxx.sh <model_path> <data_dir> <lang> <cpt-tag> [mount=$PWD] [world_size=8] [update_freq=4/6]
model_path=path/to/your/pre-trained/model
data_dir=dataset/MuSTC/en-${lang}
bash speechut/scripts/tune_speechut_st/finetune_base_mustc_enxx.sh $model_path $data_dir ${lang} tag400k
```
Please check the script [`finetune_base_mustc_enxx.sh`](speechut/scripts/tune_speechut_st/finetune_base_mustc_enxx.sh) for detailed configuration.

### Decode
You might average several model checkpoints with the best dev accuracy to stablize the performance,
```bash
python fairseq/scripts/average_checkpoints.py --inputs $model_dir/checkpoint.best_acc*.pt --output $model_dir/checkpoint.avgnbest.pt
```
Then decode the model with beam search,
```bash
# Usage: speechut/scripts/tune_speechut_st/inference_st.sh <model_path> <data_dir> <lang> [gen-set=dev] [beam_size=10] [lenpen=1.0]
model_path=path/to/your/fine-tuned/model
data_dir=dataset/MuSTC/en-${lang}
bash speechut/scripts/tune_speechut_st/inference_st.sh $model_path $data_dir ${lang} tst-COMMON
```




## Pre-train for ASR

### Data preparation
The model is pre-trained by speech-to-unit, unit-to-text and mask-unit-lm tasks.
1. For speech-to-unit task, please follow the steps of data preparation for HuBERT [here](https://github.com/facebookresearch/fairseq/tree/main/examples/hubert#data-preparation).
2. For unit-to-text task, follow the steps below:
    - Generate units from unpaired text by [T2U Generator](#T2U-Generator).
    - Pair the generated units and text data, convert them to binary files.
3. For mask-unit-lm task, combine the units generated from step1 and step2 together.

You should use [`dict.ltr.txt`](dataset/LibriSpeech/dict.ltr.txt) when preparing the text data, make sure the dictionary is the same as that used for fine-tuning.

### Pre-train base model

```bash
# Usage: speechut/scripts/pretrain_speechut/base_speechut_for_asr.sh <data_dir> <text_data_dir> [mount=$PWD] [world_size=32] [update_freq=1]
data_dir=
text_data_dir=
bash speechut/scripts/pretrain_speechut/base_speechut_for_asr.sh $data_dir $text_data_dir
```

## Pre-train for ST

### Data preparation
The model is pre-trained by speech-to-unit, unit-to-text and mask-unit-lm tasks.
1. For speech-to-unit task, please follow the steps of data preparation for HuBERT [here](https://github.com/facebookresearch/fairseq/tree/main/examples/hubert#data-preparation).
2. For unit-to-text task, we use bilingual text where the source side (i.e. English) is used to generate unit and the target side serves as the output. Follow the steps below:
    - Normalize the source (English) text by removing punctuation, converting capital letters.
    - Generate units from the source (English) text by [T2U Generator](#T2U-Generator).
    - Pair the generated units and text data, convert them to binary files.
3. For mask-unit-lm task, combine the units generated from step1 and step2 together.
You should use the same sentencepiece models and dictionaries as that used for [fine-tuning](#ST-on-MuST-C).


### Pre-train base model

```bash
# Usage: speechut/scripts/pretrain_speechut/base_speechut_for_st.sh <data_dir> <text_data_dir> <lang> [mount=$PWD] [world_size=32] [update_freq=1]
data_dir=
text_data_dir=
bash speechut/scripts/pretrain_speechut/base_speechut_for_st.sh $data_dir $text_data_dir ${lang}
```


## T2U Generator
The original paper trains an encoder-decoder model to generate reduced units from text, which is time consuming due to the autoregressive generation.
We recently update the T2U generator to a non-autoregressive model, which generates non-reduced units (can be easily post-processed to reduced units). Please follow the usage provided by [Hidden-unit Tokenizer for Text](https://github.com/microsoft/SpeechT5/tree/main/SpeechLM#hidden-unit-tokenizer-for-text) (they used the same HuBERT units as this work).


## License

This project is licensed under the license found in the LICENSE file in the root directory of this source tree.
Portions of the source code are based on the [FAIRSEQ](https://github.com/pytorch/fairseq).

[Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct)

## Reference

If you find our work is useful in your research, please cite the following paper:

```bibtex
@article{zhang2022speechut,
  title   = {SpeechUT: Bridging Speech and Text with Hidden-Unit for Encoder-Decoder Based Speech-Text Pre-training},
  author  = {Zhang, Ziqiang and Zhou, Long and Ao, Junyi and Liu, Shujie and Dai, Lirong and Li, Jinyu and Wei, Furu},
  eprint={2210.03730},
  archivePrefix={arXiv},
  primaryClass={cs.CL},
  year={2022}
}
```

### Contact Information

For help or issues using SpeechUT models, please submit a GitHub issue.

For other communications related to SpeechUT, please contact Long Zhou (`lozhou@microsoft.com`).
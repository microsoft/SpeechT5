# Speech2C

> [**Speech2C**](https://arxiv.org/abs/2203.17113) (```INTERSPEECH 2022```): **Pre-Training Transformer Decoder for End-to-End ASR Model with Unpaired Speech Data**

## Pre-Trained and Fine-tuned Models

|  Model   |               Pre-training Dataset               | Fine-tuning Dataset | Model |
| :------: | :----------------------------------------------: | :-----------------: | :-----: |
| Speech2C | [960 hrs LibriSpeech](http://www.openslr.org/12) |          -          | [Google Drive](https://drive.google.com/file/d/1nGZ0LWEwlLq2pz7o805YALsMr9irV0Za/view?usp=sharing)  |
| Speech2C | [960 hrs LibriSpeech](http://www.openslr.org/12) | [10 hrs LibriSpeech](http://www.openslr.org/12) |  [Google Drive](https://drive.google.com/file/d/1nWSAc-33LmcDQHzH8IjXVJsuk0JZTWgN/view?usp=sharing) |
| Speech2C | [960 hrs LibriSpeech](http://www.openslr.org/12) | [100 hrs LibriSpeech](http://www.openslr.org/12) |  [Google Drive](https://drive.google.com/file/d/1LwbQ5Y3tKZoK3s1ayLQgsfLTFnmkKNZs/view?usp=sharing) |


## Language Model and Vocabulary
|  Model   |  Dataset | Model | Vocabulary | 
| :------: | :------: | :---: | :--------: |
| LM | [LibriSpeech LM Dataset](https://www.openslr.org/11/) | [Model](https://drive.google.com/file/d/1UDCcNJT1DlquSRw0iRAXH6GHlf6zK6-8/view?usp=sharing)  | [Vocabulary](https://dl.fbaipublicfiles.com/fairseq/wav2vec/dict.ltr.txt) |

## Setup
```
git submodule update --init Speech2C/fairseq
cd Speech2C/
pip install --editable fairseq/
```

## Data Preparation
Please follow the steps of data preparation for HuBERT in [here](https://github.com/facebookresearch/fairseq/tree/main/examples/hubert#data-preparation).

## Pre-Training
```
DATA_DIR=
LABEL_DIR=
FAIRSEQ_PATH=

python ${FAIRSEQ_PATH}/fairseq_cli/hydra_train.py \
  --config-dir speech2c/config \
  --config-name speech2c_base_librispeech \
  task.data=${DATA_DIR} task.label_dir=${LABEL_DIR} task.labels='["km"]' \
  model.label_rate=50 common.user_dir=SpeechT5/Speech2C/speech2c \
```

## Finetune

```
DATA_DIR=
LABEL_DIR=
FAIRSEQ_PATH=
W2V_PATH=
CONFIG_NAME=

python ${FAIRSEQ_PATH}/fairseq_cli/hydra_train.py \
  --config-dir speech2c/config \
  --config-name ${CONFIG_NAME} \
  task.data=${DATA_DIR} task.label_dir=${LABEL_DIR} \
  model.w2v_path=${W2V_PATH} common.user_dir=SpeechT5/Speech2C/speech2c \
```

## Inference
Note that joint CTC and decoder inference is only supported when the batch size is 1.

```
FAIRSEQ_PATH=
DATA_DIR=
LABEL_DIR=
BEAM_SIZE=
CTC_WEIGHT=
TEST_SET=
CHECKPOINT_PATH=
W2V_PATH=


python ${FAIRSEQ_PATH}/fairseq_cli/generate.py ${DATA_DIR} \
    --label-dir ${LABEL_DIR} \
    --path ${CHECKPOINT_PATH} \
    --user-dir SpeechT5/Speech2C/speech2c \
    --model-overrides "{'w2v_path': '${W2V_PATH}'}" \
    --gen-subset ${TEST_SET} \
    --task speech2c_pretraining \
    --post-process letter \
    --add-decoder \
    --labels '["ltr"]' \
    --fine-tuning \
    --scoring wer \
    --max-len-a 0 \
    --max-len-b 620 \
    --pad-audio \
    --random-crop \
    --ctc-weight ${CTC_WEIGHT} \
    --max-tokens 8000000 \
    --beam ${BEAM_SIZE} \
    --single-target \
```

## Results on Librispeech

### Evaluation on the [LibriSpeech](http://www.openslr.org/12) 10hr subset

| Model         |LM                 | test-clean   | test-other   |
| ------------- |-------------      | ----|  ----|
| wav2vec2.0 Base          | -      | 11.1 | 17.6 |
| HuBERT Base              | -      | 10.1 | 16.8 |
| **Speech2C**              | -      | **7.8** | **13.1** |
| wav2vec 2.0 Base         | 4-gram | 4.3  |9.5   |
| wav2vec 2.0 Base   | Transf. |3.2  |7.8   |
| HuBERT Base              | 4-gram	|4.3 |9.4   |
| **Speech2C**              | **Transf.**     | **3.1** | **7.0** |

### Evaluation on the [LibriSpeech](http://www.openslr.org/12) 100hr subset

| Model         |LM                 | test-clean   | test-other   |
| ------------- |-------------      | ----|  ----|
| wav2vec2.0 Base          | -      | 6.1 | 13.3 |
| wav2vec2.0 Large          | -      | 4.7 | 9.0 |
| HuBERT Base              | -      | 6.3 | 13.2 |
| SpeechT5             | -      | 4.4 | 10.4 |
| Baseline                 | -      |  5.0 | 11.9 |
| **Speech2C**                 | - | **4.3**  |**9.0**   |
| wav2vec 2.0 Base         | 4-gram | 3.4  |8.0   |
| wav2vec 2.0 Base         | Transf. | 2.6  | 6.3   |
| HuBERT Base              | 4-gram	| 3.4  |8.1   |
| SpeechT5             | Transf. | 2.4  |5.8   |
| Baseline                 | Transf. | 2.5  |6.3   |
| **Speech2C**                 | **Transf.** | **2.4**  |**5.2**   |

## License

This project is licensed under the license found in the LICENSE file in the root directory of this source tree.
Portions of the source code are based on the [FAIRSEQ](https://github.com/pytorch/fairseq).

[Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct)

## Reference

If you find our work is useful in your research, please cite the following paper:

```bibtex
@article{Ao2022Speech2C,
  title   = {Pre-Training Transformer Decoder for End-to-End ASR Model with Unpaired Speech Data},
  author  = {Junyi Ao and Ziqiang Zhang and Long Zhou and Shujie Liu and Haizhou Li and Tom Ko and Lirong Dai and Jinyu Li and Yao Qian and Furu Wei},
  eprint={2203.17113},
  archivePrefix={arXiv},
  primaryClass={cs.SD},
  year={2022}
}
```

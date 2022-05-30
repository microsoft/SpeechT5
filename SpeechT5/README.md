# SpeechT5

<!--**Pre-trained models for speech related tasks**-->

 [**SpeechT5**](https://arxiv.org/abs/2110.07205): **Unified-Modal Encoder-Decoder Pre-training for Spoken Language Processing**

Official PyTorch implementation and pretrained models of SpeechT5

- Oct 2021: release preprint in [arXiv](https://arxiv.org/abs/2110.07205)
- Feb 2022: accepted by [ACL 2022](https://www.2022.aclweb.org/)

## Setup
```
cd SpeechT5/
git submodule update --init fairseq
pip install --editable fairseq/
pip install espnet
```

## Data Preparation

### Speech data and S2T Data
Please follow the steps for preparing wav2vec 2.0 manifest in [here](https://github.com/pytorch/fairseq/tree/main/examples/wav2vec#prepare-training-data-manifest). 

We add a third column for the speaker embedding, which is provided in [here](https://drive.google.com/uc?export=download&id=16QOUURZBrW7-GYbVG_gXt3mTMlZmQoH0).
It includes the speaker embeddings for 960hr training data and dev-other data of LibriSpeech.

We also provide example manifests for your reference in [here](https://drive.google.com/drive/folders/1Ja08XjOHe6vP8lZtLVrJM8173aPQCR_y?usp=sharing).

### Text Data
Please use [fairseq-preprocess](https://fairseq.readthedocs.io/en/latest/command_line_tools.html#fairseq-preprocess) to generate the index and bin files of the text data.

## Pre-Training

### 960hr LibriSpeech + LibriSpeech-LM

```
DATA_ROOT=
SAVE_DIR=
LABEL_DIR=
TRAIN_SET="speech_train|text_train"
VALID_SET="speech_valid|text_valid"


fairseq-train ${DATA_ROOT} \
  --save-dir ${SAVE_DIR} \
  --tensorboard-logdir ${SAVE_DIR} \
  --train-subset ${TRAIN_SET} \
  --valid-subset ${VALID_SET} \
  --hubert-label-dir ${LABEL_DIR} \
  --distributed-world-size 32 \
  --distributed-port 0 \
  --ddp-backend legacy_ddp \
  --user-dir SpeechT5/speecht5 \
  --log-format json \
  --seed 1337 \
  --fp16 \
  \
  --task speecht5 \
  --t5-task pretrain \
  --label-rates 50 \
  --sample-rate 16000 \
  --random-crop \
  \
  --num-workers 0 \
  --max-tokens 1400000 \
  --max-speech-sample-size 250000 \
  --update-freq 2 \
  --batch-ratio "[1,0.0086]" \
  \
  --criterion speecht5 \
  --optimizer adam \
  --reset-optimizer \
  --adam-betas "(0.9, 0.98)" \
  --adam-eps 1e-06 \
  --weight-decay 0.01 \
  --power 1 \
  --clip-norm 5.0 \
  --lr 0.0002 \
  --lr-scheduler polynomial_decay \
  \
  --max-update 800000 \
  --warmup-updates 64000 \
  --total-num-update 800000 \
  --save-interval-updates 3000 \
  --skip-invalid-size-inputs-valid-test \
  --required-batch-size-multiple 1 \
  \
  --arch t5_transformer_base \
  --share-input-output-embed \
  --find-unused-parameters \
  --bert-init \
  --relative-position-embedding \
  --use-codebook \
  --codebook-prob 0.1 \
  --loss-weights="[10,0.1]" \
  --max-text-positions 600 \
```

## Finetune

### ASR

#### Training

```
DATA_ROOT=
SAVE_DIR=
TRAIN_SET=
VALID_SET=
LABEL_DIR=
BPE_TOKENIZER=
USER_DIR=

mkdir -p ${SAVE_DIR}
fairseq-train ${DATA_ROOT} \
  --save-dir ${SAVE_DIR} \
  --tensorboard-logdir ${SAVE_DIR} \
  --train-subset ${TRAIN_SET} \
  --valid-subset ${VALID_SET} \
  --hubert-label-dir ${LABEL_DIR} \
  --distributed-world-size 1 \
  --distributed-port 0 \
  --ddp-backend legacy_ddp \
  --user-dir ${USER_DIR} \
  --log-format json \
  --seed 1337 \
  --fp16 \
  \
  --task speecht5 \
  --t5-task s2t \
  --sample-rate 16000 \
  --num-workers 0 \
  --max-tokens 1600000 \
  --update-freq 2 \
  --bpe-tokenizer ${BPE_TOKENIZER} \
  \
  --criterion speecht5 \
  --report-accuracy \
  --zero-infinity \
  --ce-weight 0.5 \
  --ctc-weight 0.5 \
  --sentence-avg \
  \
  --optimizer adam \
  --reset-optimizer \
  --adam-betas "(0.9, 0.98)" \
  --adam-eps 1e-08 \
  --weight-decay 0.1 \
  --clip-norm 25.0 \
  --lr 0.00006 \
  --lr-scheduler tri_stage \
  --phase-ratio "[0.1, 0.4, 0.5]" \
  --final-lr-scale 0.05 \
  \
  --max-update 80000 \
  --max-text-positions 600 \
  --required-batch-size-multiple 1 \
  --save-interval-updates 3000 \
  --skip-invalid-size-inputs-valid-test \
  \
  --arch t5_transformer_base_asr \
  --share-input-output-embed \
  --find-unused-parameters \
  --bert-init \
  --relative-position-embedding \
  --freeze-encoder-updates 13000 \
```

#### Inference
Note that joint CTC/Decoder inference is only supported when batch size is 1.

```
CHECKPOINT_PATH=
DATA_ROOT=
SUBSET=
BPE_TOKENIZER=
LABEL_DIR=
USER_DIR=
BEAM=
MAX_TOKENS=
CTC_WEIGHT=
LM_WEIGHT=
LM_PATH=

fairseq-generate ${DATA_ROOT} \
  --gen-subset ${SUBSET} \
  --bpe-tokenizer ${BPE_TOKENIZER} \
  --user-dir ${USER_DIR} \
  --task speecht5 \
  --t5-task s2t \
  --path ${CHECKPOINT_PATH} \
  --hubert-label-dir ${LABEL_DIR} \
  --ctc-weight 
  --lm-weight ${LM_WEIGHT} \
  --lm-path ${LM_PATH} \
  --max-tokens ${MAX_TOKENS} \
  --beam ${BEAM} \
  --scoring wer \
  --max-len-a 0 \
  --max-len-b 620 \
  --sample-rate 16000
```


## License

This project is licensed under the license found in the LICENSE file in the root directory of this source tree.
Portions of the source code are based on the [FAIRSEQ](https://github.com/pytorch/fairseq) and [ESPnet](https://github.com/espnet/espnet) projects.

[Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct)

### Reference

If you find our work is useful in your research, please cite the following paper:

```bibtex
@article{Ao2021SpeechT5,
  title   = {SpeechT5: Unified-Modal Encoder-Decoder Pre-training for Spoken Language Processing},
  author  = {Junyi Ao and Rui Wang and Long Zhou and Chengyi Wang and Shuo Ren and Yu Wu and Shujie Liu and Tom Ko and Qing Li and Yu Zhang and Zhihua Wei and Yao Qian and Jinyu Li and Furu Wei},
  eprint={2110.07205},
  archivePrefix={arXiv},
  primaryClass={cs.CL},
  year={2021}
}
```

### Contact Information

For help or issues using SpeechT5 models, please submit a GitHub issue.

For other communications related to SpeechT5, please contact Long Zhou (`lozhou@microsoft.com`).

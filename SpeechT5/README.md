# SpeechT5

<!--**Pre-trained models for speech related tasks**-->

 [**SpeechT5**](https://arxiv.org/abs/2110.07205): **Unified-Modal Encoder-Decoder Pre-training for Spoken Language Processing**

Official PyTorch implementation and pretrained models of SpeechT5

- Oct 2021: release preprint in [arXiv](https://arxiv.org/abs/2110.07205)


## Pre-training


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

### TTS 

### ST

### VC 

### SID

## SE


## License

This project is licensed under the license found in the LICENSE file in the root directory of this source tree.
Portions of the source code are based on the [FAIRSEQ](https://github.com/pytorch/fairseq) project.

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

For other communications related to SpeechT5, please contact Long Zhou (`Long.Zhou@microsoft.com`).

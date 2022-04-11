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

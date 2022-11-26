# Scripts

## Pretrain

### LibriSpeech 960hr

* HuBERT Baseline with Rel Pos Enc

```
SAVE_DIR=
DATA_ROOT=
LABEL_DIR=

mkdir -p ${SAVE_DIR}

python fairseq_cli/hydra_train.py \
  --config-dir examples/hubert/config/pretrain \
  --config-name hubert_base_librispeech \
  distributed_training.distributed_world_size=32 distributed_training.nprocs_per_node=8 \
  common.tensorboard_logdir=${SAVE_DIR} checkpoint.save_dir=${SAVE_DIR} \
  task.data=${DATA_ROOT} task.label_dir=${LABEL_DIR} \
  task.labels='["km"]' model.label_rate=50 \
  dataset.max_tokens=1400000
```

* HuBERT + Decoder CE Loss

```
SAVE_DIR=
DATA_ROOT=
LABEL_DIR=

mkdir -p ${SAVE_DIR}

python fairseq_cli/hydra_train.py \
  --config-dir examples/hubert/config/pretrain \
  --config-name hubert_base_librispeech \
  distributed_training.distributed_world_size=32 distributed_training.nprocs_per_node=8 \
  common.tensorboard_logdir=${SAVE_DIR} checkpoint.save_dir=${SAVE_DIR} \
  task.data=${DATA_ROOT} task.label_dir=${LABEL_DIR} \
  task.labels='["km"]' model.label_rate=50 task.add_decoder=true \
  dataset.max_tokens=1400000
```

## Fine-tuning

### LibriSpeech 100hr

* HuBERT Baseline with Rel Pos Enc

```
SAVE_DIR=
DATA_ROOT=
LABEL_DIR=
W2V_PATH=

mkdir -p ${SAVE_DIR}

python fairseq_cli/hydra_train.py \
  --config-dir examples/hubert/config/finetune \
  --config-name base_100h \
  distributed_training.distributed_world_size=16 distributed_training.nprocs_per_node=8 \
  common.tensorboard_logdir=${SAVE_DIR} checkpoint.save_dir=${SAVE_DIR} \
  task.data=${DATA_ROOT} task.label_dir=${LABEL_DIR} \
  model.w2v_path=${W2V_PATH} dataset.max_tokens=1600000
```

* HuBERT + Decoder CE Loss

```
SAVE_DIR=
DATA_ROOT=
LABEL_DIR=
W2V_PATH=

mkdir -p ${SAVE_DIR}

python fairseq_cli/hydra_train.py \
  --config-dir examples/hubert/config/finetune \
  --config-name base_100h \
  distributed_training.distributed_world_size=16 distributed_training.nprocs_per_node=8 \
  common.tensorboard_logdir=${SAVE_DIR} checkpoint.save_dir=${SAVE_DIR} \
  task.data=${DATA_ROOT} task.label_dir=${LABEL_DIR} \
  criterion._name=ctc_ce model.add_decoder=true task.add_decoder=true \
  model.freeze_finetune_updates=25000 dataset.max_tokens=1600000 \
  model.w2v_path=${W2V_PATH} checkpoint.best_checkpoint_metric=dec_accuracy \
  checkpoint.maximize_best_checkpoint_metric=true \
  task.pad_audio=true task.random_crop=false optimization.lr=[0.00004]
```

### LibriSpeech 10hr

* HuBERT Baseline with Rel Pos Enc

```
SAVE_DIR=
DATA_ROOT=
LABEL_DIR=
W2V_PATH=

mkdir -p ${SAVE_DIR}

python fairseq_cli/hydra_train.py \
  --config-dir examples/hubert/config/finetune \
  --config-name base_10h \
  distributed_training.distributed_world_size=16 distributed_training.nprocs_per_node=8 \
  common.tensorboard_logdir=${SAVE_DIR} checkpoint.save_dir=${SAVE_DIR} \
  task.data=${DATA_ROOT} task.label_dir=${LABEL_DIR} \
  model.w2v_path=${W2V_PATH} dataset.max_tokens=1600000
```

* HuBERT + Decoder CE Loss

```
SAVE_DIR=
DATA_ROOT=
LABEL_DIR=
W2V_PATH=

mkdir -p ${SAVE_DIR}

python fairseq_cli/hydra_train.py \
  --config-dir examples/hubert/config/finetune \
  --config-name base_10h \
  distributed_training.distributed_world_size=16 distributed_training.nprocs_per_node=8 \
  common.tensorboard_logdir=${SAVE_DIR} checkpoint.save_dir=${SAVE_DIR} \
  task.data=${DATA_ROOT} task.label_dir=${LABEL_DIR} \
  criterion._name=ctc_ce model.add_decoder=true task.add_decoder=true \
  dataset.max_tokens=1600000 task.pad_audio=true task.random_crop=false \
  model.w2v_path=${W2V_PATH} checkpoint.best_checkpoint_metric=dec_accuracy \
  checkpoint.maximize_best_checkpoint_metric=true
```

## Inference

### Directly Inference

```
EXP_NAME=
DATA_ROOT=
SAVE_DIR=
LABEL_DIR=
test_set=
BEAM_SIZE=30
CHECKPOINT_FILENAME=checkpoint_best.pt

mkdir -p ${SAVE_DIR}/decoder_inf

python fairseq_cli/generate.py ${DATA_ROOT} --task hubert_pretraining --gen-subset ${test_set} \
  --post-process letter --add-decoder --label-dir ${LABEL_DIR} --labels '["ltr"]' --fine-tuning \
  --scoring wer --max-len-a 0 --max-len-b 620 --pad-audio --random-crop \
  --path ${SAVE_DIR}/${CHECKPOINT_FILENAME} --max-tokens 800000 --beam ${BEAM_SIZE} \
  --single-target >> ${SAVE_DIR}/decoder_inf/${test_set}_decode_beam${BEAM_SIZE}.log
tail -n 1 ${SAVE_DIR}/decoder_inf/${test_set}_decode_beam${BEAM_SIZE}.log
```

### Joint CTC and Decoder Inference

Please set batch-size = 1

```
EXP_NAME=
CTC_WEIGHT=
DATA_ROOT=
SAVE_DIR=
LABEL_DIR=
test_set=
BEAM_SIZE=30
CHECKPOINT_FILENAME=checkpoint_best.pt

mkdir -p ${SAVE_DIR}/decoder_inf

python fairseq_cli/generate.py ${DATA_ROOT} --task hubert_pretraining --gen-subset ${test_set} \
    --post-process letter --add-decoder --label-dir ${LABEL_DIR} --labels '["ltr"]' --fine-tuning \
    --scoring wer --max-len-a 0 --max-len-b 620 --pad-audio --random-crop --ctc-weight ${CTC_WEIGHT} \
    --path ${SAVE_DIR}/${CHECKPOINT_FILENAME} --batch-size 1 --beam ${BEAM_SIZE} \
    --single-target >> ${SAVE_DIR}/decoder_inf/${test_set}_decode_beam${BEAM_SIZE}_ctc${CTC_WEIGHT}.log
tail -n 1 ${SAVE_DIR}/decoder_inf/${test_set}_decode_beam${BEAM_SIZE}_ctc${CTC_WEIGHT}.log
```

### Joint CTC and Decoder Inference + LM Shallow Fusion

Please set batch-size = 1

```
EXP_NAME=
CTC_WEIGHT=
LM_WEIGHT=
DATA_ROOT=
SAVE_DIR=
LABEL_DIR=
BEAM_SIZE=
test_set=
LM_PATH=/mnt/default/v-junyiao/librispeech/lm/lm_ctc_form/checkpoint_best.pt
CHECKPOINT_FILENAME=checkpoint_best.pt

mkdir -p ${SAVE_DIR}/decoder_inf

python fairseq_cli/generate.py ${DATA_ROOT} --task hubert_pretraining --gen-subset ${test_set} \
    --post-process letter --add-decoder --label-dir ${LABEL_DIR} --labels '["ltr"]' --fine-tuning \
    --scoring wer --max-len-a 0 --max-len-b 620 --pad-audio --random-crop --ctc-weight ${CTC_WEIGHT} --lm-path ${LM_PATH} \
    --path ${SAVE_DIR}/${CHECKPOINT_FILENAME} --batch-size 1 --beam ${BEAM_SIZE} --lm-weight ${LM_WEIGHT} \
    --single-target >> ${SAVE_DIR}/decoder_inf/${test_set}_decode_beam${BEAM_SIZE}_ctc${CTC_WEIGHT}_lm${LM_WEIGHT}.log
tail -n 1 ${SAVE_DIR}/decoder_inf/${test_set}_decode_beam${BEAM_SIZE}_ctc${CTC_WEIGHT}_lm${LM_WEIGHT}.log
```

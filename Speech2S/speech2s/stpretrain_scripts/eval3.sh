#$subset=test
python examples/speech_recognition/infer.py /home/v-kunwei --task audio_finetuning \
--nbest 1 --path /mnt/output/users/v-kunwei/code/fairseq/models/es_eval/espeak_26lang_m10.pt --gen-subset test --results-path /home/v-kunwei --criterion ctc --labels ltr --max-tokens 4000000 \
--post-process letter

lmweight=0
num_gpus=8
python examples/speech_recognition/new/infer.py --config-dir /mnt/output/users/v-kunwei/code/fairseq/examples/speech_recognition/new/conf \
--config-name infer task=audio_finetuning task.data=/home/v-kunwei common.user_dir=/mnt/output/users/v-kunwei/code/fairseq/examples/data2vec \
task.labels=ltr decoding.type=viterbi \
decoding.lexicon=models/es_eval/espeak_dict.txt \
decoding.unique_wer_file=True \
dataset.gen_subset=test \
common_eval.path=/mnt/output/users/v-kunwei/code/fairseq/models/es_eval/espeak_26lang_m10.pt decoding.beam=1500 distributed_training.distributed_world_size=${num_gpus} \
decoding.results_path=/home/v-kunwei

#sclite  -h "/home/v-kunwei/hypo.units"  -r "/home/v-kunwei/ref.units"  -i rm -o all stdout > "./result.txt"

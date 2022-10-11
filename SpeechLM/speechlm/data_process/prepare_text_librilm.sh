#!/bin/bash
[ ${PWD##*/} != SpeechLM ] && echo "Error: dir not match! Switch to SpeechLM and run it again!"
src=${PWD}/speechlm/data_process

set -e
mkdir -p dataset/LibriLM/tmp && cd dataset/LibriLM

echo "Downloading and unpacking librispeech-lm-norm.txt ..."
wget -c https://www.openslr.org/resources/11/librispeech-lm-norm.txt.gz
gzip -d librispeech-lm-norm.txt.gz

echo "Tokenize the text..."
cat librispeech-lm-norm.txt | sed '1d' | python $src/wrd2ltr.py > tmp/librilm.ltr

echo "Tokenize the text to the kaldi-style phonemes ..."
python $src/phones/ltr2kaldi_phn_sil025.py -i tmp/librilm.ltr -o tmp/librilm
cat tmp/librilm.kaldi_phn_sil025 | sed 's/SIL_S/SIL/g' > tmp/librilm.phn

echo "Filter too long samples and up-sample phonemes ..."
python $src/filter_paireddata_by_len.py -i tmp/librilm -o tmp/librilm_l2k -s phn -t ltr -m 2000
python $src/phones/repeat_withou_insert_sil_less_4375.py tmp/librilm_l2k.phn $src/phones/mean5_and_std25_sil14_spn32.dict tmp/librilm_l2k_upsample.phn
python $src/filter_paireddata_by_len.py -i tmp/librilm -o tmp/librilm_l2k -s phn -t ltr -m 2800
### the max-length is set to filter the data, considering the batch size (in Large setting, 900,000/320 = 2812 tokens in a batch).


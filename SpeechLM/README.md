# SpeechLM

<!--**Pre-trained models for speech related tasks**-->

 [**SpeechLM**](https://arxiv.org/abs/2209.15329): **Enhanced Speech Pre-Training with Unpaired Textual Data**


- The code and checkpoints will be released here.
- Oct 2022: release preprint in [arXiv](https://arxiv.org/abs/2209.15329)
- (Scheduled) Oct 2022: release the code and models

## Setup
```
git submodule update --init SpeechLM/fairseq
cd SpeechLM/
pip install --editable fairseq/
pip install sacrebleu==1.5.1
```

## Data Preparation

### Fine-tuning ASR on LibriSpeech
- Please follow the steps of wav2vec 2.0 manifest [here](https://github.com/pytorch/fairseq/tree/main/examples/wav2vec#prepare-training-data-manifest) to prepare `train.tsv` and `train.ltr`.

### Fine-tuning ST on CovoST-2 En-XX
- Please follow the scripts below to prepare the required files
```

```
### Pre-training
- **Unlabeled speech:** please follow the steps of wav2vec 2.0 manifest [here](https://github.com/pytorch/fairseq/tree/main/examples/wav2vec#prepare-training-data-manifest) to prepare `train.tsv`.
- **Phoneme units for speech:** use phoneme-unit tokenizer to process the speech to prepare `train.phn`.
- **Hidden units for speech:** use hidden-unit tokenizer to process the speech to prepare `train.km`.
> The hidden-unit tokenizer used in this word is [a K-means model on the top of the Hubert Base model](https://github.com/facebookresearch/fairseq/tree/main/examples/hubert/simple_kmeans).
- Create dict for the target units `dict.phn.txt` or `dict.km.txt`.

- **Unpaired Text:** convert [LibriSpeech LM corpus](http://www.openslr.org/11/) to normalized charecters to get `librilm.phn-ltr.ltr` and `dict.ltr.txt`.
- **Phoneme tokens for text:** use the lexicon provided by LibriSpeech LM corpus to convert words to phonemes, then apply up-sampling to get `librilm.phn-ltr.phn`.
- **Phoneme tokens for text:** use the text-to-unit tokenizer to convert phonemes to units to get `librilm.phn-ltr.phn`.



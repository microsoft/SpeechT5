# YiTrans@IWSLT22

> [**YiTrans**](https://arxiv.org/abs/2206.05777) (```IWSLT 2022```): **The YiTrans End-to-End Speech Translation System for IWSLT 2022 Offline Shared Task**
> Code is being merged to this repository, thanks for your attention

## Setup
```bash
git clone https://github.com/microsoft/SpeechT5.git
git submodule update --init YiTrans/fairseq
cd YiTrans/fairseq
pip install -e .
```

## Data Preparation
### Speech/ASR data for pre-training
Please follow the steps of data preparation for HuBERT in [here](https://github.com/facebookresearch/fairseq/tree/main/examples/hubert#data-preparation).
### Monolingual text data for pre-training
Please follow the steps of data preparation for mBART in [here](https://github.com/facebookresearch/fairseq/tree/main/examples/mbart). We reuse the multilingual vocabulary.
After getting your subset.{idx,bin} files ready, renaming them as subset.lang.lang.{idx,bin}, e.g.
```
mono_deduped_filt_sort.en_XX.en_XX.bin
mono_deduped_filt_sort.en_XX.en_XX.idx
```
### Bilingual text data for pre-training
The same way of preparing monolingual data with only the difference that you should prepare for both the source language and the target languages. Renaming them as subset.src-tgt.{src,tgt}.{idx,bin}, e.g.
```
mt8corpus_filt_slct.en_XX-de_DE.de_DE.bin
mt8corpus_filt_slct.en_XX-de_DE.de_DE.idx
mt8corpus_filt_slct.en_XX-de_DE.en_XX.bin
mt8corpus_filt_slct.en_XX-de_DE.en_XX.idx
```

### ST data for fine-tuning
Please follow the steps of data preparation for S2T tasks [here](https://github.com/pytorch/fairseq/blob/main/examples/speech_to_text/docs/mustc_example.md). Your tsv file should be like this:
```
id      audio   n_frames        tgt_text        speaker src_text        src_lang        tgt_lang
ted_1_0 /mnt/speechdata/MUSTC/en-de/flac/ted_1_0.flac    25920   Hinter mir war gar keine Autokolonne.   spk.1   There was no motorcade back there.      en_XX   de_DE
ted_1_1 /mnt/speechdata/MUSTC/en-de/flac/ted_1_1.flac    219359  Haben Sie schon mal vom Phantomschmerz gehört? (Lachen) Wir saßen in einem gemieteten Ford Taurus.       spk.1   (Laughter) You've heard of phantom limb pain? (Laughter)        en_XX   de_DE
ted_1_2 /mnt/speechdata/MUSTC/en-de/flac/ted_1_2.flac    71360   Es war Zeit zum Abendessen und wir hielten Ausschau nach einem Restaurant.      spk.1   It was dinnertime, and we started looking for a place to eat.    en_XX   de_DE
```



## Pre-train
For example of pre-training the PT36 model, please follow these steps:

Step 0: Download the released [Hubert model](https://dl.fbaipublicfiles.com/hubert/hubert_large_ll60k.pt) and [mBART model](https://dl.fbaipublicfiles.com/fairseq/models/mbart50/mbart50.pretrained.tar.gz) model.

Step 1: Pre-training with unlabeled speech data and monolingual/bilingual text data 
```bash
bash YiTrans/exp_scripts/pretrain/pretrain_pt36_adaptor_step1.sh
```

Step 2: Pre-training with ASR dat and domain-filtered bilingual text data 
```bash
bash YiTrans/exp_scripts/pretrain/pretrain_pt36_adaptor_step2.sh
```
Other configurations like training PT48 can also be fould in ./YiTrans/exp_scripts/pretrain, you might need to modify the PATH variables in .sh files to adjust your data.

## Fine-tune
For example of pre-training En-De ST model on MuST-C dataset,
```bash
bash YiTrans/exp_scripts/finetune_ST/en-de/jtst_pt36s2_mustc.sh
```
Other configurations like different translation directions or datasets could be found in ./YiTrans/exp_scripts/finetune_ST, you might need to modify the PATH variables in .sh files to adjust your data.

## Cascaded system
You can also build a cascaded ST system (ASR+MT) with our codebase.
1. ASR model: fine-tune from the cascade of [Hubert Large](https://dl.fbaipublicfiles.com/hubert/hubert_large_ll60k.pt) and [mBART model](https://dl.fbaipublicfiles.com/fairseq/models/mbart50/mbart50.pretrained.tar.gz):
    ```bash
    # change the mbart_path/hubert_path to your own in the *.sh
    bash YiTrans/exp_scripts/finetune_ASR/finetune_hubert24_mbart24_en.sh
    ```
    Check the [`.sh`](exp_scripts/finetune_ASR/finetune_hubert24_mbart24_en.sh) file for more information about the configuration.

2. MT model: fine-tune from [mBART model](https://dl.fbaipublicfiles.com/fairseq/models/mbart50/mbart50.pretrained.tar.gz):

    ```bash
    # change the mbart_path to your own in the *.sh
    bash YiTrans/exp_scripts/finetune_MT/finetune_mbart_en-de.sh
    ```
    Check the [`.sh`](exp_scripts/finetune_MT/finetune_mbart_en-de.sh) file for more information about the configuration.


## Reference

If you find our work is useful in your research, please cite the following paper:

```bibtex
@article{Zhang2022Yitrans,
  title   = {The YiTrans End-to-End Speech Translation System for IWSLT 2022 Offline Shared Task},
  author  = {Zhang, Ziqiang and Ao, Junyi and Zhou, Long and Liu, Shujie and Wei, Furu and Li, Jinyu},
  eprint={2206.05777},
  archivePrefix={arXiv},
  primaryClass={cs.CL},
  year={2022}
}
```

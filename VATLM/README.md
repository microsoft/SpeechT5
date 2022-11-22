# VATLM
<!--**Pre-trained models for speech related tasks**-->

 [**VATLM: Visual-Audio-Text Pre-Training with Unified Masked Prediction for Speech Representation Learning**](https://arxiv.org/abs/2211.11275)


- (Done) Nov. 2022: release the code and models
- Nov. 2022: release preprint in [arXiv](https://arxiv.org/abs/2211.11275)

## Pre-Trained and Fine-tuned Models

|    Model    |            Pre-training Dataset            |  Fine-tuning Dataset  |                            Model                             |
| :---------: | :----------------------------------------: | :-------------------: | :----------------------------------------------------------: |
| VatLM Base  |       LRS3 + paired audio+text+audio       |           -           | [Google drive](https://drive.google.com/file/d/121ITJc22prpbd4sCy9bPWpdkKgGikkgm/view?usp=share_link) |
| VatLM Base  |       LRS3 + paired audio+text+audio       | LRS-30h  audio-visual | [Google drive](https://drive.google.com/file/d/1Bfbq0G-tASw3YrI3rzdpYgTE-UV-YaN0/view?usp=share_link) |
| VatLM Base  |       LRS3 + paired audio+text+audio       |    LRS-30h  visual    | [Google drive](https://drive.google.com/file/d/1qALD9obym0zCDoszVn2CzW0U3EUl-4v7/view?usp=share_link) |
| VatLM Base  | VoxCeleb2 + LRS3 + paired audio+text+audio |           -           | [Google drive](https://drive.google.com/file/d/1piae9Row25OEfAekVz5Bxb9YnIVyEP0A/view?usp=share_link) |
| VatLM Base  | VoxCeleb2 + LRS3 + paired audio+text+audio | LRS-30h audio-visual  | [Google drive](https://drive.google.com/file/d/13JVuUi9gIIoUM888XcAOzvN7ioazn-cv/view?usp=share_link) |
| VatLM Base  | VoxCeleb2 + LRS3 + paired audio+text+audio |    LRS-30h  visual    | [Google drive](https://drive.google.com/file/d/1pAQHf60HgqDORGzyqEjdGTIywLKO3Ko5/view?usp=share_link) |
| VatLM Base  | VoxCeleb2 + LRS3 + paired audio+text+audio | LRS-433h audio-visual | [Google drive](https://drive.google.com/file/d/1u9oMnivBelxznQcMDoM_u5EOfJuxnSuL/view?usp=share_link) |
| VatLM Base  | VoxCeleb2 + LRS3 + paired audio+text+audio |    LRS-433h visual    | [Google drive](https://drive.google.com/file/d/1g107k5tL3XyvevSe0BzMqYOQFyFQG7jf/view?usp=share_link) |
| VatLM Large | VoxCeleb2 + LRS3 + paired audio+text+audio |           -           | [Google drive](https://drive.google.com/file/d/1_vbVFpKcaaPcCx2FtI-GyzVvxAhppg_b/view?usp=share_link) |
| VatLM Large | VoxCeleb2 + LRS3 + paired audio+text+audio | LRS-30h  audio-visual | [Google drive](https://drive.google.com/file/d/1LyTCxceTZIqjVdMY6hlJjWolaIAZ0Mhs/view?usp=share_link) |
| VatLM Large | VoxCeleb2 + LRS3 + paired audio+text+audio |    LRS-30h  visual    | [Google drive](https://drive.google.com/file/d/1CuyGg5O14F9Y_WCwpCVoKYbDKVtjBRQU/view?usp=share_link) |
| VatLM Large | VoxCeleb2 + LRS3 + paired audio+text+audio | LRS-433h audio-visual | [Google drive](https://drive.google.com/file/d/12orvO3xBuzdUDrBOqjW0mdGhV2Kmsy0Q/view?usp=share_link) |
| VatLM Large | VoxCeleb2 + LRS3 + paired audio+text+audio |    LRS-433h visual    | [Google drive](https://drive.google.com/file/d/17DDTUPs0BkaJtSUTiJHLBbymt2LCGo6e/view?usp=share_link) |



## Setup

To fine-tune or pre-train more models, please follow the instructions below.

```bash
git clone https://github.com/microsoft/SpeechT5.git
cd SpeechT5/VATLM
git submodule init && git submodule update

cd VATLM/fairseq  && pip install --editable
cd VATLM/vat_hubert && pip install -r requirements.txt
```

## Data preparation

1. For audio or visual data, please follow the steps of AV-HuBERT's [script](https://github.com/facebookresearch/av_hubert/tree/main/avhubert/preparation) to pre-process the data and get the corresponding `train.tsv`,` train.km` files.

2. For unimodal audio data, the visual modality is replaced with a zero vector, and the features are extracted according to this [script](https://github.com/facebookresearch/av_hubert/tree/main/avhubert/preparation) and then kmeans [clustering](https://github.com/facebookresearch/av_hubert/tree/main/avhubert/clustering) is performed to get the corresponding labels.

3. For unimodal text data, we use a small amount of pair text-audio data to obtain paired phone-unit data, and get the corresponding phoneme sequences by looking up the [lexicon](https://drive.google.com/file/d/1dh9NEx_cCF9_Aa0UcKyl9j00GXs6LmLQ/view?usp=sharing), and the unit data are obtained by extracting features and performing kmeans [clustering](https://github.com/facebookresearch/av_hubert/tree/main/avhubert/clustering).  Then follow this [script](https://github.com/microsoft/SpeechT5/tree/main/SpeechLM#hidden-unit-tokenizer-for-text) to train the phone2unit model.

## Pre-train

- VatLM Base model (LRS3 + paired audio+text+audio)

  ```shell
  cd VATLM/vat_hubert/vathubert/scripts/pretrain
  ngpu=32
  updatefreq=1
  save_path=/path/to/save_path
  
  bash base_lsr3_pretrain_iter5.sh ${ngpu} ${updatefreq} ${save_path}
  ```

- VatLM Base model (VoxCeleb2 + paired audio+text+audio)

  ```shell
  cd VATLM/vat_hubert/vathubert/scripts/pretrain
  ngpu=32
  updatefreq=1
  save_path=/path/to/save_path
  
  bash base_vox_pretrain_iter5.sh ${ngpu} ${updatefreq} ${save_path}
  ```

- VatLM Large model (VoxCeleb2 + paired audio+text+audio)

  ```shell
  cd VATLM/vat_hubert/vathubert/scripts/pretrain
  ngpu=32
  updatefreq=2
  save_path=/path/to/save_path
  
  bash large_vox_pretrain_iter5.sh ${ngpu} ${updatefreq} ${save_path}
  ```

## Fine-tune AVSR/VSR

For example, the AVSR model can be obtained by fine-tuning the VatLM model using 30 hours of labeled data.

```shell
cd VATLM/vat_hubert/vathubert/scripts/finetune_avsr
ngpu=8
updatefreq=1
save_path=/path/to/save_path

bash base_lrs3_finetune30_av.sh ${ngpu} ${updatefreq} ${save_path}
```

## Decode

For example, decoding the fine-tuned AVSR model.

```sh
cd VATLM/vat_hubert/vathubert/
data="test"
bash decode_avhubert_lrs3.sh ${data}
```

## License

This project is licensed under the license found in the LICENSE file in the root directory of this source tree.
Portions of the source code are based on the [FAIRSEQ](https://github.com/pytorch/fairseq) and [av_hubert](https://github.com/facebookresearch/av_hubert)

[Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct)

## Reference

If you find our work is useful in your research, please cite the following paper:

```bibtex
@article{zhu2022vatlm,
      title={VATLM: Visual-Audio-Text Pre-Training with Unified Masked Prediction for Speech Representation Learning}, 
      author={Qiushi Zhu and Long Zhou and Ziqiang Zhang and Shujie Liu and Binxing Jiao and Jie Zhang and Lirong Dai and Daxin Jiang and Jinyu Li and Furu Wei},
      year={2022},
      eprint={2211.11275},
      archivePrefix={arXiv},
}
```

### Contact Information

For help or issues using VatLM models, please submit a GitHub issue.

For other communications related to VatLM, please contact Long Zhou (`lozhou@microsoft.com`).


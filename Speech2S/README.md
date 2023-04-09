# Speech2S
<!--**Pre-trained models for speech related tasks**-->

 [**Joint Pre-Training with Speech and Bilingual Text for Direct Speech to Speech Translation**](https://arxiv.org/abs/2210.17027)


- (Updating) Nov. 2022: release the code and models
- Nov. 2022: release preprint in [arXiv](https://arxiv.org/abs/2210.17027)

## Pre-Trained and Fine-tuned Models

|  Model   |               Pre-training Dataset               | Fine-tuning Dataset | Model |
| :------: | :----------------------------------------------: | :-----------------: | :-----: |
| Speech2S_enes |   Voxpopuli_en_v2 |         -          | [Google Drive](https://drive.google.com/file/d/1TYypFiEKoCixUro8FTTG23bRZYwAxhkX/view?usp=share_link)  |
| Speech2S_enes |   Voxpopuli_en_v2 | Voxpopuli_s2s |  [Google Drive](https://drive.google.com/file/d/11RxeKznSrHcoP_KK9A1VgwRt3fNh_U_C/view?usp=share_link) |
| Speech2S_esen |   Voxpopuli_es_v2 |         -          | [Google Drive](https://drive.google.com/file/d/1NoC7W-UtQZ-ugIptF1ex0ZlGJncsT1S4/view?usp=share_link) |
| Speech2S_esen |   Voxpopuli_es_v2 | Voxpopuli_s2s |  [Google Drive](https://drive.google.com/file/d/1eNcKw4ZWGmcABWXJxlf6MKocmiPrKSkH/view?usp=share_link) |


## Setup
```
cd Speech2S/speech2s
pip install --editable fairseq/
```

## Data Preparation
Please follow the steps of data preparation for S2ST in [here](https://github.com/facebookresearch/fairseq/blob/main/examples/speech_to_speech/docs/enhanced_direct_s2st_discrete_units.md).

## Pre-Training
```
cd speech2s/stpretrain_scripts
base_sc2c_enes.sh
```
## Finetune
```
cd speech2s/stpretrain_scripts
finetune_enes.sh
```
## Inference
```
cd speech2s/stpretrain_scripts
inference_ed.sh
```
## Results on Voxpopuli and Covst


## License

This project is licensed under the license found in the LICENSE file in the root directory of this source tree.
Portions of the source code are based on the [FAIRSEQ](https://github.com/pytorch/fairseq).

[Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct)

## Reference

If you find our work is useful in your research, please cite the following paper: 
```bibtex
@article{wei2022joint,
  title={Joint Pre-Training with Speech and Bilingual Text for Direct Speech to Speech Translation},
  author={Wei, Kun and Zhou, Long and Zhang, Ziqiang and Chen, Liping and Liu, Shujie and He, Lei and Li, Jinyu and Wei, Furu},
  journal={arXiv preprint arXiv:2210.17027},
  year={2022}
}
```

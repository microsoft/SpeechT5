# SpeechT5


> [**SpeechT5**](https://arxiv.org/abs/2110.07205) (```ACL 2022```): **SpeechT5: Unified-Modal Encoder-Decoder Pre-training for Spoken Language Processing**

> [**Speech2C**](https://arxiv.org/abs/2203.17113) (```INTERSPEECH 2022```): **Pre-Training Transformer Decoder for End-to-End ASR Model with Unpaired Speech Data**

> [**YiTrans**](https://arxiv.org/abs/2206.05777) (```IWSLT 2022```): **The YiTrans End-to-End Speech Translation System for IWSLT 2022 Offline Shared Task**

> [**SpeechUT**](https://arxiv.org/abs/2210.03730) (```EMNLP 2022```): **SpeechUT: Bridging Speech and Text with Hidden-Unit for Encoder-Decoder Based Speech-Text Pre-training**

> [**SpeechLM**](https://arxiv.org/abs/2209.15329) (```Arxiv 2022```): **SpeechLM: Enhanced Speech Pre-Training with Unpaired Textual Data**




Motivated by the success of T5 (Text-To-Text Transfer Transformer) in pre-trained natural language processing models, we propose a unified-modal SpeechT5 framework that explores the encoder-decoder pre-training for self-supervised speech/text representation learning.
The SpeechT5 framework consists of a shared encoder-decoder network and six modal-specific (speech/text) pre/post-nets. 
After preprocessing the input speech/text through the pre-nets, the shared encoder-decoder network models the sequence-to-sequence transformation, and then the post-nets generate the output in the speech/text modality based on the output of the decoder.

<img src="SpeechT5/speecht5_framework.png" alt="se" width="1000" />

Leveraging large-scale unlabeled speech and text data, we pre-train SpeechT5 to learn a unified-modal representation, hoping to improve the modeling capability for both speech and text.
To align the textual and speech information into this unified semantic space, we propose a cross-modal vector quantization approach that randomly mixes up speech/text states with latent units as the interface between encoder and decoder.
Extensive evaluations show the superiority of the proposed SpeechT5 framework on a wide variety of spoken language processing tasks, including automatic speech recognition, speech synthesis, speech translation, voice conversion, speech enhancement, and speaker identification.

Model introductions, evaluation results, and model inference instructions are located in the corresponding folders. The source code is here [https://github.com/microsoft/SpeechT5/tree/main/SpeechT5].

## Pre-Trained Models

|  Model   |               Pre-training Dataset               | Fine-tuning Dataset | Model |
| :------: | :----------------------------------------------: | :-----------------: | :-----: |
| SpeechT5 Base | [960 hrs LibriSpeech](http://www.openslr.org/12) + [LibriSpeech LM Dataset](https://www.openslr.org/11/) |          -          | [HuggingFace](https://huggingface.co/ajyy/SpeechT5/resolve/main/speecht5_base.pt)<br /> [Google Drive](https://drive.google.com/file/d/1Sq00uZ1pw6Z4OUaqhOWzQEJxIVWgAO5U/view?usp=sharing)  |
| SpeechT5 Base | [960 hrs LibriSpeech](http://www.openslr.org/12) + [LibriSpeech LM Dataset](https://www.openslr.org/11/) | [100 hrs LibriSpeech](http://www.openslr.org/12) | [HuggingFace](https://huggingface.co/ajyy/SpeechT5/resolve/main/speecht5_base_asr.pt)<br /> [Google Drive](https://drive.google.com/file/d/1qLKJ81JPWOGf1MHfjSmgtZyqqTqgI6kT/view?usp=sharing)  |
| SpeechT5 Large | [60k hrs Libri-Light](https://github.com/facebookresearch/libri-light) + [LibriSpeech LM Dataset](https://www.openslr.org/11/) |          -          | [Google Drive](https://drive.google.com/file/d/1M79b1jetSPOVxWVMIX-y0URvDjNskZKp/view?usp=sharing)  |

## Language Model and Vocabulary
|  Model   |  Dataset | Model | Vocabulary | SPM Model |
| :------: | :------: | :---: | :--------: | :-------: |
| LM | [LibriSpeech LM Dataset](https://www.openslr.org/11/) | [LM Model](https://drive.google.com/uc?export=download&id=1y0TGnKAMKUW5C8l8yrvGjh9RRZETPdv7)  | [Vocabulary](https://drive.google.com/uc?export=download&id=19hcQ58RHZ6CssxF8Qp6yEF1NW_AXxObK) | [SPM Model](https://drive.google.com/uc?export=download&id=1wClgQjXXoU2lmpbaEa1v2SqMbg7cAutq) |

## Downstream Task Performance

We evaluate our models on typical spoken language processing tasks, including automatic speech recognition, text to speech, speech to text translation, voice conversion, speech enhancement, and speaker identification.

### Automatic Speech Recognition

Evaluation on the [LibriSpeech](http://www.openslr.org/12)

| Model         |LM                 | dev-clean | dev-other  | test-clean   | test-other   |
| ------------- |-------------      | ------| ----- | ----|  ----|
| wav2vec2.0 Base          | -      | 6.1   | 13.5  | 6.1 | 13.3 |
| HuBERT Base              | -      | 5.5	| 13.1  | 5.8 | 13.3 |
| Baseline (w/o CTC) | -      | 5.8   | 12.3	| 6.2 | 12.3 |
| Baseline                 | -      | 4.9	| 11.7  | 5.0 | 11.9 |
| SpeechT5 (w/o CTC)   | -      | 5.4	| 10.7  | 5.8 | 10.7 |
| **SpeechT5**             | -      | **4.3**	| **10.3**  | **4.4** | **10.4** |
| DiscreteBERT             | 4-gram | 4.0   |10.9   |4.5  |12.1  |
| wav2vec 2.0 Base         | 4-gram | 2.7   |7.9    |3.4  |8.0   |
| HuBERT Base              | 4-gram	| 2.7   |7.8    |3.4  |8.1   |
| wav2vec 2.0 Base   | Transf. | 2.2   |6.3    |2.6  |6.3   |
| Baseline                 | Transf. | 2.3   |6.3    |2.5  |6.3   |
| **SpeechT5**             | Transf. | **2.1**   |**5.5**    |**2.4**  |**5.8**   |

### Text-to-Speech

Evaluation on the [LibriTTS](http://www.openslr.org/60/)


| Model         | Naturalness | MOS       | CMOS        |
| ------------- |------------ | ------    | -----       | 
| Ground Truth  | -           | 3.87      | -           |  
| Baseline      | 2.76        | 3.56	  | 0           | 
| **SpeechT5**  | 2.91        | **3.65**  | **+0.290**  | 

### Speech Translation

Evaluation on the [MUST-C v1](https://ict.fbk.eu/must-c/)

| Model         | EN-DE | EN-FR           | 
| ------------- |------------  | ------    | 
| Fairseq ST    | 22.70        | 32.90     | 
| ESPnet ST     | 22.91        | 32.69	  | 
| Adapter Tuning| 24.63        | 34.98	  | 
| Baseline      | 23.43        | 33.76	  | 
| SpeechT5 (w/o initializing decoder)  | 24.44   | 34.5  | 
| **SpeechT5**  | **25.18**        | **35.30**  | 


### Voice Conversion

Evaluation on the [CMU Arctic](http://www.festvox.org/cmu_arctic/)


| Model            | WER        | WER         | MCD          | MCD          |
| -------------    | ------    | -----   | ----    |  ----|
|                  | bdl to slt | clb to slt  | bdl to slt   | clb to slt   |
| VTN w/ ASR       |  11.1    | 10.9   | 6.5     | 6.11 |
| VTN w/ TTS       |  7.6 	   | 9.1    | 6.33    | 13.3 |
| Many-to-many VTN |  -        | -	     | 6.13    | 5.97 |
| Baseline         |  21.5   | 10.8   | 6.26    | 6.16 |
| **SpeechT5**     |  **7.8**  | **6.4** | **5.93**| **5.87** |



### Speech Enhancement

Evaluation on the [WSJ0 Hipster AmbientMixtures (WHAM!)](http://wham.whisper.ai/)


| Model                | WER        | 
| -------------        |------------  | 
| Ground Truth Speech  | 3.2        | 
| Noisy Speech         | 76.1        | 
| Baseline             | 10.9        | 
| **SpeechT5**         | **8.9**    |


### Speaker Identification

Evaluation on the [VoxCeleb1](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox1.html)

| Model                          | Acc          | 
| -------------                  |------------  | 
| SUPERB, wav2vec 2.0 Base       | 75.18%         | 
| SUPERB, HuBERT Base       | 81.42%        | 
| SUPERB, HuBERT Large       | 90.33%        | 
| SpeechNet, single task         | 86.00%        | 
| SpeechNet, multi-task with TTS | 87.90%        |
| Thin ResNet-34                 | 89.00%        |  
| Baseline                       | 91.92%        | 
| **SpeechT5**                   | **96.49%**    |

## License

This project is licensed under the license found in the LICENSE file in the root directory of this source tree.
Portions of the source code are based on the [FAIRSEQ](https://github.com/pytorch/fairseq) and [ESPnet](https://github.com/espnet/espnet) projects.

[Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct)

### Reference

If you find our work is useful in your research, please cite the following paper:

```bibtex
@article{Ao2021SpeechT5,
  title   = {SpeechT5: Unified-Modal Encoder-Decoder Pre-training for Spoken Language Processing},
  author  = {Junyi Ao and Rui Wang and Long Zhou and Chengyi Wang and Shuo Ren and Yu Wu and Shujie Liu and Tom Ko and Qing Li and Yu Zhang and Zhihua Wei and Yao Qian and Jinyu Li and Furu Wei},
  eprint={2110.07205},
  archivePrefix={arXiv},
  primaryClass={eess.AS},
  year={2021}
}
```

```bibtex
@article{Ao2022Speech2C,
  title   = {Pre-Training Transformer Decoder for End-to-End ASR Model with Unpaired Speech Data},
  author  = {Junyi Ao and Ziqiang Zhang and Long Zhou and Shujie Liu and Haizhou Li and Tom Ko and Lirong Dai and Jinyu Li and Yao Qian and Furu Wei},
  eprint={2203.17113},
  archivePrefix={arXiv},
  primaryClass={cs.SD},
  year={2022}
}
```

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

```bibtex
@article{zhang2022speechut,
  title   = {SpeechUT: Bridging Speech and Text with Hidden-Unit for Encoder-Decoder Based Speech-Text Pre-training},
  author  = {Zhang, Ziqiang and Zhou, Long and Ao, Junyi and Liu, Shujie and Dai, Lirong and Li, Jinyu and Wei, Furu},
  eprint={2210.03730},
  archivePrefix={arXiv},
  primaryClass={cs.CL},
  year={2022}
}
```

```bibtex
@article{zhang2022speechlm,
  title   = {SpeechLM: Enhanced Speech Pre-Training with Unpaired Textual Data},
  author  = {Zhang, Ziqiang and Chen, Sanyuan and Zhou, Long and Wu, Yu and Ren, Shuo and Liu, Shujie and Yao, Zhuoyuan and Gong, Xun and Dai, Lirong and Li, Jinyu and Wei, Furu},
  eprint={2209.15329},
  archivePrefix={arXiv},
  primaryClass={cs.CL},
  year={2022}
}
```

### Contact Information

For help or issues using SpeechT5 models, please submit a GitHub issue.

For other communications related to SpeechT5, please contact Long Zhou (`lozhou@microsoft.com`).

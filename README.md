# SpeechT5

Unified-modal speech-text pre-training for spoken language processing:

> [**SpeechT5**](https://arxiv.org/abs/2110.07205) (```ACL 2022```): **SpeechT5: Unified-Modal Encoder-Decoder Pre-training for Spoken Language Processing**

> [**Speech2C**](https://arxiv.org/abs/2203.17113) (```INTERSPEECH 2022```): **Pre-Training Transformer Decoder for End-to-End ASR Model with Unpaired Speech Data**

> [**YiTrans**](https://arxiv.org/abs/2206.05777) (```IWSLT 2022```): **The YiTrans End-to-End Speech Translation System for IWSLT 2022 Offline Shared Task**

> [**SpeechUT**](https://arxiv.org/abs/2210.03730) (```EMNLP 2022```): **SpeechUT: Bridging Speech and Text with Hidden-Unit for Encoder-Decoder Based Speech-Text Pre-training**

> [**SpeechLM**](https://arxiv.org/abs/2209.15329) (```Arxiv 2022```): **SpeechLM: Enhanced Speech Pre-Training with Unpaired Textual Data**

> [**Speech2S**](https://arxiv.org/abs/2210.17027) (```ICASSP 2023```): **Joint Pre-Training with Speech and Bilingual Text for Direct Speech to Speech Translation**

> [**Prosody-SpeechT5**](https://ieeexplore.ieee.org/document/10096530/) (```ICASSP 2023```): **Prosody-aware SpeechT5 for Expressive Neural TTS**

> [**VATLM**](https://arxiv.org/abs/2211.11275) (```IEEE Transactions on Multimedia```): **VATLM: Visual-Audio-Text Pre-Training with Unified Masked Prediction for Speech Representation Learning**

> [**VALL-E X**](https://arxiv.org/abs/2303.03926) (```Arxiv 2023```): **Speak Foreign Languages with Your Own Voice: Cross-Lingual Neural Codec Language Modeling**

> [**VioLA**](https://arxiv.org/abs/2305.16107) (```Arxiv 2023```): **VioLA: Unified Codec Language Models for Speech Recognition, Synthesis, and Translation**

<!-- Model introductions, evaluation results, and model inference instructions are located in the corresponding folders. The source code is [https://github.com/microsoft/SpeechT5/tree/main/ModelName]. -->


## Update

- May, 2023: VioLA [**Arxiv**](https://arxiv.org/abs/2305.16107).
- May, 2023: [**VATLM**](https://arxiv.org/abs/2211.11275) was accepted by IEEE Transactions on Multimedia.
- March, 2023: VALL-E X [**Arxiv**](https://arxiv.org/abs/2303.03926) and [**Demo**](https://aka.ms/vallex).
- February, 2023: [**Speech2S**](https://arxiv.org/abs/2210.17027) and [**Prosody-SpeechT5**](https://arxiv.org/abs/2211.11275) were accepted by ICASSP 2023.
- [HuggingFace Integration] February, 2023: [**SpeechT5**](https://aclanthology.org/2022.acl-long.393/) models are on [**HuggingFace**](https://huggingface.co/blog/speecht5).
- [Model Release] November, 2022: [**VATLM**](https://github.com/microsoft/SpeechT5/tree/main/VATLM) models are released.
- November, 2022: VATLM [**Arxiv**](https://arxiv.org/abs/2211.11275).
- November, 2022: Speech2S [**Arxiv**](https://arxiv.org/abs/2210.17027).
- [Model Release] October, 2022: [**SpeechUT**](https://github.com/microsoft/SpeechT5/tree/main/SpeechUT) models are released.
- October, 2022: [**SpeechUT**](https://arxiv.org/abs/2210.03730) was accepted by EMNLP 2022.
- [Model Release] October, 2022: [**SpeechLM**](https://github.com/microsoft/SpeechT5/tree/main/SpeechLM) models are released.
- September, 2022: SpeechLM [**Arxiv**](https://arxiv.org/abs/2209.15329).
- [Evaluation] June, 2022: The end-to-end ST system [**YiTrans**](https://arxiv.org/abs/2206.05777) achieved top results on [**IWSLT 2022**](https://iwslt.org/2022/offline) shared tasks.
- June, 2022:  [**Speech2C**](https://www.isca-speech.org/archive/interspeech_2022/ao22_interspeech.html) was accepted by InterSpeech 2022.
- [Model Release] May, 2022: [**Speech2C**](https://github.com/microsoft/SpeechT5/tree/main/Speech2C) models are released.
- [Model Release] April, 2022: [**SpeechT5**](https://github.com/microsoft/SpeechT5/tree/main/SpeechT5) models are released.
- March, 2022: Speech2C [**Arxiv**](https://arxiv.org/abs/2203.17113).
- February, 2022: [**SpeechT5**](https://aclanthology.org/2022.acl-long.393/) was accepted by ACL 2022.
- October, 2021: SpeechT5 [**Arxiv**](https://arxiv.org/abs/2110.07205).


## Pre-Trained Models


|  Model   |               Pre-training Dataset               | Fine-tuning Dataset | Model |
| :------: | :----------------------------------------------: | :-----------------: | :-----: |
| SpeechT5 Base | [960 hrs LibriSpeech](http://www.openslr.org/12) + [LibriSpeech LM Dataset](https://www.openslr.org/11/) |          -          | [HuggingFace](https://huggingface.co/ajyy/SpeechT5/resolve/main/speecht5_base.pt)<br /> [Google Drive](https://drive.google.com/file/d/1Sq00uZ1pw6Z4OUaqhOWzQEJxIVWgAO5U/view?usp=sharing)  |
| SpeechT5 Base | [960 hrs LibriSpeech](http://www.openslr.org/12) + [LibriSpeech LM Dataset](https://www.openslr.org/11/) | [100 hrs LibriSpeech](http://www.openslr.org/12) | [HuggingFace](https://huggingface.co/ajyy/SpeechT5/resolve/main/speecht5_base_asr.pt)<br /> [Google Drive](https://drive.google.com/file/d/1qLKJ81JPWOGf1MHfjSmgtZyqqTqgI6kT/view?usp=sharing)  |
| SpeechT5 Large | [60k hrs Libri-Light](https://github.com/facebookresearch/libri-light) + [LibriSpeech LM Dataset](https://www.openslr.org/11/) |          -          | [Google Drive](https://drive.google.com/file/d/1M79b1jetSPOVxWVMIX-y0URvDjNskZKp/view?usp=sharing)  |
| Speech2C | [960 hrs LibriSpeech](http://www.openslr.org/12) |          -          | [Google Drive](https://drive.google.com/file/d/1nGZ0LWEwlLq2pz7o805YALsMr9irV0Za/view?usp=sharing)  |
| Speech2C | [960 hrs LibriSpeech](http://www.openslr.org/12) | [10 hrs LibriSpeech](http://www.openslr.org/12) |  [Google Drive](https://drive.google.com/file/d/1nWSAc-33LmcDQHzH8IjXVJsuk0JZTWgN/view?usp=sharing) |
| Speech2C | [960 hrs LibriSpeech](http://www.openslr.org/12) | [100 hrs LibriSpeech](http://www.openslr.org/12) |  [Google Drive](https://drive.google.com/file/d/1LwbQ5Y3tKZoK3s1ayLQgsfLTFnmkKNZs/view?usp=sharing) |
| SpeechLM-P Base   | [960 hrs LibriSpeech](http://www.openslr.org/12) + [40M Text](http://www.openslr.org/11)                      |                      -                                            | [Google drive](https://drive.google.com/file/d/1iJvhSGghNrMT-wAY1nwVu2YaYuTy1pxx/view?usp=sharing)  |
| SpeechLM-P Base   | [960 hrs LibriSpeech](http://www.openslr.org/12) + [40M Text](http://www.openslr.org/11)                      | [100 hrs LibriSpeech](http://www.openslr.org/12)                  | [Google drive](https://drive.google.com/file/d/1mH3N7iKMWYk3rSBJErQPYf3x5ugqDq5x/view?usp=sharing)  |
| SpeechLM-H Base   | [960 hrs LibriSpeech](http://www.openslr.org/12) + [40M Text](http://www.openslr.org/11)                      |                      -                                            | [Google drive](https://drive.google.com/file/d/1eblW8U8f9t-NTuCNRrNHwr-8BeLAUAmQ/view?usp=sharing)  |
| SpeechLM-H Base   | [960 hrs LibriSpeech](http://www.openslr.org/12) + [40M Text](http://www.openslr.org/11)                      | [100 hrs LibriSpeech](http://www.openslr.org/12)                  | [Google drive](https://drive.google.com/file/d/1vXyO5DolbiWiTYZ6pkkKQsu2wJetaPlv/view?usp=sharing)  |
| SpeechLM-P Base   | [960 hrs LibriSpeech](http://www.openslr.org/12) + [40M Text](http://www.openslr.org/11)                      | [En-De CoVoST-2](https://github.com/facebookresearch/covost)      | [Azure Storage](https://valle.blob.core.windows.net/share/speechlm/finetune_covost/checkpoint_ende.pt?sv=2020-08-04&st=2023-03-01T07%3A51%3A05Z&se=2033-03-02T07%3A51%3A00Z&sr=c&sp=rl&sig=QJXmSJG9DbMKf48UDIU1MfzIro8HQOf3sqlNXiflY1I%3D)  |
| SpeechLM-P Base   | [960 hrs LibriSpeech](http://www.openslr.org/12) + [40M Text](http://www.openslr.org/11)                      | [En-Ca CoVoST-2](https://github.com/facebookresearch/covost)      | [Azure Storage](https://valle.blob.core.windows.net/share/speechlm/finetune_covost/checkpoint_enca.pt?sv=2020-08-04&st=2023-03-01T07%3A51%3A05Z&se=2033-03-02T07%3A51%3A00Z&sr=c&sp=rl&sig=QJXmSJG9DbMKf48UDIU1MfzIro8HQOf3sqlNXiflY1I%3D)  |
| SpeechLM-P Base   | [960 hrs LibriSpeech](http://www.openslr.org/12) + [40M Text](http://www.openslr.org/11)                      | [En-Ar CoVoST-2](https://github.com/facebookresearch/covost)      | [Azure Storage](https://valle.blob.core.windows.net/share/speechlm/finetune_covost/checkpoint_enar.pt?sv=2020-08-04&st=2023-03-01T07%3A51%3A05Z&se=2033-03-02T07%3A51%3A00Z&sr=c&sp=rl&sig=QJXmSJG9DbMKf48UDIU1MfzIro8HQOf3sqlNXiflY1I%3D)  |
| SpeechLM-P Base   | [960 hrs LibriSpeech](http://www.openslr.org/12) + [40M Text](http://www.openslr.org/11)                      | [En-Tr CoVoST-2](https://github.com/facebookresearch/covost)      | [Azure Storage](https://valle.blob.core.windows.net/share/speechlm/finetune_covost/checkpoint_entr.pt?sv=2020-08-04&st=2023-03-01T07%3A51%3A05Z&se=2033-03-02T07%3A51%3A00Z&sr=c&sp=rl&sig=QJXmSJG9DbMKf48UDIU1MfzIro8HQOf3sqlNXiflY1I%3D)  |
| SpeechLM-P Large  | [60k hrs LibriLight](https://github.com/facebookresearch/libri-light) + [40M Text](http://www.openslr.org/11) |                      -                                            | [Google drive](https://drive.google.com/file/d/1QjLIgTJKIylVIp5hUkfSjGPtz8Xo7Lky/view?usp=sharing)  |
| SpeechLM-P Large  | [60k hrs LibriLight](https://github.com/facebookresearch/libri-light) + [40M Text](http://www.openslr.org/11) | [960 hrs LibriSpeech](http://www.openslr.org/12)                  | [Google drive](https://drive.google.com/file/d/1YZQDVv096o8Opt0RBnkRiZXYPRDqKZnP/view?usp=sharing)  |
| SpeechLM-P Large  | [60k hrs LibriLight](https://github.com/facebookresearch/libri-light) + [40M Text](http://www.openslr.org/11) | [En-De CoVoST-2](https://github.com/facebookresearch/covost)      | [Google drive](https://drive.google.com/file/d/1qYygNWSc11TQbBI1OzC4ChlR-dNh8t9S/view?usp=sharing)  |
| SpeechLM-P Large  | [60k hrs LibriLight](https://github.com/facebookresearch/libri-light) + [40M Text](http://www.openslr.org/11) | [En-Ca CoVoST-2](https://github.com/facebookresearch/covost)      | [Google drive](https://drive.google.com/file/d/162U88mwso2aVfzzPkEM2nP_vwTpcb57T/view?usp=sharing)  |
| SpeechLM-P Large  | [60k hrs LibriLight](https://github.com/facebookresearch/libri-light) + [40M Text](http://www.openslr.org/11) | [En-Ar CoVoST-2](https://github.com/facebookresearch/covost)      | [Google drive](https://drive.google.com/file/d/1lbTSRXewEeb2t45URunD6EiJcbniyjWW/view?usp=sharing)  |
| SpeechLM-P Large  | [60k hrs LibriLight](https://github.com/facebookresearch/libri-light) + [40M Text](http://www.openslr.org/11) | [En-Tr CoVoST-2](https://github.com/facebookresearch/covost)      | [Google drive](https://drive.google.com/file/d/1Er4I_jHS175pQQph223yKtiiLQ378VvH/view?usp=sharing)  |
| SpeechUT Base (ASR)   | [960 hrs LibriSpeech](http://www.openslr.org/12) + [40M Text](http://www.openslr.org/11)                                                          |                      -                            | [Azure Storage](https://valle.blob.core.windows.net/share/speechut/base_speechut4asr_32gpu_1accum/checkpoint_298_400000.pt?sv=2020-04-08&st=2023-03-08T01%3A39%3A48Z&se=2024-03-09T01%3A39%3A00Z&sr=b&sp=r&sig=l3gJS1D%2BJfLfNfS3z33WjmSMGrOBJ63CvqGGewC6WeU%3D)|
| SpeechUT Base (ASR)   | [960 hrs LibriSpeech](http://www.openslr.org/12) + [40M Text](http://www.openslr.org/11)                                                          | [100 hrs LibriSpeech](http://www.openslr.org/12)  | [Azure Storage](https://valle.blob.core.windows.net/share/speechut/speechut_base_asr100h_checkpoint_best.pt?sv=2020-04-08&st=2023-03-08T01%3A41%3A22Z&se=2024-03-09T01%3A41%3A00Z&sr=b&sp=r&sig=%2B9lpGrqtZXa%2F6n1uZT%2Biey54ky31bYKSJytgfnBbbN4%3D)|
| SpeechUT Large (ASR)  | [60k hrs LibriSpeech](http://www.openslr.org/12) + [40M Text](http://www.openslr.org/11)                                                          |                      -                            | [Azure Storage](https://valle.blob.core.windows.net/share/speechut/large_speechut4asr_32gpu_4accum/checkpoint_22_400k.pt?sv=2020-04-08&st=2023-03-08T01%3A42%3A10Z&se=2024-03-09T01%3A42%3A00Z&sr=b&sp=r&sig=TZNcsHQAqapyj%2BAvpHtl749kZy9flTkWm8P5L4W26qs%3D)|
| SpeechUT Large (ASR)  | [60k hrs LibriSpeech](http://www.openslr.org/12) + [40M Text](http://www.openslr.org/11)                                                          | [960 hrs LibriSpeech](http://www.openslr.org/12)  | [Azure Storage](https://valle.blob.core.windows.net/share/speechut/speechut_large_asr960h_checkpoint_best.pt?sv=2020-04-08&st=2023-03-08T01%3A43%3A02Z&se=2024-03-09T01%3A43%3A00Z&sr=b&sp=r&sig=PmO%2BgSAMXRgMC7GfpS4c%2BrDPsfJGekqUzD5AJm7RrYU%3D)|
| SpeechUT Base (En-De) | [960 hrs LibriSpeech](http://www.openslr.org/12) + [408 hrs MuST-C v1](https://ict.fbk.eu/must-c/) + [4.6M Text](https://www.statmt.org/wmt16/)   |                      -                            | [Azure Storage](https://valle.blob.core.windows.net/share/speechut/base_speechut4ende_32gpu_1accum/checkpoint_217_400000.pt?sv=2020-04-08&st=2023-03-08T01%3A43%3A47Z&se=2024-03-09T01%3A43%3A00Z&sr=b&sp=r&sig=XDEesMdGQ027j7YtpSql1kZtwgfv39gruOuWYlKlJ7w%3D)|
| SpeechUT Base (En-De) | [960 hrs LibriSpeech](http://www.openslr.org/12) + [408 hrs MuST-C v1](https://ict.fbk.eu/must-c/) + [4.6M Text](https://www.statmt.org/wmt16/)   | [En-De MuST-C v1](https://ict.fbk.eu/must-c/)     | [Azure Storage](https://valle.blob.core.windows.net/share/speechut/base_speechut4ende_32gpu_1accum/fineutne_ende_checkpoint_avg.pt?sv=2020-04-08&st=2023-03-08T01%3A44%3A15Z&se=2024-03-09T01%3A44%3A00Z&sr=b&sp=r&sig=8dcenahRg46EJdwiHUalVBJgKra6JoSN7tUxdLAwzOM%3D)|
| SpeechUT Base (En-Es) | [960 hrs LibriSpeech](http://www.openslr.org/12) + [504 hrs MuST-C v1](https://ict.fbk.eu/must-c/) + [15M Text](https://www.statmt.org/wmt13/)    |                      -                            | [Azure Storage](https://valle.blob.core.windows.net/share/speechut/base_speechut4enes_32gpu_1accum/checkpoint_204_400000.pt?sv=2020-04-08&st=2023-03-08T01%3A48%3A16Z&se=2024-03-09T01%3A48%3A00Z&sr=b&sp=r&sig=hWoCM0y0SGZTD4CznC%2F5CejFczkqDYTOaISmlhCAYAU%3D)|
| SpeechUT Base (En-Es) | [960 hrs LibriSpeech](http://www.openslr.org/12) + [504 hrs MuST-C v1](https://ict.fbk.eu/must-c/) + [15M Text](https://www.statmt.org/wmt13/)    | [En-Es MuST-C v1](https://ict.fbk.eu/must-c/)     | [Azure Storage](https://valle.blob.core.windows.net/share/speechut/base_speechut4enes_32gpu_1accum/fineutne_enes_checkpoint_avg.pt?sv=2020-04-08&st=2023-03-08T01%3A48%3A41Z&se=2024-03-09T01%3A48%3A00Z&sr=b&sp=r&sig=KGfzgKfKkDVQI0JxxnS%2BsYdBQzhUqFLQAVYG0OSGBtk%3D)|
| SpeechUT Base (En-Fr) | [960 hrs LibriSpeech](http://www.openslr.org/12) + [492 hrs MuST-C v1](https://ict.fbk.eu/must-c/) + [40M Text](https://www.statmt.org/wmt14/)    |                      -                            | [Azure Storage](https://valle.blob.core.windows.net/share/speechut/base_speechut4enfr_32gpu_1accum/checkpoint_297_600000.pt?sv=2020-04-08&st=2023-03-08T01%3A49%3A09Z&se=2024-03-09T01%3A49%3A00Z&sr=b&sp=r&sig=1eqpXMLCjWpfyd7AiOHGzfk%2B8ZYqWwVWdHk1GqXgoeg%3D)|
| SpeechUT Base (En-Fr) | [960 hrs LibriSpeech](http://www.openslr.org/12) + [492 hrs MuST-C v1](https://ict.fbk.eu/must-c/) + [40M Text](https://www.statmt.org/wmt14/)    | [En-Fr MuST-C v1](https://ict.fbk.eu/must-c/)     | [Azure Storage](https://valle.blob.core.windows.net/share/speechut/base_speechut4enfr_32gpu_1accum/fineutne_enfr_checkpoint.pt?sv=2020-04-08&st=2023-03-08T01%3A49%3A34Z&se=2024-03-09T01%3A49%3A00Z&sr=b&sp=r&sig=i3jMAqvL1Vp7DRjACAbrdoQKhlv2Cmi40%2F14SJ6%2BoiU%3D)|



## SpeechT5 Introduction

Motivated by the success of T5 (Text-To-Text Transfer Transformer) in pre-trained natural language processing models, we propose a unified-modal SpeechT5 framework that explores the encoder-decoder pre-training for self-supervised speech/text representation learning.
The SpeechT5 framework consists of a shared encoder-decoder network and six modal-specific (speech/text) pre/post-nets. 
After preprocessing the input speech/text through the pre-nets, the shared encoder-decoder network models the sequence-to-sequence transformation, and then the post-nets generate the output in the speech/text modality based on the output of the decoder.

<img src="SpeechT5/speecht5_framework.png" alt="se" width="1000" />

Leveraging large-scale unlabeled speech and text data, we pre-train SpeechT5 to learn a unified-modal representation, hoping to improve the modeling capability for both speech and text.
To align the textual and speech information into this unified semantic space, we propose a cross-modal vector quantization approach that randomly mixes up speech/text states with latent units as the interface between encoder and decoder.
Extensive evaluations show the superiority of the proposed SpeechT5 framework on a wide variety of spoken language processing tasks, including automatic speech recognition, speech synthesis, speech translation, voice conversion, speech enhancement, and speaker identification.

<!--
Model introductions, evaluation results, and model inference instructions are located in the corresponding folders. The source code is here [https://github.com/microsoft/SpeechT5/tree/main/SpeechT5].
-->

## SpeechT5 Downstream Task Performance

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

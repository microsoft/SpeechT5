# Speech2C

> [**Speech2C**](https://arxiv.org/abs/2203.17113) (```INTERSPEECH 2022 Submission```): **Pre-Training Transformer Decoder for End-to-End ASR Model with Unpaired Speech Data**

## Pre-Trained and Fine-tuned Models

|  Model   |               Pre-training Dataset               | Fine-tuning Dataset | Model |
| :------: | :----------------------------------------------: | :-----------------: | :-----: |
| Speech2C | [960 hrs LibriSpeech](http://www.openslr.org/12) + [LibriSpeech LM Dataset](https://www.openslr.org/11/) |          -          |   |
| Speech2C | [960 hrs LibriSpeech](http://www.openslr.org/12) + [LibriSpeech LM Dataset](https://www.openslr.org/11/) | [10 hrs LibriSpeech](http://www.openslr.org/12) |   |
| Speech2C | [960 hrs LibriSpeech](http://www.openslr.org/12) + [LibriSpeech LM Dataset](https://www.openslr.org/11/) | [100 hrs LibriSpeech](http://www.openslr.org/12) |   |


## Language Model and Vocabulary
|  Model   |  Dataset | Model | Vocabulary | 
| :------: | :------: | :---: | :--------: |
| LM | [LibriSpeech LM Dataset](https://www.openslr.org/11/) |  |  |

## Results on Librispeech

### Evaluation on the [LibriSpeech](http://www.openslr.org/12) 10hr subset

| Model         |LM                 | test-clean   | test-other   |
| ------------- |-------------      | ----|  ----|
| wav2vec2.0 Base          | -      | 11.1 | 17.6 |
| HuBERT Base              | -      | 10.1 | 16.8 |
| **Speech2C**              | -      | **7.8** | **13.1** |
| wav2vec 2.0 Base         | 4-gram | 4.3  |9.5   |
| wav2vec 2.0 Base   | Transf. |3.2  |7.8   |
| HuBERT Base              | 4-gram	|4.3 |9.4   |
| **Speech2C**              | **Transf.**     | **3.1** | **7.0** |

### Evaluation on the [LibriSpeech](http://www.openslr.org/12) 100hr subset

| Model         |LM                 | test-clean   | test-other   |
| ------------- |-------------      | ----|  ----|
| wav2vec2.0 Base          | -      | 6.1 | 13.3 |
| wav2vec2.0 Large          | -      | 4.7 | 9.0 |
| HuBERT Base              | -      | 6.3 | 13.2 |
| SpeechT5             | -      | 4.4 | 10.4 |
| Baseline                 | -      |  5.0 | 11.9 |
| **Speech2C**                 | - | **4.3**  |**9.0**   |
| wav2vec 2.0 Base         | 4-gram | 3.4  |8.0   |
| wav2vec 2.0 Base         | Transf. | 2.6  | 6.3   |
| HuBERT Base              | 4-gram	| 3.4  |8.1   |
| SpeechT5             | Transf. | 2.4  |5.8   |
| Baseline                 | Transf. | 2.5  |6.3   |
| **Speech2C**                 | **Transf.** | **2.4**  |**5.2**   |


## Reference

If you find our work is useful in your research, please cite the following paper:

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

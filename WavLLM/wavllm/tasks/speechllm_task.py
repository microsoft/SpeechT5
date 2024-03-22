# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
from pathlib import Path
from argparse import Namespace

from fairseq.data import Dictionary, encoders
from ..data.speechllm_dataset import (
    SpeechLLMDataset,
    get_features_or_waveform,
)

from transformers import WhisperProcessor

from ..data.tokenizer import Tokenizer
from transformers import WhisperProcessor, AutoProcessor, AutoFeatureExtractor
from fairseq.tasks import FairseqTask, LegacyFairseqTask, register_task
from dataclasses import dataclass, field
from fairseq.dataclass import ChoiceEnum, FairseqDataclass
from fairseq.data import ResamplingDataset, ConcatDataset
from typing import Optional, Any, List
import os
from fairseq import search, utils

logger = logging.getLogger(__name__)


class Dictionary_for_pad(Dictionary):
    """A mapping from symbols to consecutive integers"""

    def __init__(
        self,
        *,  # begin keyword-only arguments
        pad="<pad>", ## pad_id = 0
        bos="<s>",
        eos="</s>",
        unk="<unk>",
        extra_special_symbols=None,
    ):
        self.bos_word, self.unk_word, self.pad_word, self.eos_word = bos, unk, pad, eos
        self.symbols = []
        self.count = []
        self.indices = {}
        self.pad_index = self.add_symbol(pad) ## let pad_id = 0
        self.bos_index = self.add_symbol(bos) # 1
        self.eos_index = self.add_symbol(eos) # 2
        self.unk_index = self.add_symbol(unk) # 3
        if extra_special_symbols:
            for s in extra_special_symbols:
                self.add_symbol(s)
        self.nspecial = len(self.symbols)

@dataclass
class SpeechLLMTaskConfig(FairseqDataclass):
    data: Optional[str] = field(
        default=None, metadata={"help": "manifest root path"}
    )
    max_source_positions: int = field(
        default=640000,
        metadata={"help": "max number of tokens in the source sequence"},
    )
    max_target_positions: int = field(
        default=6000,
        metadata={"help": "max number of tokens in the target sequence"},
    )
    tokenizer_path: Optional[str] = field(
        default=None, metadata={"help": "LLM tokenizer model path"}
    )
    processor_path: Optional[str] = field(
        default=None, metadata={"help": "audio encoder's processor path"}
    )
    wavlm_processor_path: Optional[str] = field(
        default=None, metadata={"help": "wavlm encoder's processor path"}
    )
    seed: int = field(
        default=12345,
        metadata={"help": "random seed"},
    )
    audio_root: Optional[str] = field(
        default="", metadata={"help": "audio root path"}
    )

    prepend_tgt_lang_tag: bool = field(
        default=False
    )
    shuffle: bool = field(
        default=True
    )
    use_audio_input: bool = field(
        default=True
    )
    is_whisper: bool = field(
        default=False
    )
    whisper_with_decoder: bool = field(
        default=True
    )
    whisper_token_len: int = field(
        default=64
    )
    freeze_audio_encoder: bool = field(
        default=True
    )
    use_sample_rate: int = field(
        default=16000,
        metadata={"help": "sample rate for speech input"},
    )
    reload_speechllm: bool = field(
        default=False
    )
    use_vicuna: bool = field(
        default=False
    )
    sft_stage: bool = field(
        default=False
    )
    use_lora: bool = field(
        default=False
    )
    lora_r: int = field(
        default=8
    )
    lora_alpha: int = field(
        default=32
    )
    lora_scale_train: bool = field(
        default=False
    )
    lora_scale_index: bool = field(
        default=False
    )
    lora_task_index: bool = field(
        default=False
    )
    lora_scale_random: bool = field(
        default=False
    )
    lora_moe: bool = field(
        default=False
    )
    lora_moe_n_experts: int = field(
        default=3
    )
    lora_moe_scaling: bool = field(
        default=False
    )
    llama_2: bool = field(
        default=False
    )
    llama_2_path: str = field(
        default=""
    )
    parallel_mode: bool = field(
        default=False
    )
    enable_fsdp: bool = field(
        default=False
    )
    continue_write_task: bool = field(
        default=False
    )
    only_text: bool = field(
        default=False
    )
    alpaca_text: bool = field(
        default=False
    )
    with_codec: bool = field(
        default=False
    )
    after_adapter: bool = field(
        default=False
    )
    get_codec_online: bool = field(
        default=False
    )
    in_context_infer: bool = field(
        default=False
    )
    in_context_train: bool = field(
        default=False
    )
    pretrained_checkpoint: str = field(
        default=""
    )
    prompt_bulid: bool = field(
        default=False
    )
    prompt_before_speech: bool = field(
        default=False
    )
    use_xformers: bool = field(
        default=False
    )
    small_scale_training: bool = field(
        default=False
    )
    second_stage_update_scale: bool = field(
        default=False
    )
    second_stage_fix_lora: bool = field(
        default=False
    )
    scale_only_one: bool = field(
        default=False
    )
    scale_with_audio: bool = field(
        default=True
    )
    scale_0_1: bool = field(
        default=True
    )
    scale_predict_time: bool = field(
        default=False
    )
    scale_predict_all_dim: bool = field(
        default=False
    )
    scale_predict_all_dim_each_layer: bool = field(
        default=False
    )
    second_stage_update_lora: bool = field(
        default=False
    )
    second_stage_add_lora: bool = field(
        default=False
    )
    lora_only_qv: bool = field(
        default=False
    )
    load_pretrained_model: bool = field(
        default=True
    )
    prompt_loss: bool = field(
        default=False
    )
    use_llama_adapter: bool = field(
        default=False
    )
    codec_weights: bool = field(
        default=False
    )
    use_wavlm: bool = field(
        default=False
    )
    wavlm_weights: bool = field(
        default=False
    )
    wavlm_output_weight: bool = field(
        default=False
    )
    wavlm_output_weight_by_prompts: bool = field(
        default=False
    )
    wavlm_first_7_layers: bool = field(
        default=False
    )
    wavlm_plus: bool = field(
        default=False
    )
    wavlm_plus_weight: bool = field(
        default=False
    )
    wavlm_plus_1layer: bool = field(
        default=False
    )
    wavlm_plus_1layer_5: bool = field(
        default=False
    )
    wavlm_plus_5layer: bool = field(
        default=False
    )
    skip_whisper: bool = field(
        default=False
    )

@register_task("speechllm_task", dataclass=SpeechLLMTaskConfig)
class SpeechLLMTask(FairseqTask):


    def __init__(self, cfg: SpeechLLMTaskConfig):
        """"""
        cfg: SpeechLLMTaskConfig
        super().__init__(cfg)
        logger.info(f"current directory is {os.getcwd()}")
        logger.info(f"Task Config {cfg}")
        self.cfg = cfg

        self.tgt_dict = Dictionary_for_pad.load(f"{self.cfg.data}/dict.txt")
        #self.data_cfg = SpeechLLMDataConfig(Path(args.data) / args.config_yaml)
        #self.speaker_to_id = self._get_speaker_to_id(

    @classmethod
    def setup_task(cls, cfg, **kwargs):
        # data_cfg = SpeechLLMDataConfig(Path(args.data) / args.config_yaml)
        # dict_path = Path(args.data) / data_cfg.vocab_filename
        # if not dict_path.is_file():
        #     raise FileNotFoundError(f"Dict not found: {dict_path.as_posix()}")
        # tgt_dict = Dictionary.load(dict_path.as_posix())
        # logger.info(
        #     f"dictionary size ({data_cfg.vocab_filename}): " f"{len(tgt_dict):,}"
        # )

        # if getattr(args, "train_subset", None) is not None:
        #     if not all(s.startswith("train") for s in args.train_subset.split(",")):
        #         raise ValueError('Train splits should be named like "train*".')
        return cls(cfg)


    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        is_train_split = split.startswith("train")
        # pre_tokenizer = self.build_tokenizer(self.args)
        # bpe_tokenizer = self.build_bpe(self.args)
        text_tokenizer = self.build_tokenizer(self.cfg.tokenizer_path)
        if self.cfg.is_whisper:
            audio_processor = self.bulid_processor(self.cfg.processor_path)
        else:
            audio_processor = None

        if self.cfg.use_wavlm:
            wavlm_feature_extractor = AutoFeatureExtractor.from_pretrained(self.cfg.wavlm_processor_path)
        else:
            wavlm_feature_extractor = None

        self.n_words = text_tokenizer.n_words
        self.tokenizer = text_tokenizer

        datasets = [ 
                SpeechLLMDataset(
                    self.cfg,
                    data_root=self.cfg.data,
                    split=subset,
                    text_tokenizer=text_tokenizer,
                    audio_processor=audio_processor,
                    wavlm_processor=wavlm_feature_extractor,
                    is_train_split=is_train_split,
                    #seed=self.cfg.seed,
                ) for subset in split.split(",")
            ]

        if is_train_split and len(datasets) > 1 and cfg.sampling_alpha != 1.0:
            # temperature-based sampling
            size_ratios = cls.get_size_ratios(datasets, alpha=cfg.sampling_alpha)
            datasets = [
                ResamplingDataset(
                    d, size_ratio=r, seed=self.cfg.seed, epoch=epoch, replace=(r >= 1.0)
                )
                for r, d in zip(size_ratios, datasets)
            ]

        self.datasets[split] = ConcatDataset(datasets) if len(datasets) > 1 else datasets[0]


    @property
    def target_dictionary(self):
        return self.tgt_dict

    @property
    def source_dictionary(self):
        return None

    def max_positions(self):
        return self.cfg.max_source_positions, self.cfg.max_target_positions

    def build_model(self, args):
        model = super().build_model(args)
        return model

    def build_generator(
        self,
        models,
        args,
        seq_gen_cls=None,
        extra_gen_cls_kwargs=None,
    ):

        if extra_gen_cls_kwargs is None:
            extra_gen_cls_kwargs = {}
        return self.build_generator_base(
            models, args, seq_gen_cls=None, extra_gen_cls_kwargs=extra_gen_cls_kwargs
        )

    def build_generator_base(
        self,
        models,
        args,
        seq_gen_cls=None,
        extra_gen_cls_kwargs=None,
        prefix_allowed_tokens_fn=None,
    ):
        """
        Build a :class:`~fairseq.SequenceGenerator` instance for this
        task.

        Args:
            models (List[~fairseq.models.FairseqModel]): ensemble of models
            args (fairseq.dataclass.configs.GenerationConfig):
                configuration object (dataclass) for generation
            extra_gen_cls_kwargs (Dict[str, Any]): extra options to pass
                through to SequenceGenerator
            prefix_allowed_tokens_fn (Callable[[int, torch.Tensor], List[int]]):
                If provided, this function constrains the beam search to
                allowed tokens only at each step. The provided function
                should take 2 arguments: the batch ID (`batch_id: int`)
                and a unidimensional tensor of token ids (`inputs_ids:
                torch.Tensor`). It has to return a `List[int]` with the
                allowed tokens for the next generation step conditioned
                on the previously generated tokens (`inputs_ids`) and
                the batch ID (`batch_id`). This argument is useful for
                constrained generation conditioned on the prefix, as
                described in "Autoregressive Entity Retrieval"
                (https://arxiv.org/abs/2010.00904) and
                https://github.com/facebookresearch/GENRE.
        """
        if getattr(args, "score_reference", False):
            from fairseq.sequence_scorer import SequenceScorer

            return SequenceScorer(
                self.target_dictionary,
                compute_alignment=getattr(args, "print_alignment", False),
            )

        from ..inference.sequence_generator import (
            SequenceGenerator,
        )

        # Choose search strategy. Defaults to Beam Search.
        sampling = getattr(args, "sampling", False)
        sampling_topk = getattr(args, "sampling_topk", -1)
        sampling_topp = getattr(args, "sampling_topp", -1.0)
        diverse_beam_groups = getattr(args, "diverse_beam_groups", -1)
        diverse_beam_strength = getattr(args, "diverse_beam_strength", 0.5)
        match_source_len = getattr(args, "match_source_len", False)
        diversity_rate = getattr(args, "diversity_rate", -1)
        constrained = getattr(args, "constraints", False)
        if prefix_allowed_tokens_fn is None:
            prefix_allowed_tokens_fn = getattr(args, "prefix_allowed_tokens_fn", None)
        if (
            sum(
                int(cond)
                for cond in [
                    sampling,
                    diverse_beam_groups > 0,
                    match_source_len,
                    diversity_rate > 0,
                ]
            )
            > 1
        ):
            raise ValueError("Provided Search parameters are mutually exclusive.")
        assert sampling_topk < 0 or sampling, "--sampling-topk requires --sampling"
        assert sampling_topp < 0 or sampling, "--sampling-topp requires --sampling"

        if sampling:
            search_strategy = search.Sampling(
                self.target_dictionary, sampling_topk, sampling_topp
            )
        elif diverse_beam_groups > 0:
            search_strategy = search.DiverseBeamSearch(
                self.target_dictionary, diverse_beam_groups, diverse_beam_strength
            )
        elif match_source_len:
            # this is useful for tagging applications where the output
            # length should match the input length, so we hardcode the
            # length constraints for simplicity
            search_strategy = search.LengthConstrainedBeamSearch(
                self.target_dictionary,
                min_len_a=1,
                min_len_b=0,
                max_len_a=1,
                max_len_b=0,
            )
        elif diversity_rate > -1:
            search_strategy = search.DiverseSiblingsSearch(
                self.target_dictionary, diversity_rate
            )
        elif constrained:
            search_strategy = search.LexicallyConstrainedBeamSearch(
                self.target_dictionary, args.constraints
            )
        elif prefix_allowed_tokens_fn:
            search_strategy = search.PrefixConstrainedBeamSearch(
                self.target_dictionary, prefix_allowed_tokens_fn
            )
        else:
            search_strategy = search.BeamSearch(self.target_dictionary)

        extra_gen_cls_kwargs = extra_gen_cls_kwargs or {}
        if seq_gen_cls is None:
            seq_gen_cls = SequenceGenerator

        return seq_gen_cls(
            models,
            self.target_dictionary,
            beam_size=getattr(args, "beam", 5),
            max_len_a=getattr(args, "max_len_a", 0),
            max_len_b=getattr(args, "max_len_b", 200),
            min_len=getattr(args, "min_len", 1),
            normalize_scores=(not getattr(args, "unnormalized", False)),
            len_penalty=getattr(args, "lenpen", 1),
            unk_penalty=getattr(args, "unkpen", 0),
            temperature=getattr(args, "temperature", 1.0),
            match_source_len=getattr(args, "match_source_len", False),
            no_repeat_ngram_size=getattr(args, "no_repeat_ngram_size", 0),
            search_strategy=search_strategy,
            n_words=self.n_words,
            **extra_gen_cls_kwargs,
        )

    def build_tokenizer(self, tokenizer_path):
        logger.info(f"tokenizer: {self.cfg.tokenizer_path}")
        text_tokenizer = Tokenizer(self.cfg.tokenizer_path)
        return text_tokenizer

    def bulid_processor(self, processor_path):
        if self.cfg.is_whisper:
            logger.info(f"processor: {processor_path}")
            audio_processor = AutoProcessor.from_pretrained(processor_path)
        else:
            audio_processor = None
        return audio_processor

    def get_interactive_tokens_and_lengths(self, lines, encode_fn):
        n_frames = [get_features_or_waveform(p).shape[0] for p in lines]
        return lines, n_frames

    def build_dataset_for_inference(self, src_tokens, src_lengths, **kwargs):
        return SpeechToTextDataset(
            "interactive", False, self.data_cfg, src_tokens, src_lengths
        )

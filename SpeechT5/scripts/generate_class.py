import ast
import logging
import os
import sys
from argparse import Namespace

import numpy as np
import torch
from fairseq import checkpoint_utils, options, tasks, utils
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq.logging import progress_bar
from omegaconf import DictConfig


def main(cfg: DictConfig):
    if isinstance(cfg, Namespace):
        cfg = convert_namespace_to_omegaconf(cfg)

    assert cfg.common_eval.path is not None, "--path required for generation!"
    assert (
        cfg.generation.replace_unk is None or cfg.dataset.dataset_impl == "raw"
    ), "--replace-unk requires a raw text dataset (--dataset-impl=raw)"

    if cfg.common_eval.results_path is not None:
        os.makedirs(cfg.common_eval.results_path, exist_ok=True)

    return _main(cfg, sys.stdout)


def _main(cfg: DictConfig, output_file):
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=os.environ.get("LOGLEVEL", "INFO").upper(),
        stream=output_file,
    )
    logger = logging.getLogger("speecht5.generate_class")

    utils.import_user_module(cfg.common)

    assert cfg.dataset.batch_size == 1, "only support batch size 1"
    logger.info(cfg)

    # Fix seed for stochastic decoding
    if cfg.common.seed is not None and not cfg.generation.no_seed_provided:
        np.random.seed(cfg.common.seed)
        utils.set_torch_seed(cfg.common.seed)

    use_cuda = torch.cuda.is_available() and not cfg.common.cpu
    if not use_cuda:
        logger.info("generate speech on cpu")

    # build task
    task = tasks.setup_task(cfg.task)

    # Load ensemble
    logger.info("loading model(s) from {}".format(cfg.common_eval.path))
    overrides = ast.literal_eval(cfg.common_eval.model_overrides)
    models, saved_cfg = checkpoint_utils.load_model_ensemble(
        utils.split_paths(cfg.common_eval.path),
        arg_overrides=overrides,
        task=task,
        suffix=cfg.checkpoint.checkpoint_suffix,
        strict=(cfg.checkpoint.checkpoint_shard_count == 1),
        num_shards=cfg.checkpoint.checkpoint_shard_count,
    )
    logger.info(saved_cfg)

    # loading the dataset should happen after the checkpoint has been loaded so we can give it the saved task config
    task.load_dataset(cfg.dataset.gen_subset, task_cfg=saved_cfg.task)

    # optimize ensemble for generation
    for model in models:
        if model is None:
            continue
        if cfg.common.fp16:
            model.half()
        if use_cuda and not cfg.distributed_training.pipeline_model_parallel:
            model.cuda()
        model.prepare_for_inference_(cfg)

    # load dataset (possibly sharded)
    itr = task.get_batch_iterator(
        dataset=task.dataset(cfg.dataset.gen_subset),
        max_tokens=cfg.dataset.max_tokens,
        max_sentences=cfg.dataset.batch_size,
        max_positions=None,
        ignore_invalid_inputs=cfg.dataset.skip_invalid_size_inputs_valid_test,
        required_batch_size_multiple=cfg.dataset.required_batch_size_multiple,
        seed=cfg.common.seed,
        num_shards=cfg.distributed_training.distributed_world_size,
        shard_id=cfg.distributed_training.distributed_rank,
        num_workers=cfg.dataset.num_workers,
        data_buffer_size=cfg.dataset.data_buffer_size,
    ).next_epoch_itr(shuffle=False)
    progress = progress_bar.progress_bar(
        itr,
        log_format=cfg.common.log_format,
        log_interval=cfg.common.log_interval,
        default_log_format=("tqdm" if not cfg.common.no_progress_bar else "simple"),
    )
    
    n_correct = 0
    n_total = 0
    assert hasattr(task.dataset(cfg.dataset.gen_subset), "tgt_dict")
    dict_class = task.dataset(cfg.dataset.gen_subset).tgt_dict
    for i, sample in enumerate(progress):
        if "net_input" not in sample or "source" not in sample["net_input"]:
            continue
        sample = utils.move_to_cuda(sample) if use_cuda else sample
        prefix_tokens = utils.move_to_cuda(
            torch.LongTensor([[dict_class.eos()] for _ in range(len(sample["net_input"]["source"]))])
        )

        outs = task.generate_class(
            models, 
            sample["net_input"],
            prefix_tokens,
        )
        prediction = outs.detach().cpu().tolist()
        categories = [dict_class[predi] for predi in prediction]

        if "target" in sample:
            target = sample["target"].squeeze(1).detach().cpu().tolist()
            labels = [dict_class[tgti] for tgti in target]

        n_total += len(categories)
        if "target" in sample:
            r_correct = []
            for ci, li in zip(categories, labels):
                if ci == li:
                    n_correct += 1
                    r_correct.append(True)
                else:
                    r_correct.append(False)

        logger.info(
            f"{i} (size: {sample['net_input']['source'].shape}) -> {prediction} ({categories}) " +
            f"<- target: {target} ({labels})\t{r_correct}" if "target" in sample else ""
        )
    logger.info(
        f"Accuracy on {cfg.dataset.gen_subset}: {n_correct*100.0/n_total:.3f} ({n_correct}/{n_total})"
    )


def cli_main():
    parser = options.get_generation_parser()
    args = options.parse_args_and_arch(parser)
    main(args)


if __name__ == "__main__":
    cli_main()

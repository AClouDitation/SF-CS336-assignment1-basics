import os
import sys
import pathlib
import numpy as np
import torch
import wandb
import datasets
import logging
import progressbar

from typing import Any, TypeAlias
from jaxtyping import Float, Int
from cs336_basics.bpe_tokenization import ENCODING, tokenizer as bpe_tokenizer
from cs336_basics.transformer import modules, utils
from cs336_basics.training import training_config, adam_w, utils as training_utils
from cs336_basics.common import memmap_utils

_CKPT_FILE_NAME = "model.pth"

Dataset: TypeAlias = datasets.arrow_dataset.Dataset

logging.basicConfig(stream=sys.stdout)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# torch.autograd.set_detect_anomaly(True)


def get_tokenizer(config: training_config.TrainingConfig.TokenizerConfig):
    return bpe_tokenizer.Tokenizer.from_files(**config._asdict())


def preprocess_data(
    tokenizer: bpe_tokenizer.Tokenizer,
    input_path: str | os.PathLike,
    output_dir: str | os.PathLike,
    use_memmap: bool,
) -> np.ndarray[tuple[int], np.dtype[np.uint32]]:

    if use_memmap:
        fn, _ = os.path.splitext(os.path.basename(input_path))
        output_path = os.path.join(output_dir, f"{fn}.npy")

        if not pathlib.Path(output_path).exists():
            return tokenizer.encode_file(
                input_path=input_path,
                output_path=output_path,
            )
        logger.info(
            "Found existing memmap at %s, loading directly from file...",
            output_path,
        )
        return np.memmap(output_path, mode="r", dtype=np.uint32)

    with open(input_path, "r") as f:
        return np.array(tokenizer.encode(f.read()), dtype=np.uint32)


def process_dataset(
    tokenizer: bpe_tokenizer.Tokenizer,
    split: str,
    dataset: Dataset,
    output_dir: str | os.PathLike,
    use_memmap: bool,
) -> np.ndarray[tuple[int], np.dtype[np.uint32]]:

    separator = ""
    if tokenizer.special_tokens:
        separator = tokenizer.special_tokens[0].decode(ENCODING)

    token_ids_iter = tokenizer.encode_iterable(
        f"{example["text"]}{separator}" for example in dataset  # type: ignore
    )
    if use_memmap:
        output_path = os.path.join(
            output_dir, f"{dataset.info.dataset_name}_{split}.npy"
        )
        if pathlib.Path(output_path).exists():
            logger.info(
                "Found existing memmap at %s, loading directly from file...",
                output_path,
            )
            return np.memmap(output_path, mode="r", dtype=np.uint32)

        buffer = []
        chunk_size = 256 * 1024 * 1024  # 256 tokens * 4 bytes/token = 1GiB
        shard = 0
        for token_id in progressbar.progressbar(token_ids_iter):
            buffer.append(token_id)
            if len(buffer) == chunk_size:
                np.save(
                    memmap_utils.get_path_for_shard(output_path, shard),
                    np.array(buffer, dtype=np.uint32),
                )
                buffer.clear()
                shard += 1

        if buffer:
            np.save(
                memmap_utils.get_path_for_shard(output_path, shard),
                np.array(buffer, dtype=np.uint32),
            )
        return memmap_utils.merge_memmaps(output_path, dtype=np.uint32)

    return np.array(list(token_ids_iter), dtype=np.uint32)


def find_ckpt(ckpt_dir: str | os.PathLike, from_ckpt: str) -> tuple[os.PathLike, int]:
    if from_ckpt == "latest":
        ckpt_paths = sorted(pathlib.Path(ckpt_dir).glob("*"))
        assert len(ckpt_paths) > 0, f"No checkpoints found in {ckpt_dir}."

        steps = int(os.path.basename(ckpt_paths[-1]))
    else:
        steps = int(from_ckpt)

    path = pathlib.Path(ckpt_dir) / f"{steps:09d}" / _CKPT_FILE_NAME
    assert path.exists(), f"Checkpoint {path} does not exist."

    return path, steps


class Trainer:

    def __init__(
        self,
        config: training_config.TrainingConfig,
        tokenizer: bpe_tokenizer.Tokenizer,
        wandb_run = None,
    ):
        self._config = config
        self._it = 0
        self._wandb_run = wandb_run

        self._lm = modules.TransformerLM(
            tokenizer.vocab_size,
            device=torch.device(config.trainer.device),
            dtype=torch.float32,  # TODO: make this configurable
            **config.model._asdict(),
        )
        self._opt = adam_w.AdamW(self._lm.parameters(), **config.adamw._asdict())
        if config.trainer.from_ckpt is not None:
            ckpt_path, ckpt_step = find_ckpt(
                config.trainer.ckpt_dir, config.trainer.from_ckpt
            )
            logger.info(f"Loading checkpoint from {ckpt_path}...")
            training_utils.load_ckpt(ckpt_path, self._lm, self._opt)
            logger.info("Model loaded.")
            self._it += ckpt_step

        self._total_steps = self._it + config.trainer.steps

    def _wandb_log(self, info: dict[str, Any]):
        if self._wandb_run is not None:
            self._wandb_run.log(info)

    def _get_batch(self, data: np.ndarray[tuple[int], np.dtype[np.uint32]]) -> tuple[
        Int[torch.Tensor, "batch_size seq_len"],
        Int[torch.Tensor, "batch_size seq_len"],
    ]:
        return training_utils.get_batch(
            data,
            batch_size=self._config.trainer.batch_size,
            seq_len=self._config.model.context_length,
            device=self._config.trainer.device,
        )

    def _gradient_clipping(self):
        return training_utils.gradient_clipping(
            self._lm.parameters(), max_l2_norm=self._config.hyperparam.max_grad_l2_norm
        )

    def _lr_schedule(self) -> float:
        return training_utils.lr_cosine_schedule(
            it=self._it,
            max_learning_rate=self._config.hyperparam.max_learning_rate,
            min_learning_rate=self._config.hyperparam.min_learning_rate,
            warmup_iters=self._config.hyperparam.warmup_iters,
            cosine_cycle_iters=self._total_steps,
        )

    def _train_step(
        self,
        sequences: Float[torch.Tensor, "batch seq_len"],
        targets: Float[torch.Tensor, "batch seq_len"],
    ) -> tuple[Float[torch.Tensor, "batch"], float]:
        lr = self._lr_schedule()
        for param_group in self._opt.param_groups:
            param_group["lr"] = lr

        logits = self._lm(sequences)
        loss = utils.cross_entropy(logits, targets)

        self._opt.zero_grad()
        loss.backward()
        self._gradient_clipping()
        self._opt.step()

        return loss, lr

    def _validate(
        self,
        sequences: Float[torch.Tensor, "batch seq_len"],
        targets: Float[torch.Tensor, "batch seq_len"],
    ) -> Float[torch.Tensor, "batch"]:
        logits = self._lm(sequences)
        return utils.cross_entropy(logits, targets)

    def should_checkpoint(self) -> bool:
        return (
            self._it % self._config.trainer.ckpt_interval == 0
            or self._it == self._total_steps
        )

    def train(
        self,
        training_data: np.ndarray[tuple[int], np.dtype[np.uint32]],
        validation_data: np.ndarray[tuple[int], np.dtype[np.uint32]],
    ):
        bar = progressbar.ProgressBar(min_value=self._it, max_value=self._total_steps)
        bar.start()
        while self._it < self._total_steps:
            self._it += 1
            sequences, targets = self._get_batch(training_data)
            lr, training_losses = self._train_step(sequences=sequences, targets=targets)
            self._wandb_log(
                {"train/lr": lr, "train/loss": training_losses, "step": self._it}
            )

            bar.update(self._it)
            if self.should_checkpoint():
                logger.info(f"Saving checkpoint at step {self._it}...")
                out_dir = self._config.trainer.ckpt_dir / f"{self._it:09d}"
                os.makedirs(out_dir, exist_ok=True)
                training_utils.save_ckpt(
                    self._lm,
                    self._opt,
                    iteration=self._it,
                    out=out_dir / _CKPT_FILE_NAME,
                )
                logger.info("Checkpoint saved.")

                validation_seq, validation_target = self._get_batch(validation_data)
                losses = self._validate(validation_seq, validation_target)
                self._wandb_log(
                    {"validation/loss": losses, "step": self._it}
                )



def main():
    config = training_config.get_config()
    wandb_run = wandb.init(
        entity="actoy", project="toymodel", config=config._asdict()
    ) if config.use_wandb else None

    logger.info("Loading tokenizer with config: %s", config.tokenizer)
    tokenizer = get_tokenizer(config.tokenizer)
    logger.info("Tokenizer loaded.")

    encoded_data_output_dir = config.trainer.tmp_dir / "tokenized_data"
    os.makedirs(encoded_data_output_dir, exist_ok=True)

    if config.trainer.training_dataset_file and config.trainer.validation_dataset_file: 
        logger.info("Loading training data from %s", config.trainer.training_dataset_file)
        training_data = preprocess_data(
            tokenizer,
            input_path=config.trainer.training_dataset_file,
            output_dir=encoded_data_output_dir,
            use_memmap=config.trainer.use_memmap,
        )
        logger.info("Training data loaded, size: %d tokens.", len(training_data))

        logger.info("Loading validation data from %s", config.trainer.validation_dataset_file)
        validation_data = preprocess_data(
            tokenizer,
            input_path=config.trainer.validation_dataset_file,
            output_dir=encoded_data_output_dir,
            use_memmap=config.trainer.use_memmap,
        )
        logger.info("Validation data loaded, size: %d tokens.", len(validation_data))
    elif dataset_path := config.trainer.huggingface_dataset:
        logger.info(
            "Loading dataset %s:%s from Hugging Face",
            dataset_path,
            config.trainer.training_split,
        )
        training_dataset = datasets.load_dataset(
            dataset_path,
            num_proc=os.cpu_count() or 1,
            split=config.trainer.training_split,
        )

        logger.info(
            "Loading dataset %s:%s from Hugging Face",
            dataset_path,
            config.trainer.validation_split,
        )
        validation_dataset = datasets.load_dataset(
            dataset_path,
            num_proc=os.cpu_count() or 1,
            split=config.trainer.validation_split,
        )

        assert isinstance(training_dataset, Dataset)
        assert isinstance(validation_dataset, Dataset)

        logger.info("Tokenizing training data...")
        training_data = process_dataset(
            tokenizer,
            split=config.trainer.training_split,
            dataset=training_dataset,
            output_dir=encoded_data_output_dir,
            use_memmap=config.trainer.use_memmap,
        )
        logger.info("Training data tokenized, size: %d tokens.", len(training_data))

        logger.info("Tokenizing validation data...")
        validation_data = process_dataset(
            tokenizer,
            split=config.trainer.validation_split,
            dataset=validation_dataset,
            output_dir=encoded_data_output_dir,
            use_memmap=config.trainer.use_memmap,
        )
        logger.info("Validation data tokenized, size: %d tokens.", len(validation_data))
    else:
        raise ValueError("No training or validation data provided.")

    logger.info("Initializing trainer...")
    trainer = Trainer(config, tokenizer, wandb_run=wandb_run)

    logger.info("Starting training for %d steps...", config.trainer.steps)
    trainer.train(training_data, validation_data)


if __name__ == "__main__":
    main()

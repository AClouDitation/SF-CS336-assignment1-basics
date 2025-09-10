import os
import pathlib
import numpy as np
import torch
import wandb

from typing import Any
from jaxtyping import Float
from cs336_basics.bpe_tokenization import tokenizer as bpe_tokenizer
from cs336_basics.transformer import modules
from cs336_basics.transformer import utils
from cs336_basics.training import training_config, adam_w, utils as training_utils

_CKPT_FILE_NAME = "model.pth"


def get_tokenizer():
    return bpe_tokenizer.Tokenizer.from_files(
        vocab_file="", merges_file="", special_tokens=["<|endoftext|>"]
    )


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
            tokenizer.encode_file(
                input_path=input_path,
                output_path=output_path,
            )
        return np.memmap(output_path, mode="r", dtype=np.uint32)

    with open(input_path, "r") as f:
        return np.array(tokenizer.encode(f.read()), dtype=np.uint32)


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
        wandb_run: wandb.Run | None = None,
    ):
        self._config = config
        self._it = 1
        self._wandb_run = wandb_run

        self._lm = modules.TransformerLM(tokenizer.vocab_size, **config.model._asdict())
        self._opt = adam_w.AdamW(self._lm.parameters(), **config.adamw._asdict())
        if config.trainer.from_ckpt is not None:
            ckpt_path, ckpt_step = find_ckpt(
                config.trainer.ckpt_dir, config.trainer.from_ckpt
            )
            print(f"Loading checkpoint from {ckpt_path}...")
            training_utils.load_ckpt(ckpt_path, self._lm, self._opt)
            print("Model loaded.")
            self._it += ckpt_step

        self._total_steps = self._it + config.trainer.steps

    def _wandb_log(self, info: dict[str, Any]):
        if self._wandb_run is not None:
            self._wandb_run.log(info)

    def _get_batch(self, data: np.ndarray[tuple[int], np.dtype[np.uint32]]):
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
        losses = utils.cross_entropy(logits, targets)

        self._opt.zero_grad()
        losses.backward()
        self._gradient_clipping()
        self._opt.step()

        return losses, lr

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
        while self._it <= self._total_steps:
            sequences, targets = self._get_batch(training_data)
            lr, training_losses = self._train_step(sequences=sequences, targets=targets)
            self._wandb_log(
                {"train/lr": lr, "train/loss": training_losses, "step": self._it}
            )

            if self.should_checkpoint():
                training_utils.save_ckpt(
                    self._lm,
                    self._opt,
                    iteration=self._it,
                    out=self._config.trainer.ckpt_dir / f"{self._it:09d}",
                )
                validation_seq, validation_target = self._get_batch(validation_data)
                losses = self._validate(validation_seq, validation_target)
                # TODO: log to wandb
                self._wandb_log(
                    {"validation/loss": losses, "step": self._it}
                )

            self._it += 1


def main():
    config = training_config.get_config()
    wandb_run = wandb.init(entity="aclouditation", project="cs336", config={})

    tokenizer = get_tokenizer()
    training_data = preprocess_data(
        tokenizer,
        input_path=config.trainer.training_dataset_file,
        output_dir=os.path.join(config.trainer.tmp_dir, "tokenized_data"),
        use_memmap=config.trainer.use_memmap,
    )

    validation_data = preprocess_data(
        tokenizer,
        input_path=config.trainer.validation_dataset_file,
        output_dir=os.path.join(config.trainer.tmp_dir, "tokenized_data"),
        use_memmap=config.trainer.use_memmap,
    )

    trainer = Trainer(config, tokenizer, wandb_run=wandb_run)
    trainer.train(training_data, validation_data)


if __name__ == "__main__":
    main()

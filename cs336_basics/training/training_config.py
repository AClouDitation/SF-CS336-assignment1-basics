import argparse
import pathlib

from typing import NamedTuple


parser = argparse.ArgumentParser(description="Train a transformer language model.")

# Trainer configs
parser.add_argument("--tmp_dir", type=str, default="~/learning/SF_CS_336/data/tmp")
parser.add_argument("--ckpt_dir", type=str, default="~/learning/SF_CS_336/data/ckpts")
parser.add_argument("--ckpt_interval", type=int, default=1000)
parser.add_argument("--training_dataset_file", type=str, required=True)
parser.add_argument("--validation_dataset_file", type=str, required=True)
parser.add_argument("--from_ckpt", type=str, default=None)
parser.add_argument("--steps", type=int, default=10_000)
parser.add_argument("--use_memmap", action="store_true")
parser.add_argument("--batch_size", type=int, default=4)
parser.add_argument("--device", type=str, default="cpu")

# Model structure
parser.add_argument("--context_length", type=int, default=1024)
parser.add_argument("--num_layers", type=int, default=12)
parser.add_argument("--d_model", type=int, default=768)
parser.add_argument("--num_heads", type=int, default=12)
parser.add_argument("--d_ff", type=int, default=3072)


# Hyperparameters
parser.add_argument("--rope_theta", type=float, default=0.1)
parser.add_argument("--max_learning_rate", type=float, default=1e-3)
parser.add_argument("--min_learning_rate", type=float, default=1e-5)
parser.add_argument("--warmup_iters", type=int, default=100)
parser.add_argument("--max_grad_l2_norm", type=float, default=1.0)
parser.add_argument("--adamw_betas", type=float, nargs=2, default=(0.9, 0.999))
parser.add_argument("--adamw_eps", type=float, default=1e-8)
parser.add_argument("--adamw_weight_decay", type=float, default=0.01)


class TrainingConfig(NamedTuple):
    class TrainerConfig(NamedTuple):
        tmp_dir: pathlib.Path
        ckpt_dir: pathlib.Path
        ckpt_interval: int
        training_dataset_file: pathlib.Path
        validation_dataset_file: pathlib.Path
        from_ckpt: str | None
        steps: int
        use_memmap: bool
        batch_size: int
        device: str

    class ModelConfig(NamedTuple):
        context_length: int
        num_layers: int
        d_model: int
        num_heads: int
        d_ff: int
        rope_theta: float

    class AdamWConfig(NamedTuple):
        betas: tuple[float, float]
        eps: float
        weight_decay: float

    class HyperParameters(NamedTuple):
        max_learning_rate: float
        min_learning_rate: float
        warmup_iters: int
        max_grad_l2_norm: float

    trainer: TrainerConfig
    model: ModelConfig
    adamw: AdamWConfig
    hyperparam: HyperParameters



def get_config() -> TrainingConfig:
    args = parser.parse_args()

    tmp_dir = pathlib.Path(args.tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    ckpt_dir = pathlib.Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    training_dataset_file = pathlib.Path(args.training_dataset_file)
    assert (
        training_dataset_file.exists()
    ), f"Training dataset file {training_dataset_file} does not exist."

    validation_dataset_file = pathlib.Path(args.validation_dataset_file)
    assert (
        validation_dataset_file.exists()
    ), f"Validation dataset file {validation_dataset_file} does not exist."

    return TrainingConfig(
        trainer=TrainingConfig.TrainerConfig(
            tmp_dir=tmp_dir,
            ckpt_dir=ckpt_dir,
            ckpt_interval=args.ckpt_interval,
            training_dataset_file=training_dataset_file,
            validation_dataset_file=validation_dataset_file,
            from_ckpt=args.from_ckpt,
            steps=args.steps,
            use_memmap=args.use_memmap,
            batch_size=args.batch_size,
            device=args.device,
        ),
        model=TrainingConfig.ModelConfig(
            context_length=args.context_length,
            num_layers=args.num_layers,
            d_model=args.d_model,
            num_heads=args.num_heads,
            d_ff=args.d_ff,
            rope_theta=args.rope_theta,
        ),
        adamw=TrainingConfig.AdamWConfig(
            betas=tuple(args.adamw_betas),
            eps=args.adamw_eps,
            weight_decay=args.adamw_weight_decay,
        ),
        hyperparam=TrainingConfig.HyperParameters(
            max_learning_rate=args.max_learning_rate,
            min_learning_rate=args.min_learning_rate,
            warmup_iters=args.warmup_iters,
            max_grad_l2_norm=args.max_grad_l2_norm,
        ),
    )

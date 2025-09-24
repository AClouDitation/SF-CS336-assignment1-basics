import argparse
import pathlib

from typing import NamedTuple


parser = argparse.ArgumentParser(description="Inference with a transformer language model.")

# Tokenizer configs
parser.add_argument("--vocab_file", type=str, required=True)
parser.add_argument("--merges_file", type=str, required=True)
parser.add_argument("--special_tokens", type=list, default=["<|endoftext|>"])

# Model
parser.add_argument("--ckpt_path", type=str, required=True)
parser.add_argument("--context_length", type=int, default=256)
parser.add_argument("--num_layers", type=int, default=4)
parser.add_argument("--d_model", type=int, default=512)
parser.add_argument("--num_heads", type=int, default=16)
parser.add_argument("--d_ff", type=int, default=1344)
parser.add_argument("--rope_theta", type=float, default=10000)

# Decoding
parser.add_argument("--temperature", type=float, default=0.0)
parser.add_argument("--top_p", type=float, default=0.9)


class InferenceConfig(NamedTuple):
    class TokenizerConfig(NamedTuple):
        vocab_file: pathlib.Path
        merges_file: pathlib.Path
        special_tokens: list[str]

    class ModelConfig(NamedTuple):
        context_length: int
        num_layers: int
        d_model: int
        num_heads: int
        d_ff: int
        rope_theta: float
    
    class DecodingConfig(NamedTuple):
        temperature: float
        top_p: float
    
    ckpt_path: pathlib.Path
    tokenizer: TokenizerConfig
    model: ModelConfig
    decoding: DecodingConfig 
    

def get_config() -> InferenceConfig:
    args = parser.parse_args()

    vocab_file = pathlib.Path(args.vocab_file)
    assert vocab_file.exists(), f"Vocabulary file {vocab_file} does not exist."

    merges_file = pathlib.Path(args.merges_file)
    assert merges_file.exists(), f"Merges file {merges_file} does not exist."

    ckpt_path = pathlib.Path(args.ckpt_path)
    assert ckpt_path.exists(), f"Checkpoint file {ckpt_path} does not exist."

    return InferenceConfig(
        ckpt_path=ckpt_path,
        tokenizer=InferenceConfig.TokenizerConfig(
            vocab_file=vocab_file,
            merges_file=merges_file,
            special_tokens=args.special_tokens,
        ),
        model=InferenceConfig.ModelConfig(
            context_length=args.context_length,
            num_layers=args.num_layers,
            d_model=args.d_model,
            num_heads=args.num_heads,
            d_ff=args.d_ff,
            rope_theta=args.rope_theta,
        ),
        decoding=InferenceConfig.DecodingConfig(
            temperature=args.temperature,
            top_p=args.top_p,
        )
    )
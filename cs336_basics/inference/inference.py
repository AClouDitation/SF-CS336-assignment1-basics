import torch
import sys
import logging
import random

from jaxtyping import Float
from cs336_basics.bpe_tokenization import tokenizer as bpe_tokenizer
from cs336_basics.transformer import modules, utils
from cs336_basics.inference import inference_config
from cs336_basics.training import utils as training_utils


logging.basicConfig(stream=sys.stdout)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Inferencer:

    def __init__(
        self,
        tokenizer: bpe_tokenizer.Tokenizer,
        config: inference_config.InferenceConfig,
    ):
        self._config = config
        self._tokenizer = tokenizer
        self._lm = modules.TransformerLM(
            vocab_size=tokenizer.vocab_size, **config.model._asdict()
        )
        training_utils.load_ckpt(config.ckpt_path, self._lm)

    def generate(self, prompt: str, max_output_len: int = 1024):
        output = prompt
        input_ids = self._tokenizer.encode(output)
        while len(input_ids) < max_output_len:
            input_ids = input_ids[-self._config.model.context_length :]
            logits = self._lm(torch.tensor(input_ids))[-1, :]

            if self._config.decoding.temperature > 0.0:
                logits = logits / self._config.decoding.temperature
            logits: Float[torch.Tensor, "d_vocab"] = utils.soft_max(logits, dim=-1)

            max_prob = 1.0
            if self._config.decoding.top_p < 1.0:
                max_prob = self._config.decoding.top_p
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = sorted_logits.cumsum(dim=-1)
                remove_start_idx = torch.nonzero(
                    cumulative_probs > self._config.decoding.top_p
                )[0]
                mask = torch.zeros_like(logits, dtype=torch.bool)
                mask.scatter_(
                    0, sorted_indices, torch.arange(logits.shape[-1]) > remove_start_idx
                )
                logits = logits.masked_fill(mask, 0.0)

            cum_logits = logits.cumsum(dim=-1)
            output_token_id = int(
                torch.searchsorted(
                    cum_logits, torch.tensor(random.uniform(0, max_prob))
                ).item()
            )
            input_ids.append(output_token_id)
            token = self._tokenizer.decode([output_token_id])
            if token.encode(bpe_tokenizer.ENCODING) in self._tokenizer.special_tokens:
                break
            print(token, end="", flush=True)
        print()


if __name__ == "__main__":

    config = inference_config.get_config()
    logger.info("Loading tokenizer with config: %s", config.tokenizer)
    tokenizer = bpe_tokenizer.Tokenizer.from_files(**config.tokenizer._asdict())
    logger.info("Tokenizer loaded.")

    inferencer = Inferencer(tokenizer, config)

    prompt = input("Enter a prompt: ")
    inferencer.generate(prompt)

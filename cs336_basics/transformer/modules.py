import torch
import einx

from torch import Tensor
from jaxtyping import Float, Int
from cs336_basics.transformer import utils


class Linear(torch.nn.Module):

    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()

        self._in_features = in_features
        self._out_features = out_features
        self.weight: Float[Tensor, "d_out d_in"] = torch.nn.Parameter(
            torch.empty(out_features, in_features, device=device, dtype=dtype)
        )
        std = (2 / (in_features + out_features)) ** 0.5
        torch.nn.init.trunc_normal_(self.weight, std=std, a=-3 * std, b=3 * std)

    def forward(self, x: Float[Tensor, "... d_in"]) -> Float[Tensor, "... d_out"]:
        return einx.dot("... d_in, d_out d_in -> ... d_out", x, self.weight)

    def total_flops(self, tensor_shape: torch.Size) -> int:
        return 2 * tensor_shape[:-2].numel() * self._in_features * self._out_features


class Embedding(torch.nn.Module):

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.weight: Float[Tensor, "d_vocab d_model"] = torch.nn.Parameter(
            torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype)
        )
        torch.nn.init.trunc_normal_(self.weight, std=1, a=-3, b=3)

    def forward(self, token_ids: Int[Tensor, "... seq"]) -> Float[Tensor, "... seq d_model"]:
        return self.weight[token_ids]
    
    def total_flops(self) -> int:
        return 0


class RMSNorm(torch.nn.Module):

    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self._d_model = d_model
        self._eps = eps
        self._dtype = dtype
        self.weight: Float[Tensor, "d_model"] = torch.nn.Parameter(
            torch.ones(d_model, device=device, dtype=dtype)
        )

    def forward(self, x: Float[Tensor, "... d_model"]) -> Float[Tensor, "... d_model"]:
        x = x.to(torch.float32)
        rms = (einx.sum("... d_model -> ...", x**2) / self._d_model + self._eps) ** 0.5
        x = einx.divide("... d_model, ... -> ... d_model", x, rms)
        x = einx.multiply("... d_model, d_model -> ... d_model", x, self.weight)
        return x.to(self._dtype)
    
    def total_flops(self, tensor_shape: torch.Size) -> int:
        return 7 * tensor_shape.numel()


class SwiGLU(torch.nn.Module):

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self._d_ff = d_ff
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x: Float[Tensor, "... d_model"]) -> Float[Tensor, "... d_model"]:
        return self.w2(utils.silu(self.w1(x)) * self.w3(x))
    
    def total_flops(self, tensor_shape: torch.Size) -> int:
        # 3 * projection: 6 * numel * d_ff
        # + silu: 3 * numel * d_ff
        # + elementwise multiply: numel * d_ff
        return 10 * tensor_shape.numel() * self._d_ff


class RotaryPositionalEmbedding(torch.nn.Module):

    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device: torch.device | None = None,
    ):
        assert d_k % 2 == 0, "d_k must be even for RoPE"

        super().__init__()
        seq = torch.arange(0, max_seq_len, device=device)
        k = torch.arange(0, d_k, 2, device=device)
        theta_mat = einx.dot(
            "seq, half_d_k -> seq half_d_k", seq, 1 / theta ** (k / d_k)
        )
        cos = theta_mat.cos().repeat_interleave(2, dim=-1)
        sin = theta_mat.sin().repeat_interleave(2, dim=-1)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def forward(
        self,
        x: Float[Tensor, "... seq_len d_k"],
        token_positions: Float[Tensor, "... seq_len"],
    ) -> Float[Tensor, "... seq_len d_k"]:
        swapped_x = torch.stack([-x[..., 1::2], x[..., ::2]], dim=-1).reshape_as(x)
        return x * self.cos[token_positions] + swapped_x * self.sin[token_positions]  # type: ignore
    
    def total_flops(self, tensor_shape: torch.Size) -> int:
        return 3 * tensor_shape.numel()


class MultiHeadSelfAttention(torch.nn.Module):
    """Multi-Head Self Attention

    FLOPS: 6 * batch_size * seq_len * d_model
         + 2 * batch_size * seq_len ^ 2 * (2 * d_model + 6)
         + 2 * batch_size * d_model ^ 3
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        rope_module: RotaryPositionalEmbedding | None = None,
    ):
        assert (
            d_model % num_heads == 0
        ), "d_model must be divisible by num_heads, d_model=%d, num_heads=%d" % (
            d_model,
            num_heads,
        )
        super().__init__()

        self.num_heads = num_heads
        self._rope_module = rope_module

        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.output_proj = Linear(d_model, d_model, device=device, dtype=dtype)

    def forward(
        self,
        x: Float[Tensor, "... seq d_model"],
        token_positions: Float[Tensor, "... seq"] | None = None,
    ) -> Float[Tensor, "... d_model"]:
        Q: Float[Tensor, "... h seq hd_k"] = einx.rearrange(
            "... seq (h hd_k) -> ... h seq hd_k", self.q_proj(x), h=self.num_heads
        )  # type: ignore
        K: Float[Tensor, "... h seq hd_k"] = einx.rearrange(
            "... seq (h hd_k) -> ... h seq hd_k", self.k_proj(x), h=self.num_heads
        )  # type: ignore
        V: Float[Tensor, "... h seq hd_v"] = einx.rearrange(
            "... seq (h hd_v) -> ... h seq hd_v", self.v_proj(x), h=self.num_heads
        )  # type: ignore

        seq_len = x.shape[-2]
        if self._rope_module is not None:
            if token_positions is None:
                token_positions = torch.arange(seq_len, device=Q.device)
            Q = self._rope_module(Q, token_positions)
            K = self._rope_module(K, token_positions)

        mask = torch.tril(torch.ones(seq_len, seq_len)).bool()
        mask = mask.to(Q.device)

        attn: Float[Tensor, "... seq d_model"] = einx.rearrange(
            "... h seq hd_k -> ... seq (h hd_k)",
            utils.scaled_dot_product_attention(Q, K, V, mask),
        )  # type:ignore

        return self.output_proj(attn)

    def total_flops(self, tensor_shape: torch.Size) -> int:
        total = (
            self.q_proj.total_flops(tensor_shape)
            + self.k_proj.total_flops(tensor_shape)
            + self.v_proj.total_flops(tensor_shape)
        )

        if self._rope_module is not None:
            total += 2 * self._rope_module.total_flops(tensor_shape)

        d_model, seq_len = tensor_shape[-1], tensor_shape[-2]
        batch_size = tensor_shape[:-3].numel() if len(tensor_shape) >= 3 else 1
        head_dim = d_model // self.num_heads

        return (
            self.q_proj.total_flops(tensor_shape)
            + self.k_proj.total_flops(tensor_shape)
            + self.v_proj.total_flops(tensor_shape)
            + self.output_proj.total_flops(tensor_shape)
            # RoPE
            + 2 * self._rope_module.total_flops(tensor_shape)
            if self._rope_module is not None
            else 0
            # Multi Head Self Attention
            + 2 * batch_size * seq_len**2 * head_dim  #  Q @ K
            + batch_size * seq_len**2  # scaling
            + 5 * batch_size * seq_len**2  # soft max
            + 2 * batch_size * seq_len**2 * head_dim  # V @ softmax
        )


class TransformerBlock(torch.nn.Module):

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        rope_module: RotaryPositionalEmbedding | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ): 
        super().__init__()
        self.attn = MultiHeadSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            rope_module=rope_module,
            device=device,
            dtype=dtype,
        )
        self.ffn = SwiGLU(d_model=d_model, d_ff=d_ff, device=device, dtype=dtype)
        self.ln1 = RMSNorm(d_model=d_model, device=device, dtype=dtype)
        self.ln2 = RMSNorm(d_model=d_model, device=device, dtype=dtype)

    def forward(
        self, x: Float[Tensor, "... sdq d_model"]
    ) -> Float[Tensor, "... seq d_model"]:
        x += self.attn(self.ln1(x))
        return x + self.ffn(self.ln2(x))

    def total_flops(self, tensor_shape: torch.Size) -> int:
        return (
            self.ln1.total_flops(tensor_shape)
            + self.attn.total_flops(tensor_shape)
            + self.ln2.total_flops(tensor_shape)
            + self.ffn.total_flops(tensor_shape)
            + 2 * tensor_shape.numel()  # Additions
        )


class TransformerLM(torch.nn.Module):

    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        num_layers: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self._max_seq_len = context_length
        self._shared_rope = RotaryPositionalEmbedding(
            rope_theta, d_model // num_heads, context_length, device=device
        )
        self.token_embeddings = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        self.layers = torch.nn.ModuleList(
            TransformerBlock(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                rope_module=self._shared_rope,
                device=device,
                dtype=dtype,
            )
            for _ in range(num_layers)
        )
        self.ln_final = RMSNorm(d_model, device=device, dtype=dtype)
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)

    def forward(self, token_ids: Int[Tensor, "... seq"]) -> Float[Tensor, "... seq vocab_size"]:
        assert token_ids.shape[-1] <= self._max_seq_len, (
            "Input sequence length %d exceeds maximum %d" % (token_ids.shape[-1], self._max_seq_len)
        )
        x: Float[Tensor, "... seq d_model"] = self.token_embeddings(token_ids)
        for layer in self.layers:
            x = layer(x)

        return self.lm_head(self.ln_final(x))

    def total_flops(self, tensor_shape: torch.Size) -> int:
        return (
            self.token_embeddings.total_flops()
            + sum(layer.total_flops(tensor_shape) for layer in self.layers)  # type: ignore
            + self.ln_final.total_flops(tensor_shape)
            + self.lm_head.total_flops(tensor_shape)
        )

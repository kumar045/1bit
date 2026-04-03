import math
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutput


# =========================================================
# Config
# =========================================================

class BitLlamaConfig(PretrainedConfig):
    model_type = "bitllama"

    def __init__(
        self,
        vocab_size=32000,
        hidden_size=768,
        intermediate_size=2048,
        num_hidden_layers=12,
        num_attention_heads=12,
        max_position_embeddings=2048,
        rms_norm_eps=1e-5,
        rope_theta=10000.0,
        group_size=128,
        initializer_range=0.02,
        use_cache=False,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        ce_weight=0.5,
        kl_weight=0.5,
        distill_temperature=2.0,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.max_position_embeddings = max_position_embeddings
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.group_size = group_size
        self.initializer_range = initializer_range
        self.use_cache = use_cache

        self.ce_weight = ce_weight
        self.kl_weight = kl_weight
        self.distill_temperature = distill_temperature

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            **kwargs,
        )


# =========================================================
# Quantization helpers
# =========================================================

def ternary_quantize_groupwise(
    w: torch.Tensor,
    group_size: int = 128,
    threshold: float = 0.5,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Groupwise ternary quantization over the last dimension.
    Maps to {-1, 0, +1} and rescales back with per-group absmean scale.
    """
    orig_shape = w.shape
    last_dim = w.shape[-1]

    pad = (group_size - (last_dim % group_size)) % group_size
    if pad > 0:
        w = F.pad(w, (0, pad))

    padded_last_dim = w.shape[-1]
    num_groups = padded_last_dim // group_size

    w_grouped = w.view(-1, num_groups, group_size)
    scale = w_grouped.abs().mean(dim=-1, keepdim=True).clamp_min(eps)
    w_norm = w_grouped / scale

    q = torch.zeros_like(w_norm)
    q = torch.where(w_norm > threshold, torch.ones_like(q), q)
    q = torch.where(w_norm < -threshold, -torch.ones_like(q), q)

    w_q = (q * scale).view(*w.shape)

    if pad > 0:
        w_q = w_q[..., :last_dim]

    return w_q.view(orig_shape)


def quantize_with_ste(w_real: torch.Tensor, group_size: int) -> torch.Tensor:
    """
    Straight-through estimator:
    forward uses quantized weights, backward flows to real weights.
    """
    w_q = ternary_quantize_groupwise(w_real, group_size=group_size)
    return w_real + (w_q - w_real).detach()


# =========================================================
# Core layers
# =========================================================

class BitRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_fp = x.float()
        x_norm = x_fp * torch.rsqrt(x_fp.pow(2).mean(-1, keepdim=True) + self.eps)
        return x_norm.type_as(x) * self.weight


class BitLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, group_size: int, bias: bool = False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            bound = 1 / math.sqrt(self.in_features)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w_q = quantize_with_ste(self.weight, self.group_size)
        return F.linear(x, w_q, self.bias)


# =========================================================
# Rotary embeddings
# =========================================================

def precompute_rope_frequencies(head_dim: int, max_position_embeddings: int, theta: float):
    half_dim = head_dim // 2
    freq_seq = torch.arange(half_dim, dtype=torch.float32)
    inv_freq = 1.0 / (theta ** (freq_seq / half_dim))
    positions = torch.arange(max_position_embeddings, dtype=torch.float32)
    freqs = torch.outer(positions, inv_freq)
    return torch.cos(freqs), torch.sin(freqs)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, position_ids: Optional[torch.Tensor] = None):
    # x: [B, H, T, D]
    d = x.shape[-1]
    x1, x2 = x[..., : d // 2], x[..., d // 2 :]

    if position_ids is None:
        cos = cos[: x.size(-2)].unsqueeze(0).unsqueeze(0).to(x.device, x.dtype)
        sin = sin[: x.size(-2)].unsqueeze(0).unsqueeze(0).to(x.device, x.dtype)
    else:
        # position_ids: [B, T]
        cos = cos[position_ids].unsqueeze(1).to(x.device, x.dtype)  # [B, 1, T, D/2]
        sin = sin[position_ids].unsqueeze(1).to(x.device, x.dtype)

    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


# =========================================================
# Attention / MLP / Block
# =========================================================

class BitLlamaAttention(nn.Module):
    def __init__(self, config: BitLlamaConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        assert self.hidden_size % self.num_heads == 0

        self.q_proj = BitLinear(self.hidden_size, self.hidden_size, config.group_size, bias=False)
        self.k_proj = BitLinear(self.hidden_size, self.hidden_size, config.group_size, bias=False)
        self.v_proj = BitLinear(self.hidden_size, self.hidden_size, config.group_size, bias=False)
        self.o_proj = BitLinear(self.hidden_size, self.hidden_size, config.group_size, bias=False)

        cos, sin = precompute_rope_frequencies(
            self.head_dim,
            config.max_position_embeddings,
            config.rope_theta,
        )
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        bsz, seqlen, _ = hidden_states.shape

        q = self.q_proj(hidden_states).view(bsz, seqlen, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(bsz, seqlen, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(bsz, seqlen, self.num_heads, self.head_dim).transpose(1, 2)

        q = apply_rope(q, self.rope_cos, self.rope_sin, position_ids=position_ids)
        k = apply_rope(k, self.rope_cos, self.rope_sin, position_ids=position_ids)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        causal_mask = torch.triu(
            torch.ones(seqlen, seqlen, device=hidden_states.device, dtype=torch.bool),
            diagonal=1,
        )
        attn_scores = attn_scores.masked_fill(causal_mask, float("-inf"))

        if attention_mask is not None:
            # attention_mask expected [B, T] with 1 for tokens, 0 for pad
            expanded = (attention_mask[:, None, None, :] == 0)
            attn_scores = attn_scores.masked_fill(expanded, float("-inf"))

        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(attn_probs, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, seqlen, self.hidden_size)

        return self.o_proj(attn_output)


class BitLlamaMLP(nn.Module):
    def __init__(self, config: BitLlamaConfig):
        super().__init__()
        self.gate_proj = BitLinear(config.hidden_size, config.intermediate_size, config.group_size, bias=False)
        self.up_proj = BitLinear(config.hidden_size, config.intermediate_size, config.group_size, bias=False)
        self.down_proj = BitLinear(config.intermediate_size, config.hidden_size, config.group_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class BitLlamaDecoderLayer(nn.Module):
    def __init__(self, config: BitLlamaConfig):
        super().__init__()
        self.input_layernorm = BitRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = BitRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.self_attn = BitLlamaAttention(config)
        self.mlp = BitLlamaMLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


# =========================================================
# Base model
# =========================================================

class BitLlamaPreTrainedModel(PreTrainedModel):
    config_class = BitLlamaConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = False

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, BitLinear)):
            if hasattr(module, "weight") and module.weight is not None:
                nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if hasattr(module, "bias") and module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)


class BitLlamaModel(BitLlamaPreTrainedModel):
    def __init__(self, config: BitLlamaConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [BitLlamaDecoderLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.norm = BitRMSNorm(config.hidden_size, config.rms_norm_eps)

        self.post_init()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape

        if position_ids is None:
            position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)

        hidden_states = self.embed_tokens(input_ids)

        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
            )

        hidden_states = self.norm(hidden_states)
        return hidden_states


# =========================================================
# Causal LM head
# =========================================================

class BitLlamaForCausalLM(BitLlamaPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: BitLlamaConfig):
        super().__init__(config)
        self.model = BitLlamaModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def tie_weights(self):
        self._tie_or_clone_weights(self.lm_head, self.model.embed_tokens)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        teacher_logits: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> CausalLMOutput:
        hidden_states = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()

            ce_loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )
            loss = self.config.ce_weight * ce_loss

            if teacher_logits is not None:
                T = self.config.distill_temperature
                student_log_probs = F.log_softmax(shift_logits / T, dim=-1)
                teacher_probs = F.softmax(teacher_logits[:, :-1, :] / T, dim=-1)

                kl_loss = F.kl_div(
                    student_log_probs,
                    teacher_probs,
                    reduction="batchmean",
                ) * (T * T)

                loss = loss + self.config.kl_weight * kl_loss

        return CausalLMOutput(
            loss=loss,
            logits=logits,
  )

import math
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import PretrainedConfig, PreTrainedModel


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
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        ce_weight=0.5,
        kl_weight=0.4,
        hidden_weight=0.1,
        distill_temperature=2.0,
        output_hidden_states=True,
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

        self.ce_weight = ce_weight
        self.kl_weight = kl_weight
        self.hidden_weight = hidden_weight
        self.distill_temperature = distill_temperature
        self.output_hidden_states = output_hidden_states

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            **kwargs,
        )


def ternary_quantize_groupwise(
    w: torch.Tensor,
    group_size: int = 128,
    threshold: float = 0.5,
    eps: float = 1e-8,
) -> torch.Tensor:
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
    w_q = ternary_quantize_groupwise(w_real, group_size=group_size)
    return w_real + (w_q - w_real).detach()


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


def precompute_rope_frequencies(head_dim: int, max_position_embeddings: int, theta: float):
    half_dim = head_dim // 2
    freq_seq = torch.arange(half_dim, dtype=torch.float32)
    inv_freq = 1.0 / (theta ** (freq_seq / half_dim))
    positions = torch.arange(max_position_embeddings, dtype=torch.float32)
    freqs = torch.outer(positions, inv_freq)
    return torch.cos(freqs), torch.sin(freqs)


def apply_rope(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: Optional[torch.Tensor] = None,
):
    d = x.shape[-1]
    x1, x2 = x[..., : d // 2], x[..., d // 2 :]

    if position_ids is None:
        cos = cos[: x.size(-2)].unsqueeze(0).unsqueeze(0).to(x.device, x.dtype)
        sin = sin[: x.size(-2)].unsqueeze(0).unsqueeze(0).to(x.device, x.dtype)
    else:
        cos = cos[position_ids].unsqueeze(1).to(x.device, x.dtype)
        sin = sin[position_ids].unsqueeze(1).to(x.device, x.dtype)

    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


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

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        causal_mask = torch.triu(
            torch.ones(seqlen, seqlen, device=hidden_states.device, dtype=torch.bool),
            diagonal=1,
        )
        scores = scores.masked_fill(causal_mask, float("-inf"))

        if attention_mask is not None:
            expanded = (attention_mask[:, None, None, :] == 0)
            scores = scores.masked_fill(expanded, float("-inf"))

        probs = F.softmax(scores, dim=-1)
        out = torch.matmul(probs, v)
        out = out.transpose(1, 2).contiguous().view(bsz, seqlen, self.hidden_size)
        return self.o_proj(out)


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
        hidden_states = self.self_attn(hidden_states, attention_mask=attention_mask, position_ids=position_ids)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class HiddenProjector(nn.Module):
    """
    Projects teacher hidden states to student hidden size when needed.
    Separate projection per matched layer pair.
    """
    def __init__(self, teacher_hidden_size: int, student_hidden_size: int, num_pairs: int):
        super().__init__()
        self.teacher_hidden_size = teacher_hidden_size
        self.student_hidden_size = student_hidden_size

        if teacher_hidden_size == student_hidden_size:
            self.proj = None
        else:
            self.proj = nn.ModuleList([
                nn.Linear(teacher_hidden_size, student_hidden_size, bias=False)
                for _ in range(num_pairs)
            ])

    def forward(self, idx: int, x: torch.Tensor) -> torch.Tensor:
        if self.proj is None:
            return x
        return self.proj[idx](x)


class BitLlamaPreTrainedModel(PreTrainedModel):
    config_class = BitLlamaConfig
    base_model_prefix = "model"

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
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.layers = nn.ModuleList([BitLlamaDecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = BitRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_init()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[bool] = None,
    ):
        batch_size, seq_len = input_ids.shape

        if position_ids is None:
            position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)

        output_hidden_states = (
            self.config.output_hidden_states if output_hidden_states is None else output_hidden_states
        )

        all_hidden_states = [] if output_hidden_states else None

        hidden_states = self.embed_tokens(input_ids)
        if output_hidden_states:
            all_hidden_states.append(hidden_states)

        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask=attention_mask, position_ids=position_ids)
            if output_hidden_states:
                all_hidden_states.append(hidden_states)

        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states[-1] = hidden_states
            all_hidden_states = tuple(all_hidden_states)

        return hidden_states, all_hidden_states


class BitLlamaForCausalLM(BitLlamaPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: BitLlamaConfig):
        super().__init__(config)
        self.model = BitLlamaModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # built lazily once we know teacher hidden size / number of matched pairs
        self.hidden_projector = None
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def tie_weights(self):
        self._tie_or_clone_weights(self.lm_head, self.model.embed_tokens)

    @staticmethod
    def _match_hidden_states(
        student_hiddens: Tuple[torch.Tensor, ...],
        teacher_hiddens: Tuple[torch.Tensor, ...],
    ) -> List[Tuple[int, torch.Tensor, torch.Tensor]]:
        """
        Evenly maps teacher layers to student layers.
        Returns (pair_idx, student_hidden, teacher_hidden).
        """
        s_len = len(student_hiddens)
        t_len = len(teacher_hiddens)
        pairs = []

        for i in range(s_len):
            t_idx = round(i * (t_len - 1) / max(s_len - 1, 1))
            pairs.append((i, student_hiddens[i], teacher_hiddens[t_idx]))

        return pairs

    def _build_hidden_projector_if_needed(
        self,
        teacher_hidden_states: Tuple[torch.Tensor, ...],
        student_hidden_states: Tuple[torch.Tensor, ...],
    ):
        if self.hidden_projector is not None:
            return

        num_pairs = len(student_hidden_states)
        teacher_hidden_size = teacher_hidden_states[0].shape[-1]
        student_hidden_size = student_hidden_states[0].shape[-1]

        self.hidden_projector = HiddenProjector(
            teacher_hidden_size=teacher_hidden_size,
            student_hidden_size=student_hidden_size,
            num_pairs=num_pairs,
        ).to(student_hidden_states[0].device)

    @staticmethod
    def masked_hidden_mse(
        student_hidden: torch.Tensor,
        teacher_hidden: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        attention_mask: [B, T], 1 = valid token, 0 = pad
        """
        s = F.normalize(student_hidden.float(), dim=-1)
        t = F.normalize(teacher_hidden.float(), dim=-1)

        per_token = ((s - t) ** 2).mean(dim=-1)  # [B, T]

        if attention_mask is None:
            return per_token.mean()

        mask = attention_mask.float()
        denom = mask.sum().clamp_min(1.0)
        return (per_token * mask).sum() / denom

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        teacher_logits: Optional[torch.Tensor] = None,
        teacher_hidden_states: Optional[Tuple[torch.Tensor, ...]] = None,
        output_hidden_states: Optional[bool] = None,
        **kwargs,
    ):
        hidden_states, all_hidden_states = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_hidden_states=output_hidden_states,
        )
        logits = self.lm_head(hidden_states)

        loss = None
        ce_loss = None
        kl_loss = None
        hidden_loss = None

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

            if teacher_hidden_states is not None and all_hidden_states is not None:
                self._build_hidden_projector_if_needed(teacher_hidden_states, all_hidden_states)
                pairs = self._match_hidden_states(all_hidden_states, teacher_hidden_states)

                hidden_terms = []
                for pair_idx, s_h, t_h in pairs:
                    t_h = self.hidden_projector(pair_idx, t_h)
                    hidden_terms.append(self.masked_hidden_mse(s_h, t_h, attention_mask))

                if hidden_terms:
                    hidden_loss = sum(hidden_terms) / len(hidden_terms)
                    loss = loss + self.config.hidden_weight * hidden_loss

        return {
            "loss": loss,
            "logits": logits,
            "hidden_states": all_hidden_states,
            "ce_loss": ce_loss,
            "kl_loss": kl_loss,
            "hidden_loss": hidden_loss,
          }

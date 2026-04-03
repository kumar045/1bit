import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------
# Config
# ----------------------------

@dataclass
class ModelConfig:
    vocab_size: int = 32000
    dim: int = 768
    n_layers: int = 12
    n_heads: int = 12
    n_kv_heads: int = 12
    hidden_dim: int = 2048
    max_seq_len: int = 2048

    dropout: float = 0.0
    group_size: int = 128

    # quantization
    use_ternary_weights: bool = True
    keep_lm_head_fp: bool = True
    keep_embed_fp: bool = True

    # distillation
    distill_temperature: float = 2.0
    ce_weight: float = 0.5
    kl_weight: float = 0.5

    # training stability
    eps: float = 1e-5


# ----------------------------
# Quantization helpers
# ----------------------------

def round_ste(x: torch.Tensor) -> torch.Tensor:
    """Straight-through round."""
    return x + (torch.round(x) - x).detach()


def ternary_quantize_groupwise(
    w: torch.Tensor,
    group_size: int = 128,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Groupwise ternary quantization:
      latent fp weight -> grouped scale -> q in {-1, 0, +1} -> dequantized tensor

    Shape-agnostic over last dimension.
    """
    orig_shape = w.shape
    last_dim = w.shape[-1]

    # pad so last dim is divisible by group_size
    pad = (group_size - (last_dim % group_size)) % group_size
    if pad > 0:
        w = F.pad(w, (0, pad))

    new_last = w.shape[-1]
    num_groups = new_last // group_size

    w_grouped = w.view(-1, num_groups, group_size)  # [N, G, group]
    # simple group scale; many variants are possible
    scale = w_grouped.abs().mean(dim=-1, keepdim=True).clamp_min(eps)

    # normalize and ternarize
    w_norm = w_grouped / scale

    # threshold can be tuned; 0.5 is a simple starting point
    q = torch.zeros_like(w_norm)
    q = torch.where(w_norm > 0.5, torch.ones_like(q), q)
    q = torch.where(w_norm < -0.5, -torch.ones_like(q), q)

    w_q = q * scale
    w_q = w_q.view(*w.shape)

    if pad > 0:
        w_q = w_q[..., :last_dim]

    return w_q.view(orig_shape)


def quantize_with_ste(
    w_real: torch.Tensor,
    group_size: int = 128,
) -> torch.Tensor:
    """
    Forward uses quantized weights; backward flows to real weights.
    """
    w_q = ternary_quantize_groupwise(w_real, group_size=group_size)
    return w_real + (w_q - w_real).detach()


# ----------------------------
# RMSNorm
# ----------------------------

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_fp = x.float()
        normed = x_fp * torch.rsqrt(x_fp.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (normed.type_as(x)) * self.weight


# ----------------------------
# BitLinear
# ----------------------------

class BitLinear(nn.Module):
    """
    Trainable latent fp weights, ternary forward.
    """
    def __init__(self, in_features: int, out_features: int, bias: bool, group_size: int):
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
        w_q = quantize_with_ste(self.weight, group_size=self.group_size)
        return F.linear(x, w_q, self.bias)


# ----------------------------
# Rotary embeddings
# ----------------------------

def precompute_rope_frequencies(dim: int, max_seq_len: int, base: float = 10000.0):
    half_dim = dim // 2
    freq_seq = torch.arange(half_dim, dtype=torch.float32)
    inv_freq = 1.0 / (base ** (freq_seq / half_dim))
    t = torch.arange(max_seq_len, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)  # [T, half_dim]
    return torch.cos(freqs), torch.sin(freqs)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: [B, H, T, D]
    d = x.size(-1)
    x1, x2 = x[..., : d // 2], x[..., d // 2 :]
    cos = cos[: x.size(-2)].unsqueeze(0).unsqueeze(0)
    sin = sin[: x.size(-2)].unsqueeze(0).unsqueeze(0)
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


# ----------------------------
# Attention
# ----------------------------

class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        assert cfg.dim % cfg.n_heads == 0
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.dim // cfg.n_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = BitLinear(cfg.dim, cfg.dim, bias=False, group_size=cfg.group_size)
        self.k_proj = BitLinear(cfg.dim, cfg.dim, bias=False, group_size=cfg.group_size)
        self.v_proj = BitLinear(cfg.dim, cfg.dim, bias=False, group_size=cfg.group_size)
        self.o_proj = BitLinear(cfg.dim, cfg.dim, bias=False, group_size=cfg.group_size)

        cos, sin = precompute_rope_frequencies(self.head_dim, cfg.max_seq_len)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape

        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        q = apply_rope(q, self.rope_cos, self.rope_sin)
        k = apply_rope(k, self.rope_cos, self.rope_sin)

        att = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
        att = att.masked_fill(mask, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.dropout(att)

        y = torch.matmul(att, v)  # [B, H, T, D]
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.o_proj(y)
        return y


# ----------------------------
# MLP
# ----------------------------

class MLP(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.up_proj = BitLinear(cfg.dim, cfg.hidden_dim, bias=False, group_size=cfg.group_size)
        self.gate_proj = BitLinear(cfg.dim, cfg.hidden_dim, bias=False, group_size=cfg.group_size)
        self.down_proj = BitLinear(cfg.hidden_dim, cfg.dim, bias=False, group_size=cfg.group_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU-like
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


# ----------------------------
# Transformer block
# ----------------------------

class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.dim, eps=cfg.eps)
        self.ffn_norm = RMSNorm(cfg.dim, eps=cfg.eps)
        self.attn = CausalSelfAttention(cfg)
        self.mlp = MLP(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x))
        x = x + self.mlp(self.ffn_norm(x))
        return x


# ----------------------------
# Student model
# ----------------------------

class TernaryStudentLM(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg

        self.embed = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.norm = RMSNorm(cfg.dim, eps=cfg.eps)
        self.lm_head = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)

        # weight tying optional
        self.lm_head.weight = self.embed.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        teacher_logits: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], dict]:
        x = self.embed(input_ids)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        logits = self.lm_head(x)

        loss = None
        stats = {}

        if labels is not None:
            ce_loss = F.cross_entropy(
                logits[:, :-1].contiguous().view(-1, logits.size(-1)),
                labels[:, 1:].contiguous().view(-1),
                ignore_index=-100,
            )
            stats["ce_loss"] = ce_loss.detach()

            if teacher_logits is not None:
                T = self.cfg.distill_temperature
                s = logits[:, :-1] / T
                t = teacher_logits[:, :-1] / T
                kl_loss = F.kl_div(
                    F.log_softmax(s, dim=-1),
                    F.softmax(t, dim=-1),
                    reduction="batchmean",
                ) * (T * T)
                stats["kl_loss"] = kl_loss.detach()
                loss = self.cfg.ce_weight * ce_loss + self.cfg.kl_weight * kl_loss
            else:
                loss = ce_loss

        return logits, loss, stats


# ----------------------------
# Distillation training step
# ----------------------------

@torch.no_grad()
def get_teacher_logits(
    teacher_model: nn.Module,
    input_ids: torch.Tensor,
) -> torch.Tensor:
    teacher_model.eval()
    out = teacher_model(input_ids)
    if isinstance(out, tuple):
        return out[0]
    if hasattr(out, "logits"):
        return out.logits
    return out


def train_step(
    student: nn.Module,
    teacher: nn.Module,
    batch: dict,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip: float = 1.0,
):
    input_ids = batch["input_ids"].to(device)
    labels = batch["labels"].to(device)

    with torch.no_grad():
        teacher_logits = get_teacher_logits(teacher, input_ids)

    optimizer.zero_grad(set_to_none=True)
    _, loss, stats = student(
        input_ids=input_ids,
        labels=labels,
        teacher_logits=teacher_logits,
    )
    loss.backward()
    torch.nn.utils.clip_grad_norm_(student.parameters(), grad_clip)
    optimizer.step()

    return {
        "loss": float(loss.detach().cpu()),
        **{k: float(v.cpu()) for k, v in stats.items()},
    }


# ----------------------------
# Example usage
# ----------------------------

def build_models():
    cfg = ModelConfig(
        vocab_size=32000,
        dim=768,
        n_layers=12,
        n_heads=12,
        hidden_dim=2048,
        max_seq_len=2048,
        group_size=128,
        ce_weight=0.5,
        kl_weight=0.5,
    )

    student = TernaryStudentLM(cfg)

    # Replace with a real teacher, e.g. HF AutoModelForCausalLM
    teacher = TernaryStudentLM(cfg)
    for p in teacher.parameters():
        p.requires_grad = False

    return student, teacher, cfg


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    student, teacher, cfg = build_models()
    student.to(device)
    teacher.to(device)

    optimizer = torch.optim.AdamW(student.parameters(), lr=3e-4, betas=(0.9, 0.95), weight_decay=0.1)

    batch = {
        "input_ids": torch.randint(0, cfg.vocab_size, (2, 128)),
        "labels": torch.randint(0, cfg.vocab_size, (2, 128)),
    }

    metrics = train_step(student, teacher, batch, optimizer, device)
    print(metrics)

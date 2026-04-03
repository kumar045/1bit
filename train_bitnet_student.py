import math
from dataclasses import dataclass, field
from typing import Optional, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    HfArgumentParser,
    default_data_collator,
    set_seed,
)


# =========================================================
# Arguments
# =========================================================

@dataclass
class ScriptArguments:
    teacher_model_name: str = field(default="Qwen/Qwen2.5-1.5B-Instruct")
    dataset_name: str = field(default="wikitext")
    dataset_config: str = field(default="wikitext-2-raw-v1")
    output_dir: str = field(default="./bitnet_student_out")

    max_seq_length: int = field(default=512)
    vocab_size: int = field(default=32000)

    student_dim: int = field(default=512)
    student_layers: int = field(default=8)
    student_heads: int = field(default=8)
    student_hidden_dim: int = field(default=1536)
    group_size: int = field(default=128)

    distill_temperature: float = field(default=2.0)
    ce_weight: float = field(default=0.5)
    kl_weight: float = field(default=0.5)

    seed: int = field(default=42)


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
    Groupwise ternary quantization:
      1) split last dim into groups
      2) compute absmean scale per group
      3) normalize and map to {-1, 0, +1}
      4) dequantize back to floating tensor
    """
    orig_shape = w.shape
    last_dim = w.shape[-1]

    pad = (group_size - (last_dim % group_size)) % group_size
    if pad > 0:
        w = F.pad(w, (0, pad))

    new_last_dim = w.shape[-1]
    num_groups = new_last_dim // group_size
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
    forward uses quantized weights, backward updates real-valued weights.
    """
    w_q = ternary_quantize_groupwise(w_real, group_size=group_size)
    return w_real + (w_q - w_real).detach()


# =========================================================
# Core modules
# =========================================================

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_fp = x.float()
        normed = x_fp * torch.rsqrt(x_fp.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return normed.type_as(x) * self.weight


class BitLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = False, group_size: int = 128):
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


def precompute_rope(dim: int, max_seq_len: int, base: float = 10000.0):
    half_dim = dim // 2
    freq_seq = torch.arange(half_dim, dtype=torch.float32)
    inv_freq = 1.0 / (base ** (freq_seq / half_dim))
    t = torch.arange(max_seq_len, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)
    return torch.cos(freqs), torch.sin(freqs)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: [B, H, T, D]
    d = x.size(-1)
    x1, x2 = x[..., : d // 2], x[..., d // 2 :]
    cos = cos[: x.size(-2)].unsqueeze(0).unsqueeze(0).to(x.device, x.dtype)
    sin = sin[: x.size(-2)].unsqueeze(0).unsqueeze(0).to(x.device, x.dtype)
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


class BitSelfAttention(nn.Module):
    def __init__(self, dim: int, n_heads: int, max_seq_len: int, group_size: int):
        super().__init__()
        assert dim % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = BitLinear(dim, dim, bias=False, group_size=group_size)
        self.k_proj = BitLinear(dim, dim, bias=False, group_size=group_size)
        self.v_proj = BitLinear(dim, dim, bias=False, group_size=group_size)
        self.o_proj = BitLinear(dim, dim, bias=False, group_size=group_size)

        cos, sin = precompute_rope(self.head_dim, max_seq_len)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape

        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        q = apply_rope(q, self.rope_cos, self.rope_sin)
        k = apply_rope(k, self.rope_cos, self.rope_sin)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        causal_mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
        attn_scores = attn_scores.masked_fill(causal_mask, float("-inf"))
        attn_probs = F.softmax(attn_scores, dim=-1)

        y = torch.matmul(attn_probs, v)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.o_proj(y)
        return y


class BitMLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, group_size: int):
        super().__init__()
        self.gate_proj = BitLinear(dim, hidden_dim, bias=False, group_size=group_size)
        self.up_proj = BitLinear(dim, hidden_dim, bias=False, group_size=group_size)
        self.down_proj = BitLinear(hidden_dim, dim, bias=False, group_size=group_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class BitBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int, hidden_dim: int, max_seq_len: int, group_size: int):
        super().__init__()
        self.attn_norm = RMSNorm(dim)
        self.ffn_norm = RMSNorm(dim)
        self.attn = BitSelfAttention(dim, n_heads, max_seq_len, group_size)
        self.mlp = BitMLP(dim, hidden_dim, group_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x))
        x = x + self.mlp(self.ffn_norm(x))
        return x


class BitNetStudentLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        dim: int,
        n_layers: int,
        n_heads: int,
        hidden_dim: int,
        max_seq_len: int,
        group_size: int,
        ce_weight: float,
        kl_weight: float,
        distill_temperature: float,
    ):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)   # kept in normal precision
        self.blocks = nn.ModuleList(
            [BitBlock(dim, n_heads, hidden_dim, max_seq_len, group_size) for _ in range(n_layers)]
        )
        self.norm = RMSNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)  # kept in normal precision

        self.ce_weight = ce_weight
        self.kl_weight = kl_weight
        self.distill_temperature = distill_temperature

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        teacher_logits: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        x = self.embed(input_ids)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        logits = self.lm_head(x)

        loss = None
        ce_loss = None
        kl_loss = None

        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()

            ce_loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )

            loss = self.ce_weight * ce_loss

            if teacher_logits is not None:
                T = self.distill_temperature
                student_log_probs = F.log_softmax(shift_logits / T, dim=-1)
                teacher_probs = F.softmax(teacher_logits[:, :-1, :] / T, dim=-1)

                kl_loss = F.kl_div(
                    student_log_probs,
                    teacher_probs,
                    reduction="batchmean",
                ) * (T * T)

                loss = loss + self.kl_weight * kl_loss

        return {
            "loss": loss,
            "logits": logits,
            "ce_loss": ce_loss,
            "kl_loss": kl_loss,
        }


# =========================================================
# Trainer with teacher distillation
# =========================================================

class DistillationTrainer(Trainer):
    def __init__(self, teacher_model: nn.Module, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher_model = teacher_model.eval()
        for p in self.teacher_model.parameters():
            p.requires_grad = False

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels")
        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask", None)

        with torch.no_grad():
            teacher_out = self.teacher_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            teacher_logits = teacher_out.logits

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            teacher_logits=teacher_logits,
        )

        loss = outputs["loss"]
        return (loss, outputs) if return_outputs else loss


# =========================================================
# Data helpers
# =========================================================

def tokenize_function(examples, tokenizer, max_seq_length):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=max_seq_length,
        padding="max_length",
    )


def add_labels(example):
    example["labels"] = example["input_ids"].copy()
    return example


# =========================================================
# Main
# =========================================================

def main():
    parser = HfArgumentParser((ScriptArguments, TrainingArguments))
    script_args, training_args = parser.parse_args_into_dataclasses()

    set_seed(script_args.seed)

    tokenizer = AutoTokenizer.from_pretrained(script_args.teacher_model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    teacher_model = AutoModelForCausalLM.from_pretrained(
        script_args.teacher_model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )

    vocab_size = max(script_args.vocab_size, len(tokenizer))

    student_model = BitNetStudentLM(
        vocab_size=vocab_size,
        dim=script_args.student_dim,
        n_layers=script_args.student_layers,
        n_heads=script_args.student_heads,
        hidden_dim=script_args.student_hidden_dim,
        max_seq_len=script_args.max_seq_length,
        group_size=script_args.group_size,
        ce_weight=script_args.ce_weight,
        kl_weight=script_args.kl_weight,
        distill_temperature=script_args.distill_temperature,
    )

    raw_datasets = load_dataset(script_args.dataset_name, script_args.dataset_config)

    tokenized = raw_datasets.map(
        lambda x: tokenize_function(x, tokenizer, script_args.max_seq_length),
        batched=True,
        remove_columns=raw_datasets["train"].column_names,
    )
    tokenized = tokenized.map(add_labels)

    trainer = DistillationTrainer(
        teacher_model=teacher_model,
        model=student_model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"] if "validation" in tokenized else None,
        tokenizer=tokenizer,
        data_collator=default_data_collator,
    )

    trainer.train()
    trainer.save_model(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    main()

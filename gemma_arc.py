import math
from typing import Optional, Tuple, Any
import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------------------------------------------------
# Configuration and Low-Level Utilities
# ----------------------------------------------------------------------

class GemmaConfig:
    """Defines the hyperparameter blueprint for the Gemma architecture."""
    def __init__(
        self,
        vocab_size=256000,
        hidden_size=3072,
        intermediate_size=24576,
        num_hidden_layers=28,
        num_attention_heads=16,
        num_key_value_heads=16,
        head_dim=256,
        rms_norm_eps=1e-6,
        pad_token_id=0,
        rope_theta=10000.0,
        attention_dropout=0.0,
        hidden_act="gelu_pytorch_tanh",
        attn_logit_softcapping=None,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.rms_norm_eps = rms_norm_eps
        self.pad_token_id = pad_token_id
        self.rope_theta = rope_theta
        self.attention_dropout = attention_dropout
        self.hidden_act = hidden_act
        self.attn_logit_softcapping = attn_logit_softcapping

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dimensions for RoPE application."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> Tuple:
    """Injects positional awareness into queries and keys."""
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Expands KV heads for Grouped-Query Attention (GQA)."""
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

# ----------------------------------------------------------------------
# Core Sub-Layer Modules
# ----------------------------------------------------------------------

class GemmaRMSNorm(nn.Module):
    """Variance-based normalization with identity initialization offset."""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # High precision variance computation prevents underflow
        output = self._norm(x.float())
        # Gemma specifically applies the (1 + weight) transformation
        output = output * (1.0 + self.weight.float())
        return output.to(x.dtype)

class GemmaRotaryEmbedding(nn.Module):
    """Pre-computes sequence position rotation matrices."""
    def __init__(self, dim, max_position_embeddings=128000, base=10000.0):
        super().__init__()
        self.dim = dim
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.float32) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x: torch.Tensor, seq_len: int) -> Tuple:
        t = torch.arange(seq_len, device=x.device, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(x.dtype), emb.sin().to(x.dtype)

class GemmaMLP(nn.Module):
    """Gated Multi-Layer Perceptron utilizing approximate GeLU."""
    def __init__(self, config: GemmaConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        
        if config.hidden_act == "gelu_pytorch_tanh":
            self.act_fn = lambda x: F.gelu(x, approximate="tanh")
        else:
            self.act_fn = F.gelu

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_output = self.act_fn(self.gate_proj(x))
        up_output = self.up_proj(x)
        return self.down_proj(gate_output * up_output)

# ----------------------------------------------------------------------
# Attention Mechanism
# ----------------------------------------------------------------------

class GemmaAttention(nn.Module):
    """Grouped-Query Attention with optional Logit Soft-Capping."""
    def __init__(self, config: GemmaConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.scaling = self.head_dim ** -0.5
        self.attention_dropout = config.attention_dropout
        self.softcap = config.attn_logit_softcapping

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional = None,
        position_embeddings: Optional] = None,
    ) -> torch.Tensor:
        
        batch_size, q_len, _ = hidden_states.size()
        
        # Linear projection and dimensional reshape
        query_states = self.q_proj(hidden_states).view(batch_size, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(batch_size, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(batch_size, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        if position_embeddings is not None:
            cos, sin = position_embeddings
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # GQA KV expansion
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        # Scaled dot-product attention
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.scaling
        
        # Logit soft-capping (Gemma 2 paradigm)
        if self.softcap is not None:
            attn_weights = torch.tanh(attn_weights / self.softcap) * self.softcap

        # Mask integration
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        # Normalization and output projection
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = F.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(batch_size, q_len, self.num_heads * self.head_dim)
        
        return self.o_proj(attn_output)

# ----------------------------------------------------------------------
# Macro-Architecture: Layers and Sequence Modeling
# ----------------------------------------------------------------------

class GemmaDecoderLayer(nn.Module):
    """A unified block containing residual connections, attention, and MLP."""
    def __init__(self, config: GemmaConfig):
        super().__init__()
        self.self_attn = GemmaAttention(config)
        self.mlp = GemmaMLP(config)
        self.input_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional = None,
        position_embeddings: Optional] = None,
    ) -> torch.Tensor:
        
        # Pre-normalized Attention Block
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
        )
        hidden_states = residual + hidden_states

        # Pre-normalized MLP Block
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        
        return hidden_states

class GemmaModel(nn.Module):
    """The foundational backbone orchestrating embeddings and decoder stacks."""
    def __init__(self, config: GemmaConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.layers = nn.ModuleList()
        self.norm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = GemmaRotaryEmbedding(config.head_dim, base=config.rope_theta)

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional = None,
    ) -> torch.Tensor:
        
        # Embedding application and crucial sqrt scaling
        inputs_embeds = self.embed_tokens(input_ids)
        hidden_states = inputs_embeds * (self.config.hidden_size ** 0.5)

        # Dynamic RoPE generation
        seq_length = input_ids.shape
        cos, sin = self.rotary_emb(hidden_states, seq_len=seq_length)
        position_embeddings = (cos, sin)

        # Standard autoregressive causal mask generation
        if attention_mask is None:
            attention_mask = torch.tril(torch.ones((seq_length, seq_length), device=input_ids.device))
            attention_mask = attention_mask.unsqueeze(0).unsqueeze(0)
            attention_mask = (1.0 - attention_mask) * torch.finfo(hidden_states.dtype).min

        # Deep sequential processing
        for decoder_layer in self.layers:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_embeddings=position_embeddings,
            )

        return self.norm(hidden_states)

class GemmaForCausalLM(nn.Module):
    """The terminal wrapper applying weight tying and language modeling heads."""
    def __init__(self, config: GemmaConfig):
        super().__init__()
        self.config = config
        self.model = GemmaModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # Mandatory weight tying for parameter efficiency
        self.lm_head.weight = self.model.embed_tokens.weight

    def forward(self, input_ids: torch.LongTensor, attention_mask: Optional = None) -> torch.Tensor:
        hidden_states = self.model(input_ids=input_ids, attention_mask=attention_mask)
        logits = self.lm_head(hidden_states)
        return logits

from typing import Optional, Tuple

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM
from transformers.activations import ACT2FN
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm, Qwen3RotaryEmbedding


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    unsqueeze_dim: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class Qwen3EmbeddingModule(nn.Module):
    """Token embedding module shared by all blocks."""

    def __init__(self, base_model: "Qwen3ForCausalLM"):
        super().__init__()
        self.embed_tokens = base_model.model.embed_tokens

    def forward(self, input_ids: torch.LongTensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

# class Qwen3EmbeddingModule(nn.Module):
#     """Token embedding module shared by all blocks."""

#     def __init__(self, base_model: "Qwen3ForCausalLM"):
#         super().__init__()
#         self.embed_tokens = base_model.model.embed_tokens

#     def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
#         # 强制转换为 long，因为 embed_tokens 需要 long
#         input_ids = input_ids.to(torch.long)
#         return self.embed_tokens(input_ids)
    

class CustomQwen3MLP(nn.Module):
    def __init__(self, config: Qwen3Config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class CustomQwen3Attention(nn.Module):
    def __init__(self, config: Qwen3Config, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout

        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(
            config.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
        )
        self.q_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int, n_heads: int) -> torch.Tensor:
        return tensor.view(bsz, seq_len, n_heads, self.head_dim).transpose(1, 2)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: torch.Tensor,
        past_key: Optional[torch.Tensor],
        past_value: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bsz, q_len, _ = hidden_states.size()
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = self._shape(query_states, q_len, bsz, self.num_heads)
        key_states = self._shape(key_states, q_len, bsz, self.num_key_value_heads)
        value_states = self._shape(value_states, q_len, bsz, self.num_key_value_heads)

        query_states = self.q_norm(query_states)
        key_states = self.k_norm(key_states)

        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key is not None and past_value is not None:
            key_states = torch.cat([past_key, key_states], dim=2)
            value_states = torch.cat([past_value, value_states], dim=2)

        new_key = key_states[:, :, -q_len:, :]
        new_value = value_states[:, :, -q_len:, :]

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(-2, -1)) * self.scaling
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)

        attn_output = attn_output.transpose(1, 2).reshape(bsz, q_len, self.num_heads * self.head_dim)
        attn_output = self.o_proj(attn_output)

        return attn_output, new_key, new_value


# class CustomQwen3DecoderLayer(nn.Module):
#     def __init__(self, config: Qwen3Config, layer_idx: int):
#         super().__init__()
#         self.self_attn = CustomQwen3Attention(config, layer_idx)
#         self.mlp = CustomQwen3MLP(config)
#         self.input_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
#         self.post_attention_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

#     def forward(
#         self,
#         hidden_states: torch.Tensor,
#         cos: torch.Tensor,
#         sin: torch.Tensor,
#         attention_mask: torch.Tensor,
#         past_key: Optional[torch.Tensor],
#         past_value: Optional[torch.Tensor],
#     ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
#         residual = hidden_states
#         hidden_states = self.input_layernorm(hidden_states)
#         attn_output, present_key, present_value = self.self_attn(
#             hidden_states,
#             cos,
#             sin,
#             attention_mask,
#             past_key,
#             past_value,
#         )
#         hidden_states = residual + attn_output

#         residual = hidden_states
#         hidden_states = self.post_attention_layernorm(hidden_states)
#         hidden_states = self.mlp(hidden_states)
#         hidden_states = residual + hidden_states

#         return hidden_states, present_key, present_value


# class Qwen3BlockStackModule(nn.Module):
#     """Consecutive decoder layers exported together (supports KV cache IO)."""

#     def __init__(self, base_model: "Qwen3ForCausalLM", start_layer: int, end_layer: int):
#         super().__init__()
#         config = base_model.config
#         self.config = config
#         self.start_layer = start_layer
#         self.end_layer = end_layer
#         self.rotary_emb = base_model.model.rotary_emb

#         self.layers = nn.ModuleList()
#         for idx in range(start_layer, end_layer):
#             custom_layer = CustomQwen3DecoderLayer(config, idx)
#             custom_layer.load_state_dict(base_model.model.layers[idx].state_dict())
#             self.layers.append(custom_layer)

#     def forward(
#         self,
#         hidden_states: torch.Tensor,
#         attention_mask: torch.Tensor,
#         position_ids: torch.Tensor,
#         past_key: Optional[torch.Tensor] = None,
#         past_value: Optional[torch.Tensor] = None,
#     ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
#         bsz, seq_len, _ = hidden_states.shape
#         position_ids = position_ids.to(torch.long)
#         cos, sin = self.rotary_emb(hidden_states, position_ids)
#         cos = cos.to(hidden_states.dtype)
#         sin = sin.to(hidden_states.dtype)

#         present_keys = []
#         present_values = []

#         for layer_idx, layer in enumerate(self.layers):
#             layer_past_k = None
#             layer_past_v = None
#             if past_key is not None and past_key.shape[0] > layer_idx:
#                 layer_past_k = past_key[layer_idx]
#             if past_value is not None and past_value.shape[0] > layer_idx:
#                 layer_past_v = past_value[layer_idx]
#             if layer_past_k is not None and layer_past_k.shape[2] == 0:
#                 layer_past_k = None
#             if layer_past_v is not None and layer_past_v.shape[2] == 0:
#                 layer_past_v = None

#             hidden_states, pk, pv = layer(
#                 hidden_states,
#                 cos,
#                 sin,
#                 attention_mask,
#                 layer_past_k,
#                 layer_past_v,
#             )
#             present_keys.append(pk)
#             present_values.append(pv)

#         present_key = torch.stack(present_keys, dim=0)
#         present_value = torch.stack(present_values, dim=0)
#         return hidden_states, present_key, present_value

class CustomQwen3DecoderLayer(nn.Module):
    """
    Single Qwen3 decoder layer with rotary embedding and KV cache.
    Forward signature matches Qwen3BlockStackModule.
    """

    def __init__(self, base_model: "Qwen3ForCausalLM", layer_idx: int):
        super().__init__()
        config = base_model.config
        self.config = config
        self.layer_idx = layer_idx

        self.rotary_emb = base_model.model.rotary_emb

        self.self_attn = CustomQwen3Attention(config, layer_idx)
        self.mlp = CustomQwen3MLP(config)
        self.input_layernorm = Qwen3RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = Qwen3RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

        # load weights
        self.load_state_dict(
            base_model.model.layers[layer_idx].state_dict(),
            strict=False,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        past_key: Optional[torch.Tensor] = None,
        past_value: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        # ---- rotary embedding ----
        # position_ids = position_ids.to(torch.long)
        cos, sin = self.rotary_emb(hidden_states, position_ids)
        cos = cos.to(hidden_states.dtype)
        sin = sin.to(hidden_states.dtype)

        # ---- empty KV → None (ONNX-friendly) ----
        if past_key is not None and past_key.shape[2] == 0:
            past_key = None
        if past_value is not None and past_value.shape[2] == 0:
            past_value = None

        # ---- attention + residual ----
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        attn_output, present_key, present_value = self.self_attn(
            hidden_states,
            cos,
            sin,
            attention_mask,
            past_key,
            past_value,
        )
        hidden_states = residual + attn_output

        # ---- MLP + residual ----
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states, present_key, present_value



class Qwen3OutputModule(nn.Module):
    def __init__(self, base_model: "Qwen3ForCausalLM"):
        super().__init__()
        self.norm = base_model.model.norm
        self.lm_head = base_model.lm_head

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hs = self.norm(hidden_states)
        logits = self.lm_head(hs)
        return logits


def load_base_qwen3(model_path: str) -> "Qwen3ForCausalLM":
    base = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="cpu",
        trust_remote_code=False,
    )
    base.eval()
    return base

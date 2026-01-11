import json
from pathlib import Path
import numpy as np

NEG_INF = -1e9


class PipelineConfig:
    def __init__(self, onnx_dir: str):
        cfg_path = Path(onnx_dir) / "config.json"
        with open(cfg_path, "r", encoding="utf-8") as fr:
            cfg = json.load(fr)
        self.hidden_size = cfg["hidden_size"]
        self.max_position_embeddings = cfg["max_position_embeddings"]
        self.num_hidden_layers = cfg["num_hidden_layers"]
        self.num_attention_heads = cfg["num_attention_heads"]
        self.num_key_value_heads = cfg["num_key_value_heads"]
        self.head_dim = self.hidden_size // self.num_attention_heads
        self.layer_blocks = [7, 7, 7, 7]
        assert sum(self.layer_blocks) == self.num_hidden_layers


def save_array(path: Path, array: np.ndarray):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, array)


def load_array(path: Path) -> np.ndarray:
    path = Path(path)
    return np.load(path)


def build_static_attention_mask(past_len: int, q_len: int, max_cache_len: int, max_input_len: int) -> np.ndarray:
    if q_len > max_input_len:
        raise ValueError(f"q_len {q_len} exceeds configured max_input_len {max_input_len}")
    if past_len > max_cache_len:
        raise ValueError(f"past_len {past_len} exceeds configured max_cache_len {max_cache_len}")
    total = max_cache_len + max_input_len
    mask = np.full((max_input_len, total), NEG_INF, dtype=np.float32)
    if q_len == 0:
        return mask.reshape(1, 1, max_input_len, total)
    if past_len > 0:
        mask[:q_len, :past_len] = 0.0
    for row in range(q_len):
        cols_end = max_cache_len + row + 1
        mask[row, max_cache_len:cols_end] = 0.0
    return mask.reshape(1, 1, max_input_len, total)


def build_static_position_ids(past_len: int, q_len: int, max_input_len: int) -> np.ndarray:
    if q_len > max_input_len:
        raise ValueError(f"q_len {q_len} exceeds configured max_input_len {max_input_len}")
    pos = np.zeros((1, max_input_len), dtype=np.int64)
    if q_len > 0:
        pos[0, :q_len] = np.arange(past_len, past_len + q_len, dtype=np.int64)
        if q_len < max_input_len:
            pos[0, q_len:] = pos[0, q_len - 1]
    return pos


def ensure_static_kv(path: Path, layers: int, kv_heads: int, head_dim: int, max_cache_len: int):
    """Ensure the KV cache file exists with the fixed static shape."""
    path = Path(path)
    target_shape = (layers, 1, kv_heads, max_cache_len, head_dim)
    if path.exists():
        arr = np.load(path)
        if arr.shape == target_shape:
            return
        print(f"[stage_utils] KV cache shape mismatch at {path}, resetting to {target_shape}.")
    empty = np.zeros(target_shape, dtype=np.float16)
    save_array(path, empty)

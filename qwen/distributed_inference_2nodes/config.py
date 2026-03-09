"""
2 节点分布式推理配置模块
支持 Qwen 模型的 2 节点分布式推理
"""
from dataclasses import dataclass, field
from typing import List, Dict, Optional
import os
import json


@dataclass
class DistributedConfig2Nodes:
    """2 节点分布式推理配置"""
    # 模型配置
    om_dir: str = ""  # OM 模型目录
    
    # 模型参数（从 config.json 加载）
    hidden_size: int = 2048
    num_hidden_layers: int = 28
    num_attention_heads: int = 16
    num_key_value_heads: int = 8
    head_dim: int = 128
    vocab_size: int = 151936
    max_position_embeddings: int = 40960
    
    # KV Cache 配置
    max_cache_len: int = 1024  # past_key/past_value 的序列长度
    max_input_len: int = 16    # 当前输入的最大长度
    # attention_mask 形状: [1, 1, max_input_len, max_cache_len + max_input_len]
    # 即 [1, 1, 16, 1040]
    
    # 采样配置
    temperature: float = 1.0
    top_k: int = 0
    top_p: float = 1.0
    greedy: bool = True
    max_gen_len: int = 100
    
    # 设备配置
    device_id: int = 0
    
    # 网络配置
    node_id: int = 0  # 当前节点 ID (0-1)
    total_nodes: int = 2  # 总节点数
    
    # 节点角色映射（2 节点架构）
    # Node 0: embed.om + layers_0_6.om + layers_7_13.om (主节点，层 0-13)
    # Node 1: layers_14_20.om + layers_21_27.om + output.om (尾节点，层 14-27)
    
    # 网络地址配置
    node_addresses: Dict[int, Dict[str, any]] = field(default_factory=lambda: {
        0: {"ip": "127.0.0.1", "port": 9000},
        1: {"ip": "127.0.0.1", "port": 9001},
    })
    
    # 每个节点的模型文件
    node_models: Dict[int, List[str]] = field(default_factory=lambda: {
        0: ["embed.om", "layers_0_6.om", "layers_7_13.om"],
        1: ["layers_14_20.om", "layers_21_27.om", "output.om"],
    })
    
    # 每个节点的 transformer 层数（用于 KV Cache）
    # Node 0: 层 0-13 = 14 层
    # Node 1: 层 14-27 = 14 层
    node_layers: Dict[int, int] = field(default_factory=lambda: {
        0: 14,   # layers 0-13
        1: 14,   # layers 14-27
    })
    
    # 每个节点的 block 模型数量
    node_block_count: Dict[int, int] = field(default_factory=lambda: {
        0: 2,   # layers_0_6.om + layers_7_13.om
        1: 2,   # layers_14_20.om + layers_21_27.om
    })
    
    # EOS token ID
    eos_token_id: int = 151645
    bos_token_id: int = 151643
    
    def __post_init__(self):
        if self.om_dir and os.path.isdir(self.om_dir):
            config_path = os.path.join(self.om_dir, "config.json")
            if os.path.exists(config_path):
                self._load_model_config(config_path)
    
    def _load_model_config(self, config_path: str):
        """从 config.json 加载模型配置"""
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        
        self.hidden_size = cfg.get("hidden_size", self.hidden_size)
        self.num_hidden_layers = cfg.get("num_hidden_layers", self.num_hidden_layers)
        self.num_attention_heads = cfg.get("num_attention_heads", self.num_attention_heads)
        self.num_key_value_heads = cfg.get("num_key_value_heads", self.num_key_value_heads)
        self.head_dim = cfg.get("head_dim", self.hidden_size // self.num_attention_heads)
        self.vocab_size = cfg.get("vocab_size", self.vocab_size)
        self.max_position_embeddings = cfg.get("max_position_embeddings", self.max_position_embeddings)
        self.eos_token_id = cfg.get("eos_token_id", self.eos_token_id)
        self.bos_token_id = cfg.get("bos_token_id", self.bos_token_id)
    
    def get_model_paths(self, node_id: int = None) -> List[str]:
        """获取指定节点的模型路径列表"""
        if node_id is None:
            node_id = self.node_id
        model_files = self.node_models.get(node_id, [])
        return [os.path.join(self.om_dir, f) for f in model_files]
    
    def get_next_node_address(self) -> Optional[Dict[str, any]]:
        """获取下一个节点的地址"""
        next_id = self.node_id + 1
        if next_id >= self.total_nodes:
            return None
        return self.node_addresses.get(next_id)
    
    def get_head_node_address(self) -> Dict[str, any]:
        """获取头节点（Node 0）的地址"""
        return self.node_addresses.get(0)
    
    def get_listen_port(self) -> int:
        """获取当前节点的监听端口"""
        return self.node_addresses.get(self.node_id, {}).get("port", 9000 + self.node_id)
    
    def get_num_layers(self) -> int:
        """获取当前节点的层数"""
        return self.node_layers.get(self.node_id, 14)
    
    def get_block_count(self) -> int:
        """获取当前节点的 block 模型数量"""
        return self.node_block_count.get(self.node_id, 2)
    
    def is_head_node(self) -> bool:
        """是否是头节点"""
        return self.node_id == 0
    
    def is_tail_node(self) -> bool:
        """是否是尾节点"""
        return self.node_id == self.total_nodes - 1

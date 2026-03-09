"""
Llama 分布式推理框架

基于 Qwen 架构重构的 Llama 分布式推理框架
采用消息驱动的节点通信机制

主要模块：
- config: 配置管理
- network: 网络通信
- acl_model: ACL 模型封装
- kvcache: KV Cache 管理
- utils: 工具函数
- node_head: 头节点
- node_middle: 中间节点
- node_tail: 尾节点
"""

__version__ = "1.0.0"
__author__ = "AI Assistant"

from .config import DistributedConfig
from .network import (
    NodeServer,
    NodeClient,
    DistributedMessage,
    send_msg,
    recv_msg
)
from .acl_model import ACLModel
from .kvcache import KVCache, create_kvcache
from .utils import (
    build_attention_mask,
    build_position_ids,
    pad_input_ids,
    reshape_hidden_output,
    reshape_kv_output,
    sample_logits
)

__all__ = [
    # Config
    "DistributedConfig",
    
    # Network
    "NodeServer",
    "NodeClient",
    "DistributedMessage",
    "send_msg",
    "recv_msg",
    
    # Model
    "ACLModel",
    
    # KV Cache
    "KVCache",
    "create_kvcache",
    
    # Utils
    "build_attention_mask",
    "build_position_ids",
    "pad_input_ids",
    "reshape_hidden_output",
    "reshape_kv_output",
    "sample_logits",
]

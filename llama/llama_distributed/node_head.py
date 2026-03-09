"""
Node 0: 头节点（主节点）
负责 M0 (embed + layers 0-4)，是流水线的起点
接收来自尾节点的生成 token，进行下一轮推理
"""
import argparse
import numpy as np
import time
from pathlib import Path

from config import DistributedConfig
from network import NodeServer, NodeClient, DistributedMessage
from acl_model import ACLModel
from kvcache import create_kvcache
from utils import (
    build_attention_mask, build_position_ids,
    pad_input_ids, reshape_hidden_output, reshape_kv_output
)


class HeadNode:
    """头节点：M0 (embed + layers 0-4)"""
    
    def __init__(self, config: DistributedConfig):
        self.config = config
        self.node_name = "Node0-Head"
        
        # 网络组件
        self.server = None  # 接收来自尾节点的 token
        self.client = None  # 发送 hidden_states 到下一个节点
        
        # ACL 模型
        self.model = None
        
        # KV Cache
        self.kv_cache = None
        
        # 状态
        self.past_len = 0
        self.step = 0
    
    def init(self):
        """初始化节点"""
        print(f"[{self.node_name}] Initializing...")
        
        # 1. 加载模型
        model_paths = self.config.get_model_paths()
        
        print(f"[{self.node_name}] Loading model: {model_paths[0]}")
        self.model = ACLModel(model_paths[0], self.config.device_id)
        self.model.init()
        
        # 2. 初始化 KV Cache
        self.kv_cache = create_kvcache(
            cache_type="basic",
            num_layers=self.config.get_num_layers(),
            num_kv_heads=self.config.num_key_value_heads,
            head_dim=self.config.head_dim,
            max_cache_len=self.config.max_cache_len
        )
        print(f"[{self.node_name}] KV Cache initialized: {self.config.get_num_layers()} layers")
        
        # 3. 启动服务器（接收来自尾节点的 token）
        listen_port = self.config.get_listen_port()
        self.server = NodeServer(listen_port, self.node_name)
        self.server.start()
        
        # 4. 连接到下一个节点
        next_addr = self.config.get_next_node_address()
        if next_addr:
            self.client = NodeClient(
                next_addr["ip"],
                next_addr["port"],
                self.node_name
            )
            self.client.connect()
        
        # 5. 等待尾节点连接
        print(f"[{self.node_name}] Waiting for tail node connection...")
        self.server.accept_connection()
        
        print(f"[{self.node_name}] Initialization complete!")
    
    def run_model(
        self,
        input_ids: np.ndarray,
        attention_mask: np.ndarray,
        position_ids: np.ndarray,
        q_len: int
    ) -> tuple:
        """
        运行 M0 模型
        
        Returns:
            (hidden_states, past_key_values_0)
        """
        # 填充输入到固定长度
        padded_ids = pad_input_ids(input_ids, self.config.max_input_len, pad_id=0)
        
        # 获取 KV Cache
        past_key, past_value = self.kv_cache.get_cache()
        
        # 准备输入
        inputs = [
            padded_ids.astype(np.int64),
            attention_mask.astype(np.float32),
            position_ids.astype(np.int64),
        ]
        
        # 添加每一层的 KV Cache
        for i in range(self.config.get_num_layers()):
            inputs.append(past_key[i:i+1].astype(np.float32))
            inputs.append(past_value[i:i+1].astype(np.float32))
        
        # 执行推理
        outputs = self.model.execute(inputs)
        
        # 解析输出
        # outputs[0]: hidden_states
        # outputs[1:]: present_key_values (key0, value0, key1, value1, ...)
        hidden_out = reshape_hidden_output(
            outputs[0],
            batch_size=1,
            max_input_len=self.config.max_input_len,
            hidden_size=self.config.hidden_size
        )
        
        # 重组 KV Cache
        present_keys = []
        present_values = []
        for i in range(self.config.get_num_layers()):
            key_idx = 1 + i * 2
            value_idx = 1 + i * 2 + 1
            present_keys.append(outputs[key_idx])
            present_values.append(outputs[value_idx])
        
        # 堆叠成正确的形状
        target_shape = (
            self.config.get_num_layers(),
            1,
            self.config.num_key_value_heads,
            self.config.max_input_len,
            self.config.head_dim
        )
        
        present_key = np.zeros(target_shape, dtype=np.float32)
        present_value = np.zeros(target_shape, dtype=np.float32)
        
        for i in range(self.config.get_num_layers()):
            key_data = present_keys[i].view(np.float32).reshape(-1)
            value_data = present_values[i].view(np.float32).reshape(-1)
            
            layer_size = np.prod(target_shape[1:])
            present_key[i] = key_data[:layer_size].reshape(target_shape[1:])
            present_value[i] = value_data[:layer_size].reshape(target_shape[1:])
        
        # 更新 KV Cache
        self.kv_cache.update(
            present_key.astype(np.float16),
            present_value.astype(np.float16),
            q_len
        )
        
        # 返回 hidden_states 和第 0 层的 KV Cache（用于 RoPE）
        return hidden_out, present_key[0:1]
    
    def generate(
        self,
        prompt_ids: np.ndarray,
        max_new_tokens: int = 100
    ) -> list:
        """生成文本"""
        generated_ids = []
        self.past_len = 0
        self.step = 0
        
        # 当前输入
        current_ids = prompt_ids
        
        for i in range(max_new_tokens + 1):
            q_len = current_ids.shape[1]
            
            # 检查是否超出缓存
            if self.past_len + q_len > self.config.max_cache_len:
                print(f"[{self.node_name}] KV cache overflow, stopping generation")
                break
            
            # 1. 构建 attention mask 和 position ids
            attention_mask = build_attention_mask(
                self.past_len, q_len,
                self.config.max_cache_len,
                self.config.max_input_len
            )
            position_ids = build_position_ids(
                self.past_len, q_len,
                self.config.max_input_len
            )
            
            # 2. 运行 M0 模型
            hidden, past_key_values_0 = self.run_model(
                current_ids, attention_mask, position_ids, q_len
            )
            
            # 3. 构建元数据
            meta = {
                "past_len": self.past_len,
                "q_len": q_len,
                "max_cache_len": self.config.max_cache_len,
                "max_input_len": self.config.max_input_len,
            }
            
            # 4. 发送到下一个节点
            msg = DistributedMessage.create_forward_msg(
                step=self.step,
                hidden=hidden,
                input_ids=current_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values_0=past_key_values_0,
                meta=meta
            )
            self.client.send(msg.to_dict())
            
            # 5. 等待尾节点返回的 token
            result = self.server.recv()
            if result is None:
                print(f"[{self.node_name}] Connection closed")
                break
            
            result_msg = DistributedMessage.from_dict(result)
            
            if result_msg.msg_type == DistributedMessage.MSG_RESULT:
                next_token = result_msg.data.get("next_token")
                
                # 检查是否结束
                if next_token == self.config.eos_token_id:
                    print(f"[{self.node_name}] EOS token generated")
                    break
                
                generated_ids.append(next_token)
                print(f"[{self.node_name}] Step {self.step}: generated token {next_token}")
                
                # 更新状态
                self.past_len += q_len
                self.step += 1
                
                # 下一轮输入
                current_ids = np.array([[next_token]], dtype=np.int64)
            
            elif result_msg.msg_type == DistributedMessage.MSG_SHUTDOWN:
                print(f"[{self.node_name}] Received shutdown signal")
                break
        
        return generated_ids
    
    def reset(self):
        """重置状态"""
        self.past_len = 0
        self.step = 0
        self.kv_cache.reset()
        
        # 发送重置消息
        reset_msg = DistributedMessage.create_reset_msg()
        self.client.send(reset_msg.to_dict())
    
    def shutdown(self):
        """关闭节点"""
        # 发送关闭消息
        shutdown_msg = DistributedMessage.create_shutdown_msg()
        if self.client:
            self.client.send(shutdown_msg.to_dict())
        
        # 清理资源
        if self.model:
            self.model.finalize()
        if self.server:
            self.server.close()
        if self.client:
            self.client.close()


def main():
    parser = argparse.ArgumentParser(description="Node 0: Head Node (M0: embed + layers 0-4)")
    parser.add_argument("--om_dir", type=str, required=True, help="OM model directory")
    parser.add_argument("--device", type=int, default=0, help="Device ID")
    parser.add_argument("--max_cache_len", type=int, default=1024)
    parser.add_argument("--max_input_len", type=int, default=16)
    parser.add_argument("--init_tokens", type=str, required=True, help="Initial token file")
    parser.add_argument("--max_new_tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--greedy", action="store_true", default=True)
    
    # 网络配置
    parser.add_argument("--listen_port", type=int, default=8000)
    parser.add_argument("--next_ip", type=str, default="127.0.0.1")
    parser.add_argument("--next_port", type=int, default=8001)
    
    args = parser.parse_args()
    
    # 创建配置
    config = DistributedConfig(
        om_dir=args.om_dir,
        device_id=args.device,
        max_cache_len=args.max_cache_len,
        max_input_len=args.max_input_len,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        greedy=args.greedy,
        node_id=0,
    )
    
    # 更新网络配置
    config.node_addresses[0]["port"] = args.listen_port
    config.node_addresses[1]["ip"] = args.next_ip
    config.node_addresses[1]["port"] = args.next_port
    
    # 加载初始 tokens
    with open(args.init_tokens, "r") as f:
        text = f.read().strip()
        prompt_ids = [int(x) for x in text.replace("\n", " ").split()]
    prompt_ids = np.array([prompt_ids], dtype=np.int64)
    
    # 创建并运行节点
    node = HeadNode(config)
    
    try:
        node.init()
        
        start_time = time.time()
        generated = node.generate(prompt_ids, args.max_new_tokens)
        elapsed = time.time() - start_time
        
        print(f"\n{'='*50}")
        print(f"Generated {len(generated)} tokens in {elapsed:.2f}s")
        print(f"Speed: {len(generated)/elapsed:.2f} tokens/s")
        print(f"Generated IDs: {generated}")
        
    finally:
        node.shutdown()


if __name__ == "__main__":
    main()

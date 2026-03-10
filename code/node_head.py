"""
Node 0: 头节点（主节点）- 4 节点版本
负责 embed.om + layers_0_6.om，是流水线的起点
模型常驻内存，无需频繁加载/卸载
"""
import argparse
import numpy as np
import time
from pathlib import Path

from config import DistributedConfig4Nodes
from network import NodeServer, NodeClient, DistributedMessage
from acl_model import ACLModel
from kvcache import create_kvcache
from utils import (
    build_attention_mask, build_position_ids,
    decode_incremental_text, decode_token_ids,
    encode_text, load_tokenizer, pad_input_ids, reshape_hidden_output
)


class HeadNode4Nodes:
    """头节点：embed + layers_0_6（模型常驻内存）"""
    
    def __init__(self, config: DistributedConfig4Nodes):
        self.config = config
        self.node_name = "Node0-Head"
        
        # 网络组件
        self.server = None  # 接收尾节点返回的结果
        self.client = None  # 发送到下一个节点
        
        # 模型（常驻内存）
        self.embed_model = None
        self.block_model = None
        
        # KV Cache
        self.kv_cache = None
        
        # tokenizer（仅用于本地文本解码）
        self.tokenizer = None
        
        # 状态
        self.past_len = 0
        self.step = 0
    
    def init(self):
        """初始化节点"""
        print(f"[{self.node_name}] Initializing...")
        
        # 1. 加载模型（常驻内存）
        model_paths = self.config.get_model_paths()
        
        print(f"[{self.node_name}] Loading models...")
        self.embed_model = ACLModel(model_paths[0], self.config.device_id)
        self.embed_model.init()
        
        self.block_model = ACLModel(model_paths[1], self.config.device_id)
        self.block_model.init()
        
        # 2. 加载 tokenizer（仅头节点需要）
        if self.config.tokenizer_dir:
            print(f"[{self.node_name}] Loading tokenizer from: {self.config.tokenizer_dir}")
            self.tokenizer = load_tokenizer(self.config.tokenizer_dir)
        
        # 3. 初始化 KV Cache
        num_layers = self.config.get_num_layers()  # 7 层
        
        self.kv_cache = create_kvcache(
            cache_type="basic",
            num_layers=num_layers,
            num_kv_heads=self.config.num_key_value_heads,
            head_dim=self.config.head_dim,
            max_cache_len=self.config.max_cache_len
        )
        print(f"[{self.node_name}] KV Cache initialized for {num_layers} layers")
        
        # 4. 启动服务器（接收尾节点返回的结果）
        listen_port = self.config.get_listen_port()
        self.server = NodeServer(listen_port, self.node_name)
        self.server.start()
        
        # 5. 连接到下一个节点
        next_addr = self.config.get_next_node_address()
        if next_addr:
            self.client = NodeClient(next_addr["ip"], next_addr["port"], self.node_name)
            self.client.connect()
        
        # 6. 等待尾节点连接
        print(f"[{self.node_name}] Waiting for tail node connection...")
        self.server.accept_connection()
        
        print(f"[{self.node_name}] Initialization complete!")
    
    def run_embed(self, input_ids: np.ndarray) -> np.ndarray:
        """运行 embedding"""
        embed_ids = pad_input_ids(input_ids, self.config.max_input_len, pad_id=0)
        outputs = self.embed_model.execute([embed_ids])
        hidden = outputs[0].view(np.float32).reshape(
            1, self.config.max_input_len, self.config.hidden_size
        )
        return hidden
    
    def run_block(self, hidden: np.ndarray, attention_mask: np.ndarray,
                  position_ids: np.ndarray, q_len: int) -> np.ndarray:
        """运行 transformer block"""
        past_key, past_value = self.kv_cache.get_cache()
        
        inputs = [
            hidden.astype(np.float32),
            attention_mask.astype(np.float32),
            position_ids.astype(np.int64),
            past_key.astype(np.float32),
            past_value.astype(np.float32),
        ]
        
        outputs = self.block_model.execute(inputs)
        
        hidden_out = reshape_hidden_output(
            outputs[0], batch_size=1,
            max_input_len=self.config.max_input_len,
            hidden_size=self.config.hidden_size
        )
        
        # 更新 KV Cache
        num_layers = self.config.get_num_layers()
        target_shape = (num_layers, 1, self.config.num_key_value_heads,
                        self.config.max_input_len, self.config.head_dim)
        num_elements = np.prod(target_shape)
        
        present_key = outputs[1].view(np.float32).reshape(-1)[:num_elements].reshape(target_shape)
        present_value = outputs[2].view(np.float32).reshape(-1)[:num_elements].reshape(target_shape)
        
        self.kv_cache.update(present_key.astype(np.float16), present_value.astype(np.float16), q_len)
        
        return hidden_out
    
    def process_forward(self, input_ids: np.ndarray, q_len: int):
        """处理一次前向传播"""
        
        # 1. 运行 embed
        hidden = self.run_embed(input_ids)
        
        # 2. 构建 attention mask 和 position ids
        attention_mask = build_attention_mask(
            self.past_len, q_len,
            self.config.max_cache_len,
            self.config.max_input_len
        )
        position_ids = build_position_ids(
            self.past_len, q_len,
            self.config.max_input_len
        )
        
        # 3. 运行 block
        hidden = self.run_block(hidden, attention_mask, position_ids, q_len)
        
        return hidden, attention_mask, position_ids
    
    def generate(self, prompt_ids: np.ndarray, max_new_tokens: int = 100) -> list:
        """生成文本"""
        generated_ids = []
        decoded_text = ""
        self.past_len = 0
        self.step = 0
        current_ids = prompt_ids
        
        for i in range(max_new_tokens + 1):
            q_len = current_ids.shape[1]
            
            if self.past_len + q_len > self.config.max_cache_len:
                print(f"[{self.node_name}] KV cache overflow, stopping generation")
                break
            
            # 处理前向传播
            hidden, attention_mask, position_ids = self.process_forward(current_ids, q_len)
            
            # 构建元数据
            meta = {
                "past_len": self.past_len,
                "q_len": q_len,
                "max_cache_len": self.config.max_cache_len,
                "max_input_len": self.config.max_input_len,
            }
            
            # 发送到下一个节点
            msg = DistributedMessage.create_forward_msg(
                step=self.step,
                hidden=hidden,
                attention_mask=attention_mask,
                position_ids=position_ids,
                meta=meta
            )
            self.client.send(msg.to_dict())
            
            # 等待尾节点返回的 token
            result = self.server.recv()
            if result is None:
                print(f"[{self.node_name}] Connection closed")
                break
            
            result_msg = DistributedMessage.from_dict(result)
            
            if result_msg.msg_type == DistributedMessage.MSG_RESULT:
                next_token = result_msg.data.get("next_token")
                
                if next_token == self.config.eos_token_id:
                    print(f"[{self.node_name}] EOS token generated")
                    break
                
                generated_ids.append(next_token)
                print(f"[{self.node_name}] Step {self.step}: generated token {next_token}")
                
                if self.tokenizer is not None:
                    delta_text, decoded_text = decode_incremental_text(
                        self.tokenizer,
                        generated_ids,
                        previous_text=decoded_text,
                        skip_special_tokens=True
                    )
                    if delta_text:
                        print(f"[{self.node_name}] Step {self.step}: generated text {delta_text!r}")
                
                self.past_len += q_len
                self.step += 1
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
        
        reset_msg = DistributedMessage.create_reset_msg()
        self.client.send(reset_msg.to_dict())
    
    def shutdown(self):
        """关闭节点"""
        shutdown_msg = DistributedMessage.create_shutdown_msg()
        if self.client:
            self.client.send(shutdown_msg.to_dict())
        
        if self.embed_model:
            self.embed_model.finalize()
        if self.block_model:
            self.block_model.finalize()
        if self.server:
            self.server.close()
        if self.client:
            self.client.close()


def main():
    parser = argparse.ArgumentParser(description="Node 0: Head Node - 4 Nodes Version")
    parser.add_argument("--om_dir", type=str, required=True, help="OM model directory")
    parser.add_argument("--device", type=int, default=0, help="Device ID")
    parser.add_argument("--max_cache_len", type=int, default=1024)
    parser.add_argument("--max_input_len", type=int, default=16)
    parser.add_argument("--input_file", type=str, required=True, help="Input text file")
    parser.add_argument("--max_new_tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--greedy", action="store_true", default=True)
    parser.add_argument("--listen_port", type=int, default=9000)
    parser.add_argument("--next_ip", type=str, default="192.168.137.101")
    parser.add_argument("--next_port", type=int, default=9001)
    parser.add_argument(
        "--tokenizer_dir",
        type=str,
        default=r"D:\qwen_split\qwen3_1.7b",
        help="Tokenizer directory for encoding/decoding text",
    )
    
    args = parser.parse_args()
    
    config = DistributedConfig4Nodes(
        om_dir=args.om_dir,
        tokenizer_dir=args.tokenizer_dir,
        device_id=args.device,
        max_cache_len=args.max_cache_len,
        max_input_len=args.max_input_len,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        greedy=args.greedy,
        node_id=0,
    )
    
    config.node_addresses[0]["port"] = args.listen_port
    config.node_addresses[1]["ip"] = args.next_ip
    config.node_addresses[1]["port"] = args.next_port
    
    node = HeadNode4Nodes(config)
    node.init()
    
    # 从文本文件读取输入
    print(f"[Input] Reading text from file: {args.input_file}")
    with open(args.input_file, "r", encoding="utf-8") as f:
        input_text = f.read().strip()
    
    if not input_text:
        raise ValueError(f"Input file is empty: {args.input_file}")
    
    print(f"[Input] Text content: {input_text}")
    
    if node.tokenizer is None:
        raise RuntimeError("Tokenizer is required. Please specify --tokenizer_dir")
    
    # 将文本编码为 token
    prompt_ids = encode_text(node.tokenizer, input_text)
    print(f"[Input] Encoded to {len(prompt_ids)} tokens: {prompt_ids}")
    prompt_ids = np.array([prompt_ids], dtype=np.int64)
    
    try:
        start_time = time.time()
        generated = node.generate(prompt_ids, args.max_new_tokens)
        elapsed = time.time() - start_time
        
        generated_text = ""
        if node.tokenizer is not None:
            generated_text = decode_token_ids(
                node.tokenizer,
                generated,
                skip_special_tokens=True
            )
        
        print(f"\n{'='*50}")
        print(f"Generated {len(generated)} tokens in {elapsed:.2f}s")
        print(f"Speed: {len(generated)/elapsed:.2f} tokens/s")
        print(f"Generated IDs: {generated}")
        print(f"Generated text: {generated_text}")
        
    finally:
        node.shutdown()


if __name__ == "__main__":
    main()

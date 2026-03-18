"""
Node 1/2: 中间节点 - 4 节点版本
Node 1: 负责 layers_7_13.om
Node 2: 负责 layers_14_20.om
模型常驻内存，无需频繁加载/卸载
"""
import argparse
import numpy as np
from pathlib import Path

from config import DistributedConfig4Nodes
from network import NodeServer, NodeClient, DistributedMessage
from acl_model import ACLModel
from kvcache import create_kvcache
from utils import reshape_hidden_output
from tools import ToolManager, ToolAgent
from tools.builtin_tools import weather_tool, calculator_tool


class MiddleNode4Nodes:
    """中间节点：单个 transformer block（模型常驻内存）"""
    
    def __init__(self, config: DistributedConfig4Nodes):
        self.config = config
        self.node_name = f"Node{config.node_id}-Middle"
        
        # 网络组件
        self.server = None  # 接收上一个节点的数据
        self.client = None  # 发送到下一个节点
        
        # 模型（按需加载，同一时间只保留一组）
        self.block_model = None
        self._loaded_mode = None  # "system" / "prefill" / "decode" / None
        
        # KV Cache
        self.kv_cache = None
        
        # System KV cache 快照
        self._system_kv_key = None
        self._system_kv_value = None
        self._system_past_len = 0
        
        # 工具系统
        self.tool_manager = None
        self.tool_agent = None
        
        # 状态
        self.past_len = 0
        self.step = 0
    
    def init(self):
        """初始化节点"""
        print(f"[{self.node_name}] Initializing...")
        
        # 1. 模型路径
        self._system_paths = self.config.get_system_model_paths()
        self._prefill_paths = self.config.get_prefill_model_paths()
        self._decode_paths = self.config.get_decode_model_paths()
        if self._system_paths:
            print(f"[{self.node_name}] System model: {self._system_paths[0]}")
        print(f"[{self.node_name}] Prefill model: {self._prefill_paths[0]}")
        print(f"[{self.node_name}] Decode model: {self._decode_paths[0]}")
        
        # 2. 初始化 KV Cache
        num_layers = self.config.get_num_layers()  # 7 层
        
        self.kv_cache = create_kvcache(
            cache_type="basic",
            num_layers=num_layers,
            num_kv_heads=self.config.num_key_value_heads,
            head_dim=self.config.head_dim,
            max_cache_len=self.config.max_cache_len
        )
        print(f"[{self.node_name}] KV Cache initialized for {num_layers} layers")
        
        # 3. 尝试从磁盘加载 system KV cache
        self._try_load_system_kv()
        
        # 4. 初始化工具系统
        self._init_tools()
        
        # 5. 启动服务器（接收上一个节点的数据）
        listen_port = self.config.get_listen_port()
        self.server = NodeServer(listen_port, self.node_name)
        self.server.start()
        
        # 6. 连接到下一个节点
        next_addr = self.config.get_next_node_address()
        if next_addr:
            self.client = NodeClient(next_addr["ip"], next_addr["port"], self.node_name)
            if not self.client.connect():
                raise RuntimeError(f"Cannot connect to next node {next_addr['ip']}:{next_addr['port']}")
        
        # 7. 等待上一个节点连接
        print(f"[{self.node_name}] Waiting for previous node connection...")
        self.server.accept_connection()
        
        print(f"[{self.node_name}] Initialization complete!")
    
    def _try_load_system_kv(self):
        """尝试从磁盘加载预生成的 system KV cache（无需 system 模型）"""
        import json as _json
        system_kv_dir = Path(self.config.system_kv_dir) if self.config.system_kv_dir else None
        if system_kv_dir is None:
            return

        node_id = self.config.node_id
        meta_path = system_kv_dir / "system_kv_meta.json"
        key_path = system_kv_dir / f"past_key_node{node_id}.npy"
        value_path = system_kv_dir / f"past_value_node{node_id}.npy"

        if meta_path.exists() and key_path.exists() and value_path.exists():
            print(f"[{self.node_name}] Loading cached system KV from {system_kv_dir}")
            with open(meta_path, "r", encoding="utf-8") as f:
                sys_meta = _json.load(f)
            self._system_past_len = sys_meta["system_q_len"]
            self._system_kv_key = np.load(str(key_path))
            self._system_kv_value = np.load(str(value_path))
            self._restore_system_kv()
            print(f"[{self.node_name}] System KV loaded. system_past_len={self._system_past_len}")
        else:
            print(f"[{self.node_name}] No cached system KV found, will generate from system model if needed")

    def _save_system_kv(self):
        """将 system KV 快照持久化到磁盘"""
        import json as _json
        system_kv_dir = Path(self.config.system_kv_dir) if self.config.system_kv_dir else None
        if system_kv_dir is None or self._system_kv_key is None:
            return

        node_id = self.config.node_id
        system_kv_dir.mkdir(parents=True, exist_ok=True)
        np.save(str(system_kv_dir / f"past_key_node{node_id}.npy"), self._system_kv_key)
        np.save(str(system_kv_dir / f"past_value_node{node_id}.npy"), self._system_kv_value)
        # meta 由 head 节点写入，这里不重复写
        print(f"[{self.node_name}] System KV saved to {system_kv_dir}")

    def _ensure_mode(self, mode: str):
        """确保当前加载的模型与所需模式一致"""
        if self._loaded_mode == mode:
            return
        if self.block_model is not None:
            print(f"[{self.node_name}] Unloading {self._loaded_mode} block model...")
            self.block_model.finalize()
            self.block_model = None
        if mode == "system":
            paths = self._system_paths
        elif mode == "prefill":
            paths = self._prefill_paths
        else:
            paths = self._decode_paths
        print(f"[{self.node_name}] Loading {mode} model...")
        self.block_model = ACLModel(paths[0], self.config.device_id)
        self.block_model.init()
        self._loaded_mode = mode
        print(f"[{self.node_name}] {mode} model loaded.")

    def _restore_system_kv(self):
        """恢复 KV cache 到 system KV 快照状态"""
        if self._system_kv_key is not None:
            self.kv_cache.past_key[:] = self._system_kv_key
            self.kv_cache.past_value[:] = self._system_kv_value
            self.kv_cache.current_len = self._system_past_len
            self.past_len = self._system_past_len
        else:
            self.kv_cache.reset()
            self.past_len = 0

    def run_block(self, hidden: np.ndarray, attention_mask: np.ndarray,
                  position_ids: np.ndarray, q_len: int, mode: str = "decode") -> np.ndarray:
        """运行 transformer block"""
        self._ensure_mode(mode)
        if mode == "system":
            cur_input_len = self.config.system_len
            inputs = [
                hidden.astype(np.float32),
                attention_mask.astype(np.float32),
                position_ids.astype(np.int64),
            ]
        elif mode == "prefill":
            cur_input_len = self.config.prefill_len
            past_key, past_value = self.kv_cache.get_cache()
            inputs = [
                hidden.astype(np.float32),
                attention_mask.astype(np.float32),
                position_ids.astype(np.int64),
                past_key.astype(np.float32),
                past_value.astype(np.float32),
            ]
        else:
            cur_input_len = 1
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
            max_input_len=cur_input_len,
            hidden_size=self.config.hidden_size
        )

        num_layers = self.config.get_num_layers()

        if mode == "system":
            kv_seq_dim = cur_input_len
        elif mode == "prefill":
            kv_seq_dim = cur_input_len
        else:
            kv_seq_dim = self.config.max_cache_len + cur_input_len

        target_shape = (num_layers, 1, self.config.num_key_value_heads,
                        kv_seq_dim, self.config.head_dim)
        num_elements = np.prod(target_shape)

        k_bytes = outputs[1].astype(np.uint8).tobytes()
        v_bytes = outputs[2].astype(np.uint8).tobytes()

        present_key_full = np.frombuffer(k_bytes, dtype=np.float32)[:num_elements].reshape(target_shape)
        present_value_full = np.frombuffer(v_bytes, dtype=np.float32)[:num_elements].reshape(target_shape)

        if mode == "system":
            present_key = present_key_full[:, :, :, :q_len, :]
            present_value = present_value_full[:, :, :, :q_len, :]
        elif mode == "prefill":
            present_key = present_key_full[:, :, :, :q_len, :]
            present_value = present_value_full[:, :, :, :q_len, :]
        else:
            past_len = self.kv_cache.get_current_len()
            present_key = present_key_full[:, :, :, past_len:past_len + q_len, :]
            present_value = present_value_full[:, :, :, past_len:past_len + q_len, :]

        self.kv_cache.update(present_key.astype(np.float16), present_value.astype(np.float16), q_len)

        # System 阶段完成后保存 KV 快照
        if mode == "system" and self._system_kv_key is None:
            self._system_kv_key = self.kv_cache.past_key.copy()
            self._system_kv_value = self.kv_cache.past_value.copy()
            self._system_past_len = q_len
            self._save_system_kv()
            print(f"[{self.node_name}] System KV snapshot saved. q_len={q_len}")

        return hidden_out
    
    def process_loop(self):
        """处理循环"""
        print(f"[{self.node_name}] Starting processing loop...")
        
        while True:
            data = self.server.recv()
            if data is None:
                print(f"[{self.node_name}] Connection closed")
                break
            
            msg = DistributedMessage.from_dict(data)
            
            if msg.msg_type == DistributedMessage.MSG_FORWARD:
                hidden = msg.data["hidden"]
                attention_mask = msg.data["attention_mask"]
                position_ids = msg.data["position_ids"]
                meta = msg.data.get("meta", {})
                
                q_len = meta.get("q_len", 1)
                mode = meta.get("mode", "decode")
                self.past_len = meta.get("past_len", 0)
                
                # 运行 block
                hidden_out = self.run_block(hidden, attention_mask, position_ids, q_len, mode)
                
                # 转发到下一个节点
                forward_msg = DistributedMessage.create_forward_msg(
                    step=msg.step,
                    hidden=hidden_out,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    meta=meta
                )
                self.client.send(forward_msg.to_dict())
                
                self.step = msg.step
                print(f"[{self.node_name}] Step {self.step}: forwarded")
            
            elif msg.msg_type == DistributedMessage.MSG_TOOL_CALL:
                # 处理工具调用请求
                target_device = msg.data.get('target_device_id')
                
                if target_device == self.config.device_id:
                    # 本地执行工具
                    print(f"[{self.node_name}] Executing tool locally on Device {self.config.device_id}")
                    result_msg = self.tool_agent.handle_tool_call(msg)
                    
                    # 将结果转发到下一个节点（最终会到达尾节点，再返回给头节点）
                    self.client.send(result_msg.to_dict())
                    print(f"[{self.node_name}] Tool result forwarded to next node")
                else:
                    # 转发到下一个节点
                    self.client.send(msg.to_dict())
            
            elif msg.msg_type == DistributedMessage.MSG_TOOL_RESULT:
                # 转发工具结果到下一个节点
                self.client.send(msg.to_dict())
            
            elif msg.msg_type == DistributedMessage.MSG_RESET:
                self.reset()
                # 转发重置消息
                self.client.send(msg.to_dict())
                print(f"[{self.node_name}] Reset")
            
            elif msg.msg_type == DistributedMessage.MSG_SHUTDOWN:
                # 转发关闭消息
                self.client.send(msg.to_dict())
                print(f"[{self.node_name}] Received shutdown signal")
                break
    
    def reset(self):
        """重置状态 - 恢复到 system KV cache 状态"""
        self.step = 0
        self._restore_system_kv()
    
    def _init_tools(self):
        """初始化工具系统"""
        print(f"[{self.node_name}] Initializing tool system...")
        
        # 创建工具管理器
        devices = [0, 1, 2, 3]
        self.tool_manager = ToolManager(devices, device_memory_limit=500)
        
        # 注册内置工具
        self.tool_manager.register_tool('get_weather', weather_tool.TOOL_CONFIG)
        self.tool_manager.register_tool('calculator', calculator_tool.TOOL_CONFIG)
        
        # 创建本地工具代理
        self.tool_agent = ToolAgent(
            device_id=self.config.device_id,
            tool_manager=self.tool_manager,
            node_name=self.node_name
        )
        self.tool_agent.start()
        
        print(f"[{self.node_name}] Tool agent started on Device {self.config.device_id}")
    
    def shutdown(self):
        """关闭节点"""
        if self.tool_agent:
            self.tool_agent.stop()
        if self.block_model:
            self.block_model.finalize()
        if self.server:
            self.server.close()
        if self.client:
            self.client.close()


def main():
    parser = argparse.ArgumentParser(description="Middle Node - 4 Nodes Version")
    parser.add_argument("--node_id", type=int, required=True, choices=[1, 2], help="Node ID (1 or 2)")
    parser.add_argument("--system_om_dir", type=str, default="", help="System OM model directory (not needed if system_kv_dir has cache)")
    parser.add_argument("--prefill_om_dir", type=str, required=True, help="Prefill OM model directory")
    parser.add_argument("--decode_om_dir", type=str, required=True, help="Decode OM model directory")
    parser.add_argument("--system_kv_dir", type=str, default="./system_kv_cache", help="System KV cache directory")
    parser.add_argument("--device", type=int, default=0, help="Device ID")
    parser.add_argument("--max_cache_len", type=int, default=1024)
    parser.add_argument("--system_len", type=int, default=256)
    parser.add_argument("--prefill_len", type=int, default=512)
    parser.add_argument("--listen_port", type=int, default=None, help="Listen port (default: 9000 + node_id)")
    parser.add_argument("--next_ip", type=str, default=None, help="Next node IP")
    parser.add_argument("--next_port", type=int, default=None, help="Next node port")
    
    args = parser.parse_args()
    
    # 设置默认端口
    if args.listen_port is None:
        args.listen_port = 9000 + args.node_id
    if args.next_port is None:
        args.next_port = 9000 + args.node_id + 1
    
    # 设置默认 IP
    default_ips = {
        1: "192.168.137.102",  # Node 1 -> Node 2
        2: "192.168.137.103",  # Node 2 -> Node 3
    }
    if args.next_ip is None:
        args.next_ip = default_ips.get(args.node_id, "127.0.0.1")
    
    config = DistributedConfig4Nodes(
        system_om_dir=args.system_om_dir,
        prefill_om_dir=args.prefill_om_dir,
        decode_om_dir=args.decode_om_dir,
        system_kv_dir=args.system_kv_dir,
        system_len=args.system_len,
        device_id=args.device,
        max_cache_len=args.max_cache_len,
        prefill_len=args.prefill_len,
        node_id=args.node_id,
    )
    
    config.node_addresses[args.node_id]["port"] = args.listen_port
    config.node_addresses[args.node_id + 1]["ip"] = args.next_ip
    config.node_addresses[args.node_id + 1]["port"] = args.next_port
    
    node = MiddleNode4Nodes(config)
    
    try:
        node.init()
        node.process_loop()
    finally:
        node.shutdown()


if __name__ == "__main__":
    main()

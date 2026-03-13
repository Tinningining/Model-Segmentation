"""
Node 0: 头节点（带工具调用能力）- 4 节点版本
负责 embed.om + layers_0_6.om，是流水线的起点
支持在生成过程中调用外部工具
"""
from __future__ import annotations
import argparse
import numpy as np
import time
import threading
from collections import deque
from pathlib import Path

from config import DistributedConfig4Nodes
from network import NodeServer, NodeClient, DistributedMessage
from acl_model import ACLModel
from kvcache import create_kvcache
from utils import (
    build_attention_mask, build_position_ids,
    decode_incremental_text, decode_token_ids,
    encode_text, load_tokenizer, pad_input_ids, reshape_hidden_output,
    build_chat_prompt, build_tool_system_prompt, build_tool_result_prompt,
    build_tools_openai_schema,
)
from tools import (
    ToolManager,
    ToolCoordinator,
    Device0PreferredScheduler,
    ToolAgent,
    StreamingToolCallParser,
    AsyncToolExecutor,
)
from tools.builtin_tools import weather_tool, calculator_tool, time_tool, unit_converter_tool, translate_tool


class HeadNodeWithTools:
    """头节点：embed + layers_0_6（模型常驻内存）+ 工具调用"""
    
    def __init__(self, config: DistributedConfig4Nodes):
        self.config = config
        self.node_name = "Node0-Head-Tools"
        
        # 网络组件
        self.server = None
        self.client = None
        
        # 模型（常驻内存）
        self.embed_model = None
        self.block_model = None
        
        # KV Cache
        self.kv_cache = None
        
        # tokenizer
        self.tokenizer = None
        
        # 工具系统
        self.tool_manager = None
        self.tool_coordinator = None
        self.tool_agent = None
        
        # 节点客户端映射 (device_id -> NodeClient)
        self.device_clients = {}
        
        # inbound message buffer (避免 tool_result 与 token result 在同一 socket 上互相阻塞/错配)
        self._inbound_lock = threading.Lock()
        self._buffered_token_results = deque()  # MSG_RESULT
        self._buffered_tool_results = {}  # request_id -> DistributedMessage
        self._buffered_other_msgs = deque()

        # 状态
        self.past_len = 0
        self.step = 0
    
    def init(self):
        """初始化节点"""
        print(f"[{self.node_name}] Initializing...")
        
        # 1. 加载模型
        model_paths = self.config.get_model_paths()
        
        print(f"[{self.node_name}] Loading models...")
        self.embed_model = ACLModel(model_paths[0], self.config.device_id)
        self.embed_model.init()
        
        self.block_model = ACLModel(model_paths[1], self.config.device_id)
        self.block_model.init()
        
        # 2. 加载 tokenizer
        if self.config.tokenizer_dir:
            print(f"[{self.node_name}] Loading tokenizer from: {self.config.tokenizer_dir}")
            self.tokenizer = load_tokenizer(self.config.tokenizer_dir)
        
        # 3. 初始化 KV Cache
        num_layers = self.config.get_num_layers()
        
        self.kv_cache = create_kvcache(
            cache_type="basic",
            num_layers=num_layers,
            num_kv_heads=self.config.num_key_value_heads,
            head_dim=self.config.head_dim,
            max_cache_len=self.config.max_cache_len
        )
        print(f"[{self.node_name}] KV Cache initialized for {num_layers} layers")
        
        # 4. 初始化工具系统
        self._init_tools()
        
        # 5. 启动服务器
        listen_port = self.config.get_listen_port()
        self.server = NodeServer(listen_port, self.node_name)
        self.server.start()
        
        # 6. 连接到下一个节点
        next_addr = self.config.get_next_node_address()
        if next_addr:
            self.client = NodeClient(next_addr["ip"], next_addr["port"], self.node_name)
            self.client.connect()
        
        # 7. 等待尾节点连接
        print(f"[{self.node_name}] Waiting for tail node connection...")
        self.server.accept_connection()
        
        print(f"[{self.node_name}] Initialization complete!")
    
    def _init_tools(self):
        """初始化工具系统（支持多设备）"""
        print(f"[{self.node_name}] Initializing tool system...")
        
        # 创建工具管理器 - 支持所有4个设备
        devices = [0, 1, 2, 3]
        self.tool_manager = ToolManager(devices, device_memory_limit=500)
        
        # 注册内置工具
        self.tool_manager.register_tool('get_weather', weather_tool.TOOL_CONFIG)
        self.tool_manager.register_tool('calculator', calculator_tool.TOOL_CONFIG)
        self.tool_manager.register_tool('get_time', time_tool.TOOL_CONFIG)
        self.tool_manager.register_tool('unit_convert', unit_converter_tool.TOOL_CONFIG)
        self.tool_manager.register_tool('translate', translate_tool.TOOL_CONFIG)
        
        # 创建调度器
        scheduler = Device0PreferredScheduler(devices, main_device_id=0)
        
        # 创建本地工具代理
        self.tool_agent = ToolAgent(
            device_id=0,
            tool_manager=self.tool_manager,
            node_name=self.node_name
        )
        self.tool_agent.start()
        
        # 创建协调器（支持远程调用）
        self.tool_coordinator = ToolCoordinator(
            self.tool_manager,
            scheduler,
            local_device_id=0,
            remote_call_handler=self._handle_remote_tool_call
        )
        
        print(f"[{self.node_name}] Tool system initialized with {len(self.tool_manager.list_tools())} tools")
        print(f"[{self.node_name}] Supports devices: {devices}")
    
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
        hidden = self.run_embed(input_ids)
        
        attention_mask = build_attention_mask(
            self.past_len, q_len,
            self.config.max_cache_len,
            self.config.max_input_len
        )
        position_ids = build_position_ids(
            self.past_len, q_len,
            self.config.max_input_len
        )
        
        hidden = self.run_block(hidden, attention_mask, position_ids, q_len)
        
        return hidden, attention_mask, position_ids
    
    def _buffer_inbound_msg(self, msg: DistributedMessage):
        """缓存非当前期望消息，避免不同消息在同一 socket 上互相干扰"""
        with self._inbound_lock:
            if msg.msg_type == DistributedMessage.MSG_RESULT:
                self._buffered_token_results.append(msg)
            elif msg.msg_type == DistributedMessage.MSG_TOOL_RESULT:
                request_id = msg.data.get("request_id")
                if request_id:
                    self._buffered_tool_results[request_id] = msg
                else:
                    self._buffered_other_msgs.append(msg)
            else:
                self._buffered_other_msgs.append(msg)

    def _recv_from_server(
        self,
        expect_type: str,
        timeout: float = None,
        request_id: str = None,
    ) -> 'Optional[DistributedMessage]':
        """
        从 Node0.server (tail->head 的 socket) 接收指定类型消息。
        - 若收到非期望类型，会先缓存，继续等待
        - 若 expect_type 为 TOOL_RESULT 且指定 request_id，会匹配对应 request_id
        """
        # 1) 先从缓存中取
        with self._inbound_lock:
            if expect_type == DistributedMessage.MSG_RESULT and self._buffered_token_results:
                return self._buffered_token_results.popleft()

            if expect_type == DistributedMessage.MSG_TOOL_RESULT and request_id:
                if request_id in self._buffered_tool_results:
                    return self._buffered_tool_results.pop(request_id)

        # 2) 再从 socket 拉取，遇到非期望消息则缓存
        if timeout is not None and self.server and self.server.client_conn:
            self.server.client_conn.settimeout(timeout)

        try:
            while True:
                raw = self.server.recv()
                if raw is None:
                    return None

                msg = DistributedMessage.from_dict(raw)

                if msg.msg_type == expect_type:
                    if expect_type == DistributedMessage.MSG_TOOL_RESULT and request_id:
                        if msg.data.get("request_id") != request_id:
                            self._buffer_inbound_msg(msg)
                            continue
                    return msg

                self._buffer_inbound_msg(msg)
        finally:
            if timeout is not None and self.server and self.server.client_conn:
                self.server.client_conn.settimeout(None)

    def generate(
        self,
        prompt_ids: np.ndarray,
        max_new_tokens: int = 5000,
        max_tool_iterations: int = 5
    ) -> list:
        """
        生成文本（流式并行工具调用）：
        1. 边生成边解析 tool_call
        2. 检测到完整 tool_call 立即异步执行
        3. 生成结束后统一等待工具结果并注入继续推理
        """
        all_generated_ids = []
        current_ids = prompt_ids

        # 生成开始前先全链路 reset，确保 KV cache 干净
        self.reset()

        for tool_iter in range(max_tool_iterations):
            print(f"[{self.node_name}] === Tool Iteration {tool_iter + 1}/{max_tool_iterations} ===")
            
            parser = StreamingToolCallParser()
            executor = AsyncToolExecutor(self.tool_coordinator, max_workers=4)
            decoded_text = ""
            round_generated_ids = []
            has_tool_calls = False
            reached_eos = False
            
            try:
                for _ in range(max_new_tokens + 1):
                    q_len = current_ids.shape[1]
                    
                    if self.past_len + q_len > self.config.max_cache_len:
                        print(f"[{self.node_name}] KV cache overflow, stopping generation")
                        reached_eos = True
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
                    
                    # 等待尾节点返回 token（同时会缓存途中返回的 tool_result）
                    result_msg = self._recv_from_server(DistributedMessage.MSG_RESULT)
                    if result_msg is None:
                        print(f"[{self.node_name}] Connection closed")
                        reached_eos = True
                        break
                    
                    if result_msg.msg_type == DistributedMessage.MSG_RESULT:
                        next_token = result_msg.data.get("next_token")
                        
                        if next_token == self.config.eos_token_id:
                            print(f"[{self.node_name}] EOS token generated")
                            reached_eos = True
                            break
                        
                        round_generated_ids.append(next_token)
                        all_generated_ids.append(next_token)
                        print(f"[{self.node_name}] Step {self.step}: generated token {next_token}")
                        
                        if self.tokenizer is not None:
                            delta_text, decoded_text = decode_incremental_text(
                                self.tokenizer,
                                round_generated_ids,
                                previous_text=decoded_text,
                                skip_special_tokens=True
                            )
                            if delta_text:
                                print(f"[{self.node_name}] Step {self.step}: generated text {delta_text!r}")
                                
                                # 流式解析：仅喂增量文本
                                completed_calls = parser.feed(delta_text)
                                for tool_call in completed_calls:
                                    has_tool_calls = True
                                    print(
                                        f"[{self.node_name}] Streaming detected tool_call: "
                                        f"{tool_call['name']} (id={tool_call['id']})"
                                    )
                                    executor.execute_async(tool_call)
                        
                        self.past_len += q_len
                        self.step += 1
                        current_ids = np.array([[next_token]], dtype=np.int64)
                    
                    elif result_msg.msg_type == DistributedMessage.MSG_SHUTDOWN:
                        print(f"[{self.node_name}] Received shutdown signal")
                        reached_eos = True
                        break
                
                # EOS 后再做一次最终检查：解析器可能在最后几个 token 才凑齐完整 JSON
                if not has_tool_calls:
                    # 尝试从已有 buffer 中提取（可能最后一个 token 刚好闭合了 JSON）
                    final_calls = parser.get_all_calls()
                    if final_calls:
                        has_tool_calls = True
                        already_submitted = set(executor.futures.keys())
                        for tool_call in final_calls:
                            if tool_call["id"] not in already_submitted:
                                print(
                                    f"[{self.node_name}] Late-detected tool_call: "
                                    f"{tool_call['name']} (id={tool_call['id']})"
                                )
                                executor.execute_async(tool_call)

                if not has_tool_calls:
                    # 本轮确实没有工具调用，直接结束
                    break
                
                pending_count = executor.result_buffer.get_pending_count()
                if pending_count > 0:
                    print(f"[{self.node_name}] Waiting for {pending_count} streaming tool(s) to complete...")
                    tool_results = executor.wait_all_complete(timeout=30.0)
                else:
                    tool_results = executor.result_buffer.get_all_results()
                
                results_text = self.tool_coordinator.format_tool_results(tool_results)
                print(f"[{self.node_name}] Aggregated tool results:\n{results_text}")
                
                # 注入工具结果，开始下一轮推理
                if self.tokenizer is None:
                    print(f"[{self.node_name}] Tokenizer unavailable, cannot continue with tool results")
                    break

                # 工具执行完成后必须 reset（清 KV cache + 广播 MSG_RESET）
                self.reset()

                # 构建第二轮推理 prompt：将所有工具结果注入 Qwen chat 模板
                all_calls = parser.get_all_calls()
                # 第二轮不再传入工具描述（精简 prompt 长度），直接让模型回答
                result_system = build_tool_result_prompt(
                    user_message=getattr(self, '_current_user_message', ''),
                    tool_calls=all_calls,
                    tool_results=tool_results,
                    tools=None,
                )
                tool_prompt = build_chat_prompt(
                    system=result_system,
                    user=getattr(self, '_current_user_message', ''),
                )
                tool_ids = encode_text(self.tokenizer, tool_prompt)
                
                # 如果 prompt 超过 max_input_len，从左侧截断（保留最近的上下文）
                if len(tool_ids) > self.config.max_input_len:
                    print(
                        f"[{self.node_name}] Tool prompt too long ({len(tool_ids)} tokens), "
                        f"truncating to {self.config.max_input_len}"
                    )
                    tool_ids = tool_ids[-self.config.max_input_len:]
                
                current_ids = np.array([tool_ids], dtype=np.int64)
                
                # 若已到 EOS，则依靠工具结果触发下一轮；若未到 EOS，同样进入下一轮
                if reached_eos:
                    print(f"[{self.node_name}] Continuing with tool results after EOS")
            finally:
                executor.shutdown()
        
        return all_generated_ids
    
    def reset(self):
        """重置状态（本地 + 全链路）"""
        self.past_len = 0
        self.step = 0
        self.kv_cache.reset()

        # 清理 inbound buffer，避免 reset 后误消费旧消息
        with self._inbound_lock:
            self._buffered_token_results.clear()
            self._buffered_tool_results.clear()
            self._buffered_other_msgs.clear()
        
        reset_msg = DistributedMessage.create_reset_msg()
        self.client.send(reset_msg.to_dict())
    
    def _handle_remote_tool_call(self, device_id: int, tool_call_msg: DistributedMessage) -> DistributedMessage:
        """处理远程工具调用请求（带超时、缓存与错误处理）"""
        print(f"[{self.node_name}] Sending tool call to Device {device_id}")
        
        # 根据device_id确定目标节点
        # Device 0 -> Node 0 (local)
        # Device 1 -> Node 1
        # Device 2 -> Node 2
        # Device 3 -> Node 3
        
        if device_id == 0:
            # 本地执行，不应该走到这里
            return self.tool_agent.handle_tool_call(tool_call_msg)
        
        # 通过流水线发送工具调用请求
        # 工具调用消息会沿着流水线传递，直到到达目标设备
        try:
            self.client.send(tool_call_msg.to_dict())
            print(f"[{self.node_name}] Tool call sent to pipeline")
        except Exception as e:
            return DistributedMessage.create_tool_result_msg(
                request_id=tool_call_msg.data.get('request_id'),
                success=False,
                error=f"Failed to send tool call: {e}"
            )
        
        # 等待工具结果（在同一 socket 上与 token result 混流；这里用缓存机制避免阻塞/错配）
        timeout = 30.0
        request_id = tool_call_msg.data.get("request_id")
        try:
            result_msg = self._recv_from_server(
                DistributedMessage.MSG_TOOL_RESULT,
                timeout=timeout,
                request_id=request_id,
            )
            if result_msg is None:
                return DistributedMessage.create_tool_result_msg(
                    request_id=request_id,
                    success=False,
                    error=f"No response from Device {device_id}"
                )
            return result_msg
        except Exception as e:
            return DistributedMessage.create_tool_result_msg(
                request_id=request_id,
                success=False,
                error=f"Tool call error: {e}"
            )
    
    def _get_device_client(self, device_id: int):
        """获取到指定设备的客户端连接"""
        if device_id in self.device_clients:
            return self.device_clients[device_id]
        
        # 创建新连接
        # 根据device_id映射到节点地址
        node_id = device_id  # Device ID = Node ID
        
        if node_id >= len(self.config.node_addresses):
            print(f"[{self.node_name}] Invalid device_id: {device_id}")
            return None
        
        addr = self.config.node_addresses[node_id]
        client = NodeClient(addr["ip"], addr["port"], f"{self.node_name}-to-Device{device_id}")
        
        if client.connect(retry_interval=0.5, max_retries=3):
            self.device_clients[device_id] = client
            return client
        
        return None
    
    def shutdown(self):
        """关闭节点"""
        # 停止工具代理
        if self.tool_agent:
            self.tool_agent.stop()
        
        # 关闭设备客户端连接
        for client in self.device_clients.values():
            client.close()
        
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
    parser = argparse.ArgumentParser(description="Node 0: Head Node with Tools - 4 Nodes Version")
    parser.add_argument("--om_dir", type=str, required=True, help="OM model directory")
    parser.add_argument("--device", type=int, default=0, help="Device ID")
    parser.add_argument("--max_cache_len", type=int, default=1024)
    parser.add_argument("--max_input_len", type=int, default=512)
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
    
    node = HeadNodeWithTools(config)
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
    
    # 构建带工具描述的 Qwen chat prompt
    # 1) 将内部工具配置转换为 OpenAI function calling 格式
    tool_configs = {
        'get_weather': weather_tool.TOOL_CONFIG,
        'calculator': calculator_tool.TOOL_CONFIG,
        'get_time': time_tool.TOOL_CONFIG,
        'unit_convert': unit_converter_tool.TOOL_CONFIG,
        'translate': translate_tool.TOOL_CONFIG,
    }
    openai_tools = build_tools_openai_schema(tool_configs)
    
    # 2) 构建 system prompt（含工具描述 + JSON 输出指令）
    system_prompt = build_tool_system_prompt(openai_tools)
    
    # 3) 用 Qwen im_start/im_end 模板包装
    full_prompt = build_chat_prompt(system=system_prompt, user=input_text)
    print(f"[Input] Full prompt:\n{full_prompt[:300]}...")
    
    # 4) 保存用户消息和工具列表供工具结果注入时使用
    node._current_user_message = input_text
    node._openai_tools = openai_tools
    
    # 将 prompt 编码为 token
    prompt_ids = encode_text(node.tokenizer, full_prompt)
    print(f"[Input] Encoded to {len(prompt_ids)} tokens")
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

"""
单机 ONNX 模型执行 - 支持工具调用
借鉴 file_pipeline 的 ONNX 执行方式和 code 的工具调用能力
"""
import argparse
import json
import time
from pathlib import Path
import numpy as np

from config import LocalConfig
from onnx_model import ONNXModelRunner
from kvcache import KVCache
from utils import (
    load_tokenizer,
    encode_text,
    decode_token_ids,
    decode_incremental_text,
    build_system_attention_mask,
    build_prefill_with_past_attention_mask,
    build_attention_mask,
    build_position_ids,
    pad_input_ids,
    sample_token,
    build_chat_prompt,
    build_tool_system_prompt,
    build_tool_result_prompt,
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
from tools.builtin_tools import (
    weather_tool,
    calculator_tool,
    time_tool,
    unit_converter_tool,
    translate_tool,
)


class LocalONNXRunner:
    """单机 ONNX 模型运行器（支持工具调用）"""
    
    def __init__(self, config: LocalConfig):
        self.config = config
        self.tokenizer = None
        self.kv_cache = None
        self.system_kv_snapshot = None
        
        # 模型运行器（按需加载）
        self.system_runner = None
        self.prefill_runner = None
        self.decode_runner = None
        self.current_mode = None
        
        # 工具系统
        self.tool_manager = None
        self.tool_coordinator = None
        self.tool_agent = None
        
        self.past_len = 0
        self.step = 0
    
    def init(self):
        """初始化"""
        print("[LocalRunner] Initializing...")
        
        # 加载 tokenizer
        if self.config.tokenizer_dir:
            print(f"[LocalRunner] Loading tokenizer from: {self.config.tokenizer_dir}")
            self.tokenizer = load_tokenizer(self.config.tokenizer_dir)
        
        # 初始化 KV cache
        self.kv_cache = KVCache(
            num_layers=self.config.num_hidden_layers,
            num_kv_heads=self.config.num_key_value_heads,
            head_dim=self.config.head_dim,
            max_cache_len=self.config.max_cache_len,
        )
        print(f"[LocalRunner] KV Cache initialized for {self.config.num_hidden_layers} layers")
        
        # 初始化工具系统
        self._init_tools()
        
        # 加载或生成 system KV cache
        self._init_system_kv()
        
        print("[LocalRunner] Initialization complete!")
    
    def _init_tools(self):
        """初始化工具系统"""
        print("[LocalRunner] Initializing tool system...")
        
        # 单机版本只有一个设备
        devices = [0]
        self.tool_manager = ToolManager(devices, device_memory_limit=500)
        
        # 注册内置工具
        self.tool_manager.register_tool('get_weather', weather_tool.TOOL_CONFIG)
        self.tool_manager.register_tool('calculator', calculator_tool.TOOL_CONFIG)
        self.tool_manager.register_tool('get_time', time_tool.TOOL_CONFIG)
        self.tool_manager.register_tool('unit_convert', unit_converter_tool.TOOL_CONFIG)
        self.tool_manager.register_tool('translate', translate_tool.TOOL_CONFIG)
        
        # 创建本地调度器
        scheduler = Device0PreferredScheduler(devices)
        
        # 创建工具代理
        self.tool_agent = ToolAgent(
            device_id=0,
            tool_manager=self.tool_manager,
            node_name="LocalRunner"
        )
        self.tool_agent.start()
        
        # 创建协调器（本地执行，无需远程调用）
        self.tool_coordinator = ToolCoordinator(
            self.tool_manager,
            scheduler,
            local_device_id=0,
            remote_call_handler=None  # 单机版本不需要远程调用
        )
        
        print(f"[LocalRunner] Tool system initialized with {len(self.tool_manager.list_tools())} tools")
    
    def _init_system_kv(self):
        """加载或生成 system KV cache"""
        system_kv_dir = Path(self.config.system_kv_dir) if self.config.system_kv_dir else None
        if system_kv_dir is None:
            print("[LocalRunner] No system_kv_dir configured, skipping system KV")
            return
        
        system_kv_dir.mkdir(parents=True, exist_ok=True)
        meta_path = system_kv_dir / "system_kv_meta.json"
        kv_path = system_kv_dir / "system_kv_cache.npz"
        
        if meta_path.exists() and kv_path.exists():
            # 复用已有的 system KV cache
            print(f"[LocalRunner] Loading cached system KV from {system_kv_dir}")
            with open(meta_path, "r", encoding="utf-8") as f:
                sys_meta = json.load(f)
            
            kv_data = np.load(str(kv_path))
            self.system_kv_snapshot = {
                'past_key': kv_data['past_key'],
                'past_value': kv_data['past_value'],
                'current_len': sys_meta['system_q_len']
            }
            
            # 恢复到 system KV 状态
            self.kv_cache.restore_snapshot(self.system_kv_snapshot)
            self.past_len = self.system_kv_snapshot['current_len']
            print(f"[LocalRunner] System KV loaded. system_past_len={self.past_len}")
        else:
            print("[LocalRunner] System KV cache not found, will generate on first run")
    
    def _ensure_mode(self, mode: str):
        """确保当前加载的模型与所需模式一致"""
        if self.current_mode == mode:
            return
        
        # 卸载当前模型
        if self.current_mode:
            print(f"[LocalRunner] Unloading {self.current_mode} models...")
            if self.current_mode == "system":
                self.system_runner = None
            elif self.current_mode == "prefill":
                self.prefill_runner = None
            else:
                self.decode_runner = None
        
        # 加载新模型
        print(f"[LocalRunner] Loading {mode} models...")
        if mode == "system":
            paths = self.config.get_system_model_paths()
            self.system_runner = ONNXModelRunner(paths)
        elif mode == "prefill":
            paths = self.config.get_prefill_model_paths()
            self.prefill_runner = ONNXModelRunner(paths)
        else:
            paths = self.config.get_decode_model_paths()
            self.decode_runner = ONNXModelRunner(paths)
        
        self.current_mode = mode
        print(f"[LocalRunner] {mode} models loaded")
    
    def _get_runner(self):
        """获取当前模式的模型运行器"""
        if self.current_mode == "system":
            return self.system_runner
        elif self.current_mode == "prefill":
            return self.prefill_runner
        else:
            return self.decode_runner
    
    def _run_system_stage(self, system_prompt_text: str):
        """运行 system 阶段生成 KV cache"""
        if self.system_kv_snapshot is not None:
            return  # 已有 system KV
        
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer required for system stage")
        
        system_ids = encode_text(self.tokenizer, system_prompt_text)
        q_len = len(system_ids)
        system_len = self.config.system_len
        
        if q_len > system_len:
            print(f"[LocalRunner] System prompt {q_len} tokens > system_len {system_len}, truncating")
            system_ids = system_ids[:system_len]
            q_len = system_len
        
        input_ids = np.array([system_ids], dtype=np.int64)
        
        # 加载 system 模型
        self._ensure_mode("system")
        runner = self._get_runner()
        
        # Embed
        embed_ids = pad_input_ids(input_ids, system_len, pad_id=0)
        hidden = runner.run_embed(embed_ids)
        
        # Attention mask & position ids
        attention_mask = build_system_attention_mask(q_len, system_len)
        position_ids = build_position_ids(0, q_len, system_len)
        
        # 运行所有 blocks（无 past KV）
        for block_idx in range(4):
            hidden, present_key, present_value = runner.run_block(
                block_idx, hidden, attention_mask, position_ids
            )
            
            # 更新 KV cache
            if block_idx == 0:
                # 第一个 block，初始化 KV cache
                self.kv_cache.reset()
            
            # 提取有效的 KV（前 q_len 个位置）
            pk = present_key[:, :, :, :q_len, :].astype(np.float16)
            pv = present_value[:, :, :, :q_len, :].astype(np.float16)
            
            # 更新对应层的 KV
            num_layers_per_block = 7
            start_layer = block_idx * num_layers_per_block
            end_layer = start_layer + pk.shape[0]
            
            self.kv_cache.past_key[start_layer:end_layer, :, :, :q_len, :] = pk
            self.kv_cache.past_value[start_layer:end_layer, :, :, :q_len, :] = pv
        
        self.kv_cache.current_len = q_len
        self.past_len = q_len
        
        # 保存 system KV 快照
        self.system_kv_snapshot = self.kv_cache.save_snapshot()
        
        # 持久化到磁盘
        system_kv_dir = Path(self.config.system_kv_dir)
        system_kv_dir.mkdir(parents=True, exist_ok=True)
        
        np.savez_compressed(
            str(system_kv_dir / "system_kv_cache.npz"),
            past_key=self.system_kv_snapshot['past_key'],
            past_value=self.system_kv_snapshot['past_value']
        )
        
        meta = {
            "system_q_len": q_len,
            "max_cache_len": self.config.max_cache_len
        }
        with open(system_kv_dir / "system_kv_meta.json", "w", encoding="utf-8") as fw:
            json.dump(meta, fw, indent=2)
        
        print(f"[LocalRunner] System KV generated and saved. q_len={q_len}")
    
    def process_forward(self, input_ids: np.ndarray, q_len: int, mode: str = "decode"):
        """处理一次前向传播"""
        self._ensure_mode(mode)
        runner = self._get_runner()
        
        if mode == "system":
            cur_input_len = self.config.system_len
        elif mode == "prefill":
            cur_input_len = self.config.prefill_len
        else:
            cur_input_len = 1
        
        # Embed
        embed_ids = pad_input_ids(input_ids, cur_input_len, pad_id=0)
        hidden = runner.run_embed(embed_ids)
        
        # Attention mask & position ids
        if mode == "system":
            attention_mask = build_system_attention_mask(q_len, cur_input_len)
        elif mode == "prefill":
            # Prefill 总是使用 max_cache_len + max_input_len 的 attention mask
            attention_mask = build_prefill_with_past_attention_mask(
                q_len, cur_input_len, self.past_len, self.config.max_cache_len
            )
        else:
            attention_mask = build_attention_mask(
                self.past_len, q_len, self.config.max_cache_len, cur_input_len
            )
        
        position_ids = build_position_ids(self.past_len, q_len, cur_input_len)
        
        # 运行所有 blocks
        for block_idx in range(4):
            num_layers_per_block = 7
            start_layer = block_idx * num_layers_per_block
            end_layer = start_layer + num_layers_per_block
            
            if mode == "system":
                # System: 无 past KV 输入
                hidden, present_key, present_value = runner.run_block(
                    block_idx, hidden, attention_mask, position_ids
                )
            else:
                # Prefill/Decode: 总是需要 past KV（即使是全零）
                past_key = self.kv_cache.past_key[start_layer:end_layer]
                past_value = self.kv_cache.past_value[start_layer:end_layer]
                
                hidden, present_key, present_value = runner.run_block(
                    block_idx, hidden, attention_mask, position_ids,
                    past_key, past_value
                )
            
            # 更新 KV cache
            num_layers_per_block = 7
            start_layer = block_idx * num_layers_per_block
            end_layer = start_layer + present_key.shape[0]
            
            if mode == "system":
                # System: 直接写入前 q_len 个位置
                pk = present_key[:, :, :, :q_len, :].astype(np.float16)
                pv = present_value[:, :, :, :q_len, :].astype(np.float16)
                self.kv_cache.past_key[start_layer:end_layer, :, :, :q_len, :] = pk
                self.kv_cache.past_value[start_layer:end_layer, :, :, :q_len, :] = pv
            elif mode == "prefill":
                # Prefill: 追加到 past_len 位置
                pk = present_key[:, :, :, :q_len, :].astype(np.float16)
                pv = present_value[:, :, :, :q_len, :].astype(np.float16)
                s = self.past_len
                e = s + q_len
                self.kv_cache.past_key[start_layer:end_layer, :, :, s:e, :] = pk
                self.kv_cache.past_value[start_layer:end_layer, :, :, s:e, :] = pv
            else:
                # Decode: 追加到 past_len 位置
                pk = present_key[:, :, :, :q_len, :].astype(np.float16)
                pv = present_value[:, :, :, :q_len, :].astype(np.float16)
                s = self.past_len
                e = s + q_len
                self.kv_cache.past_key[start_layer:end_layer, :, :, s:e, :] = pk
                self.kv_cache.past_value[start_layer:end_layer, :, :, s:e, :] = pv
        
        if mode == "system":
            self.kv_cache.current_len = q_len
        else:
            self.kv_cache.current_len = self.past_len + q_len
        
        # LM head
        logits = runner.run_lm_head(hidden)
        
        return logits, hidden
    
    def generate(
        self,
        prompt_ids: np.ndarray,
        max_new_tokens: int = 100,
        max_tool_iterations: int = 5
    ) -> list:
        """
        生成文本（流式并行工具调用）
        """
        all_generated_ids = []
        current_ids = prompt_ids
        
        # 生成开始前先 reset（恢复到 system KV 状态）
        self.reset()
        
        for tool_iter in range(max_tool_iterations):
            print(f"[LocalRunner] === Tool Iteration {tool_iter + 1}/{max_tool_iterations} ===")
            
            parser = StreamingToolCallParser()
            decoded_text = ""
            round_generated_ids = []
            tool_calls_list = []
            reached_eos = False
            
            for _ in range(max_new_tokens + 1):
                q_len = current_ids.shape[1]
                
                if self.past_len + q_len > self.config.max_cache_len:
                    print("[LocalRunner] KV cache overflow, stopping generation")
                    reached_eos = True
                    break
                
                # 确定当前模式
                # 如果有 system KV (past_len > 0) 且是多 token 输入，使用 prefill
                # 否则使用 decode
                mode = "prefill" if self.past_len > 0 and q_len > 1 else "decode"
                
                # 处理前向传播
                logits, hidden = self.process_forward(current_ids, q_len, mode)
                
                # 采样下一个 token
                next_token = sample_token(
                    logits[0, q_len - 1, :],
                    temperature=self.config.temperature,
                    top_k=self.config.top_k,
                    top_p=self.config.top_p,
                    greedy=self.config.greedy
                )
                
                if next_token == self.config.eos_token_id:
                    print("[LocalRunner] EOS token generated")
                    reached_eos = True
                    break
                
                round_generated_ids.append(next_token)
                all_generated_ids.append(next_token)
                
                if self.tokenizer is not None:
                    delta_text, decoded_text = decode_incremental_text(
                        self.tokenizer,
                        round_generated_ids,
                        previous_text=decoded_text,
                        skip_special_tokens=True
                    )
                    if delta_text:
                        print(f"[LocalRunner] Step {self.step}: token {next_token}, text {delta_text!r}")
                        
                        # 流式解析工具调用
                        completed_calls = parser.feed(delta_text)
                        for tool_call in completed_calls:
                            print(
                                f"[LocalRunner] Detected tool_call: "
                                f"{tool_call['name']} (id={tool_call['id']})"
                            )
                            tool_calls_list.append(tool_call)
                    else:
                        print(f"[LocalRunner] Step {self.step}: token {next_token}")
                else:
                    print(f"[LocalRunner] Step {self.step}: token {next_token}")
                
                self.past_len += q_len
                self.step += 1
                current_ids = np.array([[next_token]], dtype=np.int64)
            
            # EOS 后再做一次最终检查
            if not tool_calls_list:
                final_calls = parser.get_all_calls()
                if final_calls:
                    already_added = set(tc["id"] for tc in tool_calls_list)
                    for tool_call in final_calls:
                        if tool_call["id"] not in already_added:
                            print(
                                f"[LocalRunner] Late-detected tool_call: "
                                f"{tool_call['name']} (id={tool_call['id']})"
                            )
                            tool_calls_list.append(tool_call)
            
            if not tool_calls_list:
                break
            
            # 同步执行所有工具
            print(f"[LocalRunner] Executing {len(tool_calls_list)} tool(s)...")
            tool_results = self.tool_coordinator.execute_tools(tool_calls_list)
            
            results_text = self.tool_coordinator.format_tool_results(tool_results)
            print(f"[LocalRunner] Aggregated tool results:\n{results_text}")
            
            # 工具执行完成后必须 reset
            self.reset()
            
            # 构建第二轮推理 prompt
            result_system = build_tool_result_prompt(
                user_message=getattr(self, '_current_user_message', ''),
                tool_calls=tool_calls_list,
                tool_results=tool_results,
                tools=None,
            )
            tool_prompt = build_chat_prompt(
                system=result_system,
                user=getattr(self, '_current_user_message', ''),
            )
            tool_ids = encode_text(self.tokenizer, tool_prompt)
            
            if len(tool_ids) > self.config.prefill_len:
                print(
                    f"[LocalRunner] Tool prompt too long ({len(tool_ids)} tokens), "
                    f"truncating to {self.config.prefill_len}"
                )
                tool_ids = tool_ids[-self.config.prefill_len:]
            
            current_ids = np.array([tool_ids], dtype=np.int64)
        
        return all_generated_ids
    
    def reset(self):
        """重置状态 - 恢复到 system KV cache 状态"""
        self.step = 0
        if self.system_kv_snapshot:
            self.kv_cache.restore_snapshot(self.system_kv_snapshot)
            self.past_len = self.system_kv_snapshot['current_len']
        else:
            self.kv_cache.reset()
            self.past_len = 0
    
    def shutdown(self):
        """关闭"""
        if self.tool_agent:
            self.tool_agent.stop()


def main():
    parser = argparse.ArgumentParser(description="Local ONNX Runner with Tool Support")
    parser.add_argument("--system_onnx_dir", type=str, default="", help="System ONNX model directory")
    parser.add_argument("--prefill_onnx_dir", type=str, required=True, help="Prefill ONNX model directory")
    parser.add_argument("--decode_onnx_dir", type=str, required=True, help="Decode ONNX model directory")
    parser.add_argument("--tokenizer_dir", type=str, required=True, help="Tokenizer directory")
    parser.add_argument("--system_kv_dir", type=str, default="./system_kv_cache", help="System KV cache directory")
    parser.add_argument("--input_file", type=str, required=True, help="Input text file (user query)")
    parser.add_argument("--max_new_tokens", type=int, default=100)
    parser.add_argument("--max_cache_len", type=int, default=1024)
    parser.add_argument("--system_len", type=int, default=256)
    parser.add_argument("--prefill_len", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--greedy", action="store_true", default=True)
    
    args = parser.parse_args()
    
    config = LocalConfig(
        system_onnx_dir=args.system_onnx_dir,
        prefill_onnx_dir=args.prefill_onnx_dir,
        decode_onnx_dir=args.decode_onnx_dir,
        tokenizer_dir=args.tokenizer_dir,
        system_kv_dir=args.system_kv_dir,
        system_len=args.system_len,
        prefill_len=args.prefill_len,
        max_cache_len=args.max_cache_len,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        greedy=args.greedy,
    )
    
    runner = LocalONNXRunner(config)
    runner.init()
    
    # 如果没有 system KV cache，先生成
    if runner.system_kv_snapshot is None and args.system_onnx_dir:
        print("[LocalRunner] Generating system KV cache...")
        tool_configs = {
            'get_weather': weather_tool.TOOL_CONFIG,
            'calculator': calculator_tool.TOOL_CONFIG,
            'get_time': time_tool.TOOL_CONFIG,
            'unit_convert': unit_converter_tool.TOOL_CONFIG,
            'translate': translate_tool.TOOL_CONFIG,
        }
        openai_tools = build_tools_openai_schema(tool_configs)
        system_prompt = build_tool_system_prompt(openai_tools)
        runner._run_system_stage(system_prompt)
    
    # 从文本文件读取用户输入（user_prompt.txt）
    print(f"[Input] Reading user prompt from file: {args.input_file}")
    with open(args.input_file, "r", encoding="utf-8") as f:
        user_text = f.read().strip()
    
    if not user_text:
        raise ValueError(f"Input file is empty: {args.input_file}")
    
    print(f"[Input] User text: {user_text}")
    
    # 保存用户消息供工具结果注入时使用
    runner._current_user_message = user_text
    
    # 编码用户输入为 token（不包含 system prompt，因为已经在 system KV 中）
    # 使用 Qwen chat 格式：<|im_start|>user\n{user_text}<|im_end|>\n<|im_start|>assistant\n
    user_prompt = f"<|im_start|>user\n{user_text}<|im_end|>\n<|im_start|>assistant\n"
    prompt_ids = encode_text(runner.tokenizer, user_prompt)
    print(f"[Input] User prompt encoded to {len(prompt_ids)} tokens")
    prompt_ids = np.array([prompt_ids], dtype=np.int64)
    
    try:
        start_time = time.time()
        generated = runner.generate(prompt_ids, args.max_new_tokens)
        elapsed = time.time() - start_time
        
        generated_text = ""
        if runner.tokenizer is not None:
            generated_text = decode_token_ids(
                runner.tokenizer,
                generated,
                skip_special_tokens=True
            )
        
        print(f"\n{'='*50}")
        print(f"Generated {len(generated)} tokens in {elapsed:.2f}s")
        print(f"Speed: {len(generated)/elapsed:.2f} tokens/s")
        print(f"Generated IDs: {generated}")
        print(f"Generated text: {generated_text}")
        
    finally:
        runner.shutdown()


if __name__ == "__main__":
    main()

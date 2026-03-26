"""
单机 ONNX 模型执行 - 支持多轮对话和工具调用
使用 prefill 模型处理所有输入（system prompt、历史记忆、用户问题等）
"""
import argparse
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

from config import LocalConfig
from onnx_model import ONNXModelRunner
from kvcache import KVCache
from utils import (
    load_tokenizer,
    encode_text,
    decode_token_ids,
    decode_incremental_text,
    build_prefill_with_past_attention_mask,
    build_attention_mask,
    build_position_ids,
    pad_input_ids,
    sample_token,
    build_chat_prompt,
    build_tool_system_prompt,
    build_round2_system_prompt,
    build_tool_result_prompt,
    build_tools_openai_schema,
)
from tools import (
    ToolManager,
    ToolCoordinator,
    Device0PreferredScheduler,
    ToolAgent,
    StreamingToolCallParser,
)
from tools.builtin_tools import (
    weather_tool,
    calculator_tool,
    time_tool,
    unit_converter_tool,
    translate_tool,
)


class MultiTurnONNXRunner:
    """支持多轮对话的单机 ONNX 模型运行器"""
    
    def __init__(self, config: LocalConfig):
        self.config = config
        self.tokenizer = None
        self.kv_cache = None
        
        # KV cache 快照管理
        self.system_only_kv_snapshot = None       # 仅含第一轮 system prompt 的 KV（永不变）
        self.base_kv_snapshot = None              # 第一轮：system + 所有历史记忆的 KV
        self.round2_system_only_kv_snapshot = None  # 仅含第二轮 system prompt 的 KV（永不变）
        self.round2_base_kv_snapshot = None       # 第二轮：system + 所有历史记忆的 KV
        self.question_kv_snapshot = None          # 当前问题第一轮推理后的 KV
        
        # 对话记忆
        self.conversation_memory: List[Dict[str, Any]] = []
        
        # 模型运行器（按需加载）
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
        print("[MultiTurn] Initializing...")
        
        # 加载 tokenizer
        if self.config.tokenizer_dir:
            print(f"[MultiTurn] Loading tokenizer from: {self.config.tokenizer_dir}")
            self.tokenizer = load_tokenizer(self.config.tokenizer_dir)
        
        # 初始化 KV cache
        self.kv_cache = KVCache(
            num_layers=self.config.num_hidden_layers,
            num_kv_heads=self.config.num_key_value_heads,
            head_dim=self.config.head_dim,
            max_cache_len=self.config.max_cache_len,
        )
        print(f"[MultiTurn] KV Cache initialized for {self.config.num_hidden_layers} layers")
        
        # 初始化工具系统
        self._init_tools()
        
        # 生成初始 KV cache（工具描述）
        self._init_base_kv()
        
        print("[MultiTurn] Initialization complete!")
    
    def _init_tools(self):
        """初始化工具系统"""
        print("[MultiTurn] Initializing tool system...")
        
        devices = [0]
        self.tool_manager = ToolManager(devices, device_memory_limit=500)
        
        # 注册内置工具
        self.tool_manager.register_tool('get_weather', weather_tool.TOOL_CONFIG)
        self.tool_manager.register_tool('calculator', calculator_tool.TOOL_CONFIG)
        self.tool_manager.register_tool('get_time', time_tool.TOOL_CONFIG)
        self.tool_manager.register_tool('unit_convert', unit_converter_tool.TOOL_CONFIG)
        self.tool_manager.register_tool('translate', translate_tool.TOOL_CONFIG)
        
        scheduler = Device0PreferredScheduler(devices)
        
        self.tool_agent = ToolAgent(
            device_id=0,
            tool_manager=self.tool_manager,
            node_name="MultiTurnRunner"
        )
        self.tool_agent.start()
        
        self.tool_coordinator = ToolCoordinator(
            self.tool_manager,
            scheduler,
            local_device_id=0,
            remote_call_handler=None
        )
        
        print(f"[MultiTurn] Tool system initialized with {len(self.tool_manager.list_tools())} tools")
    
    def _init_base_kv(self):
        """生成初始 KV cache（使用 prefill 模型处理工具描述）"""
        print("[MultiTurn] Generating base KV cache with prefill model...")
        
        tool_configs = {
            'get_weather': weather_tool.TOOL_CONFIG,
            'calculator': calculator_tool.TOOL_CONFIG,
            'get_time': time_tool.TOOL_CONFIG,
            'unit_convert': unit_converter_tool.TOOL_CONFIG,
            'translate': translate_tool.TOOL_CONFIG,
        }
        openai_tools = build_tools_openai_schema(tool_configs)
        # 始终使用 has_history=True 格式，这样后续追加历史时 system prompt 不变
        system_prompt = build_tool_system_prompt(openai_tools, has_history=True)
        
        # 用正确的 chat template 包裹 system prompt
        full_prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        
        # 使用 prefill 模型处理第一轮 system prompt
        self._process_prompt_with_prefill(full_prompt)
        self.system_only_kv_snapshot = self.kv_cache.save_snapshot()
        self.base_kv_snapshot = self.kv_cache.save_snapshot()
        
        # 同时初始化第二轮 system prompt 的 KV
        round2_system_prompt = build_round2_system_prompt(has_history=True)
        round2_full_prompt = f"<|im_start|>system\n{round2_system_prompt}<|im_end|>\n"
        self._process_prompt_with_prefill(round2_full_prompt)
        self.round2_system_only_kv_snapshot = self.kv_cache.save_snapshot()
        self.round2_base_kv_snapshot = self.kv_cache.save_snapshot()
        
        print(f"[MultiTurn] Base KV generated. past_len={self.past_len}")
    
    def _format_memory_text(self) -> str:
        """将对话记忆格式化为文本"""
        if not self.conversation_memory:
            return ""
        
        memory_parts = []
        for idx, mem in enumerate(self.conversation_memory, 1):
            memory_parts.append(f"## 历史对话 {idx}")
            memory_parts.append(f"用户: {mem['question']}")
            memory_parts.append(f"助手: {mem['final_answer']}")
            memory_parts.append("")
        
        return "\n".join(memory_parts)
    
    def _ensure_mode(self, mode: str):
        """确保当前加载的模型与所需模式一致"""
        if self.current_mode == mode:
            return
        
        if self.current_mode:
            print(f"[MultiTurn] Unloading {self.current_mode} models...")
            if self.current_mode == "prefill":
                self.prefill_runner = None
            else:
                self.decode_runner = None
        
        print(f"[MultiTurn] Loading {mode} models...")
        if mode == "prefill":
            paths = self.config.get_prefill_model_paths()
            self.prefill_runner = ONNXModelRunner(paths)
        else:
            paths = self.config.get_decode_model_paths()
            self.decode_runner = ONNXModelRunner(paths)
        
        self.current_mode = mode
        print(f"[MultiTurn] {mode} models loaded")
    
    def _get_runner(self):
        """获取当前模式的模型运行器"""
        if self.current_mode == "prefill":
            return self.prefill_runner
        else:
            return self.decode_runner
    
    def _process_prompt_with_prefill(self, prompt_text: str, reset_kv: bool = True):
        """
        使用 prefill 模型处理 prompt 文本
        
        Args:
            prompt_text: 要处理的文本
            reset_kv: 是否重置 KV cache。False 时在已有 KV 基础上追加
        """
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer required")
        
        prompt_ids = encode_text(self.tokenizer, prompt_text)
        q_len = len(prompt_ids)
        prefill_len = self.config.prefill_len
        
        if q_len > prefill_len:
            print(f"[MultiTurn] Prompt {q_len} tokens > prefill_len {prefill_len}, truncating")
            prompt_ids = prompt_ids[:prefill_len]
            q_len = prefill_len
        
        input_ids = np.array([prompt_ids], dtype=np.int64)
        
        # 加载 prefill 模型
        self._ensure_mode("prefill")
        runner = self._get_runner()
        
        # Embed
        embed_ids = pad_input_ids(input_ids, prefill_len, pad_id=0)
        hidden = runner.run_embed(embed_ids)
        
        # 确定起始位置
        if reset_kv:
            start_pos = 0
        else:
            start_pos = self.past_len
        
        # Attention mask & position ids（支持在已有 KV 后追加）
        attention_mask = build_prefill_with_past_attention_mask(
            q_len, prefill_len, start_pos, self.config.max_cache_len
        )
        position_ids = build_position_ids(start_pos, q_len, prefill_len)
        
        # 运行所有 blocks
        for block_idx in range(4):
            num_layers_per_block = 7
            start_layer = block_idx * num_layers_per_block
            end_layer = start_layer + num_layers_per_block
            
            # Prefill 模型总是需要 past KV
            past_key = self.kv_cache.past_key[start_layer:end_layer]
            past_value = self.kv_cache.past_value[start_layer:end_layer]
            
            hidden, present_key, present_value = runner.run_block(
                block_idx, hidden, attention_mask, position_ids,
                past_key, past_value
            )
            
            if block_idx == 0 and reset_kv:
                self.kv_cache.reset()
            
            # 写入 KV cache（追加到 start_pos 位置）
            pk = present_key[:, :, :, :q_len, :].astype(np.float16)
            pv = present_value[:, :, :, :q_len, :].astype(np.float16)
            end_pos = start_pos + q_len
            self.kv_cache.past_key[start_layer:end_layer, :, :, start_pos:end_pos, :] = pk
            self.kv_cache.past_value[start_layer:end_layer, :, :, start_pos:end_pos, :] = pv
        
        self.kv_cache.current_len = start_pos + q_len
        self.past_len = start_pos + q_len
    
    def process_forward(self, input_ids: np.ndarray, q_len: int, mode: str = "decode"):
        """处理一次前向传播"""
        self._ensure_mode(mode)
        runner = self._get_runner()
        
        if mode == "prefill":
            cur_input_len = self.config.prefill_len
        else:
            cur_input_len = 1
        
        # Embed
        embed_ids = pad_input_ids(input_ids, cur_input_len, pad_id=0)
        hidden = runner.run_embed(embed_ids)
        
        # Attention mask & position ids
        if mode == "prefill":
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
            
            past_key = self.kv_cache.past_key[start_layer:end_layer]
            past_value = self.kv_cache.past_value[start_layer:end_layer]
            
            hidden, present_key, present_value = runner.run_block(
                block_idx, hidden, attention_mask, position_ids,
                past_key, past_value
            )
            
            # 追加到 past_len 位置
            pk = present_key[:, :, :, :q_len, :].astype(np.float16)
            pv = present_value[:, :, :, :q_len, :].astype(np.float16)
            s = self.past_len
            e = s + q_len
            self.kv_cache.past_key[start_layer:end_layer, :, :, s:e, :] = pk
            self.kv_cache.past_value[start_layer:end_layer, :, :, s:e, :] = pv
        
        self.kv_cache.current_len = self.past_len + q_len
        
        # LM head
        logits = runner.run_lm_head(hidden)
        
        return logits
    
    def generate_single_round(
        self,
        prompt_ids: np.ndarray,
        max_new_tokens: int = 100
    ) -> Tuple[List[int], List[Dict], List[Dict]]:
        """
        单轮生成（第一轮：用户问题 → 工具调用）
        
        Returns:
            (generated_ids, tool_calls, tool_results)
        """
        parser = StreamingToolCallParser()
        decoded_text = ""
        generated_ids = []
        tool_calls_list = []
        
        current_ids = prompt_ids
        
        for _ in range(max_new_tokens + 1):
            q_len = current_ids.shape[1]
            
            if self.past_len + q_len > self.config.max_cache_len:
                print("[MultiTurn] KV cache overflow, stopping generation")
                break
            
            # 确定模式
            mode = "prefill" if self.past_len > 0 and q_len > 1 else "decode"
            
            # 前向传播
            logits = self.process_forward(current_ids, q_len, mode)
            
            # 采样
            next_token = sample_token(
                logits[0, q_len - 1, :],
                temperature=self.config.temperature,
                top_k=self.config.top_k,
                top_p=self.config.top_p,
                greedy=self.config.greedy
            )
            
            if next_token == self.config.eos_token_id:
                print("[MultiTurn] EOS token generated")
                break
            
            generated_ids.append(next_token)
            
            if self.tokenizer is not None:
                delta_text, decoded_text = decode_incremental_text(
                    self.tokenizer,
                    generated_ids,
                    previous_text=decoded_text,
                    skip_special_tokens=True
                )
                if delta_text:
                    print(f"[MultiTurn] Step {self.step}: token {next_token}, text {delta_text!r}")
                    
                    # 流式解析工具调用
                    completed_calls = parser.feed(delta_text)
                    for tool_call in completed_calls:
                        print(f"[MultiTurn] Detected tool_call: {tool_call['name']} (id={tool_call['id']})")
                        tool_calls_list.append(tool_call)
                else:
                    print(f"[MultiTurn] Step {self.step}: token {next_token}")
            else:
                print(f"[MultiTurn] Step {self.step}: token {next_token}")
            
            self.past_len += q_len
            self.step += 1
            current_ids = np.array([[next_token]], dtype=np.int64)
        
        # 最终检查
        if not tool_calls_list:
            final_calls = parser.get_all_calls()
            if final_calls:
                already_added = set(tc["id"] for tc in tool_calls_list)
                for tool_call in final_calls:
                    if tool_call["id"] not in already_added:
                        print(f"[MultiTurn] Late-detected tool_call: {tool_call['name']}")
                        tool_calls_list.append(tool_call)
        
        # 执行工具
        tool_results = []
        if tool_calls_list:
            print(f"[MultiTurn] Executing {len(tool_calls_list)} tool(s)...")
            tool_results = self.tool_coordinator.execute_tools(tool_calls_list)
            results_text = self.tool_coordinator.format_tool_results(tool_results)
            print(f"[MultiTurn] Tool results:\n{results_text}")
        
        return generated_ids, tool_calls_list, tool_results
    
    def process_question(
        self,
        user_text: str,
        max_new_tokens: int = 100
    ) -> str:
        """
        处理一个完整的问题（两轮推理）
        
        Args:
            user_text: 用户问题
            max_new_tokens: 最大生成 token 数
        
        Returns:
            最终回答文本
        """
        print(f"\n{'='*60}")
        print(f"[MultiTurn] Processing question: {user_text}")
        print(f"{'='*60}")
        
        # 1. 历史记忆已在上一轮结束时增量追加到 base_kv_snapshot，无需重建
        # （首轮无历史，base_kv_snapshot == system_only_kv_snapshot）
        
        # 2. 恢复到基础 KV 状态
        self.reset_to_base()
        
        # 3. 第一轮推理：用户问题 → 工具调用
        print(f"\n[MultiTurn] === Round 1: User Question → Tool Calls ===")
        # 在用户提问后添加输出格式提示和 /no_think
        output_format_hint = (
            '【重要】如需调用工具，必须严格输出JSON格式：{"tool_name":"工具名","arguments":{"参数":"值"}}\n'
            "禁止使用函数调用格式如 tool(arg:value)。不需要工具则直接用自然语言回答。\n"
            "/no_think"
        )
        user_prompt = f"<|im_start|>user\n{user_text}\n\n{output_format_hint}<|im_end|>\n<|im_start|>assistant\n"
        prompt_ids = encode_text(self.tokenizer, user_prompt)
        prompt_ids = np.array([prompt_ids], dtype=np.int64)
        
        round1_ids, tool_calls, tool_results = self.generate_single_round(
            prompt_ids, max_new_tokens
        )
        
        # 4. 保存第一轮推理后的 KV 快照
        self.question_kv_snapshot = self.kv_cache.save_snapshot()
        print(f"[MultiTurn] Saved question KV snapshot. past_len={self.past_len}")
        
        # 5. 如果没有工具调用，直接返回
        if not tool_calls:
            print("[MultiTurn] No tool calls, returning first round result")
            final_text = decode_token_ids(self.tokenizer, round1_ids, skip_special_tokens=True)
            return final_text
        
        # 6. 第二轮推理前：处理工具结果提示
        print(f"\n[MultiTurn] === Round 2 Preparation: Tool Results ===")
        
        has_history = len(self.conversation_memory) > 0
        
        # 6.1 恢复到第二轮基础 KV（round2_system + 所有历史，已增量维护）
        self.kv_cache.restore_snapshot(self.round2_base_kv_snapshot)
        self.past_len = self.round2_base_kv_snapshot['current_len']
        
        # 7. 第二轮推理：工具结果 → 最终回答
        print(f"\n[MultiTurn] === Round 2: Tool Results → Final Answer ===")
        
        # 构建工具结果 prompt
        result_system = build_tool_result_prompt(
            user_message=user_text,
            tool_calls=tool_calls,
            tool_results=tool_results,
            has_history=has_history,
            tools=None,
        )
        # 在工具结果后添加输出格式提示和 /no_think
        output_format_hint = (
            '【重要】如需继续调用工具，必须严格输出JSON格式：{"tool_name":"工具名","arguments":{"参数":"值"}}\n'
            "禁止使用函数调用格式如 tool(arg:value)。否则直接用自然语言回答用户的问题。\n"
            "/no_think"
        )
        # 工具结果作为 user 消息追加（在 round2_base 基础上增量追加）
        tool_prompt = f"<|im_start|>user\n{result_system}\n{user_text}\n\n{output_format_hint}<|im_end|>\n<|im_start|>assistant\n"
        tool_ids = encode_text(self.tokenizer, tool_prompt)
        
        if len(tool_ids) > self.config.prefill_len:
            print(f"[MultiTurn] Tool prompt too long ({len(tool_ids)} tokens), truncating")
            tool_ids = tool_ids[-self.config.prefill_len:]
        
        tool_prompt_ids = np.array([tool_ids], dtype=np.int64)
        
        round2_ids, _, _ = self.generate_single_round(tool_prompt_ids, max_new_tokens)
        
        # 8. 获取最终回答文本
        final_text = decode_token_ids(self.tokenizer, round2_ids, skip_special_tokens=True)
        
        # 9. 存储到对话记忆（包含最终回答）
        self.conversation_memory.append({
            'question': user_text,
            'tool_calls': tool_calls,
            'tool_results': tool_results,
            'final_answer': final_text
        })
        
        # 10. 增量追加最新历史到 base_kv_snapshot
        #     从 system_only_kv_snapshot 恢复，然后追加所有历史（或从上一个 base 追加最新一条）
        self._append_latest_history_to_base()
        
        # 11. 返回最终回答
        return final_text
    
    def reset_to_base(self):
        """重置到基础 KV 状态"""
        self.step = 0
        if self.base_kv_snapshot:
            self.kv_cache.restore_snapshot(self.base_kv_snapshot)
            self.past_len = self.base_kv_snapshot['current_len']
        else:
            self.kv_cache.reset()
            self.past_len = 0
    
    def _append_latest_history_to_base(self):
        """
        增量追加最新一条历史记录到 base_kv_snapshot 和 round2_base_kv_snapshot。
        """
        if not self.conversation_memory:
            return
        
        latest = self.conversation_memory[-1]
        # 格式化为标准 chat template，模型能正确理解这是历史对话
        new_entry = (
            f"<|im_start|>user\n{latest['question']}<|im_end|>\n"
            f"<|im_start|>assistant\n{latest['final_answer']}<|im_end|>\n"
        )
        
        # 更新第一轮 base（从上一个 base 追加）
        self.kv_cache.restore_snapshot(self.base_kv_snapshot)
        self.past_len = self.base_kv_snapshot['current_len']
        self._process_prompt_with_prefill(new_entry, reset_kv=False)
        self.base_kv_snapshot = self.kv_cache.save_snapshot()
        
        # 更新第二轮 base（从上一个 round2_base 追加）
        self.kv_cache.restore_snapshot(self.round2_base_kv_snapshot)
        self.past_len = self.round2_base_kv_snapshot['current_len']
        self._process_prompt_with_prefill(new_entry, reset_kv=False)
        self.round2_base_kv_snapshot = self.kv_cache.save_snapshot()
        
        print(f"[MultiTurn] Base KVs updated with history {len(self.conversation_memory)}. round1_past_len={self.base_kv_snapshot['current_len']}, round2_past_len={self.round2_base_kv_snapshot['current_len']}")

    def reset_to_question_start(self):
        """重置到当前问题开始时的 KV 状态"""
        self.step = 0
        if self.question_kv_snapshot:
            self.kv_cache.restore_snapshot(self.question_kv_snapshot)
            self.past_len = self.question_kv_snapshot['current_len']
        else:
            self.reset_to_base()
    
    def shutdown(self):
        """关闭"""
        if self.tool_agent:
            self.tool_agent.stop()


def main():
    parser = argparse.ArgumentParser(description="Multi-turn ONNX Runner with Tool Support")
    parser.add_argument("--prefill_onnx_dir", type=str, required=True, help="Prefill ONNX model directory")
    parser.add_argument("--decode_onnx_dir", type=str, required=True, help="Decode ONNX model directory")
    parser.add_argument("--tokenizer_dir", type=str, required=True, help="Tokenizer directory")
    parser.add_argument("--max_new_tokens", type=int, default=100)
    parser.add_argument("--max_cache_len", type=int, default=1024)
    parser.add_argument("--prefill_len", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--greedy", action="store_true", default=True)
    parser.add_argument("--interactive", action="store_true", help="Interactive mode")
    parser.add_argument("--questions", nargs="+", help="List of questions to ask")
    
    args = parser.parse_args()
    
    config = LocalConfig(
        system_onnx_dir="",  # 不使用 system 模型
        prefill_onnx_dir=args.prefill_onnx_dir,
        decode_onnx_dir=args.decode_onnx_dir,
        tokenizer_dir=args.tokenizer_dir,
        system_kv_dir="",
        system_len=0,  # 不使用 system_len
        prefill_len=args.prefill_len,
        max_cache_len=args.max_cache_len,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        greedy=args.greedy,
    )
    
    runner = MultiTurnONNXRunner(config)
    runner.init()
    
    try:
        if args.interactive:
            # 交互模式
            print("\n[MultiTurn] Entering interactive mode. Type 'quit' to exit.")
            while True:
                user_input = input("\nYou: ").strip()
                if user_input.lower() in ['quit', 'exit', 'q']:
                    break
                
                if not user_input:
                    continue
                
                start_time = time.time()
                answer = runner.process_question(user_input, args.max_new_tokens)
                elapsed = time.time() - start_time
                
                print(f"\nAssistant: {answer}")
                print(f"(Time: {elapsed:.2f}s)")
        
        elif args.questions:
            # 批量问题模式
            for idx, question in enumerate(args.questions, 1):
                print(f"\n[Question {idx}/{len(args.questions)}]")
                start_time = time.time()
                answer = runner.process_question(question, args.max_new_tokens)
                elapsed = time.time() - start_time
                
                print(f"\nQ: {question}")
                print(f"A: {answer}")
                print(f"(Time: {elapsed:.2f}s)")
        
        else:
            print("Please specify --interactive or --questions")
    
    finally:
        runner.shutdown()


if __name__ == "__main__":
    main()

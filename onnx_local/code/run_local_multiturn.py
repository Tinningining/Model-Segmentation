"""
单机 ONNX 模型执行 - 支持多轮对话和工具调用
每次提问都会调用 system 模型处理历史记忆
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
    build_system_attention_mask,
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
        self.base_kv_snapshot = None  # 当前对话状态的基础 KV
        self.question_kv_snapshot = None  # 当前问题第一轮推理后的 KV
        
        # 对话记忆
        self.conversation_memory: List[Dict[str, Any]] = []
        
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
        
        # 生成初始 system KV cache（工具描述）
        self._init_base_system_kv()
        
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
    
    def _init_base_system_kv(self):
        """生成初始 system KV cache（仅包含工具描述）"""
        print("[MultiTurn] Generating base system KV cache...")
        
        tool_configs = {
            'get_weather': weather_tool.TOOL_CONFIG,
            'calculator': calculator_tool.TOOL_CONFIG,
            'get_time': time_tool.TOOL_CONFIG,
            'unit_convert': unit_converter_tool.TOOL_CONFIG,
            'translate': translate_tool.TOOL_CONFIG,
        }
        openai_tools = build_tools_openai_schema(tool_configs)
        # 初始化时没有历史记忆
        system_prompt = build_tool_system_prompt(openai_tools, has_history=False)
        
        self._run_system_stage(system_prompt, has_past_kv=False)
        self.base_kv_snapshot = self.kv_cache.save_snapshot()
        
        print(f"[MultiTurn] Base system KV generated. past_len={self.past_len}")
    
    def _format_memory_text(self) -> str:
        """将对话记忆格式化为文本（用户问题 + 模型最终回答的JSON格式）"""
        if not self.conversation_memory:
            return ""
        
        memory_parts = []
        for idx, mem in enumerate(self.conversation_memory, 1):
            memory_parts.append(f"## 历史对话 {idx}")
            memory_parts.append(f"用户: {mem['question']}")
            memory_parts.append(f"助手: {mem['final_answer']}")
            memory_parts.append("")
        
        return "\n".join(memory_parts)
    
    def _update_system_kv_with_memory(self, memory_text: str):
        """使用记忆文本更新 system KV cache"""
        print(f"[MultiTurn] Updating system KV with memory ({len(memory_text)} chars)...")
        
        # 构建包含记忆的 system prompt
        full_memory_prompt = f"<|im_start|>system\n{memory_text}<|im_end|>\n"
        
        self._run_system_stage(full_memory_prompt, has_past_kv=True)
        self.base_kv_snapshot = self.kv_cache.save_snapshot()
        
        print(f"[MultiTurn] System KV updated with memory. past_len={self.past_len}")
    
    def _ensure_mode(self, mode: str):
        """确保当前加载的模型与所需模式一致"""
        if self.current_mode == mode:
            return
        
        if self.current_mode:
            print(f"[MultiTurn] Unloading {self.current_mode} models...")
            if self.current_mode == "system":
                self.system_runner = None
            elif self.current_mode == "prefill":
                self.prefill_runner = None
            else:
                self.decode_runner = None
        
        print(f"[MultiTurn] Loading {mode} models...")
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
        print(f"[MultiTurn] {mode} models loaded")
    
    def _get_runner(self):
        """获取当前模式的模型运行器"""
        if self.current_mode == "system":
            return self.system_runner
        elif self.current_mode == "prefill":
            return self.prefill_runner
        else:
            return self.decode_runner
    
    def _run_system_stage(self, system_prompt_text: str, has_past_kv: bool = False):
        """
        运行 system 阶段生成/更新 KV cache
        
        Args:
            system_prompt_text: system prompt 文本
            has_past_kv: 是否有历史 KV cache（True 表示追加模式）
        """
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer required for system stage")
        
        system_ids = encode_text(self.tokenizer, system_prompt_text)
        q_len = len(system_ids)
        system_len = self.config.system_len
        
        if q_len > system_len:
            print(f"[MultiTurn] System prompt {q_len} tokens > system_len {system_len}, truncating")
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
        if has_past_kv:
            # 追加模式：使用 prefill 风格的 attention mask
            attention_mask = build_prefill_with_past_attention_mask(
                q_len, system_len, self.past_len, self.config.max_cache_len
            )
        else:
            # 初始模式：使用 system attention mask
            attention_mask = build_system_attention_mask(q_len, system_len)
        
        position_ids = build_position_ids(self.past_len, q_len, system_len)
        
        # 运行所有 blocks
        for block_idx in range(4):
            num_layers_per_block = 7
            start_layer = block_idx * num_layers_per_block
            end_layer = start_layer + num_layers_per_block
            
            if has_past_kv:
                # 追加模式：需要 past KV
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
            else:
                # 初始模式：无 past KV
                hidden, present_key, present_value = runner.run_block(
                    block_idx, hidden, attention_mask, position_ids
                )
                
                if block_idx == 0:
                    self.kv_cache.reset()
                
                # 直接写入前 q_len 个位置
                pk = present_key[:, :, :, :q_len, :].astype(np.float16)
                pv = present_value[:, :, :, :q_len, :].astype(np.float16)
                self.kv_cache.past_key[start_layer:end_layer, :, :, :q_len, :] = pk
                self.kv_cache.past_value[start_layer:end_layer, :, :, :q_len, :] = pv
        
        if has_past_kv:
            self.kv_cache.current_len = self.past_len + q_len
            self.past_len = self.past_len + q_len
        else:
            self.kv_cache.current_len = q_len
            self.past_len = q_len
    
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
        
        # 1. 如果有历史记忆，先通过 system 模型处理
        if self.conversation_memory:
            memory_text = self._format_memory_text()
            self._update_system_kv_with_memory(memory_text)
        
        # 2. 恢复到基础 KV 状态
        self.reset_to_base()
        
        # 3. 第一轮推理：用户问题 → 工具调用
        print(f"\n[MultiTurn] === Round 1: User Question → Tool Calls ===")
        user_prompt = f"<|im_start|>user\n{user_text}<|im_end|>\n<|im_start|>assistant\n"
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
        
        # 6. 第二轮推理前：通过 system 模型处理历史记忆
        print(f"\n[MultiTurn] === Round 2 Preparation: Adding History to System ===")
        
        # 6.1 生成第二轮的初始 system KV（包含工具结果提示）
        self.kv_cache.reset()
        self.past_len = 0
        
        has_history = len(self.conversation_memory) > 0
        round2_system_prompt = build_round2_system_prompt(has_history=has_history)
        self._run_system_stage(round2_system_prompt, has_past_kv=False)
        round2_base_snapshot = self.kv_cache.save_snapshot()
        
        # 6.2 如果有历史记忆，追加到 system KV
        if self.conversation_memory:
            memory_text = self._format_memory_text()
            full_memory = f"<|im_start|>system\n{memory_text}<|im_end|>\n"
            self._run_system_stage(full_memory, has_past_kv=True)
        
        # 6.3 保存第二轮的 base KV
        round2_base_snapshot = self.kv_cache.save_snapshot()
        
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
        tool_prompt = build_chat_prompt(system=result_system, user=user_text)
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
        
        # 10. 恢复到问题开始状态，作为下一个问题的基础
        self.reset_to_question_start()
        self.base_kv_snapshot = self.question_kv_snapshot
        
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
    parser.add_argument("--system_onnx_dir", type=str, required=True, help="System ONNX model directory")
    parser.add_argument("--prefill_onnx_dir", type=str, required=True, help="Prefill ONNX model directory")
    parser.add_argument("--decode_onnx_dir", type=str, required=True, help="Decode ONNX model directory")
    parser.add_argument("--tokenizer_dir", type=str, required=True, help="Tokenizer directory")
    parser.add_argument("--max_new_tokens", type=int, default=100)
    parser.add_argument("--max_cache_len", type=int, default=1024)
    parser.add_argument("--system_len", type=int, default=256)
    parser.add_argument("--prefill_len", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--greedy", action="store_true", default=True)
    parser.add_argument("--interactive", action="store_true", help="Interactive mode")
    parser.add_argument("--questions", nargs="+", help="List of questions to ask")
    
    args = parser.parse_args()
    
    config = LocalConfig(
        system_onnx_dir=args.system_onnx_dir,
        prefill_onnx_dir=args.prefill_onnx_dir,
        decode_onnx_dir=args.decode_onnx_dir,
        tokenizer_dir=args.tokenizer_dir,
        system_kv_dir="",  # 不使用持久化的 system KV
        system_len=args.system_len,
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

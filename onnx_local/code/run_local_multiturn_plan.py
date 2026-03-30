"""
单机 ONNX 模型执行 - 支持多轮对话和并行/串行工具调用
基于 run_local_multiturn.py，增加执行计划模式支持
"""
import argparse
import os
from run_local_multiturn import MultiTurnONNXRunner
from config import LocalConfig
from kvcache import KVCache
from utils import (
    build_tool_system_prompt,
    build_tools_openai_schema,
    encode_text,
    decode_token_ids,
)
from tools import ExecutionPlanParser, PlanExecutor
from tools.builtin_tools import (
    weather_tool,
    calculator_tool,
    time_tool,
    unit_converter_tool,
    translate_tool,
)
import numpy as np
import time


class PlanModeRunner(MultiTurnONNXRunner):
    """支持执行计划模式的运行器"""
    
    def __init__(self, config: LocalConfig, enable_plan_mode: bool = True):
        super().__init__(config, enable_plan_mode)
        self.system_runner = None  # System 模型运行器
        
    def _init_tools(self):
        """初始化工具系统（覆盖父类方法以添加执行计划支持）"""
        super()._init_tools()
        
        if self.enable_plan_mode:
            # 初始化执行计划解析器
            self.plan_parser = ExecutionPlanParser()
            tool_names = self.tool_manager.list_tools()
            self.plan_parser.set_available_tools(tool_names)
            
            # 初始化执行计划调度器
            self.plan_executor = PlanExecutor(self.tool_coordinator, max_workers=4)
            
            print("[PlanMode] Execution plan system initialized")
    
    def _init_base_kv(self):
        """生成初始 KV cache（使用 system 模型处理固定 prompt，支持文件缓存）"""
        # 检查是否有缓存文件
        cache_dir = self.config.system_kv_dir
        if cache_dir:
            # Plan mode 使用不同的缓存文件名
            cache_suffix = "_plan" if self.enable_plan_mode else ""
            system_cache_path = os.path.join(cache_dir, f"system_only_kv{cache_suffix}")
            round2_cache_path = os.path.join(cache_dir, f"round2_system_only_kv{cache_suffix}")
            
            # 尝试从缓存加载
            if KVCache.snapshot_exists(system_cache_path) and KVCache.snapshot_exists(round2_cache_path):
                print("[PlanMode] Loading base KV cache from files...")
                
                # 加载第一轮 system KV
                if self.kv_cache.load_snapshot_from_file(system_cache_path):
                    self.system_only_kv_snapshot = self.kv_cache.save_snapshot()
                    self.base_kv_snapshot = self.kv_cache.save_snapshot()
                    
                    # 加载第二轮 system KV
                    if self.kv_cache.load_snapshot_from_file(round2_cache_path):
                        self.round2_system_only_kv_snapshot = self.kv_cache.save_snapshot()
                        self.round2_base_kv_snapshot = self.kv_cache.save_snapshot()
                        
                        print(f"[PlanMode] Base KV loaded from cache. past_len={self.past_len}")
                        return
                
                print("[PlanMode] Cache loading failed, regenerating...")
        
        # 没有缓存或加载失败，重新生成
        print("[PlanMode] Generating base KV cache with plan mode enabled...")
        
        tool_configs = {
            'get_weather': weather_tool.TOOL_CONFIG,
            'calculator': calculator_tool.TOOL_CONFIG,
            'get_time': time_tool.TOOL_CONFIG,
            'unit_convert': unit_converter_tool.TOOL_CONFIG,
            'translate': translate_tool.TOOL_CONFIG,
        }
        openai_tools = build_tools_openai_schema(tool_configs)
        
        # 使用支持执行计划的 system prompt
        system_prompt = build_tool_system_prompt(
            openai_tools, 
            has_history=True, 
            enable_plan_mode=self.enable_plan_mode
        )
        
        full_prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        
        # 如果有 system 模型，使用它处理固定 prompt
        if self.config.system_onnx_dir and self.system_runner:
            print("[PlanMode] Using system model for fixed prompt...")
            self._process_prompt_with_system(full_prompt)
        else:
            # 否则使用 prefill 模型
            print("[PlanMode] Using prefill model for fixed prompt...")
            self._process_prompt_with_prefill(full_prompt)
        
        self.system_only_kv_snapshot = self.kv_cache.save_snapshot()
        self.base_kv_snapshot = self.kv_cache.save_snapshot()
        
        # 保存到文件
        if cache_dir:
            cache_suffix = "_plan" if self.enable_plan_mode else ""
            self.kv_cache.save_snapshot_to_file(os.path.join(cache_dir, f"system_only_kv{cache_suffix}"))
        
        # 第二轮 system prompt
        from utils import build_round2_system_prompt
        round2_system_prompt = build_round2_system_prompt(has_history=True)
        round2_full_prompt = f"<|im_start|>system\n{round2_system_prompt}<|im_end|>\n"
        
        # 同样使用 system 模型或 prefill 模型
        if self.config.system_onnx_dir and self.system_runner:
            self._process_prompt_with_system(round2_full_prompt)
        else:
            self._process_prompt_with_prefill(round2_full_prompt)
        
        self.round2_system_only_kv_snapshot = self.kv_cache.save_snapshot()
        self.round2_base_kv_snapshot = self.kv_cache.save_snapshot()
        
        # 保存到文件
        if cache_dir:
            cache_suffix = "_plan" if self.enable_plan_mode else ""
            self.kv_cache.save_snapshot_to_file(os.path.join(cache_dir, f"round2_system_only_kv{cache_suffix}"))
        
        print(f"[PlanMode] Base KV generated. past_len={self.past_len}")
    
    def _process_prompt_with_system(self, prompt: str):
        """使用 system 模型处理 prompt"""
        prompt_ids = encode_text(self.tokenizer, prompt)
        
        if len(prompt_ids) > self.config.system_len:
            print(f"[PlanMode] System prompt {len(prompt_ids)} tokens > system_len {self.config.system_len}, truncating")
            prompt_ids = prompt_ids[-self.config.system_len:]
        
        prompt_ids = np.array([prompt_ids], dtype=np.int64)
        q_len = prompt_ids.shape[1]
        
        # Pad input_ids
        from utils import pad_input_ids, build_system_attention_mask, build_position_ids
        padded_ids = pad_input_ids(prompt_ids, self.config.system_len)
        
        # Build attention mask (no past KV for system model)
        attention_mask = build_system_attention_mask(q_len, self.config.system_len)
        
        # Build position ids
        position_ids = build_position_ids(0, q_len, self.config.system_len)
        
        # Run system model
        hidden = self.system_runner.run_embed(padded_ids)
        
        # 收集所有 block 的 KV
        all_present_keys = []
        all_present_values = []
        
        for block_idx in range(4):
            hidden, present_key, present_value = self.system_runner.run_block(
                block_idx, hidden, attention_mask, position_ids
            )
            all_present_keys.append(present_key)
            all_present_values.append(present_value)
        
        # 合并所有 block 的 KV（每个 block 包含 7 层）
        # present_key shape: (7, 1, num_kv_heads, system_len, head_dim)
        # 但我们只需要前 q_len 个位置
        new_key = np.concatenate(all_present_keys, axis=0)  # (28, 1, num_kv_heads, system_len, head_dim)
        new_value = np.concatenate(all_present_values, axis=0)
        
        # 只取实际使用的部分
        new_key = new_key[:, :, :, :q_len, :]  # (28, 1, num_kv_heads, q_len, head_dim)
        new_value = new_value[:, :, :, :q_len, :]
        
        # Update KV cache
        self.kv_cache.update(new_key, new_value, q_len)
        self.past_len += q_len
    
    def process_question(self, user_text: str, max_new_tokens: int = 100) -> str:
        """
        处理问题（支持执行计划模式）
        
        Args:
            user_text: 用户问题
            max_new_tokens: 最大生成 token 数
        
        Returns:
            最终回答文本
        """
        print(f"\n{'='*60}")
        print(f"[PlanMode] Processing question: {user_text}")
        print(f"{'='*60}")
        
        # 恢复到基础 KV 状态
        self.reset_to_base()
        
        # 第一轮推理：用户问题 → 执行计划
        print(f"\n[PlanMode] === Round 1: User Question → Execution Plan ===")
        
        if self.enable_plan_mode:
            output_format_hint = (
                '【重要】如需调用工具，必须输出完整的execution_plan格式JSON。\n'
                "JSON必须完整：所有 { 必须有对应的 }，所有 [ 必须有对应的 ]。\n"
                "确保JSON在结束前包含所有必需的闭合括号。\n"
                "不需要工具则直接用自然语言回答。\n"
                "/no_think"
            )
        else:
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
        
        # 保存第一轮推理后的 KV 快照
        self.question_kv_snapshot = self.kv_cache.save_snapshot()
        print(f"[PlanMode] Saved question KV snapshot. past_len={self.past_len}")
        
        # 获取第一轮输出文本
        round1_text = decode_token_ids(self.tokenizer, round1_ids, skip_special_tokens=True)
        
        # 尝试解析执行计划
        if self.enable_plan_mode and self.plan_parser:
            execution_plan = self.plan_parser.parse(round1_text)
            
            if execution_plan:
                print(f"[PlanMode] Detected execution plan: mode={execution_plan.mode}, steps={len(execution_plan.steps)}")
                
                # 执行计划
                tool_results = self.plan_executor.execute(execution_plan)
                
                # 转换为旧格式的 tool_calls（用于第二轮）
                tool_calls = []
                for step in execution_plan.steps:
                    for call in step.get_all_calls():
                        tool_calls.append({
                            "id": f"call_{len(tool_calls)}",
                            "name": call.tool_name,
                            "arguments": call.arguments
                        })
            elif tool_calls:
                # 回退到传统模式（已在 generate_single_round 中执行）
                print("[PlanMode] No execution plan found, using traditional tool calls")
            else:
                # 无工具调用，直接返回
                print("[PlanMode] No tool calls, returning first round result")
                return round1_text
        elif not tool_calls:
            # 传统模式且无工具调用
            print("[PlanMode] No tool calls, returning first round result")
            return round1_text
        
        # 第二轮推理：工具结果 → 最终回答
        print(f"\n[PlanMode] === Round 2: Tool Results → Final Answer ===")
        
        has_history = len(self.conversation_memory) > 0
        
        # 恢复到第二轮基础 KV
        self.kv_cache.restore_snapshot(self.round2_base_kv_snapshot)
        self.past_len = self.round2_base_kv_snapshot['current_len']
        
        # 构建工具结果 prompt
        from utils import build_tool_result_prompt
        result_system = build_tool_result_prompt(
            user_message=user_text,
            tool_calls=tool_calls,
            tool_results=tool_results,
            has_history=has_history,
            tools=None,
        )
        
        output_format_hint = (
            '【重要】如需继续调用工具，必须严格输出JSON格式：{"tool_name":"工具名","arguments":{"参数":"值"}}\n'
            "禁止使用函数调用格式如 tool(arg:value)。否则直接用自然语言回答用户的问题。\n"
            "/no_think"
        )
        
        tool_prompt = f"<|im_start|>user\n{result_system}\n{user_text}\n\n{output_format_hint}<|im_end|>\n<|im_start|>assistant\n"
        tool_ids = encode_text(self.tokenizer, tool_prompt)
        
        if len(tool_ids) > self.config.prefill_len:
            print(f"[PlanMode] Tool prompt too long ({len(tool_ids)} tokens), truncating")
            tool_ids = tool_ids[-self.config.prefill_len:]
        
        tool_prompt_ids = np.array([tool_ids], dtype=np.int64)
        
        round2_ids, _, _ = self.generate_single_round(tool_prompt_ids, max_new_tokens)
        
        # 获取最终回答文本
        final_text = decode_token_ids(self.tokenizer, round2_ids, skip_special_tokens=True)
        
        # 存储到对话记忆
        self.conversation_memory.append({
            'question': user_text,
            'tool_calls': tool_calls,
            'tool_results': tool_results,
            'final_answer': final_text
        })
        
        # 增量追加最新历史到 base_kv_snapshot
        self._append_latest_history_to_base()
        
        return final_text
    
    def _check_cache_exists(self) -> bool:
        """检查 KV 缓存文件是否存在"""
        cache_dir = self.config.system_kv_dir
        if not cache_dir:
            return False
        
        cache_suffix = "_plan" if self.enable_plan_mode else ""
        system_cache_path = os.path.join(cache_dir, f"system_only_kv{cache_suffix}")
        round2_cache_path = os.path.join(cache_dir, f"round2_system_only_kv{cache_suffix}")
        
        return KVCache.snapshot_exists(system_cache_path) and KVCache.snapshot_exists(round2_cache_path)
    
    def init(self):
        """初始化（覆盖父类以加载 system 模型）"""
        # 先检查缓存是否存在
        cache_exists = self._check_cache_exists()
        
        if cache_exists:
            print("[PlanMode] KV cache found, skipping system model loading")
            self.system_runner = None
        elif self.config.system_onnx_dir:
            # 缓存不存在且配置了 system 模型，才加载它
            print("[PlanMode] No KV cache found, loading system model...")
            from onnx_model import ONNXModelRunner
            system_model_paths = self.config.get_system_model_paths()
            self.system_runner = ONNXModelRunner(system_model_paths)
            print("[PlanMode] System model loaded")
        else:
            self.system_runner = None
        
        # 再调用父类初始化（会调用 _init_base_kv）
        super().init()
    
    def shutdown(self):
        """关闭"""
        super().shutdown()
        if self.plan_executor:
            self.plan_executor.shutdown()


def main():
    parser = argparse.ArgumentParser(description="Multi-turn ONNX Runner with Execution Plan Support")
    parser.add_argument("--system_onnx_dir", type=str, default="", help="System ONNX model directory (optional)")
    parser.add_argument("--prefill_onnx_dir", type=str, required=True, help="Prefill ONNX model directory")
    parser.add_argument("--decode_onnx_dir", type=str, required=True, help="Decode ONNX model directory")
    parser.add_argument("--tokenizer_dir", type=str, required=True, help="Tokenizer directory")
    parser.add_argument("--system_kv_dir", type=str, default="./system_kv_cache", help="Directory to cache system KV snapshots")
    parser.add_argument("--max_new_tokens", type=int, default=200)
    parser.add_argument("--max_cache_len", type=int, default=1024)
    parser.add_argument("--system_len", type=int, default=1024, help="System model max input length")
    parser.add_argument("--prefill_len", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--greedy", action="store_true", default=True)
    parser.add_argument("--enable_plan_mode", action="store_true", default=True, help="Enable execution plan mode")
    parser.add_argument("--interactive", action="store_true", help="Interactive mode")
    parser.add_argument("--questions", nargs="+", help="List of questions to ask")
    
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
    
    runner = PlanModeRunner(config, enable_plan_mode=args.enable_plan_mode)
    runner.init()
    
    try:
        if args.interactive:
            print("\n[PlanMode] Entering interactive mode. Type 'quit' to exit.")
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

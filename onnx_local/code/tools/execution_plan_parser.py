"""
执行计划解析器
解析模型输出的 execution_plan JSON 格式
"""
import json
import re
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field


@dataclass
class ToolCall:
    """工具调用"""
    tool_name: str
    arguments: Dict[str, Any]
    output_ref: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        result = {
            "tool_name": self.tool_name,
            "arguments": self.arguments
        }
        if self.output_ref:
            result["output_ref"] = self.output_ref
        return result


@dataclass
class ExecutionStep:
    """执行步骤"""
    step_id: int
    call: Optional[ToolCall] = None  # 单个调用（串行）
    parallel_calls: Optional[List[ToolCall]] = None  # 并行调用
    depends_on: List[int] = field(default_factory=list)
    
    def is_parallel(self) -> bool:
        """是否为并行步骤"""
        return self.parallel_calls is not None and len(self.parallel_calls) > 0
    
    def get_all_calls(self) -> List[ToolCall]:
        """获取所有工具调用"""
        if self.is_parallel():
            return self.parallel_calls
        elif self.call:
            return [self.call]
        return []


@dataclass
class ExecutionPlan:
    """执行计划"""
    mode: str  # "parallel", "sequential", "mixed"
    steps: List[ExecutionStep]
    
    def is_valid(self) -> bool:
        """验证计划是否有效"""
        if not self.steps:
            return False
        if self.mode not in ["parallel", "sequential", "mixed"]:
            return False
        return True


class ExecutionPlanParser:
    """执行计划解析器"""
    
    def __init__(self):
        self.available_tools = set()
    
    def set_available_tools(self, tool_names: List[str]):
        """设置可用工具列表"""
        self.available_tools = set(tool_names)
    
    def parse(self, model_output: str) -> Optional[ExecutionPlan]:
        """
        从模型输出中提取执行计划
        
        Args:
            model_output: 模型生成的文本
            
        Returns:
            ExecutionPlan 对象，如果解析失败返回 None
        """
        # 尝试提取 JSON
        json_str = self._extract_json(model_output)
        if not json_str:
            return None
        
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            print(f"[ExecutionPlanParser] JSON decode error: {e}")
            print(f"[ExecutionPlanParser] Full JSON length: {len(json_str)}")
            # print(f"[ExecutionPlanParser] JSON around error (chars 270-290): '{json_str[270:290]}'")
            print(f"[ExecutionPlanParser] Full JSON: {json_str}")
            return None
        
        # 新格式：直接使用 data，不再有 execution_plan 外层
        plan_data = data
        
        # 解析 mode
        mode = plan_data.get("mode", "")
        if mode not in ["parallel", "sequential", "mixed"]:
            print(f"[ExecutionPlanParser] Invalid mode: {mode}")
            return None
        
        # 解析 steps
        steps_data = plan_data.get("steps", [])
        if not steps_data:
            print("[ExecutionPlanParser] No steps found")
            return None
        
        steps = []
        for step_data in steps_data:
            step = self._parse_step(step_data)
            if step:
                steps.append(step)
        
        if not steps:
            return None
        
        plan = ExecutionPlan(mode=mode, steps=steps)
        
        # 验证
        if not self.validate(plan):
            return None
        
        return plan
    
    def _extract_json(self, text: str) -> Optional[str]:
        """从文本中提取 JSON 字符串"""
        # 移除所有 think 标签及其内容（包括换行符）
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL | re.IGNORECASE)
        
        # 移除多余的空白字符
        text = text.strip()
        
        # 查找第一个 { 和最后一个 }
        start_idx = text.find('{')
        end_idx = text.rfind('}')
        
        if start_idx == -1 or end_idx == -1 or start_idx >= end_idx:
            return None
        
        json_str = text[start_idx:end_idx + 1]
        
        # 验证是否包含 mode 和 steps（新格式不再有 execution_plan 外层）
        if '"mode"' not in json_str and "'mode'" not in json_str:
            return None
        if '"steps"' not in json_str and "'steps'" not in json_str:
            return None
        
        # 尝试修复不完整的 JSON（如果缺少结尾的 ] 或 }）
        try:
            json.loads(json_str)
        except json.JSONDecodeError as e:
            # 如果是缺少结尾符号，尝试补全
            error_msg = str(e)
            if "Expecting" in error_msg or "Unterminated" in error_msg:
                # 计算需要补全的括号数量
                open_braces = json_str.count('{')
                close_braces = json_str.count('}')
                open_brackets = json_str.count('[')
                close_brackets = json_str.count(']')
                
                missing_brackets = open_brackets - close_brackets
                missing_braces = open_braces - close_braces
                
                # 智能补全：尝试不同的补全顺序
                if missing_brackets > 0 or missing_braces > 0:
                    print(f"[ExecutionPlanParser] Attempting to fix incomplete JSON:")
                    print(f"[ExecutionPlanParser]   Missing {missing_brackets} ']' and {missing_braces} '}}'")
                    
                    # 尝试多种补全顺序
                    attempts = [
                        ']' * missing_brackets + '}' * missing_braces,  # 先 ] 后 }
                        '}' * missing_braces + ']' * missing_brackets,  # 先 } 后 ]
                    ]
                    
                    # 如果只缺一种括号，只尝试一次
                    if missing_brackets == 0:
                        attempts = ['}' * missing_braces]
                    elif missing_braces == 0:
                        attempts = [']' * missing_brackets]
                    
                    for i, suffix in enumerate(attempts):
                        test_json = json_str + suffix
                        try:
                            json.loads(test_json)
                            json_str = test_json
                            print(f"[ExecutionPlanParser] JSON successfully fixed with attempt {i+1}: '{suffix}'")
                            break
                        except json.JSONDecodeError:
                            if i == len(attempts) - 1:
                                print(f"[ExecutionPlanParser] All fix attempts failed")
                                return None
        
        return json_str
    
    def _parse_step(self, step_data: Dict[str, Any]) -> Optional[ExecutionStep]:
        """解析单个步骤"""
        step_id = step_data.get("step_id")
        if step_id is None:
            return None
        
        # 解析 depends_on
        depends_on = step_data.get("depends_on", [])
        if not isinstance(depends_on, list):
            depends_on = []
        
        # 解析 parallel_calls
        if "parallel_calls" in step_data:
            parallel_calls_data = step_data["parallel_calls"]
            if not isinstance(parallel_calls_data, list):
                return None
            
            parallel_calls = []
            for call_data in parallel_calls_data:
                tool_call = self._parse_tool_call(call_data)
                if tool_call:
                    parallel_calls.append(tool_call)
            
            if not parallel_calls:
                return None
            
            return ExecutionStep(
                step_id=step_id,
                parallel_calls=parallel_calls,
                depends_on=depends_on
            )
        
        # 解析 call
        elif "call" in step_data:
            call_data = step_data["call"]
            tool_call = self._parse_tool_call(call_data)
            if not tool_call:
                return None
            
            # 检查 output_ref
            if "output_ref" in step_data:
                tool_call.output_ref = step_data["output_ref"]
            
            return ExecutionStep(
                step_id=step_id,
                call=tool_call,
                depends_on=depends_on
            )
        
        return None
    
    def _parse_tool_call(self, call_data: Dict[str, Any]) -> Optional[ToolCall]:
        """解析工具调用"""
        tool_name = call_data.get("tool_name")
        if not tool_name:
            return None
        
        arguments = call_data.get("arguments", {})
        if not isinstance(arguments, dict):
            arguments = {}
        
        output_ref = call_data.get("output_ref")
        
        return ToolCall(
            tool_name=tool_name,
            arguments=arguments,
            output_ref=output_ref
        )
    
    def validate(self, plan: ExecutionPlan) -> bool:
        """
        验证执行计划的合法性
        
        检查：
        1. 依赖关系是否合法（不能依赖不存在的步骤）
        2. 是否存在循环依赖
        3. 工具名称是否有效
        """
        if not plan.is_valid():
            return False
        
        step_ids = {step.step_id for step in plan.steps}
        
        # 检查依赖关系
        for step in plan.steps:
            for dep_id in step.depends_on:
                if dep_id not in step_ids:
                    print(f"[ExecutionPlanParser] Step {step.step_id} depends on non-existent step {dep_id}")
                    return False
                if dep_id >= step.step_id:
                    print(f"[ExecutionPlanParser] Step {step.step_id} depends on future/self step {dep_id}")
                    return False
        
        # 检查循环依赖（简单检查：依赖的步骤ID必须小于当前步骤ID）
        # 更复杂的循环检测在 DependencyResolver 中进行
        
        # 检查工具名称（如果设置了可用工具列表）
        if self.available_tools:
            for step in plan.steps:
                for tool_call in step.get_all_calls():
                    if tool_call.tool_name not in self.available_tools:
                        print(f"[ExecutionPlanParser] Unknown tool: {tool_call.tool_name}")
                        return False
        
        return True

"""
执行计划调度器
按照依赖关系执行工具调用计划
"""
import concurrent.futures
from typing import List, Dict, Any, Optional
from .execution_plan_parser import ExecutionPlan, ExecutionStep, ToolCall
from .dependency_resolver import DependencyResolver


class PlanExecutor:
    """执行计划调度器"""
    
    def __init__(self, tool_coordinator, max_workers: int = 4):
        """
        Args:
            tool_coordinator: 工具协调器
            max_workers: 最大并行工作线程数
        """
        self.tool_coordinator = tool_coordinator
        self.max_workers = max_workers
        self.dependency_resolver = DependencyResolver()
        self.executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="PlanExecutor"
        )
    
    def execute(self, plan: ExecutionPlan) -> List[Dict[str, Any]]:
        """
        执行完整的工具调用计划
        
        流程：
        1. 构建依赖图
        2. 拓扑排序得到执行顺序
        3. 按组执行（组内并行，组间串行）
        4. 每组执行完后更新结果缓存
        5. 返回所有结果
        
        Args:
            plan: 执行计划
            
        Returns:
            所有工具调用的结果列表
        """
        print(f"[PlanExecutor] Executing plan with mode: {plan.mode}")
        
        # 清空缓存
        self.dependency_resolver.clear_cache()
        
        # 构建依赖图
        graph = self.dependency_resolver.build_dependency_graph(plan.steps)
        print(f"[PlanExecutor] Dependency graph: {graph}")
        
        # 拓扑排序
        execution_groups = self.dependency_resolver.topological_sort(graph)
        if not execution_groups:
            print("[PlanExecutor] Failed to create execution order (cycle detected?)")
            return []
        
        print(f"[PlanExecutor] Execution groups: {execution_groups}")
        
        # 创建 step_id -> step 的映射
        step_map = {step.step_id: step for step in plan.steps}
        
        # 按组执行
        all_results = []
        for group_idx, step_ids in enumerate(execution_groups):
            print(f"\n[PlanExecutor] === Executing Group {group_idx + 1}: steps {step_ids} ===")
            
            group_results = self._execute_step_group(step_ids, step_map)
            all_results.extend(group_results)
            
            # 缓存结果
            for step_id, results in zip(step_ids, group_results):
                step = step_map[step_id]
                
                # 如果是并行调用，缓存每个调用的结果
                if step.is_parallel():
                    for tool_call, result in zip(step.parallel_calls, results):
                        # 提取实际结果值
                        actual_result = self._extract_result_value(result)
                        if tool_call.output_ref:
                            self.dependency_resolver.cache_result(tool_call.output_ref, actual_result)
                            print(f"[PlanExecutor] Cached {tool_call.output_ref} = {actual_result}")
                else:
                    # 单个调用
                    actual_result = self._extract_result_value(results[0])
                    if step.call and step.call.output_ref:
                        self.dependency_resolver.cache_result(step.call.output_ref, actual_result)
                        print(f"[PlanExecutor] Cached {step.call.output_ref} = {actual_result}")
                    
                    # 同时缓存为 stepN_result
                    self.dependency_resolver.cache_step_result(step_id, actual_result)
        
        print(f"\n[PlanExecutor] Execution complete. Total results: {len(all_results)}")
        return all_results
    
    def _execute_step_group(self, step_ids: List[int], step_map: Dict[int, ExecutionStep]) -> List[List[Dict[str, Any]]]:
        """
        执行一组步骤（组内并行）
        
        Args:
            step_ids: 步骤ID列表
            step_map: step_id -> ExecutionStep 映射
            
        Returns:
            每个步骤的结果列表
        """
        if len(step_ids) == 1:
            # 单个步骤，直接执行
            step = step_map[step_ids[0]]
            results = self._execute_single_step(step)
            return [results]
        
        # 多个步骤，并行执行
        futures = []
        for step_id in step_ids:
            step = step_map[step_id]
            future = self.executor.submit(self._execute_single_step, step)
            futures.append(future)
        
        # 等待所有完成
        results = []
        for future in futures:
            try:
                result = future.result(timeout=60)  # 60秒超时
                results.append(result)
            except Exception as e:
                print(f"[PlanExecutor] Step execution failed: {e}")
                results.append([{"success": False, "error": str(e)}])
        
        return results
    
    def _execute_single_step(self, step: ExecutionStep) -> List[Dict[str, Any]]:
        """
        执行单个步骤
        
        Args:
            step: 执行步骤
            
        Returns:
            工具调用结果列表
        """
        print(f"[PlanExecutor] Executing step {step.step_id}")
        
        if step.is_parallel():
            # 并行调用多个工具
            return self._execute_parallel_calls(step.parallel_calls)
        elif step.call:
            # 单个工具调用
            result = self._execute_tool_call(step.call)
            return [result]
        else:
            print(f"[PlanExecutor] Step {step.step_id} has no calls")
            return []
    
    def _execute_parallel_calls(self, tool_calls: List[ToolCall]) -> List[Dict[str, Any]]:
        """
        并行执行多个工具调用
        
        Args:
            tool_calls: 工具调用列表
            
        Returns:
            结果列表
        """
        print(f"[PlanExecutor] Parallel executing {len(tool_calls)} tools")
        
        # 解析参数引用
        resolved_calls = []
        for tool_call in tool_calls:
            resolved_args = self.dependency_resolver.resolve_references(tool_call.arguments)
            resolved_calls.append({
                "name": tool_call.tool_name,
                "arguments": resolved_args
            })
        
        # 转换为旧格式（兼容现有 tool_coordinator）
        legacy_calls = []
        for idx, call in enumerate(resolved_calls):
            legacy_calls.append({
                "id": f"call_{idx}",
                "name": call["name"],
                "arguments": call["arguments"]
            })
        
        # 使用 tool_coordinator 执行
        results = self.tool_coordinator.execute_tools(legacy_calls)
        
        return results
    
    def _execute_tool_call(self, tool_call: ToolCall) -> Dict[str, Any]:
        """
        执行单个工具调用
        
        Args:
            tool_call: 工具调用
            
        Returns:
            执行结果
        """
        print(f"[PlanExecutor] Executing tool: {tool_call.tool_name}")
        
        # 解析参数引用
        resolved_args = self.dependency_resolver.resolve_references(tool_call.arguments)
        print(f"[PlanExecutor] Resolved arguments: {resolved_args}")
        
        # 转换为旧格式
        legacy_call = {
            "id": "call_0",
            "name": tool_call.tool_name,
            "arguments": resolved_args
        }
        
        # 使用 tool_coordinator 执行
        results = self.tool_coordinator.execute_tools([legacy_call])
        
        if results:
            return results[0]
        else:
            return {"success": False, "error": "No result returned"}
    
    def _extract_result_value(self, result: Dict[str, Any]) -> Any:
        """
        从工具执行结果中提取实际值
        
        工具执行返回格式：
        {
            "success": True,
            "result": {...},  # 实际结果
            "tool_name": "calculator",
            "device_id": 0
        }
        
        Args:
            result: 工具执行结果
            
        Returns:
            实际的结果值（result 字段的内容）
        """
        if isinstance(result, dict):
            # 如果有 result 字段，提取它
            if "result" in result:
                inner_result = result["result"]
                # 如果 inner_result 也是字典且有 result 字段，继续提取
                if isinstance(inner_result, dict) and "result" in inner_result:
                    return inner_result["result"]
                return inner_result
            # 如果没有 result 字段但有 success 字段，可能是错误
            elif "success" in result and not result["success"]:
                return result.get("error", "Unknown error")
        
        # 否则返回原值
        return result
    
    def shutdown(self):
        """关闭执行器"""
        self.executor.shutdown(wait=True)

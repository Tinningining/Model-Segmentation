"""
依赖解析器
处理工具调用间的依赖关系、拓扑排序和参数引用替换
"""
import re
from typing import Dict, Any, List, Optional
from collections import defaultdict, deque


class DependencyResolver:
    """依赖解析器"""
    
    def __init__(self):
        self.results_cache: Dict[str, Any] = {}  # output_ref -> result
        self.step_results: Dict[int, Any] = {}   # step_id -> result
    
    def build_dependency_graph(self, steps: List[Any]) -> Dict[int, List[int]]:
        """
        构建依赖图
        
        Args:
            steps: ExecutionStep 列表
            
        Returns:
            依赖图：step_id -> [依赖的 step_ids]
        """
        graph = {}
        for step in steps:
            graph[step.step_id] = step.depends_on.copy() if step.depends_on else []
        return graph
    
    def topological_sort(self, graph: Dict[int, List[int]]) -> List[List[int]]:
        """
        拓扑排序，返回可并行执行的步骤组
        
        Args:
            graph: 依赖图 step_id -> [依赖的 step_ids]
            
        Returns:
            [[step1, step2], [step3], [step4, step5]]
            同一组内的步骤可并行执行，组间按顺序执行
        """
        # 计算入度
        in_degree = {node: 0 for node in graph}
        for node in graph:
            for dep in graph[node]:
                if dep in in_degree:
                    in_degree[node] += 1
        
        # 检测循环依赖
        if self._has_cycle(graph):
            print("[DependencyResolver] Cycle detected in dependency graph!")
            return []
        
        result = []
        remaining = set(graph.keys())
        
        while remaining:
            # 找出当前可执行的节点（入度为0且所有依赖已完成）
            ready = []
            for node in remaining:
                deps = graph[node]
                if all(dep not in remaining for dep in deps):
                    ready.append(node)
            
            if not ready:
                # 不应该发生（已检测循环）
                print("[DependencyResolver] No ready nodes but remaining nodes exist!")
                break
            
            result.append(sorted(ready))  # 排序保证确定性
            remaining -= set(ready)
        
        return result
    
    def _has_cycle(self, graph: Dict[int, List[int]]) -> bool:
        """检测图中是否存在环"""
        WHITE, GRAY, BLACK = 0, 1, 2
        color = {node: WHITE for node in graph}
        
        def dfs(node):
            if color[node] == GRAY:
                return True  # 发现环
            if color[node] == BLACK:
                return False  # 已访问
            
            color[node] = GRAY
            for neighbor in graph.get(node, []):
                if neighbor in color and dfs(neighbor):
                    return True
            color[node] = BLACK
            return False
        
        for node in graph:
            if color[node] == WHITE:
                if dfs(node):
                    return True
        return False
    
    def resolve_references(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        替换参数中的引用
        
        支持的引用格式：
        - $stepN_result: 第 N 步的完整结果
        - $stepN_result.field: 第 N 步结果的特定字段
        - $ref_name: 自定义引用名
        - $ref_name.field: 自定义引用的特定字段
        
        Args:
            arguments: 原始参数字典
            
        Returns:
            替换后的参数字典
        """
        resolved = {}
        
        for key, value in arguments.items():
            if isinstance(value, str):
                resolved[key] = self._resolve_string_value(value)
            elif isinstance(value, dict):
                resolved[key] = self.resolve_references(value)
            elif isinstance(value, list):
                resolved[key] = [
                    self._resolve_string_value(item) if isinstance(item, str) else item
                    for item in value
                ]
            else:
                resolved[key] = value
        
        return resolved
    
    def _resolve_string_value(self, value: str) -> Any:
        """解析字符串值中的引用"""
        # 匹配 $xxx 或 $xxx.yyy 格式
        pattern = r'\$([a-zA-Z0-9_]+)(?:\.([a-zA-Z0-9_]+))?'
        
        # 如果整个字符串就是一个引用，直接返回引用的值
        match = re.fullmatch(pattern, value)
        if match:
            ref_name = match.group(1)
            field_name = match.group(2)
            return self._get_reference_value(ref_name, field_name)
        
        # 如果字符串包含多个引用，进行替换
        def replace_ref(match):
            ref_name = match.group(1)
            field_name = match.group(2)
            val = self._get_reference_value(ref_name, field_name)
            return str(val) if val is not None else match.group(0)
        
        return re.sub(pattern, replace_ref, value)
    
    def _get_reference_value(self, ref_name: str, field_name: Optional[str] = None) -> Any:
        """
        获取引用的值
        
        Args:
            ref_name: 引用名（如 "step1_result" 或 "beijing_weather"）
            field_name: 字段名（可选）
            
        Returns:
            引用的值
        """
        # 尝试从 results_cache 获取（自定义引用名）
        if ref_name in self.results_cache:
            result = self.results_cache[ref_name]
            if field_name:
                return self._get_field_value(result, field_name)
            return result
        
        # 尝试解析 stepN_result 格式
        step_match = re.match(r'step(\d+)_result', ref_name)
        if step_match:
            step_id = int(step_match.group(1))
            if step_id in self.step_results:
                result = self.step_results[step_id]
                if field_name:
                    return self._get_field_value(result, field_name)
                return result
        
        print(f"[DependencyResolver] Reference not found: ${ref_name}")
        return None
    
    def _get_field_value(self, obj: Any, field_name: str) -> Any:
        """从对象中获取字段值"""
        if isinstance(obj, dict):
            return obj.get(field_name)
        elif hasattr(obj, field_name):
            return getattr(obj, field_name)
        else:
            print(f"[DependencyResolver] Field '{field_name}' not found in result")
            return None
    
    def cache_result(self, output_ref: str, result: Any):
        """
        缓存步骤结果
        
        Args:
            output_ref: 引用名（如 "$step1_result" 或 "$beijing_weather"）
            result: 结果值
        """
        # 去掉 $ 前缀
        if output_ref.startswith('$'):
            output_ref = output_ref[1:]
        
        self.results_cache[output_ref] = result
        
        # 如果是 stepN_result 格式，也存入 step_results
        step_match = re.match(r'step(\d+)_result', output_ref)
        if step_match:
            step_id = int(step_match.group(1))
            self.step_results[step_id] = result
    
    def cache_step_result(self, step_id: int, result: Any):
        """
        缓存步骤结果（使用步骤ID）
        
        Args:
            step_id: 步骤ID
            result: 结果值
        """
        self.step_results[step_id] = result
        self.results_cache[f'step{step_id}_result'] = result
    
    def clear_cache(self):
        """清空缓存"""
        self.results_cache.clear()
        self.step_results.clear()

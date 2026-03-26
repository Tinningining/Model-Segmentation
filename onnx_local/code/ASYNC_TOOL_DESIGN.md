# 异步/同步工具调用设计文档

## 1. 概述

本文档描述如何在现有 `onnx_local/code` 代码基础上，实现支持串行（同步）和并行（异步）工具调用的功能。

### 1.1 核心目标

- **第一轮推理**：模型分析用户问题，拆解任务，判断工具间的依赖关系，输出结构化的工具调用计划
- **工具执行**：根据依赖关系，串行执行有依赖的工具，并行执行无依赖的工具
- **第二轮推理**：将所有工具结果注入，生成最终回答

### 1.2 现有代码分析

| 文件 | 功能 | 复用情况 |
|------|------|----------|
| `run_local_multiturn.py` | 两轮推理主流程 | 需要修改 `generate_single_round` 和 `process_question` |
| `tools/async_executor.py` | 异步执行器 | 可复用，需扩展支持依赖调度 |
| `tools/streaming_parser.py` | 流式解析工具调用 | 需要修改以支持新格式 |
| `tools/tool_coordinator.py` | 工具协调器 | 可复用 |

---

## 2. 工具调用输出格式设计

### 2.1 格式选择

模型输出 JSON 格式的工具调用计划，包含两种执行模式：

```json
{
  "execution_plan": {
    "mode": "parallel" | "sequential" | "mixed",
    "steps": [...]
  }
}
```

### 2.2 并行模式（Parallel）

所有工具无依赖，可同时执行：

```json
{
  "execution_plan": {
    "mode": "parallel",
    "steps": [
      {
        "step_id": 1,
        "parallel_calls": [
          {"tool_name": "get_weather", "arguments": {"city": "北京"}},
          {"tool_name": "get_weather", "arguments": {"city": "上海"}}
        ]
      }
    ]
  }
}
```

**示例场景**：用户问"北京和上海的天气如何？"

### 2.3 串行模式（Sequential）

工具间有依赖关系，必须按顺序执行：

```json
{
  "execution_plan": {
    "mode": "sequential",
    "steps": [
      {
        "step_id": 1,
        "call": {"tool_name": "get_weather", "arguments": {"city": "北京"}},
        "output_ref": "$step1_result"
      },
      {
        "step_id": 2,
        "call": {"tool_name": "unit_convert", "arguments": {
          "value": "$step1_result.temperature",
          "from_unit": "celsius",
          "to_unit": "fahrenheit"
        }},
        "depends_on": [1],
        "output_ref": "$step2_result"
      }
    ]
  }
}
```

**关键设计**：
- `depends_on`: 声明依赖的步骤 ID 列表
- `output_ref`: 当前步骤结果的引用名
- `$stepN_result.field`: 引用第 N 步结果的特定字段

**示例场景**：用户问"北京现在多少华氏度？"
1. 先调用 `get_weather` 获取摄氏温度
2. 再调用 `unit_convert` 将摄氏转华氏（依赖步骤1的结果）

### 2.4 混合模式（Mixed）

部分并行，部分串行：

```json
{
  "execution_plan": {
    "mode": "mixed",
    "steps": [
      {
        "step_id": 1,
        "parallel_calls": [
          {"tool_name": "get_weather", "arguments": {"city": "北京"}, "output_ref": "$beijing_weather"},
          {"tool_name": "get_weather", "arguments": {"city": "上海"}, "output_ref": "$shanghai_weather"}
        ]
      },
      {
        "step_id": 2,
        "call": {"tool_name": "calculator", "arguments": {
          "expression": "$beijing_weather.temperature - $shanghai_weather.temperature"
        }},
        "depends_on": [1]
      }
    ]
  }
}
```

**示例场景**：用户问"北京和上海的温差是多少？"
1. 并行获取两地天气
2. 串行计算温差（依赖步骤1的两个结果）

---

## 3. 系统架构

```
┌─────────────────────────────────────────────────────────────────┐
│                    MultiTurnONNXRunner                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────┐    ┌──────────────────┐    ┌───────────────┐  │
│  │ Round 1     │───▶│ ExecutionPlan    │───▶│ PlanExecutor  │  │
│  │ 推理        │    │ Parser           │    │ 执行调度器     │  │
│  └─────────────┘    └──────────────────┘    └───────┬───────┘  │
│                                                      │          │
│                     ┌────────────────────────────────┘          │
│                     ▼                                           │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                  DependencyResolver                       │  │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐       │  │
│  │  │ 依赖图构建   │  │ 拓扑排序    │  │ 参数替换    │       │  │
│  │  └─────────────┘  └─────────────┘  └─────────────┘       │  │
│  └──────────────────────────────────────────────────────────┘  │
│                     │                                           │
│                     ▼                                           │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                  AsyncToolExecutor (扩展)                 │  │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐       │  │
│  │  │ 并行执行    │  │ 串行执行    │  │ 结果缓存    │       │  │
│  │  └─────────────┘  └─────────────┘  └─────────────┘       │  │
│  └──────────────────────────────────────────────────────────┘  │
│                     │                                           │
│                     ▼                                           │
│  ┌─────────────┐    ┌──────────────────┐                       │
│  │ Round 2     │◀───│ 工具结果汇总     │                       │
│  │ 推理        │    │                  │                       │
│  └─────────────┘    └──────────────────┘                       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 4. 核心模块设计

### 4.1 ExecutionPlanParser（执行计划解析器）

**文件**：`tools/execution_plan_parser.py`

```python
class ExecutionPlanParser:
    """解析模型输出的执行计划"""
    
    def parse(self, model_output: str) -> ExecutionPlan:
        """
        从模型输出中提取执行计划
        
        Returns:
            ExecutionPlan 对象，包含 mode 和 steps
        """
        pass
    
    def validate(self, plan: ExecutionPlan) -> bool:
        """验证执行计划的合法性（依赖关系、工具名称等）"""
        pass
```

**ExecutionPlan 数据结构**：

```python
@dataclass
class ToolCall:
    tool_name: str
    arguments: Dict[str, Any]
    output_ref: Optional[str] = None

@dataclass
class ExecutionStep:
    step_id: int
    call: Optional[ToolCall] = None           # 单个调用（串行）
    parallel_calls: Optional[List[ToolCall]] = None  # 并行调用
    depends_on: List[int] = field(default_factory=list)

@dataclass
class ExecutionPlan:
    mode: str  # "parallel", "sequential", "mixed"
    steps: List[ExecutionStep]
```

### 4.2 DependencyResolver（依赖解析器）

**文件**：`tools/dependency_resolver.py`

```python
class DependencyResolver:
    """解析和处理工具调用间的依赖关系"""
    
    def __init__(self):
        self.results_cache: Dict[str, Any] = {}  # output_ref -> result
    
    def build_dependency_graph(self, plan: ExecutionPlan) -> Dict[int, List[int]]:
        """构建依赖图：step_id -> [依赖的 step_ids]"""
        pass
    
    def topological_sort(self, graph: Dict[int, List[int]]) -> List[List[int]]:
        """
        拓扑排序，返回可并行执行的步骤组
        
        Returns:
            [[step1, step2], [step3], [step4, step5]]
            同一组内的步骤可并行执行
        """
        pass
    
    def resolve_references(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        替换参数中的引用（如 $step1_result.temperature）
        
        Example:
            {"value": "$step1_result.temperature"} 
            -> {"value": 25}
        """
        pass
    
    def cache_result(self, output_ref: str, result: Any):
        """缓存步骤结果供后续引用"""
        pass
```

**引用解析规则**：

| 引用格式 | 含义 | 示例 |
|----------|------|------|
| `$stepN_result` | 第 N 步的完整结果 | `$step1_result` |
| `$stepN_result.field` | 第 N 步结果的特定字段 | `$step1_result.temperature` |
| `$ref_name` | 自定义引用名 | `$beijing_weather.temperature` |

### 4.3 PlanExecutor（计划执行器）

**文件**：`tools/plan_executor.py`

```python
class PlanExecutor:
    """执行工具调用计划"""
    
    def __init__(self, tool_coordinator, async_executor):
        self.tool_coordinator = tool_coordinator
        self.async_executor = async_executor
        self.dependency_resolver = DependencyResolver()
    
    def execute(self, plan: ExecutionPlan) -> List[Dict[str, Any]]:
        """
        执行完整的工具调用计划
        
        流程：
        1. 构建依赖图
        2. 拓扑排序得到执行顺序
        3. 按组执行（组内并行，组间串行）
        4. 每组执行完后更新结果缓存
        5. 返回所有结果
        """
        pass
    
    def _execute_step_group(self, step_ids: List[int], plan: ExecutionPlan) -> List[Dict]:
        """执行一组步骤（组内并行）"""
        pass
    
    def _execute_single_step(self, step: ExecutionStep) -> Dict[str, Any]:
        """执行单个步骤"""
        pass
```

---

## 5. System Prompt 设计

### 5.1 工具调用格式说明

```
你是AI助手，可用工具：
- get_weather(city*:string, unit:string="celsius") — 获取城市天气
- calculator(expression*:string) — 计算数学表达式
- unit_convert(value*:number, from_unit*:string, to_unit*:string) — 单位转换
- get_time(timezone:string="UTC") — 获取当前时间
- translate(text*:string, target_lang*:string) — 翻译文本

【工具调用规则】
1. 分析用户问题，判断需要哪些工具
2. 判断工具间是否有依赖关系：
   - 无依赖：可并行执行
   - 有依赖：必须串行执行，后续工具的参数引用前序工具的结果
3. 输出执行计划（JSON格式）

【输出格式】

并行模式（工具间无依赖）：
{"execution_plan":{"mode":"parallel","steps":[{"step_id":1,"parallel_calls":[{"tool_name":"工具名","arguments":{"参数":"值"}}]}]}}

串行模式（工具间有依赖）：
{"execution_plan":{"mode":"sequential","steps":[{"step_id":1,"call":{"tool_name":"工具名","arguments":{"参数":"值"}},"output_ref":"$step1_result"},{"step_id":2,"call":{"tool_name":"工具名","arguments":{"参数":"$step1_result.字段"}},"depends_on":[1]}]}}

混合模式（部分并行，部分串行）：
{"execution_plan":{"mode":"mixed","steps":[{"step_id":1,"parallel_calls":[{"tool_name":"工具1","arguments":{...},"output_ref":"$ref1"},{"tool_name":"工具2","arguments":{...},"output_ref":"$ref2"}]},{"step_id":2,"call":{"tool_name":"工具3","arguments":{"参数":"$ref1.字段"}},"depends_on":[1]}]}}

【重要】
- 使用 $stepN_result 或自定义 $ref_name 引用前序结果
- 使用 .字段名 访问结果的特定字段
- depends_on 声明依赖的步骤ID列表
- 不需要工具则直接用自然语言回答
```

---

## 6. 执行流程

### 6.1 完整流程图

```
用户输入
    │
    ▼
┌─────────────────────────────────────┐
│ Round 1: 生成执行计划                │
│ - 分析问题                          │
│ - 判断依赖关系                       │
│ - 输出 JSON 格式的 execution_plan   │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│ 解析执行计划                         │
│ - ExecutionPlanParser.parse()       │
│ - 验证格式和依赖关系                 │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│ 构建依赖图 & 拓扑排序                │
│ - DependencyResolver                │
│ - 输出: [[1], [2,3], [4]]           │
│   (组内并行，组间串行)               │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│ 按组执行工具                         │
│                                     │
│ Group 1: [step1]                    │
│   └─ 执行 step1，缓存结果            │
│                                     │
│ Group 2: [step2, step3]             │
│   ├─ 解析引用 ($step1_result.xxx)   │
│   └─ 并行执行 step2, step3          │
│                                     │
│ Group 3: [step4]                    │
│   ├─ 解析引用                        │
│   └─ 执行 step4                     │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│ Round 2: 生成最终回答                │
│ - 注入所有工具结果                   │
│ - 生成自然语言回答                   │
└─────────────────────────────────────┘
    │
    ▼
最终回答
```

### 6.2 示例执行过程

**用户问题**："北京和上海的温差是多少华氏度？"

**Round 1 输出**：
```json
{
  "execution_plan": {
    "mode": "mixed",
    "steps": [
      {
        "step_id": 1,
        "parallel_calls": [
          {"tool_name": "get_weather", "arguments": {"city": "北京"}, "output_ref": "$beijing"},
          {"tool_name": "get_weather", "arguments": {"city": "上海"}, "output_ref": "$shanghai"}
        ]
      },
      {
        "step_id": 2,
        "call": {
          "tool_name": "calculator",
          "arguments": {"expression": "$beijing.temperature - $shanghai.temperature"}
        },
        "depends_on": [1],
        "output_ref": "$diff_celsius"
      },
      {
        "step_id": 3,
        "call": {
          "tool_name": "unit_convert",
          "arguments": {
            "value": "$diff_celsius.result",
            "from_unit": "celsius",
            "to_unit": "fahrenheit"
          }
        },
        "depends_on": [2]
      }
    ]
  }
}
```

**执行过程**：

1. **拓扑排序**：`[[1], [2], [3]]`

2. **执行 Group 1**（step 1）：
   - 并行调用 `get_weather("北京")` 和 `get_weather("上海")`
   - 缓存结果：
     - `$beijing = {"temperature": 15, "condition": "晴"}`
     - `$shanghai = {"temperature": 18, "condition": "多云"}`

3. **执行 Group 2**（step 2）：
   - 解析引用：`$beijing.temperature` → `15`，`$shanghai.temperature` → `18`
   - 调用 `calculator("15 - 18")`
   - 缓存结果：`$diff_celsius = {"result": -3}`

4. **执行 Group 3**（step 3）：
   - 解析引用：`$diff_celsius.result` → `-3`
   - 调用 `unit_convert(-3, "celsius", "fahrenheit")`
   - 结果：`{"result": -5.4}`

**Round 2 输入**：
```
工具执行结果：
Step 1: get_weather(北京) = {"temperature": 15, ...}
        get_weather(上海) = {"temperature": 18, ...}
Step 2: calculator("15 - 18") = {"result": -3}
Step 3: unit_convert(-3, celsius, fahrenheit) = {"result": -5.4}

用户问题：北京和上海的温差是多少华氏度？
```

**最终回答**："北京和上海的温差是 -5.4 华氏度（上海比北京高约 3 摄氏度）。"

---

## 7. 代码修改清单

### 7.1 新增文件

| 文件 | 功能 |
|------|------|
| `tools/execution_plan_parser.py` | 解析模型输出的执行计划 |
| `tools/dependency_resolver.py` | 依赖关系解析和引用替换 |
| `tools/plan_executor.py` | 执行计划调度器 |

### 7.2 修改文件

| 文件 | 修改内容 |
|------|----------|
| `run_local_multiturn.py` | 集成新的执行流程 |
| `utils.py` | 更新 `build_tool_system_prompt` 支持新格式 |
| `tools/streaming_parser.py` | 支持解析 `execution_plan` 格式 |
| `tools/async_executor.py` | 扩展支持依赖调度 |

### 7.3 关键代码片段

**run_local_multiturn.py 修改**：

```python
def process_question(self, user_text: str, max_new_tokens: int = 100) -> str:
    # ... 第一轮推理 ...
    round1_output = self.generate_single_round(prompt_ids, max_new_tokens)
    
    # 解析执行计划
    plan_parser = ExecutionPlanParser()
    plan = plan_parser.parse(round1_output)
    
    if plan is None:
        # 无工具调用，直接返回
        return round1_output
    
    # 执行工具调用计划
    plan_executor = PlanExecutor(self.tool_coordinator, self.async_executor)
    all_results = plan_executor.execute(plan)
    
    # 第二轮推理
    # ...
```

---

## 8. 测试用例

### 8.1 并行模式测试

```python
# 输入
"北京和上海的天气如何？"

# 期望输出格式
{
  "execution_plan": {
    "mode": "parallel",
    "steps": [{"step_id": 1, "parallel_calls": [...]}]
  }
}

# 验证
- 两个 get_weather 调用并行执行
- 执行时间 ≈ 单次调用时间（而非 2 倍）
```

### 8.2 串行模式测试

```python
# 输入
"北京现在多少华氏度？"

# 期望输出格式
{
  "execution_plan": {
    "mode": "sequential",
    "steps": [
      {"step_id": 1, "call": {"tool_name": "get_weather", ...}, "output_ref": "$step1_result"},
      {"step_id": 2, "call": {"tool_name": "unit_convert", "arguments": {"value": "$step1_result.temperature", ...}}, "depends_on": [1]}
    ]
  }
}

# 验证
- step2 的 value 参数正确替换为 step1 的温度值
- 执行顺序正确
```

### 8.3 混合模式测试

```python
# 输入
"北京和上海的温差是多少？"

# 验证
- step1 并行获取两地天气
- step2 串行计算温差，正确引用 step1 的结果
```

---

## 9. 实现优先级

| 优先级 | 任务 | 预计工作量 |
|--------|------|-----------|
| P0 | ExecutionPlanParser | 2h |
| P0 | DependencyResolver | 3h |
| P0 | PlanExecutor | 3h |
| P1 | 修改 run_local_multiturn.py | 2h |
| P1 | 更新 System Prompt | 1h |
| P2 | 单元测试 | 2h |
| P2 | 集成测试 | 2h |

**总计**：约 15 小时

---

## 10. 风险与缓解

| 风险 | 缓解措施 |
|------|----------|
| 模型输出格式不稳定 | 增加格式校验和容错处理 |
| 循环依赖 | 拓扑排序时检测环 |
| 引用解析失败 | 提供默认值或报错提示 |
| 并行执行超时 | 设置超时机制，部分结果也可用 |

---

## 11. 后续扩展

1. **多轮工具调用**：支持 Round 2 继续调用工具
2. **条件执行**：根据前序结果决定是否执行后续步骤
3. **错误恢复**：某步骤失败时的重试或跳过策略
4. **执行计划可视化**：展示依赖图和执行进度

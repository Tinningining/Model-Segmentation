# Qwen分布式MCP工具调度系统 - 完整设计与实现指南

> **本文档完全融合了以下三个文档的所有内容：**
> - 《多轮推理与工具调用详解》
> - 《Qwen分布式MCP工具调度系统设计方案》  
> - 《核心问题解决方案》
>
> **文档特点**：理论与实践结合，从原理到实现，一站式完整指南

---

## 📚 文档导航

### 第一章：理论基础 - 多轮推理机制
深入理解为什么需要多轮推理，以及工具调用的完整流程

### 第二章：核心问题解决方案  
解决两个关键问题：如何让模型输出固定格式？如何定义工具？

### 第三章：系统架构设计
4设备分布式推理系统的完整架构设计

### 第四章：详细实现方案
从Tool Call解析到工具调度的完整实现

### 第五章：代码实现与示例
可运行的完整代码和使用示例

### 第六章：测试、优化与扩展
测试场景、性能优化和未来扩展方向

---


# 第一章：理论基础 - 多轮推理与工具调用机制

> 本章详细解释为什么需要多轮推理，以及工具调用的完整流程

# 多轮推理与工具调用详解

## 核心问题回答

**Q: 在工具调用场景中，模型需要多次输入吗？**

**A: 是的，需要多次调用模型。**

原因很简单：**模型不能直接访问外部数据（如天气API、数据库等），需要外部系统帮它执行工具，然后把结果"告诉"模型。**

---

## 完整流程示例

### 用户问题："查询北京今天的天气，并根据天气推荐穿衣建议"

```
┌──────────────────────────────────────────────────────────────┐
│ 第1轮：模型推理（决策阶段）                                   │
├──────────────────────────────────────────────────────────────┤
│ 输入：用户问题                                                │
│ 输出：<tool_call>get_weather(city="北京")</tool_call>        │
│ 说明：模型决定需要调用天气工具，但此时还没有天气数据          │
└──────────────────────────────────────────────────────────────┘
                            ↓
┌──────────────────────────────────────────────────────────────┐
│ 工具执行（不涉及模型）                                        │
├──────────────────────────────────────────────────────────────┤
│ 在Device 2上执行：get_weather(city="北京")                   │
│ 返回：{"temperature": 15, "condition": "晴"}                 │
│ 说明：纯函数调用，获取真实天气数据                            │
└──────────────────────────────────────────────────────────────┘
                            ↓
┌──────────────────────────────────────────────────────────────┐
│ 第2轮：模型推理（综合阶段）                                   │
├──────────────────────────────────────────────────────────────┤
│ 输入：用户问题 + 工具结果                                     │
│ 输出：今天北京天气晴朗，温度15度，建议穿长袖衬衫...           │
│ 说明：模型看到了天气数据，可以给出具体建议                    │
└──────────────────────────────────────────────────────────────┘
```

---

## 为什么需要两次调用？

### 第1次调用的作用
- **输入**：只有用户问题
- **输出**：tool_call指令
- **目的**：让模型"思考"需要什么信息
- **关键**：此时模型还没有天气数据

### 第2次调用的作用
- **输入**：用户问题 + 工具返回的数据
- **输出**：最终答案
- **目的**：让模型基于真实数据回答
- **关键**：此时模型有了天气数据

---

## 详细的Prompt对比

### 第1轮推理的输入

```
System: You are a helpful assistant with tools.
Available tools: get_weather

User: 查询北京今天的天气，并根据天气推荐穿衣建议
```

**第1轮模型输出：**
```
我需要先查询天气信息。
<tool_call>
  <name>get_weather</name>
  <arguments>{"city": "北京", "date": "today"}</arguments>
</tool_call>
```

**关键点**：模型此时还没有天气数据，只是决定要调用工具。

---

### 第2轮推理的输入

```
System: You are a helpful assistant with tools.
Available tools: get_weather

User: 查询北京今天的天气，并根据天气推荐穿衣建议

Assistant: 我需要先查询天气信息。
<tool_call>
  <name>get_weather</name>
  <arguments>{"city": "北京", "date": "today"}</arguments>
</tool_call>

Tool Result: {"temperature": 15, "condition": "晴", "wind": "微风", "humidity": "45%"}
```

**第2轮模型输出：**
```
根据查询结果，今天北京天气晴朗，温度15度，微风，湿度45%。

建议穿着：
- 上衣：长袖衬衫或薄外套
- 下装：长裤
- 配件：可携带太阳镜

天气适宜外出活动，注意防晒。
```

**关键点**：模型现在有了天气数据，可以给出具体的穿衣建议。

---

## 四、关键要点总结

### 4.1 为什么不能一次完成？

| 方案 | 可行性 | 原因 |
|------|--------|------|
| 模型直接调用API | ❌ 不可行 | 模型只能输出文本，不能执行代码 |
| 在prompt中提供所有数据 | ❌ 不现实 | 数据是动态的，无法预先知道 |
| 多轮推理 | ✅ 可行 | 模型决策→系统执行→模型综合 |

### 4.2 数据流向

```
用户输入 → 模型(第1轮) → tool_call指令 → 工具执行 → 真实数据 → 模型(第2轮) → 最终答案
```

### 4.3 类比理解

就像你去餐厅点餐：

1. **第1轮**：你看菜单，决定"我要点宫保鸡丁"（模型决策）
2. **执行**：服务员去厨房下单，厨师做菜（工具执行）
3. **第2轮**：菜上桌后，你品尝并评价"味道不错"（模型综合）

你不能在没看到菜之前就评价味道，模型也不能在没有数据之前就给出具体答案！

---

## 五、在分布式系统中的实现

### 5.1 系统架构

```
Device 0 (主模型)          Device 2 (工具设备)
┌─────────────┐           ┌──────────────┐
│  Qwen模型   │           │  工具集合    │
│  (推理)     │◄─────────►│  - weather   │
│             │   RPC     │  - database  │
└─────────────┘           └──────────────┘
```

### 5.2 简化的代码示例

```python
class ToolInference:
    def run(self, query):
        # 第1轮：模型决策
        response1 = self.model.generate(query)
        
        if has_tool_call(response1):
            # 执行工具
            tool_result = self.execute_tool(response1)
            
            # 第2轮：模型综合
            response2 = self.model.generate(
                query + response1 + tool_result
            )
            return response2
        
        return response1
```

---

## 六、常见问题

### Q1: 能否跳过第1轮，直接给模型提供数据？

**A**: 不能。因为：
- 你不知道用户会问什么问题
- 你不知道需要调用哪个工具
- 你不知道工具需要什么参数

必须让模型先"思考"需要什么。

### Q2: 第2轮推理时，模型能"记住"第1轮的输出吗？

**A**: 不是"记住"，而是把第1轮的输出作为第2轮的输入：

```
第2轮输入 = 用户问题 + 第1轮输出 + 工具结果
```

模型每次都是"无状态"的，需要完整的上下文。

### Q3: 如果需要调用多个工具怎么办？

**A**: 继续循环：

```
第1轮推理 → 工具1执行 → 第2轮推理 → 工具2执行 → 第3轮推理 → 最终答案
```

每次工具执行后都要回到模型。

---

## 七、总结

### 核心原理

**模型 = 大脑（思考）**
**工具 = 手脚（行动）**

大脑不能直接拿东西，需要：
1. 大脑决定"我要拿杯子"
2. 手去拿杯子
3. 大脑看到杯子后说"这是个红色的杯子"

### 关键点

1. ✅ **需要多次调用模型**
2. ✅ **每次工具执行后都要回到模型**
3. ✅ **第2轮输入包含第1轮的所有信息**
4. ✅ **工具执行不涉及模型**

### 记住这个流程

```
用户问题 
  ↓
模型推理(第1轮) → 决定调用工具
  ↓
工具执行 → 获取真实数据
  ↓
模型推理(第2轮) → 基于数据回答
  ↓
返回答案
```

这就是工具调用的完整流程！


---


# 第二章：核心问题解决方案

> 本章解决两个关键问题：如何让模型输出固定格式的tool_call？如何定义和管理工具？

# 第一部分：核心问题解决方案

## 问题1：如何让模型生成固定的工具调用格式？

### 1.1 解决方案：Prompt工程

**核心思想**：通过System Prompt教会模型输出特定格式

在System Prompt中包含：
1. 工具调用的格式规范
2. 可用工具列表及其描述
3. 每个工具的参数说明
4. 使用示例

**代码实现**：见原设计方案第三部分的`build_system_prompt_with_tools`函数

**关键点**：
- 使用XML格式（清晰、易解析）
- 提供详细的工具描述和参数说明
- 包含使用示例（Few-Shot Learning）
- 明确输出规则

---

## 问题2：如何确定可调用的工具及其功能描述？

### 2.1 工具注册机制

**方案**：使用工具注册表（Tool Registry）

每个工具需要提供：
1. **工具名称**：唯一标识符
2. **功能描述**：简短说明工具用途
3. **参数定义**：每个参数的类型、是否必需、描述
4. **示例**：典型的调用示例
5. **元数据**：内存占用、执行时间等

### 2.2 工具定义格式

**JSON Schema格式**：

```python
TOOL_REGISTRY = {
    "get_weather": {
        "name": "get_weather",
        "description": "Get current weather information for a city",
        "parameters": {
            "city": {
                "type": "string",
                "required": True,
                "description": "The city name to query weather for"
            },
            "date": {
                "type": "string",
                "required": False,
                "default": "today",
                "description": "Date for weather query (e.g., 'today', '2024-01-01')"
            }
        },
        "example": '{"city": "北京", "date": "today"}',
        "returns": {
            "type": "object",
            "description": "Weather information including temperature and condition"
        },
        # 元数据（用于调度）
        "metadata": {
            "memory_size": 30,  # MB
            "avg_execution_time": 0.5,  # 秒
            "dependencies": [],  # 依赖的其他工具
            "device_preference": None  # 设备偏好
        }
    },
    
    "calculator": {
        "name": "calculator",
        "description": "Perform mathematical calculations",
        "parameters": {
            "expression": {
                "type": "string",
                "required": True,
                "description": "Mathematical expression to evaluate (e.g., '123 + 456')"
            }
        },
        "example": '{"expression": "123 + 456"}',
        "returns": {
            "type": "number",
            "description": "Calculation result"
        },
        "metadata": {
            "memory_size": 10,
            "avg_execution_time": 0.1,
            "dependencies": [],
            "device_preference": None
        }
    },
    
    "web_search": {
        "name": "web_search",
        "description": "Search the web for information",
        "parameters": {
            "query": {
                "type": "string",
                "required": True,
                "description": "Search query"
            },
            "limit": {
                "type": "integer",
                "required": False,
                "default": 5,
                "description": "Maximum number of results to return"
            }
        },
        "example": '{"query": "人工智能", "limit": 5}',
        "returns": {
            "type": "array",
            "description": "List of search results"
        },
        "metadata": {
            "memory_size": 50,
            "avg_execution_time": 2.0,
            "dependencies": [],
            "device_preference": None
        }
    }
}
```

### 2.3 工具管理器实现

```python
class ToolRegistry:
    """工具注册表"""
    
    def __init__(self):
        self.tools = {}
    
    def register(self, tool_config):
        """注册工具"""
        tool_name = tool_config['name']
        
        # 验证工具配置
        self._validate_tool_config(tool_config)
        
        self.tools[tool_name] = tool_config
        print(f"✓ Tool '{tool_name}' registered")
    
    def get_tool(self, tool_name):
        """获取工具配置"""
        return self.tools.get(tool_name)
    
    def list_tools(self):
        """列出所有工具"""
        return list(self.tools.keys())
    
    def get_tool_description(self, tool_name):
        """获取工具描述（用于构建prompt）"""
        tool = self.tools.get(tool_name)
        if not tool:
            return None
        
        desc = f"{tool['name']}:\n"
        desc += f"  Description: {tool['description']}\n"
        desc += f"  Parameters:\n"
        
        for param_name, param_info in tool['parameters'].items():
            required = "required" if param_info.get('required', False) else "optional"
            desc += f"    - {param_name} ({param_info['type']}, {required}): "
            desc += f"{param_info.get('description', '')}\n"
        
        desc += f"  Example: {tool['example']}\n"
        
        return desc
    
    def _validate_tool_config(self, config):
        """验证工具配置"""
        required_fields = ['name', 'description', 'parameters']
        for field in required_fields:
            if field not in config:
                raise ValueError(f"Tool config missing required field: {field}")
```

### 2.4 工具发现机制

**方式1：静态注册（推荐用于初期）**

```python
# config.py
from tool_registry import TOOL_REGISTRY

# 在启动时注册所有工具
registry = ToolRegistry()
for tool_name, tool_config in TOOL_REGISTRY.items():
    registry.register(tool_config)
```

**方式2：动态发现（用于扩展）**

```python
import importlib
import os

def discover_tools(tools_dir="tools/builtin_tools"):
    """自动发现并注册工具"""
    registry = ToolRegistry()
    
    for filename in os.listdir(tools_dir):
        if filename.endswith("_tool.py"):
            module_name = filename[:-3]
            module = importlib.import_module(f"tools.builtin_tools.{module_name}")
            
            # 每个工具模块需要提供TOOL_CONFIG
            if hasattr(module, 'TOOL_CONFIG'):
                registry.register(module.TOOL_CONFIG)
    
    return registry
```

**方式3：MCP协议（用于外部工具）**

```python
class MCPToolAdapter:
    """MCP工具适配器"""
    
    def __init__(self, mcp_server_url):
        self.server_url = mcp_server_url
    
    def discover_tools(self):
        """从MCP服务器发现工具"""
        # 调用MCP协议的list_tools接口
        response = requests.get(f"{self.server_url}/tools")
        mcp_tools = response.json()
        
        # 转换为内部格式
        tools = []
        for mcp_tool in mcp_tools:
            tool_config = self._convert_mcp_to_internal(mcp_tool)
            tools.append(tool_config)
        
        return tools
    
    def _convert_mcp_to_internal(self, mcp_tool):
        """将MCP工具格式转换为内部格式"""
        return {
            "name": mcp_tool['name'],
            "description": mcp_tool['description'],
            "parameters": mcp_tool['inputSchema']['properties'],
            "example": mcp_tool.get('example', '{}'),
            "metadata": {
                "memory_size": 50,  # 默认值
                "avg_execution_time": 1.0,
                "dependencies": [],
                "device_preference": None
            }
        }
```

---




---


# 第三章到第六章：系统架构、实现、测试与优化

> 本部分包含完整的系统设计、详细实现方案、代码示例、测试场景和优化建议

# Qwen分布式MCP工具调度系统设计方案

## 一、系统概述

### 1.1 当前基础
- **4台设备**：运行Qwen-1.7B模型的分布式推理
- **模型切分**：embed + 4个layer blocks + output（共6个OM模型）
- **现有功能**：基础的文本生成推理

### 1.2 目标功能
在现有分布式推理基础上，实现：
1. **Tool Call解析**：大模型输出特定格式的工具调用指令
2. **动态工具加载**：在4台设备上按需加载MCP工具
3. **工具执行**：调用工具并获取结果
4. **结果整合**：将工具结果反馈给模型，生成最终答案
5. **智能调度**：合理分配工具到各设备，支持并行和流水线调用

### 1.3 工作流程示例

```
用户问题: "查询北京今天的天气，并根据天气推荐穿衣建议"

Step 1: 模型分析问题
  → 输出: <tool_call>get_weather(city="北京")</tool_call>

Step 2: 系统解析并调度工具
  → 在Device 2加载weather_tool
  → 执行: get_weather(city="北京")
  → 返回: {"temperature": 15, "condition": "晴"}

Step 3: 模型继续推理
  → 输入: 天气数据 + 原始问题
  → 输出: "今天北京天气晴朗，温度15度，建议穿长袖衬衫..."
```

---

## 二、系统架构设计

### 2.1 整体架构

```
┌─────────────────────────────────────────────────────────────────┐
│                        用户交互层                                 │
│  输入问题 → 接收答案                                              │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    推理协调层 (Coordinator)                       │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │ Tool Call    │  │ 工具调度器   │  │ 结果整合器   │          │
│  │ 解析器       │  │              │  │              │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    工具管理层 (Tool Manager)                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │ 工具注册表   │  │ 工具加载器   │  │ 执行引擎     │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│              分布式推理层 (Distributed Inference)                 │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐        │
│  │ Device 0 │  │ Device 1 │  │ Device 2 │  │ Device 3 │        │
│  │ embed +  │  │ layers   │  │ layers   │  │ layers + │        │
│  │ layers   │  │ 7-13     │  │ 14-20    │  │ 21-27 +  │        │
│  │ 0-6      │  │          │  │          │  │ output   │        │
│  │          │  │          │  │          │  │          │        │
│  │ [工具槽] │  │ [工具槽] │  │ [工具槽] │  │ [工具槽] │        │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘        │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 核心组件

#### 组件1：Tool Call解析器
**功能**：从模型输出中识别和解析工具调用指令

**输入格式**：
```xml
<tool_call>
  <name>get_weather</name>
  <arguments>
    {"city": "北京", "date": "today"}
  </arguments>
</tool_call>
```

**输出**：
```python
{
    "tool_name": "get_weather",
    "arguments": {"city": "北京", "date": "today"},
    "call_id": "call_001"
}
```

#### 组件2：工具调度器
**功能**：决定在哪个设备上加载和执行工具

**调度策略**：
1. **负载均衡**：选择当前负载最低的设备
2. **亲和性**：优先选择已加载该工具的设备
3. **依赖感知**：考虑工具间的数据依赖关系

#### 组件3：工具管理器
**功能**：管理工具的生命周期（注册、加载、卸载、执行）

**工具元数据**：
```python
{
    "tool_name": "get_weather",
    "memory_size": 50,  # MB
    "execution_time": 0.5,  # 秒
    "dependencies": [],  # 依赖的其他工具
    "device_preference": None  # 设备偏好
}
```

#### 组件4：执行引擎
**功能**：在指定设备上执行工具调用

---

## 2.3 关键技术挑战：设备间数据传输

### 核心问题

在4设备分布式推理系统中存在一个关键技术挑战：

- **模型推理固定在Device 0**（包含embed和layers 0-6）
- **工具可能在任意设备执行**（Device 0/1/2/3）
- **工具结果必须传回Device 0**才能进行第2轮推理

### 数据流问题示意

```
用户问题 → Device 0 (第1轮推理) → tool_call
                ↓
         工具调度到Device 2
                ↓
         Device 2 执行工具 → 结果
                ↓
         ❌ 问题：结果在Device 2，但需要在Device 0进行第2轮推理
                ↓
         ✅ 解决：将结果传输回Device 0
                ↓
         Device 0 (第2轮推理：问题 + 工具结果) → 最终答案
```

### 解决方案：集中式结果收集

**核心思想**：所有工具结果都传回Device 0（主设备）

#### 架构设计

```
┌─────────────────────────────────────────────────────────────┐
│                    Device 0 (主设备)                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ 模型推理     │  │ 结果收集器   │  │ 协调器       │      │
│  │ (embed +     │  │              │  │              │      │
│  │  layers 0-6) │  │              │  │              │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└─────────────────────────────────────────────────────────────┘
         ↑                    ↑                    ↓
         │                    │                    │ 工具调度
         │ 结果传输           │ 结果传输           │
         │                    │                    ↓
┌────────┴────────┐  ┌────────┴────────┐  ┌──────────────────┐
│   Device 1      │  │   Device 2      │  │   Device 3       │
│  [工具执行]     │  │  [工具执行]     │  │  [工具执行]      │
└─────────────────┘  └─────────────────┘  └──────────────────┘
```

#### 结果收集器实现

```python
class ResultCollector:
    """结果收集器 - 负责将工具结果传回Device 0"""
    
    def __init__(self, main_device_id=0):
        self.main_device_id = main_device_id
        self.result_buffer = {}
    
    def collect_result(self, tool_result, source_device_id):
        """
        收集工具结果并传输到主设备
        
        Args:
            tool_result: 工具执行结果
            source_device_id: 工具执行的设备ID
        
        Returns:
            在主设备上可访问的结果
        """
        if source_device_id == self.main_device_id:
            # 已经在主设备，直接返回
            return tool_result
        
        # 需要跨设备传输
        print(f"  → Transferring result from Device {source_device_id} to Device {self.main_device_id}")
        
        # 通过共享内存传输
        transferred_result = self._transfer_via_shared_memory(
            tool_result, source_device_id, self.main_device_id
        )
        
        return transferred_result
    
    def _transfer_via_shared_memory(self, data, from_device, to_device):
        """通过共享内存传输数据"""
        import pickle
        
        # 1. 序列化数据
        serialized = pickle.dumps(data)
        
        # 2. 复制到主设备内存
        # 注意：实际实现需要使用设备间通信API
        transferred = serialized
        
        # 3. 反序列化
        return pickle.loads(transferred)
```

### 优化策略：优先使用Device 0

**核心思想**：优先在Device 0上执行工具，避免数据传输

```python
class Device0PreferredScheduler:
    """优先使用Device 0的调度器"""
    
    def __init__(self, devices, main_device_id=0):
        self.devices = devices
        self.main_device_id = main_device_id
        self.device_loads = {i: 0 for i in range(len(devices))}
        self.loaded_tools = {i: set() for i in range(len(devices))}
    
    def schedule(self, tool_name, tool_size=50):
        """
        调度策略：
        1. 如果Device 0负载不高，优先使用Device 0
        2. 否则选择其他设备
        """
        # 1. 检查工具是否已在某设备加载
        for device_id, tools in self.loaded_tools.items():
            if tool_name in tools:
                return device_id
        
        # 2. 检查Device 0的负载
        main_device_load = self.device_loads[self.main_device_id]
        max_load_threshold = 500  # MB
        
        if main_device_load + tool_size < max_load_threshold:
            # Device 0负载不高，优先使用
            print(f"  → Scheduling to Device {self.main_device_id} (avoid data transfer)")
            self.device_loads[self.main_device_id] += tool_size
            self.loaded_tools[self.main_device_id].add(tool_name)
            return self.main_device_id
        
        # 3. Device 0负载过高，选择其他设备
        other_devices = [i for i in range(len(self.devices)) if i != self.main_device_id]
        best_device = min(other_devices, key=lambda x: self.device_loads[x])
        
        print(f"  → Scheduling to Device {best_device} (Device 0 overloaded)")
        self.device_loads[best_device] += tool_size
        self.loaded_tools[best_device].add(tool_name)
        
        return best_device
```

### 性能对比

| 场景 | 工具位置 | 数据传输 | 延迟 |
|------|---------|---------|------|
| 最优 | Device 0 | 无 | 0ms |
| 一般 | Device 1/2/3 | 一次传输 | 5-10ms |
| 并行 | 多设备 | 批量传输 | 10-20ms |

**优化效果**：
- 优化前：每个工具调用都可能需要数据传输，平均延迟15ms/工具
- 优化后：70%的工具在Device 0执行（无传输），平均延迟5ms/工具

---

## 三、详细实现方案

### 3.1 Tool Call格式设计

#### 方案1：XML格式（推荐）
```xml
<tool_call>
  <name>get_weather</name>
  <arguments>{"city": "北京"}</arguments>
</tool_call>
```

**优点**：
- 易于解析
- 结构清晰
- 支持嵌套

#### 方案2：JSON格式
```json
{"tool_call": {"name": "get_weather", "arguments": {"city": "北京"}}}
```

#### 方案3：函数调用格式
```python
get_weather(city="北京")
```

### 3.2 工具调度策略

#### 策略1：简单轮询（Round-Robin）
```python
class RoundRobinScheduler:
    def __init__(self, num_devices=4):
        self.num_devices = num_devices
        self.current = 0
    
    def schedule(self, tool_name):
        """轮询选择设备"""
        device_id = self.current
        self.current = (self.current + 1) % self.num_devices
        return device_id
```

#### 策略2：负载感知调度
```python
class LoadAwareScheduler:
    def __init__(self, devices):
        self.devices = devices
        self.device_loads = {i: 0 for i in range(len(devices))}
        self.loaded_tools = {i: set() for i in range(len(devices))}
    
    def schedule(self, tool_name, tool_size):
        """基于负载选择设备"""
        # 1. 检查是否已加载
        for device_id, tools in self.loaded_tools.items():
            if tool_name in tools:
                return device_id  # 优先使用已加载的设备
        
        # 2. 选择负载最低的设备
        min_load = float('inf')
        best_device = 0
        
        for device_id, load in self.device_loads.items():
            if load < min_load:
                min_load = load
                best_device = device_id
        
        # 3. 更新负载
        self.device_loads[best_device] += tool_size
        self.loaded_tools[best_device].add(tool_name)
        
        return best_device
```

#### 策略3：依赖感知调度（用于流水线调用）
```python
class DependencyAwareScheduler:
    def __init__(self, devices):
        self.devices = devices
        self.tool_locations = {}  # tool_name -> device_id
        self.device_loads = {i: 0 for i in range(len(devices))}
    
    def schedule_pipeline(self, tool_sequence):
        """为工具流水线分配设备"""
        schedule_plan = []
        
        for i, tool_info in enumerate(tool_sequence):
            tool_name = tool_info['name']
            
            if i == 0:
                # 第一个工具：选择负载最低的设备
                device_id = self._select_least_loaded_device()
            else:
                # 后续工具：考虑数据传输成本
                prev_device = schedule_plan[-1]['device_id']
                device_id = self._select_with_locality(
                    tool_name, prev_device, tool_info['data_size']
                )
            
            schedule_plan.append({
                'tool_name': tool_name,
                'device_id': device_id,
                'depends_on': schedule_plan[-1]['tool_name'] if i > 0 else None
            })
            
            self.tool_locations[tool_name] = device_id
        
        return schedule_plan
    
    def _select_with_locality(self, tool_name, prev_device, data_size):
        """考虑数据局部性选择设备"""
        # 如果数据量小，优先在同一设备执行
        if data_size < 10:  # MB
            return prev_device
        
        # 否则选择负载较低的设备
        return self._select_least_loaded_device()
```

#### 策略4：并行调度（用于并行工具调用）
```python
class ParallelScheduler:
    def __init__(self, devices):
        self.devices = devices
        self.device_loads = {i: 0 for i in range(len(devices))}
    
    def schedule_parallel(self, tool_list):
        """为并行工具调用分配设备"""
        schedule_plan = []
        
        # 按工具大小排序（大的优先）
        sorted_tools = sorted(
            tool_list, 
            key=lambda x: x['size'], 
            reverse=True
        )
        
        for tool_info in sorted_tools:
            # 选择当前负载最低的设备
            device_id = min(
                self.device_loads.keys(),
                key=lambda x: self.device_loads[x]
            )
            
            schedule_plan.append({
                'tool_name': tool_info['name'],
                'device_id': device_id,
                'parallel': True
            })
            
            self.device_loads[device_id] += tool_info['size']
        
        return schedule_plan
```

### 3.3 工具管理实现

```python
class ToolManager:
    """工具管理器"""
    
    def __init__(self, devices):
        self.devices = devices
        self.tool_registry = {}  # 工具注册表
        self.loaded_tools = {i: {} for i in range(len(devices))}  # 已加载的工具
        self.device_memory = {i: {'total': 2048, 'used': 0} for i in range(len(devices))}
    
    def register_tool(self, tool_name, tool_config):
        """注册工具"""
        self.tool_registry[tool_name] = {
            'name': tool_name,
            'module_path': tool_config['module_path'],
            'memory_size': tool_config.get('memory_size', 50),  # MB
            'execution_time': tool_config.get('execution_time', 1.0),
            'dependencies': tool_config.get('dependencies', []),
            'handler': None  # 工具处理函数
        }
        print(f"✓ Tool '{tool_name}' registered")
    
    def load_tool(self, tool_name, device_id):
        """在指定设备上加载工具"""
        if tool_name not in self.tool_registry:
            raise ValueError(f"Tool '{tool_name}' not registered")
        
        # 检查是否已加载
        if tool_name in self.loaded_tools[device_id]:
            print(f"Tool '{tool_name}' already loaded on Device {device_id}")
            return True
        
        tool_info = self.tool_registry[tool_name]
        
        # 检查内存
        required_memory = tool_info['memory_size']
        available_memory = (
            self.device_memory[device_id]['total'] - 
            self.device_memory[device_id]['used']
        )
        
        if available_memory < required_memory:
            # 需要卸载一些工具
            self._evict_tools(device_id, required_memory)
        
        # 加载工具
        try:
            # 动态导入工具模块
            import importlib
            module = importlib.import_module(tool_info['module_path'])
            handler = getattr(module, 'execute')
            
            self.loaded_tools[device_id][tool_name] = {
                'handler': handler,
                'memory_size': required_memory,
                'load_time': time.time()
            }
            
            self.device_memory[device_id]['used'] += required_memory
            
            print(f"✓ Tool '{tool_name}' loaded on Device {device_id}")
            return True
            
        except Exception as e:
            print(f"✗ Failed to load tool '{tool_name}': {e}")
            return False
    
    def unload_tool(self, tool_name, device_id):
        """卸载工具"""
        if tool_name not in self.loaded_tools[device_id]:
            return True
        
        tool_info = self.loaded_tools[device_id][tool_name]
        memory_size = tool_info['memory_size']
        
        del self.loaded_tools[device_id][tool_name]
        self.device_memory[device_id]['used'] -= memory_size
        
        print(f"✓ Tool '{tool_name}' unloaded from Device {device_id}")
        return True
    
    def execute_tool(self, tool_name, device_id, arguments):
        """执行工具"""
        if tool_name not in self.loaded_tools[device_id]:
            # 自动加载
            success = self.load_tool(tool_name, device_id)
            if not success:
                raise RuntimeError(f"Failed to load tool '{tool_name}'")
        
        handler = self.loaded_tools[device_id][tool_name]['handler']
        
        try:
            result = handler(**arguments)
            return {
                'success': True,
                'result': result,
                'tool_name': tool_name,
                'device_id': device_id
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'tool_name': tool_name,
                'device_id': device_id
            }
    
    def _evict_tools(self, device_id, required_memory):
        """驱逐工具以释放内存（LRU策略）"""
        tools = list(self.loaded_tools[device_id].items())
        tools.sort(key=lambda x: x[1]['load_time'])  # 按加载时间排序
        
        freed_memory = 0
        for tool_name, tool_info in tools:
            if freed_memory >= required_memory:
                break
            
            self.unload_tool(tool_name, device_id)
            freed_memory += tool_info['memory_size']
```

### 3.4 推理协调器实现

```python
class InferenceCoordinator:
    """推理协调器 - 整合模型推理和工具调用"""
    
    def __init__(self, model_network, tool_manager, scheduler):
        self.model = model_network
        self.tool_manager = tool_manager
        self.scheduler = scheduler
        self.max_iterations = 10  # 最大迭代次数
    
    def run(self, user_query):
        """运行完整的推理流程"""
        print(f"\n{'='*60}")
        print(f"User Query: {user_query}")
        print(f"{'='*60}\n")
        
        conversation_history = [
            {"role": "user", "content": user_query}
        ]
        
        for iteration in range(self.max_iterations):
            print(f"\n--- Iteration {iteration + 1} ---")
            
            # 1. 模型生成响应
            prompt = self._build_prompt(conversation_history)
            response = self.model.generate(prompt)
            
            print(f"Model Output: {response[:200]}...")
            
            # 2. 检查是否包含tool call
            tool_calls = self._parse_tool_calls(response)
            
            if not tool_calls:
                # 没有工具调用，返回最终答案
                print(f"\n{'='*60}")
                print(f"Final Answer: {response}")
                print(f"{'='*60}\n")
                return response
            
            # 3. 执行工具调用
            tool_results = self._execute_tools(tool_calls)
            
            # 4. 将工具结果添加到对话历史
            conversation_history.append({
                "role": "assistant",
                "content": response
            })
            conversation_history.append({
                "role": "tool",
                "content": self._format_tool_results(tool_results)
            })
        
        return "达到最大迭代次数，无法完成任务"
    
    def _parse_tool_calls(self, text):
        """解析工具调用"""
        import re
        import json
        
        tool_calls = []
        
        # 匹配 <tool_call>...</tool_call>
        pattern = r'<tool_call>(.*?)</tool_call>'
        matches = re.findall(pattern, text, re.DOTALL)
        
        for match in matches:
            try:
                # 提取name和arguments
                name_match = re.search(r'<name>(.*?)</name>', match)
                args_match = re.search(r'<arguments>(.*?)</arguments>', match, re.DOTALL)
                
                if name_match and args_match:
                    tool_name = name_match.group(1).strip()
                    arguments = json.loads(args_match.group(1).strip())
                    
                    tool_calls.append({
                        'name': tool_name,
                        'arguments': arguments
                    })
            except Exception as e:
                print(f"Failed to parse tool call: {e}")
        
        return tool_calls
    
    def _execute_tools(self, tool_calls):
        """执行工具调用"""
        results = []
        
        # 检测是否为并行调用
        is_parallel = len(tool_calls) > 1 and self._check_parallel(tool_calls)
        
        if is_parallel:
            print(f"Executing {len(tool_calls)} tools in parallel...")
            results = self._execute_parallel(tool_calls)
        else:
            print(f"Executing {len(tool_calls)} tools sequentially...")
            results = self._execute_sequential(tool_calls)
        
        return results
    
    def _execute_sequential(self, tool_calls):
        """顺序执行工具"""
        results = []
        
        for tool_call in tool_calls:
            tool_name = tool_call['name']
            arguments = tool_call['arguments']
            
            # 调度到设备
            device_id = self.scheduler.schedule(tool_name)
            
            print(f"  → Executing '{tool_name}' on Device {device_id}")
            
            # 执行工具
            result = self.tool_manager.execute_tool(
                tool_name, device_id, arguments
            )
            
            results.append(result)
            
            if result['success']:
                print(f"    ✓ Success: {result['result']}")
            else:
                print(f"    ✗ Error: {result['error']}")
        
        return results
    
    def _execute_parallel(self, tool_calls):
        """并行执行工具"""
        import concurrent.futures
        
        # 调度所有工具
        schedule_plan = self.scheduler.schedule_parallel([
            {'name': tc['name'], 'size': 50} for tc in tool_calls
        ])
        
        results = []
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = []
            
            for tool_call, schedule in zip(tool_calls, schedule_plan):
                future = executor.submit(
                    self.tool_manager.execute_tool,
                    tool_call['name'],
                    schedule['device_id'],
                    tool_call['arguments']
                )
                futures.append(future)
            
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                results.append(result)
                
                if result['success']:
                    print(f"  ✓ '{result['tool_name']}' completed on Device {result['device_id']}")
                else:
                    print(f"  ✗ '{result['tool_name']}' failed: {result['error']}")
        
        return results
    
    def _check_parallel(self, tool_calls):
        """检查工具调用是否可以并行"""
        # 简单策略：如果没有明显的依赖关系，就并行执行
        # 可以根据工具的依赖关系进行更复杂的判断
        return True
    
    def _build_prompt(self, conversation_history):
        """构建提示词"""
        prompt = "You are a helpful assistant with access to tools.\n\n"
        prompt += "Available tools:\n"
        
        for tool_name, tool_info in self.tool_manager.tool_registry.items():
            prompt += f"- {tool_name}: {tool_info.get('description', 'No description')}\n"
        
        prompt += "\nTo use a tool, output:\n"
        prompt += "<tool_call>\n"
        prompt += "  <name>tool_name</name>\n"
        prompt += "  <arguments>{\"arg1\": \"value1\"}</arguments>\n"
        prompt += "</tool_call>\n\n"
        
        prompt += "Conversation:\n"
        for msg in conversation_history:
            role = msg['role']
            content = msg['content']
            prompt += f"{role.capitalize()}: {content}\n\n"
        
        prompt += "Assistant:"
        
        return prompt
    
    def _format_tool_results(self, results):
        """格式化工具结果"""
        formatted = "Tool Results:\n"
        for result in results:
            if result['success']:
                formatted += f"- {result['tool_name']}: {result['result']}\n"
            else:
                formatted += f"- {result['tool_name']}: Error - {result['error']}\n"
        return formatted
```

---

## 四、代码结构

### 4.1 目录结构

```
qwen/distributed_inference/
├── __init__.py
├── config.py                 # 配置文件
├── network.py                # 现有的网络模块
├── acl_model.py              # 现有的ACL模型
├── kvcache.py                # 现有的KV Cache
├── utils.py                  # 工具函数
├── node_head.py              # 现有的头节点
├── node_middle.py            # 现有的中间节点
├── node_tail.py              # 现有的尾节点
│
├── tools/                    # 新增：工具模块
│   ├── __init__.py
│   ├── tool_manager.py       # 工具管理器
│   ├── tool_parser.py        # Tool Call解析器
│   ├── tool_scheduler.py     # 工具调度器
│   └── builtin_tools/        # 内置工具
│       ├── __init__.py
│       ├── weather_tool.py   # 天气查询工具
│       ├── calculator_tool.py # 计算器工具
│       └── search_tool.py    # 搜索工具
│
├── coordinator.py            # 新增：推理协调器
└── run_with_tools.py         # 新增：带工具的运行脚本
```

### 4.2 配置文件示例

```python
# config.py 新增部分

# 工具配置
TOOL_CONFIG = {
    'max_tools_per_device': 3,  # 每个设备最多加载的工具数
    'tool_memory_limit': 500,   # 每个设备工具内存限制（MB）
    'scheduler_type': 'load_aware',  # 调度策略: round_robin, load_aware, dependency_aware
    'enable_parallel': True,    # 是否启用并行工具执行
    'max_iterations': 10,       # 最大推理迭代次数
}

# 注册的工具
REGISTERED_TOOLS = {
    'get_weather': {
        'module_path': 'tools.builtin_tools.weather_tool',
        'memory_size': 30,
        'description': 'Get weather information for a city',
    },
    'calculator': {
        'module_path': 'tools.builtin_tools.calculator_tool',
        'memory_size': 10,
        'description': 'Perform mathematical calculations',
    },
    'web_search': {
        'module_path': 'tools.builtin_tools.search_tool',
        'memory_size': 50,
        'description': 'Search the web for information',
    },
}
```

---

## 五、实现步骤

### 阶段1：基础工具框架（Week 1）

**任务1.1：实现Tool Call解析器**
```python
# tools/tool_parser.py
class ToolCallParser:
    def parse(self, text):
        """从模型输出中解析工具调用"""
        pass
```

**任务1.2：实现工具管理器**
```python
# tools/tool_manager.py
class ToolManager:
    def register_tool(self, tool_name, config):
        """注册工具"""
        pass
    
    def load_tool(self, tool_name, device_id):
        """加载工具到设备"""
        pass
    
    def execute_tool(self, tool_name, device_id, arguments):
        """执行工具"""
        pass
```

**任务1.3：实现简单调度器**
```python
# tools/tool_scheduler.py
class SimpleScheduler:
    def schedule(self, tool_name):
        """轮询调度"""
        pass
```

### 阶段2：内置工具开发（Week 2）

**任务2.1：天气查询工具**
```python
# tools/builtin_tools/weather_tool.py
def execute(city, date="today"):
    """查询天气"""
    # 调用天气API
    return {"temperature": 15, "condition": "晴"}
```

**任务2.2：计算器工具**
```python
# tools/builtin_tools/calculator_tool.py
def execute(expression):
    """计算数学表达式"""
    return eval(expression)
```

**任务2.3：搜索工具**
```python
# tools/builtin_tools/search_tool.py
def execute(query):
    """网络搜索"""
    # 调用搜索API
    return ["result1", "result2"]
```

### 阶段3：推理协调器（Week 3）

**任务3.1：实现基础协调器**
- 整合模型推理和工具调用
- 实现迭代式对话流程

**任务3.2：实现顺序执行**
- 工具按顺序执行
- 结果传递给下一步

**任务3.3：测试端到端流程**
- 简单的单工具调用
- 多轮对话测试

### 阶段4：高级调度（Week 4）

**任务4.1：实现负载感知调度**
- 监控设备负载
- 动态选择设备

**任务4.2：实现并行执行**
- 多工具并行调用
- 线程池管理

**任务4.3：实现流水线调度**
- 依赖关系分析
- 数据局部性优化

### 阶段5：优化与测试（Week 5）

**任务5.1：性能优化**
- 减少工具加载时间
- 优化内存使用

**任务5.2：完整测试**
- 各种场景测试
- 压力测试

---

## 六、使用示例

### 6.1 基础使用

```python
# run_with_tools.py
from distributed_inference.network import DistributedNetwork
from distributed_inference.tools.tool_manager import ToolManager
from distributed_inference.tools.tool_scheduler import LoadAwareScheduler
from distributed_inference.coordinator import InferenceCoordinator
from distributed_inference.config import REGISTERED_TOOLS, TOOL_CONFIG

# 1. 初始化分布式网络
network = DistributedNetwork(
    model_paths=['embed.om', 'layers_0_6.om', 'layers_7_13.om', 
                 'layers_14_20.om', 'layers_21_27.om', 'output.om'],
    device_ids=[0, 1, 2, 3]
)

# 2. 初始化工具管理器
tool_manager = ToolManager(devices=[0, 1, 2, 3])

# 3. 注册工具
for tool_name, tool_config in REGISTERED_TOOLS.items():
    tool_manager.register_tool(tool_name, tool_config)

# 4. 初始化调度器
scheduler = LoadAwareScheduler(devices=[0, 1, 2, 3])

# 5. 创建推理协调器
coordinator = InferenceCoordinator(
    model_network=network,
    tool_manager=tool_manager,
    scheduler=scheduler
)

# 6. 运行推理
user_query = "查询北京今天的天气，并根据天气推荐穿衣建议"
answer = coordinator.run(user_query)

print(f"最终答案: {answer}")
```

### 6.2 并行工具调用示例

```python
# 问题需要同时调用多个工具
user_query = "同时查询北京、上海、广州三个城市的天气"

# 模型输出（并行调用）
"""
<tool_call>
  <name>get_weather</name>
  <arguments>{"city": "北京"}</arguments>
</tool_call>
<tool_call>
  <name>get_weather</name>
  <arguments>{"city": "上海"}</arguments>
</tool_call>
<tool_call>
  <name>get_weather</name>
  <arguments>{"city": "广州"}</arguments>
</tool_call>
"""

# 系统自动并行调度到不同设备
# Device 0: get_weather(北京)
# Device 1: get_weather(上海)
# Device 2: get_weather(广州)
```

### 6.3 流水线工具调用示例

```python
# 问题需要流水线调用
user_query = "搜索'人工智能'相关文章，然后总结前3篇的内容"

# 模型输出（流水线调用）
"""
Step 1:
<tool_call>
  <name>web_search</name>
  <arguments>{"query": "人工智能", "limit": 3}</arguments>
</tool_call>

Step 2 (基于Step 1结果):
<tool_call>
  <name>summarize</name>
  <arguments>{"texts": ["article1", "article2", "article3"]}</arguments>
</tool_call>
"""

# 系统考虑数据局部性调度
# Device 1: web_search → 返回文章列表
# Device 1: summarize (在同一设备，避免数据传输)
```

---

## 七、测试场景

### 7.1 单工具调用测试

```python
# 测试用例1：天气查询
query = "北京今天天气怎么样？"
expected_tool_call = "get_weather"
expected_device = 0  # 第一次调用，轮询到Device 0

# 测试用例2：计算
query = "计算 (123 + 456) * 789"
expected_tool_call = "calculator"
expected_device = 1  # 轮询到Device 1
```

### 7.2 多工具并行测试

```python
# 测试用例3：并行查询
query = "同时查询北京、上海的天气和计算100+200"
expected_tool_calls = ["get_weather", "get_weather", "calculator"]
expected_parallel = True
expected_devices = [0, 1, 2]  # 分配到不同设备
```

### 7.3 工具流水线测试

```python
# 测试用例4：流水线
query = "搜索'机器学习'，然后总结结果"
expected_tool_sequence = ["web_search", "summarize"]
expected_locality = True  # 应该在同一设备执行
```

### 7.4 负载均衡测试

```python
# 测试用例5：负载均衡
# 连续调用多个工具，验证负载分布
queries = [
    "查询天气1",
    "查询天气2", 
    "查询天气3",
    "查询天气4",
    "查询天气5"
]

# 预期：工具均匀分布到4个设备
expected_distribution = {0: 2, 1: 1, 2: 1, 3: 1}
```

---

## 八、性能优化建议

### 8.1 工具预加载

```python
# 对于高频使用的工具，可以预加载到所有设备
PRELOAD_TOOLS = ['get_weather', 'calculator']

for device_id in range(4):
    for tool_name in PRELOAD_TOOLS:
        tool_manager.load_tool(tool_name, device_id)
```

### 8.2 工具缓存策略

```python
# 使用LRU缓存，保留最近使用的工具
class LRUToolCache:
    def __init__(self, capacity=3):
        self.capacity = capacity
        self.cache = OrderedDict()
    
    def get(self, tool_name):
        if tool_name in self.cache:
            self.cache.move_to_end(tool_name)
            return self.cache[tool_name]
        return None
    
    def put(self, tool_name, tool_handler):
        if tool_name in self.cache:
            self.cache.move_to_end(tool_name)
        else:
            if len(self.cache) >= self.capacity:
                self.cache.popitem(last=False)
            self.cache[tool_name] = tool_handler
```

### 8.3 异步工具加载

```python
import asyncio

async def async_load_tool(tool_manager, tool_name, device_id):
    """异步加载工具，不阻塞推理"""
    await asyncio.to_thread(
        tool_manager.load_tool, 
        tool_name, 
        device_id
    )

# 在推理过程中预测下一个可能使用的工具，提前异步加载
```

---

## 九、扩展方向

### 9.1 支持更多工具类型

- **数据库工具**：查询数据库
- **文件操作工具**：读写文件
- **API调用工具**：调用外部API
- **代码执行工具**：执行Python/JavaScript代码

### 9.2 智能预测

```python
class ToolPredictor:
    """基于历史预测下一个可能使用的工具"""
    
    def __init__(self):
        self.history = []
        self.transition_prob = {}
    
    def predict_next_tool(self, current_tool):
        """预测下一个工具"""
        if current_tool in self.transition_prob:
            return max(
                self.transition_prob[current_tool].items(),
                key=lambda x: x[1]
            )[0]
        return None
```

### 9.3 工具组合

```python
# 定义工具组合（常用的工具序列）
TOOL_COMBINATIONS = {
    'weather_and_advice': ['get_weather', 'clothing_advice'],
    'search_and_summarize': ['web_search', 'summarize'],
    'calculate_and_visualize': ['calculator', 'plot_chart']
}

# 识别到组合模式时，一次性调度整个流水线
```

---

## 十、总结

### 10.1 核心特性

1. **灵活的工具调用**：支持XML/JSON等多种格式
2. **智能调度**：负载均衡、亲和性、依赖感知
3. **并行执行**：多工具同时执行，提高效率
4. **流水线优化**：考虑数据局部性，减少传输
5. **动态管理**：按需加载/卸载，LRU驱逐策略

### 10.2 实现优先级

**P0（必须）：**
- Tool Call解析器
- 工具管理器（注册、加载、执行）
- 简单轮询调度器
- 推理协调器（基础版）
- 2-3个示例工具

**P1（重要）：**
- 负载感知调度
- 并行执行支持
- LRU驱逐策略
- 完整的错误处理

**P2（优化）：**
- 流水线调度
- 工具预加载
- 异步加载
- 智能预测

### 10.3 预期效果

- **功能完整**：支持单工具、并行、流水线调用
- **性能良好**：合理的负载分布，高效的资源利用
- **易于扩展**：新工具只需实现execute函数即可接入
- **稳定可靠**：完善的错误处理和日志记录

---

## 附录：快速开始指南

### A.1 第一周目标

创建基础框架，实现单工具调用：

```bash
# 1. 创建目录结构
cd qwen/distributed_inference
mkdir -p tools/builtin_tools

# 2. 实现核心文件
touch tools/__init__.py
touch tools/tool_parser.py
touch tools/tool_manager.py
touch tools/tool_scheduler.py
touch coordinator.py

# 3. 实现第一个工具
touch tools/builtin_tools/weather_tool.py

# 4. 测试
python run_with_tools.py
```

### A.2 示例工具模板

```python
# tools/builtin_tools/your_tool.py

def execute(**kwargs):
    """
    工具执行函数
    
    Args:
        **kwargs: 工具参数
    
    Returns:
        dict: 执行结果
    """
    try:
        # 实现你的工具逻辑
        result = do_something(kwargs)
        return result
    except Exception as e:
        raise RuntimeError(f"Tool execution failed: {e}")

# 工具元数据（可选）
TOOL_METADATA = {
    'name': 'your_tool',
    'description': '工具描述',
    'parameters': {
        'param1': {'type': 'string', 'required': True},
        'param2': {'type': 'int', 'required': False, 'default': 0}
    }
}
```

这个方案现在更加具体和可操作，直接针对你的4设备Qwen分布式推理场景！

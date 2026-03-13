"""
Qwen Agent 配置文件
用于连接VLLM部署的Qwen模型
"""

# VLLM服务配置
VLLM_CONFIG = {
    'base_url': 'http://localhost:8000/v1',
    'api_key': 'EMPTY',  # VLLM不需要真实的API key
}

# Qwen Agent LLM配置
LLM_CONFIG = {
    'model': 'qwen2.5-7b-instruct',
    'model_server': 'http://localhost:8000/v1',
    'api_key': 'EMPTY',
}

# 生成参数
GENERATION_CONFIG = {
    'temperature': 0.7,
    'top_p': 0.8,
    'top_k': 20,
    'max_tokens': 2048,
    'repetition_penalty': 1.05,
}

# 工具定义示例
SAMPLE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "获取指定城市的天气信息",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "城市名称，如：北京、上海、广州"
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "温度单位"
                    }
                },
                "required": ["city"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "在网络上搜索信息",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "搜索关键词"
                    },
                    "num_results": {
                        "type": "integer",
                        "description": "返回结果数量",
                        "default": 5
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "执行数学计算",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "数学表达式，如：2+2, sqrt(16), sin(3.14)"
                    }
                },
                "required": ["expression"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "获取当前时间",
            "parameters": {
                "type": "object",
                "properties": {
                    "timezone": {
                        "type": "string",
                        "description": "时区，如：Asia/Shanghai, UTC, America/New_York"
                    }
                },
                "required": []
            }
        }
    }
]

# MCP工具定义示例（用于MCP协议测试）
MCP_TOOLS = [
    {
        "name": "read_file",
        "description": "读取指定路径的文件内容",
        "inputSchema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "文件路径"
                }
            },
            "required": ["path"]
        }
    },
    {
        "name": "write_file",
        "description": "将内容写入指定路径的文件",
        "inputSchema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "文件路径"
                },
                "content": {
                    "type": "string",
                    "description": "要写入的内容"
                }
            },
            "required": ["path", "content"]
        }
    },
    {
        "name": "execute_command",
        "description": "执行系统命令",
        "inputSchema": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "要执行的命令"
                },
                "working_directory": {
                    "type": "string",
                    "description": "工作目录"
                }
            },
            "required": ["command"]
        }
    }
]

# 测试用例配置
TEST_CASES = {
    "tool_calling": [
        {
            "name": "天气查询",
            "user_message": "北京今天天气怎么样？",
            "expected_tool": "get_weather",
            "expected_params": {"city": "北京"}
        },
        {
            "name": "网络搜索",
            "user_message": "帮我搜索一下Python教程",
            "expected_tool": "search_web",
            "expected_params": {"query": "Python教程"}
        },
        {
            "name": "数学计算",
            "user_message": "计算一下 123 * 456 等于多少",
            "expected_tool": "calculate",
            "expected_params": {"expression": "123 * 456"}
        },
        {
            "name": "时间查询",
            "user_message": "现在几点了？",
            "expected_tool": "get_current_time",
            "expected_params": {}
        }
    ],
    "instruction_following": [
        {
            "name": "JSON格式输出",
            "instruction": "请以JSON格式输出以下信息：姓名张三，年龄25，城市北京",
            "expected_format": "json"
        },
        {
            "name": "列表格式输出",
            "instruction": "请列出5种常见的编程语言，每行一个",
            "expected_format": "list"
        },
        {
            "name": "结构化输出",
            "instruction": "请用Markdown表格格式输出：苹果-红色-5元，香蕉-黄色-3元，橙子-橙色-4元",
            "expected_format": "markdown_table"
        }
    ]
}

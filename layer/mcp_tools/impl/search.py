from mcp_tools.base import MCPTool


class SearchTool(MCPTool):
    name = "search"
    description = "search(query: string) - 搜索外部信息"
    schema = {
        "type": "object",
        "properties": {
            "query": {"type": "string"}
        },
        "required": ["query"]
    }

    def _load(self):
        # 示例：建立连接、加载索引等
        self._conn = "dummy_connection"

    def _run(self, query: str):
        # 示例：真实系统中可替换为搜索引擎
        return f"搜索结果：{query} → 巴黎是法国首都"

    def _unload(self):
        self._conn = None
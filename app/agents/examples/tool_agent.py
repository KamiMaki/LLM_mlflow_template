"""Google ADK 工具 Agent 範例。

示範如何建立帶有工具的 ADK Agent，使用 llm_service 的 LiteLlm model。

Usage:
    python -m app.agents.examples.tool_agent
"""

from app.agents import create_agent, run_agent_sync


def search_knowledge_base(query: str) -> dict:
    """搜尋內部知識庫。"""
    return {"results": [f"Document about {query}"]}


agent = create_agent(
    name="knowledge_agent",
    instruction="你是一個知識庫助手，使用工具搜尋並回答問題。",
    tools=[search_knowledge_base],
)

if __name__ == "__main__":
    result = run_agent_sync(agent, "什麼是我們的退款政策？")
    print(result)

"""Google ADK Agent 基礎模組。

使用 llm_service factory 取得 LiteLlm model，建立 Google ADK Agent。

Usage:
    from app.agents import create_agent, run_agent_sync

    agent = create_agent(
        name="helper",
        instruction="回答用戶問題",
        tools=[my_tool],
    )
    result = run_agent_sync(agent, "今天天氣如何？")
"""

from __future__ import annotations

import asyncio
from typing import Any

from google.adk.agents import Agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from llm_service import get_adk_model


def create_agent(
    name: str,
    instruction: str,
    tools: list | None = None,
    model: Any | None = None,
    **llm_overrides: Any,
) -> Agent:
    """建立 Google ADK Agent，自動注入 llm_service 的 LiteLlm model。

    Args:
        name: Agent 名稱。
        instruction: Agent 指示（system prompt）。
        tools: 工具函式列表。
        model: 自訂 model，None 時自動從 factory 取得。
        **llm_overrides: 傳給 get_adk_model 的覆寫參數。

    Returns:
        Google ADK Agent 實例。
    """
    if model is None:
        model = get_adk_model(**llm_overrides)

    return Agent(
        name=name,
        model=model,
        instruction=instruction,
        tools=tools or [],
    )


async def run_agent(
    agent: Agent,
    query: str,
    user_id: str = "user_1",
    session_id: str = "session_1",
) -> str:
    """非同步執行 agent 並回傳最終結果。

    Args:
        agent: Google ADK Agent。
        query: 使用者查詢。
        user_id: 使用者 ID。
        session_id: Session ID。

    Returns:
        Agent 的最終回應文字。
    """
    session_service = InMemorySessionService()
    runner = Runner(
        agent=agent,
        app_name=agent.name,
        session_service=session_service,
    )

    content = types.Content(
        role="user",
        parts=[types.Part(text=query)],
    )

    final = "No response."
    async for event in runner.run_async(
        user_id=user_id,
        session_id=session_id,
        new_message=content,
    ):
        if event.is_final_response() and event.content and event.content.parts:
            final = event.content.parts[0].text
            break

    return final


def run_agent_sync(
    agent: Agent,
    query: str,
    **kwargs: Any,
) -> str:
    """同步執行 agent（封裝 asyncio.run）。"""
    return asyncio.run(run_agent(agent, query, **kwargs))

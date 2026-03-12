"""Google ADK Agent 基礎模組。

使用 LLMService 的 config 建立 Google ADK Agent。

Usage:
    from app.agents import create_agent, run_agent_sync
    from llm_service import LLMService

    service = LLMService()
    agent = create_agent(
        name="helper",
        instruction="回答用戶問題",
        service=service,
    )
    result = run_agent_sync(agent, "今天天氣如何？")
"""

from __future__ import annotations

import asyncio
from typing import Any, TYPE_CHECKING

from google.adk.agents import Agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

if TYPE_CHECKING:
    from llm_service import LLMService


def _build_adk_model(service: LLMService) -> Any:
    """從 LLMService 的當前 model config 建立 ADK LiteLlm model。"""
    from google.adk.models.lite_llm import LiteLlm

    resolved = service._resolve()
    model_name = resolved.model_name
    if "/" not in model_name:
        model_name = f"openai/{model_name}"

    headers = {**resolved.extra_headers}
    if resolved.api_key:
        headers["Authorization"] = f"Bearer {resolved.api_key}"

    return LiteLlm(model=model_name, api_base=resolved.api_base, extra_headers=headers)


def create_agent(
    name: str,
    instruction: str,
    tools: list | None = None,
    model: Any | None = None,
    service: LLMService | None = None,
) -> Agent:
    """建立 Google ADK Agent。

    Args:
        name: Agent 名稱。
        instruction: Agent 指示（system prompt）。
        tools: 工具函式列表。
        model: 自訂 model（LiteLlm），優先於 service。
        service: LLMService 實例，用於自動建立 model。
    """
    if model is None:
        if service is None:
            from llm_service import LLMService
            service = LLMService()
        model = _build_adk_model(service)

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
    """非同步執行 agent 並回傳最終結果。"""
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

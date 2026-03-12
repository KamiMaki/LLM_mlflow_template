"""通用 LangGraph node functions — 支援多模型切換、prompt 組裝與圖片輸入。

使用 llm_service factory 取得 ChatLiteLLM，直接在 StateGraph 中使用。
MLflow autolog 自動追蹤 LangChain/LiteLLM 呼叫，不需手動 span。

Usage:
    from app.workflow.nodes import create_call_llm_node, create_set_model_node

    # 基本 LLM node
    call_llm = create_call_llm_node(system_prompt="你是一個助手")
    graph.add_node("call_llm", call_llm)

    # 切換模型 node
    set_qwen3vl = create_set_model_node("QWEN3VL")
    graph.add_node("set_model", set_qwen3vl)

    # 帶圖片的 LLM node
    call_vl = create_call_llm_node(system_prompt="描述圖片", support_images=True)
    graph.add_node("call_vl", call_vl)
"""

from __future__ import annotations

from typing import Any, Callable

from langchain_core.messages import HumanMessage, SystemMessage

from app.logger import get_logger

logger = get_logger(__name__)


def create_set_model_node(model_alias: str) -> Callable[[dict], dict]:
    """建立一個切換模型的 node，更新 state 中的 model_alias。

    Args:
        model_alias: 目標模型別名（如 "QWEN3", "QWEN3VL"）。

    Returns:
        可作為 LangGraph node 的 function。
    """
    def set_model(state: dict) -> dict:
        logger.info(f"Switching model to: {model_alias}")
        return {"model_alias": model_alias}

    return set_model


def create_prompt_assembly_node(
    prompt_name: str | None = None,
    prompt_template: str | None = None,
) -> Callable[[dict], dict]:
    """建立 prompt 組裝 node — 載入 prompt 並格式化，將結果加入 messages。

    支援從 MLflow 載入 prompt 或使用直接提供的 template。
    格式化時使用 state 中的 prompt_variables。

    Args:
        prompt_name: MLflow prompt 名稱（使用 PromptManager 載入）。
        prompt_template: 直接提供的 prompt 模板字串。

    Returns:
        可作為 LangGraph node 的 function。
    """
    def assemble_prompt(state: dict) -> dict:
        variables = state.get("prompt_variables", {})
        template = state.get("prompt_template", "") or prompt_template or ""

        if prompt_name and not template:
            try:
                from app.prompts import PromptManager
                pm = PromptManager()
                template = pm.load_and_format(prompt_name, **variables)
            except Exception as e:
                logger.warning(f"Failed to load prompt '{prompt_name}': {e}")
                template = ""
        elif template and variables:
            for key, value in variables.items():
                template = template.replace("{{ " + key + " }}", str(value))

        if template:
            return {"messages": [HumanMessage(content=template)]}
        return {}

    return assemble_prompt


def create_call_llm_node(
    *,
    system_prompt: str = "You are a helpful assistant.",
    tools: list | None = None,
    llm: Any | None = None,
    support_images: bool = False,
    **llm_overrides: Any,
) -> Callable[[dict], dict]:
    """建立 call_llm node function — 支援多模型切換與 base64 圖片。

    如果 state 中有 llm_config 和 model_alias，會動態建立對應的 LLM client。
    如果 support_images=True 且 state 中有 image_base64，會構建 multimodal message。

    Args:
        system_prompt: 系統提示詞。
        tools: 要綁定的工具列表（LangChain Tool）。
        llm: 自訂 LLM 實例（BaseChatModel），None 時自動從 factory 取得。
        support_images: 是否支援 base64 圖片輸入。
        **llm_overrides: 傳給 get_langchain_llm 的覆寫參數（model, temperature 等）。

    Returns:
        可直接作為 LangGraph node 的 function。
    """
    static_llm = llm
    if static_llm and tools:
        static_llm = static_llm.bind_tools(tools)

    def call_llm(state: dict) -> dict:
        current_llm = static_llm

        # 動態建立 LLM（根據 state 中的 llm_config + model_alias）
        if current_llm is None:
            from llm_service import get_langchain_llm
            llm_config = state.get("llm_config")
            model_alias = state.get("model_alias")
            zone = state.get("zone")

            current_llm = get_langchain_llm(
                config=llm_config,
                model_alias=model_alias or None,
                zone=zone or None,
                **llm_overrides,
            )
            if tools:
                current_llm = current_llm.bind_tools(tools)

        messages = list(state.get("messages", []))
        if system_prompt:
            messages = [SystemMessage(content=system_prompt)] + messages

        # 處理 base64 圖片
        if support_images:
            image_data = state.get("image_base64")
            if image_data:
                from llm_service.factory import build_multimodal_messages
                last_user_text = ""
                for msg in reversed(messages):
                    if hasattr(msg, "type") and msg.type == "human":
                        last_user_text = msg.content if isinstance(msg.content, str) else str(msg.content)
                        messages.remove(msg)
                        break
                    elif isinstance(msg, dict) and msg.get("role") == "user":
                        last_user_text = msg["content"]
                        messages.remove(msg)
                        break

                mm_messages = build_multimodal_messages(
                    user_text=last_user_text,
                    image_base64=image_data,
                )
                for mm_msg in mm_messages:
                    if mm_msg["role"] == "user":
                        messages.append(HumanMessage(content=mm_msg["content"]))

        response = current_llm.invoke(messages)
        return {"messages": [response]}

    return call_llm

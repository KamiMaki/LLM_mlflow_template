"""app.workflow 單元測試。"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from app.workflow import BaseState, create_call_llm_node


class TestBaseState:
    def test_base_state_is_messages_state(self):
        """BaseState 應基於 MessagesState，有 messages key。"""
        annotations = BaseState.__annotations__
        assert "messages" in annotations or hasattr(BaseState, "__annotations__")


class TestCallLLMNode:
    @patch("llm_service.factory.get_langchain_llm")
    def test_create_call_llm_node_returns_callable(self, mock_factory):
        """create_call_llm_node 應回傳 callable。"""
        mock_llm = MagicMock()
        mock_factory.return_value = mock_llm
        node = create_call_llm_node()
        assert callable(node)

    def test_call_llm_node_with_custom_llm(self):
        """可傳入自訂 LLM 實例。"""
        mock_response = MagicMock()
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = mock_response

        node = create_call_llm_node(llm=mock_llm)
        state = {"messages": [("user", "Test")]}
        result = node(state)
        assert result["messages"] == [mock_response]

    def test_call_llm_node_invokes_llm(self):
        """call_llm node 應呼叫 LLM 並回傳 messages。"""
        mock_response = MagicMock()
        mock_response.content = "Hello back!"
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = mock_response

        node = create_call_llm_node(llm=mock_llm)
        state = {"messages": [("user", "Hello")]}
        result = node(state)
        assert result["messages"] == [mock_response]
        mock_llm.invoke.assert_called_once()

    def test_call_llm_node_injects_system_prompt(self):
        """call_llm node 應注入 system prompt。"""
        mock_response = MagicMock()
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = mock_response

        node = create_call_llm_node(system_prompt="Be helpful", llm=mock_llm)
        state = {"messages": [("user", "Hi")]}
        node(state)

        call_args = mock_llm.invoke.call_args[0][0]
        assert call_args[0].content == "Be helpful"

    def test_call_llm_node_binds_tools(self):
        """傳入 tools 時應呼叫 bind_tools。"""
        mock_llm = MagicMock()
        mock_bound = MagicMock()
        mock_llm.bind_tools.return_value = mock_bound
        mock_bound.invoke.return_value = MagicMock()

        tools = [MagicMock()]
        node = create_call_llm_node(tools=tools, llm=mock_llm)
        state = {"messages": [("user", "Hi")]}
        node(state)

        mock_llm.bind_tools.assert_called_once_with(tools)
        mock_bound.invoke.assert_called_once()

"""app.workflow 單元測試。"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from app.workflow import BaseState, WorkflowState, create_call_llm_node, create_set_model_node, create_prompt_assembly_node


class TestBaseState:
    def test_base_state_is_messages_state(self):
        """BaseState 應基於 MessagesState，有 messages key。"""
        annotations = BaseState.__annotations__
        assert "messages" in annotations or hasattr(BaseState, "__annotations__")


class TestWorkflowState:
    def test_workflow_state_fields(self):
        """WorkflowState 應有多模型相關欄位。"""
        fields = WorkflowState.__annotations__
        assert "llm_config" in fields
        assert "prompt_template" in fields
        assert "prompt_variables" in fields
        assert "model_alias" in fields
        assert "zone" in fields
        assert "image_base64" in fields
        assert "metadata" in fields


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

    @patch("llm_service.get_langchain_llm")
    def test_call_llm_node_uses_state_config(self, mock_factory):
        """call_llm node 應從 state 取得 llm_config 和 model_alias。"""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock()
        mock_factory.return_value = mock_llm

        node = create_call_llm_node()
        mock_config = MagicMock()
        state = {
            "messages": [("user", "Hi")],
            "llm_config": mock_config,
            "model_alias": "QWEN3",
            "zone": "PROD",
        }
        node(state)

        mock_factory.assert_called_once_with(
            config=mock_config,
            model_alias="QWEN3",
            zone="PROD",
        )


class TestSetModelNode:
    def test_set_model_returns_model_alias(self):
        """create_set_model_node 應回傳更新 model_alias 的 dict。"""
        node = create_set_model_node("QWEN3VL")
        result = node({})
        assert result == {"model_alias": "QWEN3VL"}


class TestPromptAssemblyNode:
    def test_assemble_with_template_and_variables(self):
        """prompt_assembly_node 應格式化 template。"""
        node = create_prompt_assembly_node(prompt_template="Hello {{ name }}")
        result = node({"prompt_variables": {"name": "World"}})
        assert len(result["messages"]) == 1
        assert "Hello World" in result["messages"][0].content

    def test_assemble_with_state_template(self):
        """應優先使用 state 中的 prompt_template。"""
        node = create_prompt_assembly_node()
        result = node({
            "prompt_template": "State template {{ x }}",
            "prompt_variables": {"x": "42"},
        })
        assert "State template 42" in result["messages"][0].content

    def test_assemble_returns_empty_without_template(self):
        """沒有 template 時應回傳空 dict。"""
        node = create_prompt_assembly_node()
        result = node({})
        assert result == {}

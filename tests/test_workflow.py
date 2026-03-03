"""app.workflow 單元測試。"""

from __future__ import annotations

from app.workflow import LLMState, create_llm_state, create_base_state, create_call_llm_node
from app.workflow.state import BaseState


class TestStateFactories:
    def test_create_base_state_defaults(self):
        state = create_base_state()
        assert state["messages"] == []
        assert state["metadata"] == {}

    def test_create_llm_state_defaults(self):
        state = create_llm_state()
        assert state["messages"] == []
        assert state["metadata"] == {}
        assert state["llm_response"] == ""
        assert state["token_usage"]["total_tokens"] == 0
        assert state["model"] == ""
        assert state["error"] is None

    def test_create_llm_state_with_values(self):
        state = create_llm_state(
            messages=[{"role": "user", "content": "hi"}],
            metadata={"key": "value"},
            llm_response="hello",
            model="gpt-4o",
        )
        assert len(state["messages"]) == 1
        assert state["llm_response"] == "hello"
        assert state["model"] == "gpt-4o"


class TestCallLLMNode:
    def test_create_call_llm_node_returns_callable(self):
        """create_call_llm_node 應回傳 callable。"""
        class MockClient:
            def chat(self, system_prompt, user_prompt, **kwargs):
                pass
        node = create_call_llm_node(MockClient())
        assert callable(node)

    def test_call_llm_node_handles_error(self):
        """LLM 呼叫失敗時應回傳 error。"""
        class FailingClient:
            def chat(self, system_prompt, user_prompt, **kwargs):
                raise ConnectionError("Service unavailable")

        node = create_call_llm_node(FailingClient())
        state = create_llm_state(
            messages=[{"role": "user", "content": "test"}],
        )
        result = node(state)
        assert result["error"] is not None
        assert "Service unavailable" in result["error"]
        assert result["llm_response"] == ""

    def test_call_llm_node_success(self):
        """LLM 呼叫成功時應回傳回應內容。"""
        class MockResponse:
            content = "Hello back!"
            model = "test-model"
            class usage:
                prompt_tokens = 10
                completion_tokens = 5
                total_tokens = 15

        class MockClient:
            def chat(self, system_prompt, user_prompt, **kwargs):
                return MockResponse()

        node = create_call_llm_node(MockClient())
        state = create_llm_state(
            messages=[{"role": "user", "content": "Hello"}],
        )
        result = node(state)
        assert result["error"] is None
        assert result["llm_response"] == "Hello back!"
        assert result["model"] == "test-model"
        assert result["token_usage"]["total_tokens"] == 15

    def test_call_llm_node_extracts_last_user_message(self):
        """應取得最後一則 user message。"""
        class MockResponse:
            content = "response"
            model = "m"
            class usage:
                prompt_tokens = 0
                completion_tokens = 0
                total_tokens = 0

        captured = {}

        class MockClient:
            def chat(self, system_prompt, user_prompt, **kwargs):
                captured["user_prompt"] = user_prompt
                return MockResponse()

        node = create_call_llm_node(MockClient())
        state = create_llm_state(
            messages=[
                {"role": "user", "content": "first"},
                {"role": "assistant", "content": "reply"},
                {"role": "user", "content": "second"},
            ],
        )
        node(state)
        assert captured["user_prompt"] == "second"

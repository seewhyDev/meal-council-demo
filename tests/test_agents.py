"""Verify prompt wiring and OpenAI failures without sending network requests."""

from contextlib import ExitStack
import os
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

import httpx
from openai import APIConnectionError, APIStatusError, APITimeoutError, AuthenticationError, RateLimitError

import agents
import prompts


USER_CONTEXT = ("얼큰한 국물", "어제 라면과 돈까스", "1만 5천원 이하", "지치고 배고픔")


class AgentContextTests(unittest.TestCase):
    def test_each_agent_receives_context_and_all_required_predecessors(self):
        opinions = ("경제성의 고유 의견", "영양의 고유 의견", "미식의 고유 의견")
        functions_and_prompts = (
            (agents.run_budget_agent, prompts.BUDGET_AGENT_PROMPT),
            (agents.run_nutrition_agent, prompts.NUTRITION_AGENT_PROMPT),
            (agents.run_gourmet_agent, prompts.GOURMET_AGENT_PROMPT),
            (agents.run_moderator_agent, prompts.MODERATOR_AGENT_PROMPT),
        )
        self.assertEqual(len({prompt for _, prompt in functions_and_prompts}), 4)
        for index, (function, expected_system) in enumerate(functions_and_prompts):
            with self.subTest(agent=function.__name__), patch("agents.call_agent", return_value="추천 결과") as call:
                result = function(*USER_CONTEXT, *opinions[:index])
                self.assertEqual(result, "추천 결과")
                call.assert_called_once()
                invocation = call.call_args
                system = invocation.args[0] if invocation.args else invocation.kwargs["system_prompt"]
                user = invocation.args[1] if len(invocation.args) > 1 else invocation.kwargs["user_prompt"]
                self.assertEqual(system, expected_system)
                for value in (*USER_CONTEXT, *opinions[:index]):
                    self.assertIn(value, user)
                positions = [user.index(opinion) for opinion in opinions[:index]]
                self.assertEqual(positions, sorted(positions))


class OpenAICallTests(unittest.TestCase):
    def setUp(self):
        self.patches = ExitStack()
        self.addCleanup(self.patches.close)
        self.real_get_api_key = agents.get_api_key
        self.patches.enter_context(patch.dict(os.environ, {}, clear=True))
        self.patches.enter_context(patch("agents.dotenv_values", return_value={}))
        self.patches.enter_context(patch("agents.get_api_key", return_value="test-key"))
        self.constructor = self.patches.enter_context(patch("agents.OpenAI"))
        self.client = MagicMock()
        self.constructor.return_value = self.client
        self.client.__enter__.return_value = self.client
        self.create = self.client.chat.completions.create
        self.create.return_value = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="  추천 결과  "), finish_reason="stop")])

    def test_sdk_receives_default_model_and_system_user_messages(self):
        self.assertEqual(agents.call_agent("시스템 역할", "사용자 상황"), "추천 결과")
        self.constructor.assert_called_once()
        self.assertEqual(self.constructor.call_args.kwargs["api_key"], "test-key")
        self.assertEqual(self.constructor.call_args.kwargs["max_retries"], 0)
        self.create.assert_called_once()
        params = self.create.call_args.kwargs
        self.assertEqual(params["model"], "gpt-5-mini")
        self.assertEqual(params["messages"], [{"role": "system", "content": "시스템 역할"}, {"role": "user", "content": "사용자 상황"}])
        self.assertNotIn("max_tokens", params)

    def test_model_can_be_overridden_with_environment(self):
        with patch.dict(os.environ, {"OPENAI_MODEL": "custom-model"}):
            agents.call_agent("역할", "상황")
        self.assertEqual(self.create.call_args.kwargs["model"], "custom-model")

    def test_empty_or_truncated_response_raises_actionable_error(self):
        invalid_responses = [
            SimpleNamespace(choices=[]),
            SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=None), finish_reason="stop")]),
            SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=" \n "), finish_reason="stop")]),
            SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="일부만 생성됨"), finish_reason="length")]),
        ]
        for response in invalid_responses:
            with self.subTest(response=response):
                self.create.return_value = response
                with self.assertRaises(agents.AgentError) as caught:
                    agents.call_agent("역할", "상황")
                self.assertTrue(str(caught.exception).strip())

    def test_sdk_failures_are_actionable_and_do_not_expose_details(self):
        secret = "DO-NOT-DISPLAY-test-secret"
        request = httpx.Request("POST", "https://example.test/v1/chat/completions")
        failures = [
            RuntimeError(secret),
            APIConnectionError(message=secret, request=request),
            APITimeoutError(request=request),
            AuthenticationError(secret, response=httpx.Response(401, request=request), body=None),
            RateLimitError(secret, response=httpx.Response(429, request=request), body=None),
            APIStatusError(secret, response=httpx.Response(500, request=request), body=None),
        ]
        for failure in failures:
            with self.subTest(exception=type(failure).__name__):
                self.create.reset_mock()
                self.create.side_effect = failure
                with self.assertRaises(agents.AgentError) as caught:
                    agents.call_agent("역할", "상황")
                self.assertTrue(str(caught.exception).strip())
                self.assertNotIn(secret, str(caught.exception))
                self.create.assert_called_once()

    def test_missing_key_fails_before_client_creation(self):
        with patch("agents.get_api_key", new=self.real_get_api_key):
            with self.assertRaises(agents.AgentError) as caught:
                agents.call_agent("역할", "상황")
        self.assertIn("OPENAI_API_KEY", str(caught.exception))
        self.constructor.assert_not_called()


if __name__ == "__main__":
    unittest.main()

"""Exercise the real Streamlit chat with mocked, free-of-charge AI calls."""

from contextlib import ExitStack
from copy import deepcopy
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

import agents


APP = Path(__file__).resolve().parents[1] / "app.py"
ANSWERS = ("매콤한 국물", "점심 돈까스, 저녁 라면", "15,000원", "피곤하고 스트레스 받아요")
RESULTS = (
    "예산 안에서 든든합니다. 💰 추천 메뉴: 순두부찌개",
    "경제성 의견에 동의하며 채소를 곁들입니다. 🥗 추천 메뉴: 순두부찌개",
    "앞선 의견에 동의하며 얼큰하게 먹겠습니다. 😋 추천 메뉴: 얼큰 순두부찌개",
    "⚖️ 위원회 최종 의견\n\n예산과 기분에 적합합니다.\n\n🍽️ 최종 추천: 얼큰 순두부찌개\n\n한 줄 추천 이유:\n따뜻하고 든든합니다.",
)
AGENT_NAMES = ("budget", "nutrition", "gourmet", "moderator")


class ChatFlowTests(unittest.TestCase):
    def setUp(self):
        self.patches = ExitStack()
        self.addCleanup(self.patches.close)
        self.real_get_api_key = agents.get_api_key
        self.real_call_agent = agents.call_agent
        self.patches.enter_context(patch("agents.get_api_key", return_value="test-key"))
        self.api = self.patches.enter_context(patch("agents.call_agent", side_effect=RESULTS))
        self.app = AppTest.from_file(str(APP), default_timeout=10).run()
        self.assertFalse(self.app.exception)

    def answer(self, value):
        self.app.chat_input[0].set_value(value).run()
        self.assertFalse(self.app.exception)

    def finish(self):
        for answer in ANSWERS:
            self.answer(answer)

    def test_initial_state_and_questions_advance_one_at_a_time(self):
        state = self.app.session_state
        self.assertEqual(state["stage"], "ask_desired_food")
        self.assertEqual(len(state["messages"]), 1)
        self.assertEqual(state["messages"][0]["role"], "assistant")
        self.assertIn("오늘 어떤 음식", state["messages"][0]["content"])
        self.assertEqual(state["agent_results"], dict.fromkeys(AGENT_NAMES))
        for key in ("desired_food", "yesterday_foods", "budget", "mood"):
            self.assertIsNone(state[key])

        for answer, key, next_stage in zip(
            ANSWERS[:3],
            ("desired_food", "yesterday_foods", "budget"),
            ("ask_yesterday_food", "ask_budget", "ask_mood"),
        ):
            self.answer(answer)
            self.assertEqual(state[key], answer)
            self.assertEqual(state["stage"], next_stage)
            self.assertEqual(state["messages"][-2], {"role": "user", "content": answer})
            self.assertEqual(state["messages"][-1]["role"], "assistant")
        self.api.assert_not_called()

    def test_whitespace_input_does_not_advance_or_add_messages(self):
        before = deepcopy(self.app.session_state["messages"])
        self.answer("   \t  ")
        self.assertEqual(self.app.session_state["stage"], "ask_desired_food")
        self.assertEqual(self.app.session_state["messages"], before)
        self.assertIsNone(self.app.session_state["desired_food"])
        self.api.assert_not_called()

    def test_complete_chat_calls_agents_in_order_with_preceding_opinions(self):
        import streamlit as st

        stages = []

        def respond(*args, **kwargs):
            stages.append(st.session_state["stage"])
            return RESULTS[len(stages) - 1]

        self.api.side_effect = respond
        self.finish()
        self.assertEqual(stages, ["debate"] * 4)
        self.assertEqual(self.app.session_state["stage"], "finished")
        self.assertEqual(self.app.session_state["mood"], ANSWERS[-1])
        self.assertEqual(self.app.session_state["agent_results"], dict(zip(AGENT_NAMES, RESULTS)))
        self.assertEqual(self.api.call_count, 4)
        for index, invocation in enumerate(self.api.call_args_list):
            user_prompt = invocation.args[1] if len(invocation.args) > 1 else invocation.kwargs["user_prompt"]
            for answer in ANSWERS:
                self.assertIn(answer, user_prompt)
            for previous in RESULTS[:index]:
                self.assertIn(previous, user_prompt)

        messages = self.app.session_state["messages"]
        self.assertEqual(len(self.app.chat_message), len(messages))
        self.assertEqual([m["content"] for m in messages if m["role"] == "user"], list(ANSWERS))
        for response in RESULTS:
            self.assertEqual(sum(response in m["content"] for m in messages), 1)
        self.assertTrue(self.app.success)
        self.assertIn("얼큰 순두부찌개", self.app.success[0].value)

    def test_reruns_preserve_chat_without_repeating_completed_calls(self):
        self.finish()
        messages = deepcopy(self.app.session_state["messages"])
        for _ in range(2):
            self.app.run()
            self.assertFalse(self.app.exception)
            self.assertEqual(self.app.session_state["messages"], messages)
        self.assertEqual(self.api.call_count, 4)

    def test_partial_failure_waits_for_retry_and_preserves_successful_agents(self):
        self.api.side_effect = [RESULTS[0], agents.AgentError("일시적인 호출 오류"), *RESULTS[1:]]
        self.finish()
        self.assertEqual(self.app.session_state["stage"], "debate")
        self.assertEqual(self.api.call_count, 2)
        self.assertEqual(self.app.session_state["agent_results"]["budget"], RESULTS[0])
        self.assertIsNone(self.app.session_state["agent_results"]["nutrition"])
        self.assertTrue(self.app.error or self.app.warning)
        messages = deepcopy(self.app.session_state["messages"])
        self.app.run()
        self.assertEqual(self.api.call_count, 2)
        self.assertEqual(self.app.session_state["messages"], messages)

        self.app.button(key="retry_debate").click().run()
        self.assertFalse(self.app.exception)
        self.assertEqual(self.app.session_state["stage"], "finished")
        self.assertEqual(self.api.call_count, 5)
        self.assertEqual(self.app.session_state["agent_results"], dict(zip(AGENT_NAMES, RESULTS)))
        for response in RESULTS:
            self.assertEqual(sum(response in m["content"] for m in self.app.session_state["messages"]), 1)

    def test_restart_clears_all_answers_results_and_history(self):
        self.finish()
        self.app.button(key="restart").click().run()
        self.assertFalse(self.app.exception)
        state = self.app.session_state
        self.assertEqual(state["stage"], "ask_desired_food")
        for key in ("desired_food", "yesterday_foods", "budget", "mood"):
            self.assertIsNone(state[key])
        self.assertEqual(state["agent_results"], dict.fromkeys(AGENT_NAMES))
        self.assertEqual(len(state["messages"]), 1)
        self.assertIn("오늘 어떤 음식", state["messages"][0]["content"])
        self.assertEqual(self.api.call_count, 4)

    def test_missing_key_guidance_and_retry_after_configuration(self):
        with (
            patch.dict(os.environ, {}, clear=True),
            patch("agents.dotenv_values", return_value={}) as config,
            patch("agents.get_api_key", new=self.real_get_api_key),
            patch("agents.call_agent", new=self.real_call_agent),
            patch("agents.OpenAI") as constructor,
        ):
            self.app.run()
            self.assertFalse(self.app.exception)
            notices = [element.value for kind in (self.app.error, self.app.warning, self.app.info, self.app.markdown) for element in kind]
            self.assertTrue(any("OPENAI_API_KEY" in notice for notice in notices))
            self.finish()
            self.assertEqual(self.app.session_state["stage"], "debate")
            constructor.assert_not_called()

            config.return_value = {"OPENAI_API_KEY": "test-key"}
            self.app.run()
            constructor.assert_not_called()
            client = constructor.return_value
            client.__enter__.return_value = client
            client.chat.completions.create.side_effect = [
                SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=result), finish_reason="stop")])
                for result in RESULTS
            ]
            self.app.button(key="retry_debate").click().run()
            self.assertFalse(self.app.exception)
            self.assertEqual(self.app.session_state["stage"], "finished")
            self.assertEqual(client.chat.completions.create.call_count, 4)

    def test_interrupted_request_waits_for_explicit_retry(self):
        import streamlit as st

        def interrupt_nutrition(*args, **kwargs):
            if self.api.call_count == 2:
                st.rerun()
            return RESULTS[0]

        self.api.side_effect = interrupt_nutrition
        self.finish()
        self.assertEqual(self.app.session_state["stage"], "debate")
        self.assertEqual(self.app.session_state["agent_results"]["budget"], RESULTS[0])
        self.assertIsNone(self.app.session_state["agent_results"]["nutrition"])
        self.app.run()
        self.assertFalse(self.app.exception)
        self.assertEqual(self.api.call_count, 2)
        self.assertTrue(self.app.error)

        self.api.side_effect = RESULTS[1:]
        self.app.button(key="retry_debate").click().run()
        self.assertFalse(self.app.exception)
        self.assertEqual(self.app.session_state["stage"], "finished")
        self.assertEqual(self.api.call_count, 5)

    def test_empty_agent_result_does_not_finish_or_call_later_agents(self):
        self.api.side_effect = None
        self.api.return_value = "   "
        self.finish()
        self.assertEqual(self.app.session_state["stage"], "debate")
        self.assertEqual(self.app.session_state["agent_results"], dict.fromkeys(AGENT_NAMES))
        self.assertEqual(self.api.call_count, 1)
        self.assertTrue(self.app.error or self.app.warning)

    def test_unexpected_exception_is_safe_and_does_not_retry_automatically(self):
        self.api.side_effect = RuntimeError("DO-NOT-DISPLAY-test-secret")
        self.finish()
        self.app.run()
        self.assertFalse(self.app.exception)
        self.assertEqual(self.api.call_count, 1)
        self.assertEqual(self.app.session_state["stage"], "debate")
        displayed = "\n".join(element.value for kind in (self.app.error, self.app.warning, self.app.markdown) for element in kind)
        self.assertNotIn("DO-NOT-DISPLAY-test-secret", displayed)
        self.assertTrue(self.app.error or self.app.warning)


if __name__ == "__main__":
    unittest.main()

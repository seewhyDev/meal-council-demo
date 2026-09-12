"""공식 OpenAI SDK로 위원회 전문가를 한 명씩 호출한다."""

import os
from pathlib import Path

from dotenv import dotenv_values
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    OpenAI,
    RateLimitError,
)

from prompts import (
    BUDGET_AGENT_PROMPT,
    GOURMET_AGENT_PROMPT,
    MODERATOR_AGENT_PROMPT,
    NUTRITION_AGENT_PROMPT,
)

ENV_PATH = Path(__file__).resolve().with_name(".env")


class AgentError(Exception):
    """채팅 화면에 그대로 표시할 수 있는 안전한 오류 메시지."""


def _get_setting(name: str, default: str = "") -> str:
    # 매번 읽으면 앱을 재시작하지 않고 .env 수정 후 재시도할 수 있다.
    # 이미 설정된 환경변수가 .env 파일보다 우선한다.
    if name in os.environ:
        return os.environ[name].strip()
    try:
        value = dotenv_values(ENV_PATH).get(name)
    except Exception:
        raise AgentError(".env 파일을 읽을 수 없습니다. 파일 설정을 확인해주세요.") from None
    return value.strip() if value else default


def get_api_key() -> str:
    """필요할 때만 API 키를 확인하여, 키 없이도 앱을 시작할 수 있게 한다."""
    api_key = _get_setting("OPENAI_API_KEY")
    if not api_key or api_key in {"your_api_key", "..."}:
        raise AgentError("OPENAI_API_KEY가 없습니다. .env 파일에 API 키를 설정한 뒤 다시 시도해주세요.")
    return api_key


def call_agent(system_prompt: str, user_prompt: str) -> str:
    """한 번 요청하고 완료된 답변만 반환한다. 재시도는 UI에서 명시적으로 한다."""
    try:
        api_key = get_api_key()
        model = _get_setting("OPENAI_MODEL", "gpt-5-mini") or "gpt-5-mini"
        options = {}
        if model == "gpt-5-mini" or model.startswith("gpt-5-mini-"):
            # 짧은 메뉴 토론에 맞춰 추론 토큰과 대기 시간을 줄인다.
            options["reasoning_effort"] = "low"

        with OpenAI(api_key=api_key, max_retries=0, timeout=60.0) as client:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_completion_tokens=4096,
                **options,
            )

        if not response.choices:
            raise AgentError("전문가가 빈 응답을 반환했습니다. 다시 시도해주세요.")

        choice = response.choices[0]
        if choice.finish_reason == "length":
            raise AgentError("전문가의 답변이 길이 제한으로 중단되었습니다. 다시 시도해주세요.")
        content = choice.message.content
        if not isinstance(content, str) or not content.strip():
            raise AgentError("전문가가 빈 응답을 반환했습니다. 다시 시도해주세요.")
        return content.strip()
    except AgentError:
        raise
    except AuthenticationError:
        raise AgentError("API 인증에 실패했습니다. OPENAI_API_KEY 설정을 확인해주세요.") from None
    except RateLimitError:
        raise AgentError("API 사용 한도에 도달했습니다. 사용량과 결제 설정을 확인하거나 잠시 후 다시 시도해주세요.") from None
    except APITimeoutError:
        raise AgentError("응답 대기 시간이 초과되었습니다. 잠시 후 다시 시도해주세요.") from None
    except APIConnectionError:
        raise AgentError("OpenAI에 연결할 수 없습니다. 네트워크를 확인한 뒤 다시 시도해주세요.") from None
    except APIStatusError:
        raise AgentError("OpenAI API 요청에 실패했습니다. 모델 설정을 확인하거나 잠시 후 다시 시도해주세요.") from None
    except Exception:
        # SDK 오류 원문에는 요청 정보 등이 포함될 수 있으므로 노출하지 않는다.
        raise AgentError("전문가 호출 중 예상하지 못한 오류가 발생했습니다. 잠시 후 다시 시도해주세요.") from None


def _user_context(
    desired_food: str, yesterday_foods: str, budget: str, mood: str
) -> str:
    return f"""오늘 먹고 싶은 음식:
{desired_food}

어제 먹은 음식:
{yesterday_foods}

한 끼 예산:
{budget}

현재 기분:
{mood}"""


def run_budget_agent(
    desired_food: str, yesterday_foods: str, budget: str, mood: str
) -> str:
    user_prompt = _user_context(desired_food, yesterday_foods, budget, mood)
    user_prompt += "\n\n위 정보를 바탕으로 경제성 전문가로서 메뉴를 추천하라."
    return call_agent(BUDGET_AGENT_PROMPT, user_prompt)


def run_nutrition_agent(
    desired_food: str,
    yesterday_foods: str,
    budget: str,
    mood: str,
    budget_response: str,
) -> str:
    user_prompt = _user_context(desired_food, yesterday_foods, budget, mood)
    user_prompt += f"""

[경제성 전문가 의견]
{budget_response}

경제성 전문가 의견을 검토한 후
영양 관점에서 동의 또는 반박하면서 메뉴를 추천하라."""
    return call_agent(NUTRITION_AGENT_PROMPT, user_prompt)


def run_gourmet_agent(
    desired_food: str,
    yesterday_foods: str,
    budget: str,
    mood: str,
    budget_response: str,
    nutrition_response: str,
) -> str:
    user_prompt = _user_context(desired_food, yesterday_foods, budget, mood)
    user_prompt += f"""

[경제성 전문가]
{budget_response}

[영양 전문가]
{nutrition_response}

두 전문가의 의견을 검토한 후
미식과 현재 만족도 관점에서 자신의 의견을 제시하라."""
    return call_agent(GOURMET_AGENT_PROMPT, user_prompt)


def run_moderator_agent(
    desired_food: str,
    yesterday_foods: str,
    budget: str,
    mood: str,
    budget_response: str,
    nutrition_response: str,
    gourmet_response: str,
) -> str:
    user_prompt = _user_context(desired_food, yesterday_foods, budget, mood)
    user_prompt += f"""

[경제성 전문가]
{budget_response}

[영양 전문가]
{nutrition_response}

[미식 전문가]
{gourmet_response}

모든 의견을 종합하여
사용자에게 가장 적절한 최종 메뉴 하나를 결정하라."""
    return call_agent(MODERATOR_AGENT_PROMPT, user_prompt)

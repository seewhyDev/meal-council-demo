"""오늘 뭐 먹지 AI 위원회 — Streamlit 채팅과 순차 토론."""

import streamlit as st

import agents


QUESTIONS = {
    "ask_desired_food": {
        "field": "desired_food",
        "next": "ask_yesterday_food",
        "text": """오늘 어떤 음식이 땡기시나요? 😋

정확한 메뉴가 아니어도 괜찮습니다.

예:
- 매콤한 음식
- 따뜻한 국물
- 고기
- 면 요리
- 담백한 음식
- 아무거나""",
    },
    "ask_yesterday_food": {
        "field": "yesterday_foods",
        "next": "ask_budget",
        "text": """어제는 어떤 음식을 드셨나요? 🍱

하나 이상 자유롭게 입력해주세요.

예: 점심 제육볶음, 저녁 라면""",
    },
    "ask_budget": {
        "field": "budget",
        "next": "ask_mood",
        "text": """이번 한 끼에 어느 정도까지 쓰고 싶으신가요? 💸

예:
- 8,000원
- 15,000원 정도
- 2만원 이하
- 가격은 상관없음""",
    },
    "ask_mood": {
        "field": "mood",
        "next": "debate",
        "text": """지금 기분이나 상태는 어떤가요? 🙂

예:
- 피곤함
- 스트레스 받음
- 기분 좋음
- 우울함
- 허기짐
- 술 마신 다음 날
- 그냥 평범함""",
    },
}

DEBATE_INTRO = """좋습니다. 위원회를 소집하겠습니다. 🧑‍⚖️

오늘의 안건:  
"무엇을 먹어야 가장 만족스러운가?"

💰 경제성 전문가  
🥗 영양 전문가  
😋 미식 전문가  
⚖️ 중재자

회의를 시작합니다."""

AGENT_DETAILS = {
    "budget": ("💰", "경제성 전문가", "지갑 사정을 검토하고 있습니다..."),
    "nutrition": ("🥗", "영양 전문가", "어제 식단을 분석하고 있습니다..."),
    "gourmet": ("😋", "미식 전문가", "앞선 의견에 불만이 있는 것 같습니다..."),
    "moderator": ("⚖️", "중재자", "위원회의 의견을 정리하고 있습니다..."),
}


def reset_session() -> None:
    """이 앱의 대화 데이터만 초기화한다. 버튼 콜백으로도 사용한다."""
    st.session_state.update(
        stage="ask_desired_food",
        messages=[{"role": "assistant", "content": QUESTIONS["ask_desired_food"]["text"]}],
        desired_food=None,
        yesterday_foods=None,
        budget=None,
        mood=None,
        agent_results={key: None for key in AGENT_DETAILS},
        debate_error=None,
        pending_agent=None,
    )
    st.session_state.pop("chat_input", None)


def render_message(message: dict) -> None:
    agent_key = message.get("agent")
    if agent_key:
        avatar, name, _ = AGENT_DETAILS[agent_key]
    elif message["role"] == "user":
        avatar, name = "👤", "사용자"
    else:
        avatar, name = "🍽️", "AI 위원회"

    with st.chat_message(message["role"], avatar=avatar):
        if agent_key:
            st.markdown(f"**{avatar} {name}**")
        if agent_key == "moderator":
            st.success(message["content"])
        else:
            st.markdown(message["content"])


def retry_debate() -> None:
    """성공한 결과는 유지하고 실패한 요청만 다시 허용한다."""
    st.session_state.debate_error = None
    st.session_state.pending_agent = None


def render_debate_error() -> None:
    error = st.session_state.debate_error
    _, name, _ = AGENT_DETAILS[error["agent"]]
    st.error(f"⚠️ {name}를 호출하는 중 오류가 발생했습니다.\n\n{error['message']}")
    st.button("🔁 실패한 전문가 다시 시도", key="retry_debate", on_click=retry_debate)


def run_debate() -> None:
    # session_state 접근도 rerun 지점이므로 결과와 메시지 참조를 미리 확보한다.
    results = st.session_state.agent_results
    messages = st.session_state.messages

    # 요청 도중 스크립트가 중단된 경우에도 자동으로 같은 요청을 보내지 않는다.
    pending = st.session_state.pending_agent
    if pending is not None:
        if results[pending] is None and st.session_state.debate_error is None:
            st.session_state.debate_error = {
                "agent": pending,
                "message": "이전 요청이 중단되어 응답을 확인하지 못했습니다. 다시 시도해주세요.",
            }
        st.session_state.pending_agent = None

    if st.session_state.debate_error is not None:
        render_debate_error()
        return

    context = {
        field: st.session_state[field]
        for field in ("desired_food", "yesterday_foods", "budget", "mood")
    }
    # dict의 삽입 순서대로 호출하고, 매번 앞서 성공한 의견을 전달한다.
    runners = {
        "budget": agents.run_budget_agent,
        "nutrition": agents.run_nutrition_agent,
        "gourmet": agents.run_gourmet_agent,
        "moderator": agents.run_moderator_agent,
    }
    previous_opinions = {}
    for key, runner in runners.items():
        if results[key] is None:
            avatar, name, activity = AGENT_DETAILS[key]
            st.session_state.pending_agent = key
            with st.spinner(f"{avatar} {name}가 {activity}"):
                try:
                    response = runner(**context, **previous_opinions)
                    if not isinstance(response, str) or not response.strip():
                        raise agents.AgentError("빈 응답을 받았습니다. 잠시 후 다시 시도해주세요.")
                except agents.AgentError as exc:
                    st.session_state.debate_error = {"agent": key, "message": str(exc)}
                except Exception:
                    # 예외 원문에는 API 키나 사용자 정보가 포함될 수 있다.
                    st.session_state.debate_error = {
                        "agent": key,
                        "message": "예상하지 못한 오류가 발생했습니다. 잠시 후 다시 시도해주세요.",
                    }
                else:
                    # 다음 Streamlit 출력 전에 저장해야 rerun에서도 재호출하지 않는다.
                    results[key] = response.strip()
                    message = {"role": "assistant", "agent": key, "content": results[key]}
                    messages.append(message)
                st.session_state.pending_agent = None

            if st.session_state.debate_error is not None:
                render_debate_error()
                return
            render_message(message)

        previous_opinions[f"{key}_response"] = results[key]

    st.session_state.stage = "finished"


def main() -> None:
    st.set_page_config(page_title="오늘 뭐 먹지 AI 위원회", page_icon="🍽️", layout="centered")
    if "stage" not in st.session_state:
        reset_session()

    st.title("🍽️ 오늘 뭐 먹지 AI 위원회")
    st.caption("사소한 메뉴 선택을 AI 전문가들의 진지한 토론으로 해결합니다.")

    try:
        agents.get_api_key()
    except agents.AgentError as exc:
        st.warning(str(exc))

    for message in st.session_state.messages:
        render_message(message)

    stage = st.session_state.stage
    if stage in QUESTIONS:
        answer = st.chat_input("편하게 답변해주세요.", key="chat_input")
        if answer is not None:
            answer = answer.strip()
            if not answer:
                st.warning("내용을 입력해주세요. 공백만으로는 다음 질문으로 넘어갈 수 없습니다.")
                return
            question = QUESTIONS[stage]
            st.session_state[question["field"]] = answer
            st.session_state.messages.append({"role": "user", "content": answer})
            next_stage = question["next"]
            st.session_state.stage = next_stage
            next_message = DEBATE_INTRO if next_stage == "debate" else QUESTIONS[next_stage]["text"]
            st.session_state.messages.append({"role": "assistant", "content": next_message})
            st.rerun()

    if st.session_state.stage == "debate":
        st.chat_input("위원회가 의견을 나누고 있습니다.", disabled=True, key="chat_input")
        run_debate()

    if st.session_state.stage == "finished":
        st.button("🔄 다시 추천받기", key="restart", on_click=reset_session, type="primary")


if __name__ == "__main__":
    main()

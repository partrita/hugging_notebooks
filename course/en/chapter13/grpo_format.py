import marimo

__generated_with = "0.10.6"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo

    mo.md(
        """
        ## 구조 형식 보상 함수
        
        이 예제는 완료가 특정 구조 형식을 따르는지 여부를 평가하는 보상 함수를 보여줍니다.
        `<think>...</think><answer>...</answer>` 또는 `<code>...</code><explanation>...</explanation>` 형식입니다.
        
        버튼을 사용하여 보상할 구조 형식을 선택합니다.
        """
    )
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    format_buttons = mo.ui.radio(
        options=["think-answer", "code-explanation"],
        value="think-answer",
        label="보상할 형식",
    )
    format_buttons
    return (format_buttons,)


@app.cell(hide_code=True)
def _(mo, format_buttons):
    import plotly.express as px
    import re

    # 다양한 형식의 샘플 완료
    completions = [
        # 생각-답변 형식 예제
        "<think>단계별로 해결해 보겠습니다</think><answer>42</answer>",
        "답은 특별한 형식 없이 15입니다",
        "<code>print('Hello world')</code><explanation>인사말을 인쇄합니다</explanation>",
        # 코드-설명 형식 예제
        "<code>def add(a, b): return a + b</code><explanation>숫자를 더하는 함수</explanation>",
        "<code>for i in range(10): print(i)</code>",
        "<think>루프를 사용해야 합니다</think><code>while True: pass</code>",
    ]

    # 표시에 사용할 축약 버전 만들기
    short_completions = [c[:30] + "..." if len(c) > 30 else c for c in completions]

    def format_reward(completions, format_type="think-answer", **kwargs):
        """
        원하는 형식 구조를 따르는 완료에 보상합니다.

        Args:
            completions: 평가할 완료 목록
            format_type: 보상할 형식 구조

        Returns:
            보상 및 세부 정보 목록
        """
        # 다양한 형식에 대한 패턴 정의
        patterns = {
            "think-answer": r"<think>.*?</think>\s*<answer>.*?</answer>",
            "code-explanation": r"<code>.*?</code>\s*<explanation>.*?</explanation>",
        }

        # format_type에 따라 패턴 선택
        pattern = patterns.get(format_type, patterns["think-answer"])

        rewards = []
        details = []
        categories = []

        for completion in completions:
            match = re.search(pattern, completion, re.DOTALL)
            if match:
                # 정확한 형식에 대한 전체 일치
                rewards.append(1.0)
                details.append(f"올바른 {format_type} 형식")
                categories.append("정확한 형식 일치")
            elif f"<{format_type.split('-')[0]}>" in completion:
                # 부분 일치 - 형식의 여는 태그가 있음
                rewards.append(0.5)
                details.append(f"{format_type.split('-')[0]} 태그가 있지만 불완전합니다.")
                categories.append("부분 형식 일치")
            elif any(f"<{tag}>" in completion for tag in format_type.split("-")):
                # 필요한 태그 중 하나 이상을 포함합니다.
                rewards.append(0.2)
                details.append("일부 필수 태그가 있지만 형식이 잘못되었습니다.")
                categories.append("일부 태그 있음")
            else:
                # 전혀 일치하지 않음
                rewards.append(0.0)
                details.append("잘못된 형식")
                categories.append("형식 일치 없음")

        return rewards, details, categories

    # 보상 계산
    rewards, details, categories = format_reward(
        completions=completions, format_type=format_buttons.value
    )

    # 결과 표시
    results = []
    for completion, reward, detail, category in zip(
        short_completions, rewards, details, categories
    ):
        results.append(
            {
                "Completion": completion,
                "Reward": reward,
                "Detail": detail,
                "Category": category,
            }
        )

    # 테이블 보기 만들기
    mo.md(f"### {format_buttons.value} 형식에 대한 결과")
    mo.ui.table(results)

    # 완료별 보상을 비교하는 막대 차트 만들기
    fig = px.bar(
        results,
        x="Completion",
        y="Reward",
        color="Category",
        title=f"완료별 형식 보상 ({format_buttons.value})",
        hover_data=["Detail"],
    )
    mo.ui.plotly(fig)


if __name__ == "__main__":
    app.run()

import marimo

__generated_with = "0.10.6"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo

    mo.md(
        """
        ## 수학 문제 보상 함수
        
        이 예제는 검증 가능한 답변이 있는 수학 문제에 대한 보상 함수를 보여줍니다.
        
        슬라이더는 근사값에 대한 허용 오차를 제어합니다.
        """
    )
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    tolerance_slider = mo.ui.slider(
        start=0, stop=25, step=5, value=0, label="허용 오차"
    )
    tolerance_slider
    return (tolerance_slider,)


@app.cell(hide_code=True)
def _(mo, tolerance_slider):
    import plotly.express as px

    # 샘플 수학 문제와 정답
    problems = [
        "5 + 7은 무엇입니까?",
        "12 * 6을 계산하십시오.",
        "100 / 4는 무엇입니까?",
        "x에 대해 푸십시오: 3x = 15",
        "81의 제곱근은 무엇입니까?",
    ]

    # 정답
    correct_answers = [12, 72, 25, 5, 9]

    # 모델 완료 (시뮬레이션)
    model_completions = [
        12,  # 정답
        92,  # 오답
        15,  # 오답
        0,  # 오답
        9,  # 정답
    ]

    def extract_final_answer(completion):
        """
        실제 시나리오에서는 완료를 구문 분석하여 답변을 추출합니다.
        이 예에서는 직접 정수 완료를 사용하고 있습니다.
        """
        return completion

    def problem_reward(completions, answers, tolerance=0):
        """
        검증 가능한 답변이 있는 수학 문제에 대한 보상 함수

        Args:
            completions: 평가할 완료 목록
            answers: 문제에 대한 정답 목록
            tolerance: 정답에 허용되는 차이

        Returns:
            각 완료에 대한 보상 목록
        """
        rewards = []

        for completion, correct_answer in zip(completions, answers):
            try:
                # 완료에서 답변 추출
                answer = extract_final_answer(completion)

                # 답변이 얼마나 가까운지 계산
                difference = abs(answer - correct_answer)

                # 허용 오차가 있는 이진 보상
                if difference <= tolerance:
                    reward = 1.0
                else:
                    # 답변이 얼마나 가까운지에 따른 부분 점수
                    # 문제 크기에 따라 조정
                    max_diff = max(correct_answer * 0.5, 10)
                    reward = max(0, 1 - (difference / max_diff))

                rewards.append(reward)
            except Exception:
                # 답변을 구문 분석할 수 없는 경우 낮은 보상 제공
                rewards.append(0.0)

        return rewards

    # 보상 계산
    rewards = problem_reward(
        completions=model_completions,
        answers=correct_answers,
        tolerance=tolerance_slider.value,
    )

    # 결과 표시
    results = []
    for problem, correct, completion, reward in zip(
        problems, correct_answers, model_completions, rewards
    ):
        results.append(
            {
                "Problem": problem,
                "Correct Answer": correct,
                "Model Answer": completion,
                "Difference": abs(correct - completion),
                "Reward": reward,
            }
        )

    # 테이블 보기 만들기
    mo.md("### 결과")
    mo.ui.table(results)

    # 막대 차트 만들기
    fig = px.bar(
        results,
        x="Problem",
        y="Reward",
        color="Difference",
        hover_data=["Correct Answer", "Model Answer"],
        title="문제별 보상",
    )
    mo.ui.plotly(fig)


if __name__ == "__main__":
    app.run()

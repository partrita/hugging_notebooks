import marimo

__generated_with = "0.10.6"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo

    mo.md(
        "## 길이 기반 보상\n슬라이더를 조정하여 다양한 완료 길이에 따라 보상이 어떻게 변하는지 확인하세요."
    )
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    slider = mo.ui.slider(start=5, stop=50, step=5, label="이상적인 길이 (문자)")
    slider
    return (slider,)


@app.cell(hide_code=True)
def _(mo, slider):
    import plotly.express as px

    # 길이가 다른 5개의 샘플이 있는 토이 데이터세트
    completions = [
        "짧음",  # 2자
        "중간 길이 텍스트",  # 9자
        "이것은 약 20자입니다",  # 13자
        "이것은 약간 더 긴 완료입니다",  # 17자
        "이것은 단어가 더 많은 훨씬 긴 완료입니다",  # 26자
    ]

    maximum_length = max(len(completion) for completion in completions)
    minimum_length = min(len(completion) for completion in completions)

    def length_reward(completions, ideal_length):
        """
        완료 길이를 기준으로 보상을 계산합니다.

        Args:
            completions: 텍스트 완료 목록
            ideal_length: 목표 길이 (문자)

        Returns:
            각 완료에 대한 보상 점수 목록
        """
        rewards = []

        for completion in completions:
            length = len(completion)
            # 간단한 보상 함수: 음의 절대 차이
            reward = maximum_length - abs(length - ideal_length)
            reward = max(0, reward)
            reward = min(1, reward / (maximum_length - minimum_length))
            rewards.append(reward)

        return rewards

    # 예제에 대한 보상 계산
    rewards = length_reward(completions=completions, ideal_length=slider.value)

    # 예제와 해당 보상 표시
    results = []
    for completion, reward in zip(completions, rewards):
        results.append(
            {"Completion": completion, "Length": len(completion), "Reward": reward}
        )

    fig = px.bar(results, x="Completion", y="Reward", color="Length")
    mo.ui.plotly(fig)


if __name__ == "__main__":
    app.run()

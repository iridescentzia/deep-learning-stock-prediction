import pandas as pd
import plotly.graph_objects as go

symbols = ["AAPL", "MSFT", "NVDA"]


def plot_predictions(symbol):
    data = pd.read_csv(f"{symbol}_combined_predictions.csv")

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=data['Time Index'],
        y=data['실제값'],
        name="실제 주가",
        line=dict(color='firebrick', width=2)
    ))

    fig.add_trace(go.Scatter(
        x=data['Time Index'],
        y=data['MLP 예측'],
        name="MLP 예측",
        line=dict(color='blue', width=1)
    ))

    fig.add_trace(go.Scatter(
        x=data['Time Index'],
        y=data['LSTM 예측'],
        name="LSTM 예측",
        line=dict(color='green', width=1)
    ))

    fig.add_trace(go.Scatter(
        x=data['Time Index'],
        y=data['Transformer 예측'],
        name="Transformer 예측",
        line=dict(color='orange', width=1)
    ))

    fig.update_layout(
        title={
            "text": f"딥러닝 주가 예측 비교 ({symbol})",
            "x": 0.5, "y": 0.9,
            "font": {"size": 18}
        },
        xaxis={"title": "Time Index"},
        yaxis={"title": "Price"},
        template="ggplot2",
        showlegend=True
    )

    fig.show()


for symbol in symbols:
    print(f"{symbol} 주가 예측 결과 시각화 중...")
    plot_predictions(symbol)

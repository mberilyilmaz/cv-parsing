import pandas as pd
import plotly.express as px


def plot_skill_frequency(skills_df: pd.DataFrame):
    plot_df = skills_df.sort_values("count", ascending=False).head(20)
    fig = px.bar(plot_df, x="skill_name", y="count", title="Top 20 Skills")
    fig.update_layout(xaxis_tickangle=-35)
    return fig

import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import to_rgb
from wordcloud import WordCloud
import numpy as np

sns.set(style="whitegrid")

# ========== 1. Sentiment & Rating Distribution ==========
def plot_sentiment_distribution(df):
    fig = px.histogram(
        df,
        x="sentiment",
        color="sentiment",
        category_orders={"sentiment": df["sentiment"].value_counts().index},
        color_discrete_sequence=px.colors.qualitative.Set2,
        title="Sentiment Distribution",
    )
    fig.update_layout(
        xaxis_title="Sentiment",
        yaxis_title="Number of Reviews",
        bargap=0.1,
        height=400,
        width=1000,
    )
    return fig

def plot_score_distribution(df):
    fig = px.histogram(
        df,
        x="score",
        nbins=5,
        marginal="box",
        color_discrete_sequence=["#636EFA"],
        title="Score Distribution",
    )

    fig.update_layout(
        xaxis_title="Review Score",
        yaxis_title="Count",
        bargap=0.1,
        height=400,
        width=1000,
    )

    return fig


# ========== 2. Sentiment Over Time ==========
def plot_sentiment_over_time(df):
    monthly_counts = (
        df.groupby(["month", "sentiment"])
        .size()
        .reset_index(name="count")
    )

    sentiment_colors = {
        "positive": "#2ECC71",
        "neutral": "#F1C40F",
        "negative": "#E74C3C"
    }

    fig = px.line(
        monthly_counts,
        x="month",
        y="count",
        color="sentiment",
        color_discrete_map=sentiment_colors,
        markers=True,
        title="Sentiment Over Time",
    )

    fig.update_layout(
        xaxis_title="Month",
        yaxis_title="Number of Reviews",
        legend_title="Sentiment",
        height=500,
        width=1000,
        margin=dict(l=40, r=40, t=60, b=40),
    )

    return fig

# ========== 3. Language and Source ==========
def plot_top_languages(df, top_n=10):
    lang_counts = df["language_final"].dropna().value_counts().head(top_n)
    lang_df = lang_counts.reset_index()
    lang_df.columns = ["language", "count"]

    fig = px.bar(
        lang_df,
        y="language",
        x="count",
        orientation="h",
        color="language",
        color_discrete_sequence=px.colors.qualitative.Set2,
        title=f"Top {top_n} Languages",
    )

    fig.update_layout(
        xaxis_title="Number of Reviews",
        yaxis_title="Language",
        showlegend=False,
        height=450,
        width=1000,
        margin=dict(l=80, r=40, t=60, b=40),
    )

    return fig

def plot_avg_score_by_source(df):
    avg_scores = (
        df.groupby("source")["score"]
        .mean()
        .sort_values(ascending=False)
        .reset_index()
    )

    fig = px.bar(
        avg_scores,
        x="score",
        y="source",
        orientation="h",
        color="score",
        color_continuous_scale=px.colors.sequential.Sunset,
        title="Average Review Score by Source",
    )

    fig.update_layout(
        xaxis_title="Average Score",
        yaxis_title="Source",
        coloraxis_colorbar_title="Score",
        height=450,
        width=1000,
        margin=dict(l=80, r=40, t=60, b=40),
    )

    return fig


# ========== 4. Text Behavior ==========
def plot_review_length_vs_score(df):
    fig = px.box(
        df,
        x="score",
        y="message_len_words",
        color="score",
        color_discrete_sequence=px.colors.sequential.Sunset,
        title="Review Length vs. Score",
        points="outliers",
    )

    fig.update_layout(
        xaxis_title="Score",
        yaxis_title="Review Length (words)",
        height=500,
        width=1000,
        margin=dict(l=60, r=40, t=60, b=60),
        showlegend=False,
    )

    return fig


def plot_sentiment_wordclouds(df, theme="auto"):
    def _is_dark_from_rc():
        fc = mpl.rcParams.get("figure.facecolor", "white")
        r, g, b = to_rgb(fc)
        return (0.2126*r + 0.7152*g + 0.0722*b) < 0.5

    if theme == "auto":
        dark = None
        try:
            import darkdetect
            dark = bool(darkdetect.isDark())
        except Exception:
            pass
        if dark is None:
            dark = _is_dark_from_rc()
    elif theme == "dark":
        dark = True
    else:
        dark = False

    bg = "#0e1117" if dark else "white"
    title_color = "#e6edf3" if dark else "#0b0f14"

    def make_wc(sentiment, cmap):
        text = " ".join(df.loc[df["sentiment"].str.lower() == sentiment, "clean_message"].dropna())
        if not text.strip():
            return None
        return WordCloud(
            width=800,
            height=500,
            background_color=bg,
            colormap=cmap,
            max_words=200,
            random_state=42,
            prefer_horizontal=0.9,
        ).generate(text)

    wc_pos = make_wc("positive", "Greens" if not dark else "Greens_r")
    wc_neg = make_wc("negative", "Reds" if not dark else "Reds_r")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor(bg)

    if wc_pos is not None:
        axes[0].imshow(wc_pos, interpolation="bilinear")
        axes[0].set_title("Positive Reviews", color=title_color, fontsize=14)
    axes[0].axis("off")

    if wc_neg is not None:
        axes[1].imshow(wc_neg, interpolation="bilinear")
        axes[1].set_title("Negative Reviews", color=title_color, fontsize=14)
    axes[1].axis("off")

    plt.tight_layout()
    plt.show()



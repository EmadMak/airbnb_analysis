import marimo

__generated_with = "0.17.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from collections import Counter
    import warnings

    warnings.filterwarnings("ignore")
    return pd, plt, sns


@app.cell
def _(pd):
    df = pd.read_csv("data/processed/airbnb_reviews_clean.csv")
    return (df,)


@app.cell(hide_code=True)
def _(df, plt, sns):
    def plot_sentiment_distribution(df):
        plt.figure(figsize=(6, 4))
        sns.countplot(data=df, x="sentiment", order=df["sentiment"].value_counts().index, palette="magma")
        plt.title("Sentiment Distribution")
        plt.xlabel("Sentiment")
        plt.ylabel("Number of Reviews")
        plt.tight_layout()
        plt.show()

    plot_sentiment_distribution(df)
    return


@app.cell(hide_code=True)
def _(df, plt, sns):
    def plot_score_distribution(df):
        plt.figure(figsize=(6, 4))
        sns.histplot(df["score"], bins=5, kde=True, palette="magma")
        plt.title("Score Distribution")
        plt.xlabel("Review Score")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.show()

    plot_score_distribution(df)
    return


@app.cell(hide_code=True)
def _(df, plt):
    def plot_sentiment_over_time(df):
        monthly_counts = (
            df.groupby(["month", "sentiment"]).size().unstack(fill_value=0)
        )
        monthly_counts.plot(figsize=(10, 5), marker="o")
        plt.title("Sentiment Over Time")
        plt.xlabel("Month")
        plt.ylabel("Number of Reviews")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    plot_sentiment_over_time(df)
    return


@app.cell(hide_code=True)
def _(df, plt, sns):
    def plot_top_languages(df, top_n=10):
        lang_counts = df["language_final"].dropna().value_counts().head(top_n)
        plt.figure(figsize=(8, 4))
        sns.barplot(y=lang_counts.index, x=lang_counts.values, palette="magma")
        plt.title(f"Top {top_n} Languages")
        plt.xlabel("Number of Reviews")
        plt.ylabel("Language")
        plt.tight_layout()
        plt.show()

    plot_top_languages(df)
    return


@app.cell(hide_code=True)
def _(df, plt, sns):
    def plot_avg_score_by_source(df):
        avg_scores = df.groupby("source")["score"].mean().sort_values(ascending=False)
        plt.figure(figsize=(8, 4))
        sns.barplot(x=avg_scores.values, y=avg_scores.index, palette="magma")
        plt.title("Average Review Score by Source")
        plt.xlabel("Average Score")
        plt.ylabel("Source")
        plt.tight_layout()
        plt.show()

    plot_avg_score_by_source(df)
    return


@app.cell(hide_code=True)
def _(df, plt, sns):
    def plot_review_length_vs_score(df):
        plt.figure(figsize=(7, 5))
        sns.boxplot(data=df, x="score", y="message_len_words", palette="Blues")
        plt.title("Review Length vs. Score")
        plt.xlabel("Score")
        plt.ylabel("Length (words)")
        plt.tight_layout()
        plt.show()

    plot_review_length_vs_score(df)
    return


@app.cell(hide_code=True)
def _(df, plt):
    def plot_wordcloud(df, sentiment="Positive"):
        """Creates a word cloud for a given sentiment (requires wordcloud)."""
        from wordcloud import WordCloud

        text = " ".join(df.loc[df["sentiment"] == sentiment, "clean_message"].dropna())
        wc = WordCloud(width=1000, height=500, background_color="white").generate(text)
        plt.figure(figsize=(10, 5))
        plt.imshow(wc, interpolation="bilinear")
        plt.axis("off")
        plt.title(f"Word Cloud — {sentiment} Reviews")
        plt.show()

    plot_wordcloud(df)
    return


if __name__ == "__main__":
    app.run()

import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns

sns.set(style="whitegrid")

# ========== 1. Sentiment & Rating Distribution ==========
def plot_sentiment_distribution(df):
    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x="sentiment", order=df["sentiment"].value_counts().index, palette="RdYlGn")
    plt.title("Sentiment Distribution")
    plt.xlabel("Sentiment")
    plt.ylabel("Number of Reviews")
    plt.tight_layout()
    plt.show()


def plot_score_distribution(df):
    plt.figure(figsize=(6, 4))
    sns.histplot(df["score"], bins=5, kde=True, color="steelblue")
    plt.title("Score Distribution")
    plt.xlabel("Review Score")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()


# ========== 2. Sentiment Over Time ==========
def plot_sentiment_over_time(df):
    monthly_counts = (
        df.groupby(["year_month", "sentiment"]).size().unstack(fill_value=0)
    )
    monthly_counts.plot(figsize=(10, 5), marker="o")
    plt.title("Sentiment Over Time")
    plt.xlabel("Month")
    plt.ylabel("Number of Reviews")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


# ========== 3. Language and Source ==========
def plot_top_languages(df, top_n=10):
    lang_counts = df["language_final"].dropna().value_counts().head(top_n)
    plt.figure(figsize=(8, 4))
    sns.barplot(y=lang_counts.index, x=lang_counts.values, palette="viridis")
    plt.title(f"Top {top_n} Languages")
    plt.xlabel("Number of Reviews")
    plt.ylabel("Language")
    plt.tight_layout()
    plt.show()


def plot_avg_score_by_source(df):
    avg_scores = df.groupby("source")["score"].mean().sort_values(ascending=False)
    plt.figure(figsize=(8, 4))
    sns.barplot(x=avg_scores.values, y=avg_scores.index, palette="coolwarm")
    plt.title("Average Review Score by Source")
    plt.xlabel("Average Score")
    plt.ylabel("Source")
    plt.tight_layout()
    plt.show()

# ========== 4. Topics ==========
def plot_top_topics(df, top_n=10):
    """Assumes topics_parsed is a list column."""
    all_topics = df["topics_parsed"].dropna().sum()
    top_topics = Counter(all_topics).most_common(top_n)
    topics, counts = zip(*top_topics)

    plt.figure(figsize=(8, 5))
    sns.barplot(x=list(counts), y=list(topics), palette="magma")
    plt.title(f"Top {top_n} Topics")
    plt.xlabel("Count")
    plt.ylabel("Topic")
    plt.tight_layout()
    plt.show()

# ========== 5. Text Behavior ==========
def plot_review_length_vs_score(df):
    plt.figure(figsize=(7, 5))
    sns.boxplot(data=df, x="score", y="message_len_words", palette="Blues")
    plt.title("Review Length vs. Score")
    plt.xlabel("Score")
    plt.ylabel("Length (words)")
    plt.tight_layout()
    plt.show()


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




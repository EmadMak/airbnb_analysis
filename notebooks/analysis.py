import marimo

__generated_with = "0.17.0"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Airbnb Reviews Analysis

    This notebook explores user reviews from Airbnb listings to uncover insights about guest satisfaction, sentiment trends, and linguistic patterns across different sources and languages.  
    By combining descriptive statistics and visual analysis, we aim to understand how factors such as sentiment, review length, and language affect review scores — and ultimately, what makes guests happy.

    ---

    ## Table of Contents

    1. [Introduction](#introduction)
    2. [Dataset Overview](#dataset-overview)
    3. [Exploratory Data Analysis](#exploratory-data-analysis)
       - [Sentiment Distribution](#sentiment-distribution)
       - [Score Distribution](#score-distribution)
       - [Sentiment Over Time](#sentiment-over-time)
       - [Top Languages](#top-languages)
       - [Average Score by Source](#average-score-by-source)
       - [Review Length vs Score](#review-length-vs-score)
       - [Sentiment Word Clouds](#sentiment-word-clouds)
    4. [Actionable Recommendations](#actionable-recommendations)
    4. [Conclusion](#conclusion)

    ---

    ## Introduction

    Online reviews are a rich source of user-generated feedback that can reveal valuable insights into customer satisfaction, product quality, and service consistency.  
    For platforms like **Airbnb**, reviews are a critical trust mechanism — influencing booking decisions, host reputations, and the overall perception of listings.

    This notebook focuses on analyzing Airbnb reviews through multiple lenses:
    - **Sentiment analysis** to gauge overall positivity or negativity.
    - **Temporal trends** to observe how satisfaction evolves over time.
    - **Linguistic patterns** to identify which languages dominate user feedback.
    - **Correlations** between review characteristics (length, score, and sentiment).

    By combining these analyses, we can better understand guest experiences and identify areas for improvement or growth.

    ---

    ## Dataset Overview

    The dataset used in this analysis (`airbnb_reviews_clean.csv`) contains preprocessed Airbnb review data, including:
    - **Review text and sentiment** (Positive, Negative, or Neutral)
    - **Numeric scores** (ratings given by users)
    - **Language of review**
    - **Review source** (platform or channel)
    - **Date of review**

    Each row represents a single review, making it possible to explore both **individual-level patterns** (e.g., text sentiment) and **aggregate-level insights** (e.g., average scores per language or source).

    Before diving into visualizations, the dataset was cleaned to remove missing or malformed entries and prepare fields for analysis. This ensures that our insights are accurate and representative.

    ---
    """
    )
    return


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import numpy as np
    import anywidget
    import plotly.express as px
    from wordcloud import WordCloud

    import matplotlib.pyplot as plt
    import matplotlib as mpl
    from matplotlib.colors import to_rgb

    import airbnb_analysis.plots as plots

    import warnings
    warnings.filterwarnings("ignore")
    return mo, pd, plots


@app.cell
def _(pd):
    df = pd.read_csv("data/processed/airbnb_reviews_clean.csv")
    return (df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Exploratory Data Analysis""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ---

    ### Sentiment Distribution

    The chart below shows the overall distribution of sentiments across all Airbnb reviews in the dataset.  

    **Insights:**
    - The majority of reviews are **positive**, indicating that most guests had favorable experiences with their stays.  
    - **Negative reviews** represent a smaller but still significant portion, suggesting that some guests encountered issues worth addressing.  
    - **Neutral reviews** are relatively few, which may imply that guests tend to express stronger opinions—either satisfaction or dissatisfaction—rather than indifference.  

    This sentiment balance highlights that while Airbnb hosts generally meet or exceed expectations, a consistent minority of users experience problems that could be opportunities for service improvement.
    """
    )
    return


@app.cell(hide_code=True)
def _(df, plots):
    plots.plot_sentiment_distribution(df)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ---

    ### Score Distribution

    This plot illustrates how review scores are distributed across all Airbnb listings in the dataset.  

    **Insights:**
    - The majority of reviews have a score of **5**, showing that most guests rated their stays very highly.  
    - A smaller but noticeable group of reviews have a score of **1**, which likely correspond to strongly negative experiences.  
    - Scores of **2**, **3**, and **4** are comparatively rare, suggesting that guests tend to either love or dislike their stays rather than leave moderate feedback.  
    - The accompanying boxplot reinforces this imbalance, with a strong right-skew toward high ratings and a few low-score outliers.  

    Overall, the data suggests that Airbnb reviews are heavily **positively biased**, a common trend in online platforms where users with positive experiences are more likely to leave feedback.
    """
    )
    return


@app.cell(hide_code=True)
def _(df, plots):
    plots.plot_score_distribution(df)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ---

    ### Sentiment Over Time

    This visualization tracks how the volume of positive, negative, and neutral Airbnb reviews changes throughout the year.  

    **Insights:**
    - **Positive reviews** consistently dominate across all months, peaking around midsummer — possibly reflecting increased travel activity and high guest satisfaction during peak vacation periods.  
    - **Negative reviews** also rise slightly in mid-year, perhaps due to higher booking volumes or increased service pressure on hosts during busy months.  
    - **Neutral reviews** remain comparatively low and stable, indicating that guests tend to express clear opinions rather than mixed feedback.  
    - The apparent **decline in October** is not an actual sentiment trend but rather a **data collection artifact** — since the dataset was likely scraped mid-October, fewer reviews had accumulated at the time.  

    Overall, the sentiment trends reinforce that positive experiences remain the norm year-round, with occasional seasonal fluctuations tied to travel cycles.
    """
    )
    return


@app.cell(hide_code=True)
def _(df, plots):
    plots.plot_sentiment_over_time(df)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ---

    ### Top Languages

    This bar chart shows the top ten languages used by guests when writing Airbnb reviews.  

    **Insights:**
    - **Spanish** and **English** dominate the dataset, together accounting for the majority of all reviews. This suggests that a large portion of Airbnb’s active user base comes from Spanish- and English-speaking regions.  
    - **Portuguese** ranks third, followed by **French** and **Italian**, highlighting strong participation from Southern European and Latin American travelers.  
    - **German** and **Dutch** also appear in the top ten, while **Korean**, **Turkish**, and **Russian** contribute smaller but notable portions.  
    - The diversity of languages reflects Airbnb’s **global reach**, though the strong dominance of a few languages may indicate **regional concentration** of listings or **localization preferences** in review activity.  

    This multilingual spread emphasizes the importance of supporting multilingual interfaces and natural language processing tools when analyzing global Airbnb data.
    """
    )
    return


@app.cell(hide_code=True)
def _(df, plots):
    plots.plot_top_languages(df)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ---

    ### Average Score by Source

    This chart compares the average review scores across different platforms where users submitted their Airbnb feedback.  

    **Insights:**
    - **Google Play** reviews have the **highest average score**, suggesting that Android users generally report more positive experiences or tend to leave higher ratings.  
    - **App Store** reviews show a moderate average score, indicating a slightly more balanced mix of satisfied and critical users among iOS reviewers.  
    - **Trustpilot** has the **lowest average score**, which aligns with the platform’s tendency to attract more critical feedback, as users often visit dedicated review sites to report negative experiences.  
    - These differences may also reflect **audience and intent**: app store reviews are often written after quick mobile experiences, while Trustpilot reviews may be motivated by stronger opinions or service-related frustrations.  

    In short, the sentiment varies by source, hinting that the **context of review collection** significantly influences how users express satisfaction.
    """
    )
    return


@app.cell(hide_code=True)
def _(df, plots):
    plots.plot_avg_score_by_source(df)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ---

    ### Review Length vs. Score

    This boxplot explores how the length of reviews (in words) varies with the score given by users.  

    **Insights:**
    - **Lower scores (1–2)** tend to have **much longer reviews**, often accompanied by higher variance and extreme outliers. This suggests that dissatisfied guests are more likely to provide detailed explanations of their negative experiences.  
    - **Mid-range scores (3–4)** show shorter and more consistent review lengths, implying that moderately satisfied users provide brief, neutral feedback without elaboration.  
    - **High scores (5)** typically correspond to **shorter, more concise reviews**, possibly due to users expressing quick appreciation (“Great stay!”, “Highly recommended!”) without additional detail.  
    - The overall trend indicates a **negative correlation** between review length and rating — the lower the score, the more effort guests invest in describing their experiences.  

    This behavioral asymmetry is common in review data: users motivated by dissatisfaction tend to write longer, more expressive reviews than those with positive experiences.
    """
    )
    return


@app.cell(hide_code=True)
def _(df, plots):
    plots.plot_review_length_vs_score(df)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ---

    ### Sentiment Word Clouds

    The following word clouds highlight the most frequent words used in **positive** and **negative** Airbnb reviews, offering a qualitative view of customer sentiment.  

    **Insights:**
    - In **positive reviews**, words such as *“excelente,” “fácil,” “buena,” “super,”* and *“love”* dominate, emphasizing ease of use, satisfaction, and enjoyment. Guests often praise the **app experience**, the **hosts**, and the **overall stay**, suggesting that convenience and service quality are major drivers of positive feedback.  
    - In contrast, **negative reviews** are filled with terms like *“refund,” “support,” “booking,” “issue,”* and *“money.”* This vocabulary points to frustrations related to **customer service**, **payment problems**, and **technical issues** within the platform rather than the stays themselves.  
    - The linguistic contrast between the two clouds mirrors the **score and sentiment distributions**: satisfied users express gratitude and simplicity, while dissatisfied users focus on practical failures and disputes.  

    These patterns reaffirm that while Airbnb excels in delivering positive guest experiences, **pain points often arise from app performance and support handling** — key areas for potential improvement.
    """
    )
    return


@app.cell(hide_code=True)
def _(df, plots):
    plots.plot_sentiment_wordclouds(df)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ---

    ## Actionable Recommendations

    Based on the insights gathered from sentiment, score, and linguistic analyses, the following recommendations can help enhance both **guest experience** and **platform reliability**:

    ### 1. Strengthen Customer Support and Refund Handling
    - The negative word cloud and low Trustpilot scores indicate recurring issues with **refunds, support responsiveness, and booking disputes**.  
    - Airbnb should prioritize improving **support ticket resolution times**, **refund transparency**, and **automated help flows** to reduce friction and restore user confidence.

    ### 2. Enhance App Stability and Usability
    - Frequent mentions of *“app,” “issue,”* and *“update”* in negative reviews highlight user frustration with the mobile experience.  
    - Regular **UX audits**, **bug tracking dashboards**, and **beta testing programs** can help maintain a smoother, more reliable app experience across both iOS and Android.

    ### 3. Address Platform-Specific Feedback Trends
    - Users on **Trustpilot** tend to be more critical, while **Google Play** users leave overwhelmingly positive feedback.  
    - Tailor engagement strategies to each platform — e.g., **proactive outreach to dissatisfied users** on external review sites, and **encouraging detailed feedback** from satisfied users within the app.

    ### 4. Leverage Positive Review Patterns for Marketing
    - Positive reviews frequently mention **ease of use**, **great hosts**, and **excellent stays**.  
    - Airbnb can highlight these themes in **marketing campaigns** and **host training materials** to reinforce strengths and attract new users through authentic testimonials.

    ### 5. Monitor Seasonal Sentiment Trends
    - While sentiment remains mostly stable, mid-year increases in both positive and negative reviews suggest that **peak seasons amplify guest expectations**.  
    - Airbnb could deploy **seasonal quality assurance checks** and **host performance incentives** during busy months to sustain satisfaction rates.

    ### 6. Multilingual Engagement and Localization
    - With Spanish, English, and Portuguese dominating the dataset, but multiple other languages present, Airbnb should expand **localized support** and **multi-language review analysis** to better address regional feedback.

    By addressing these areas, Airbnb can move beyond maintaining satisfaction toward **actively optimizing user trust, reducing churn**, and **enhancing global user experience consistency**.

    ---
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Conclusion

    This analysis reveals that Airbnb enjoys overwhelmingly positive guest sentiment, with most reviews expressing high satisfaction across time, languages, and sources.  
    However, recurring challenges—particularly around **customer support**, **refunds**, and **app reliability**—stand out as key areas for improvement.  

    Guests who face problems tend to write longer, more detailed reviews, while satisfied users often leave brief but enthusiastic feedback, underscoring a strong emotional divide between positive and negative experiences.  
    Language diversity and platform differences further emphasize the importance of **localized engagement** and **context-aware response strategies**.  

    Overall, Airbnb’s reputation remains strong, but targeted improvements in **service responsiveness**, **technical stability**, and **user communication** can meaningfully elevate trust and consistency across its global community.
    """
    )
    return


if __name__ == "__main__":
    app.run()

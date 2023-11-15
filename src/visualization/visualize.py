from typing import Optional
from collections import Counter
import pandas as pd
from wordcloud import wordcloud
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from const import PROCESSED_DATA_ROOT
from src.data.stop_words import stop_words
import seaborn as sns

tweets = pd.read_csv(PROCESSED_DATA_ROOT.joinpath("tweets.csv"))
popular_member_themes = pd.read_csv(PROCESSED_DATA_ROOT.joinpath("popular_member_themes.csv"))
theme_year_count = pd.read_csv(PROCESSED_DATA_ROOT.joinpath("theme_year_count.csv"))

def get_word_cloud(year, ax: Optional[Axes]=None):
    assert ax is not None
    df = tweets[tweets['year']==year]
    no_sw = df["text"].apply(
        lambda text: [
            word for word in text.lower().split(' ') if word not in stop_words
        ]
    )
    corpus = []
    for ea in no_sw:
        corpus += set(ea)
    wordDict = Counter(corpus)
    word_cloud = wordcloud.WordCloud().generate_from_frequencies(wordDict)
    ax.imshow(word_cloud, interpolation="bilinear")
    ax.set_title(year)
    ax.axis("off")

def make_word_cloud_grid():
    fig, ax = plt.subplots(5, 2)
    fig.set_figheight(15)
    fig.set_figwidth(15)
    tweets = pd.read_csv(f"{PROCESSED_DATA_ROOT}/tweets.csv")
    for i in range(5):
        for j in range(2):
            get_word_cloud(tweets['year'].dropna().unique().reshape((5, 2))[i, j], ax=ax[i, j])
    plt.tight_layout()
    plt.show()
    plt.close()

def plot_theme_by_name(year, ax=None):
    assert ax is not None
    sns.barplot(data = popular_member_themes[popular_member_themes['year'] == year], 
                y='screen_name', x='score', hue='theme', ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    ax.set_title(year)

def plot_theme_year():
    sns.set_theme()
    sns.lineplot(x=theme_year_count['year'], y=theme_year_count['count'], hue=theme_year_count['theme'])

def make_theme_grid():
    fig, ax = plt.subplots(3, 3)
    fig.set_figheight(20)
    fig.set_figwidth(20)
    for i in range(3):
        for j in range(3):
            year = tweets['year'].dropna().unique()[1:].reshape((3,3))[i,j]
            plot_theme_by_name(year, ax=ax[i, j])
    plt.tight_layout()
    plt.show()
    plt.close()

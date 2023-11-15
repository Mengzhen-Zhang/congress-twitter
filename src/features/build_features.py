from collections.abc import Callable
from pathlib import Path
from typing import Any, Optional, cast
import pandas as pd
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from data.stop_words import stop_words

data_folder = "../../data"
raw_path = f"{data_folder}/raw"
interim_path = f"{data_folder}/interim"
processed_path = f"{data_folder}/processed"

def get_raw(path: str, pattern: str) -> pd.DataFrame:
    files = Path(path).glob(pattern)
    dfs: list[pd.DataFrame] = list()
    for f in files:
        try:
            dfs.append(pd.read_csv(f, lineterminator="\n"))
        except:
            print(f)
    return pd.concat(dfs, ignore_index=True)

class save_to:
    def __init__(self, path: str, name: str):
        self.path = path
        self.name = name

    def __call__(self, func: Callable[[Any], pd.DataFrame]) -> Callable:
        def inner(*args, **kwargs):
            df = func(*args, **kwargs)
            self.save_data(self.path, df, self.name)
        return inner

    @staticmethod
    def save_data(path: str, df: pd.DataFrame, name: str):
        filename = f"{path}/{name}.csv"
        df.to_csv(filename, index=False)
        print("{name} created")
        print(df.info())


class build_feautures:
    def __init__(self):
        print(f"loading raw data from {raw_path}")
        
        self.raw_tweets: pd.DataFrame = get_raw(raw_path, "tweets*.csv")
        print("raw tweets loaded")
        
        self.raw_users: pd.DataFrame = get_raw(raw_path, "users.csv")
        print("raw users loaded")
        
        self.themes = ["health", "work", "house", "tax", "auto", "trade"]
        self.tweets: Optional[pd.DataFrame] = None
        self.users: Optional[pd.DataFrame] = None
        self.mentions: Optional[pd.DataFrame] = None
        self.mentioned_members: Optional[pd.DataFrame] = None
        self.topics: Optional[pd.DataFrame] = None
        self.topics_year: Optional[pd.DataFrame] = None
        self.theme_year_count: Optional[pd.DataFrame] = None

    def initialize(self):
        self.get_tweets()
        self.get_users()
        self.get_mentions()
        self.get_mentioned_members()
        self.get_topics()
        self.get_topics_year()
        self.get_theme_year_count()

    @save_to(interim_path, "tweets")
    def get_tweets(self) -> pd.DataFrame:
        tweets: pd.DataFrame = self.raw_tweets[
            ["created_at", "id", "in_reply_to_user_id", "text", "user_id"]
        ]
        tweets.in_reply_to_user_id.astype("Int64", copy=False)
        tweets.in_reply_to_user_id.astype("object", copy=False)
        tweets["year"] = tweets["created_at"].dt.year
        tweets['sentiment'] = tweets['text'].apply(lambda text: cast(tuple, TextBlob(text).sentiment)[0])
        return tweets

    @save_to(interim_path, "users")
    def get_users(self) -> pd.DataFrame:
        self.users = self.raw_users[["id", "name", "screen_name"]]
        return self.users

    @save_to(interim_path, "mentions")
    def get_mentions(self) -> pd.DataFrame:
        mentions: pd.DataFrame = self.raw_tweets[["id"]].join(
            pd.json_normalize(self.raw_tweets.entities)["user_mentions"]
        )
        mentions.rename(columns={"id": "tweet_id"}, inplace=True)
        mentions = mentions.explode("user_mentions")
        mentions = mentions[["tweet_id"]].join(
            pd.json_normalize(mentions.user_mentions)["id"]
        )
        mentions = mentions.rename(columns={"id": "mentioned_id"})
        mentions.mentioned_id = mentions.mentioned_id.astype("Int64")
        mentions = mentions[~mentions.mentioned_id.isnull()]
        mentions = mentions.drop_duplicates()
        self.mentions = mentions
        return self.mentions

    @save_to(interim_path, "mentioned_members")
    def get_mentioned_members(self) -> pd.DataFrame:
        assert self.users is not None
        mentioned_members = self.users.merge(self.mentions, 
                                        left_on='id', 
                                        right_on='mentioned_id',
                                        how='inner')
        mentioned_members = mentioned_members.drop(columns=['id'])
        mentioned_members = mentioned_members.groupby('screen_name').count()
        mentioned_members = mentioned_members[['mentioned_id']]
        mentioned_members = mentioned_members.rename(columns={'mentioned_id': 'count'})
        mentioned_members = mentioned_members.sort_values(by='count', ascending=False)
        mentioned_members = mentioned_members.reset_index()
        self.mentioned_members = mentioned_members
        return self.mentioned_members

    @save_to(interim_path, "mentioned_members_outliers")
    def get_mentioned_members_outliers(self) -> pd.DataFrame:
        assert self.mentioned_members is not None
        Q1 = self.mentioned_members.quantile(0.25).values[0]
        Q2 = self.mentioned_members.quantile(0.75).values[0]
        IQR = Q2 - Q1
        lower_bound = Q1 - 1.5*IQR
        upper_bound = Q2 + 1.5*IQR
        self.mentioned_members_outliers = self.mentioned_members[
            (self.mentioned_members['count'] < lower_bound) | 
            (self.mentioned_members['count'] > upper_bound)
        ]
        return self.mentioned_members_outliers

    @save_to(interim_path, "topics")
    def get_topics(self) -> pd.DataFrame:
        assert self.tweets is not None
        topics = self.tweets[['id', 'created_at', 'text']]
        topics['text'] = topics['text'].str.lower()
        topics['text'] = topics['text'].str.replace('care', 'health')
        topics['text'] = topics['text'].str.replace('#obamacare', 'health')
        topics['text'] = topics['text'].str.replace('jobs', 'work')
        topics['year'] = topics['created_at'].dt.year

    @save_to(interim_path, "theme_counts")
    def get_theme_counts(self) -> pd.DataFrame:
        assert self.topics is not None
        theme_counts = []
        for theme in self.themes:
            theme_name = theme
            if theme in ["health", "#obamacare", "care"]:
                theme_name = "health"
            theme_counts.append({
                'theme': theme_name,
                'count': self.topics['text'].str.contains(theme).sum()})
        theme_counts = pd.DataFrame(theme_counts)
        for theme in self.themes:
            self.topics['theme_' + theme ] = self.topics['text'].str.contains(theme).astype('Int64')

    @save_to(interim_path, "topics_year")
    def get_topics_year(self) -> pd.DataFrame:
        assert self.topics is not None
        topics_year = self.topics.groupby('year').sum()
        topics_year = topics_year.reset_index()
        self.topics_year = topics_year
        return self.topics_year

    @save_to(interim_path, "theme_year_count")
    def get_theme_year_count(self) -> pd.DataFrame:
        assert self.topics_year is not None
        dfs = []
        for theme in self.themes:
            df = self.topics_year[['year', 'theme_' + theme]]
            df = df.rename(columns={'theme_' + theme: 'count'})
            df['theme'] = theme
            dfs.append(df)
        self.theme_year_count = pd.concat(dfs)
        return self.theme_year_count

    @save_to(interim_path, "popular_member_tweets")
    def get_popular_member_tweets(self) -> pd.DataFrame:
        popular_member_tweets = self.mentioned_members_outliers.merge(self.users, how="inner").merge(self.tweets, left_on='id', right_on='user_id', how="inner")
        popular_member_tweets = popular_member_tweets[["screen_name", "year", "text"]]
        vocabulary = ["health", "work", "house", "tax", "auto", "trade", "care", "#obamacare"]
        dfs = []
        for screen_name in popular_member_tweets['screen_name'].unique():
            df = popular_member_tweets[popular_member_tweets['screen_name'] == screen_name]
            if df.shape[0] == 0:
                continue
            vectorizer = TfidfVectorizer(input='content', stop_words = list(stop_words), vocabulary=vocabulary)
            vector = vectorizer.fit_transform(df['text'])
            df1 = pd.DataFrame(vector.toarray(), columns=vectorizer.get_feature_names_out(), index=df.index)
            df1['health'] = df1['health'] + df1['care'] + df1['#obamacare']
            df1 = df1.drop(columns=['care', '#obamacare'])
            df1['screen_name'] = screen_name
            df1['year'] = df['year']
            dfs.append(df1)
        popular_member_themes = pd.concat(dfs)
        popular_member_themes = popular_member_themes.groupby(['screen_name', 'year']).mean().reset_index()
        popular_member_themes = pd.melt(popular_member_themes,
                                       id_vars=['screen_name', 'year'],
                                       value_vars=self.themes,
                                       value_name='score',
                                       var_name='theme')

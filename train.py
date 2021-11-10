import pickle
import pandas as pd
import ast
from collections import Counter
import numpy as np
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor

TO_REMOVE = ["id", "belongs_to_collection", "homepage", "original_title", "overview",
             "tagline", "title"]


def clean_data(df, to_remove=TO_REMOVE):
    df.drop(columns=to_remove, inplace=True)
    df["has_budget"] = np.where(df["budget"] == 0, 0, 1)
    budget_average = int(df.budget.sum() / (df.budget != 0).sum())
    df['budget'] = df['budget'].replace([0], budget_average)
    df["spoken_languages"] = df["spoken_languages"].apply(lambda x: len(x))
    valid_language = (df.original_language
                      .value_counts()[lambda x: x >= 40].index.to_list())

    for language in df['original_language']:
        if language not in valid_language:
            df['original_language'] = df['original_language'].replace([language], 'other')

    df = (pd.get_dummies(df, columns=["original_language"], drop_first=False)
          .drop(columns=['original_language_other']))
    cast_members = create_dummy(df, "cast", "name", 15)
    crew_members = create_dummy(df, "crew", "name", 30)
    keywords = create_dummy(df, "keywords", "name", 30)
    df = (df.assign(
        release_date=
        lambda d: (datetime.now() - pd.to_datetime(d.release_date)).dt.days))

    df = (df.loc[(df.runtime > 20) & (df.release_date > 0)
                 & (df.revenue < 15 * (10 ** 8)), :])

    genres = create_dummy(df, "genres", "name")
    df.drop(columns=["genres", "crew", "cast", "production_companies", "keywords",
                     "production_countries"], inplace=True)
    data = (genres, cast_members, crew_members, budget_average, valid_language, keywords)
    return (df.loc[df.status == "Released", :].drop(columns=["status", 'Unnamed: 0'])
            .dropna()), data


def create_dummy(df, col_name, dict_key, threshold=0):
    parsed_col = df[col_name].apply(
        lambda x: [d[dict_key] for d in (ast.literal_eval(x))] if pd.notna(x) else [])
    dummies = []
    for lst in parsed_col:
        dummies += lst
    if threshold <= 0:
        dummies = set(dummies)
    else:
        c = Counter(dummies)
        dummies = {name for name, val in c.items() if val > threshold}
    for dummy in dummies:
        df[dummy] = parsed_col.apply(lambda x: 1 if dummy in x else 0)
    return dummies


def train():
    df, dummies = clean_data(pd.read_csv("train_test.csv"))
    means = df.drop(columns=["revenue", "vote_average"]).mean()
    mdl_revenue = (RandomForestRegressor()
                   .fit(df.drop(columns=["revenue", "vote_average"]), df.revenue))
    mdl_vote = (RandomForestRegressor()
                .fit(df.drop(columns=["revenue", "vote_average"]), df.vote_average))
    with open("mdl", mode="wb") as f:
        pickle.dump((mdl_revenue, mdl_vote), f)
    with open("dummies", mode="wb") as f:
        pickle.dump(dummies, f)
    with open("means", mode="wb") as f:
        pickle.dump(means, f)
    return df.drop(columns=["revenue", "vote_average"])


df_train = train()

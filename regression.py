################################################
#
#     Task 1 - predict movies revenue & ranking
#
################################################
import pandas as pd
import numpy as np
from datetime import datetime
import pickle
import ast

TO_REMOVE = ["id", "belongs_to_collection", "homepage", "original_title", "overview",
             "tagline", "title", "keywords", "genres", "crew", "cast",
             "production_companies", "production_countries", "original_language"]


def predict(csv_file):
    """
    This function predicts revenues and votes of movies given a csv file with movie details.
    Note: Here you should also load your model since we are not going to run the training process.
    :param csv_file: csv with movies details. Same format as the training dataset csv.
    :return: a tuple - (a python list with the movies revenues, a python list with the movies avg_votes)
    """
    df = process_data(csv_file)
    unreleased = np.where(df['status'] != "Released", df['status'], 0)
    data_to_predict = df.drop(columns=["status"])
    with open("mdl", mode="rb") as f:
        revenue_mdl, vote_mdl = pickle.load(f)
    revenue_predict = revenue_mdl.predict(data_to_predict).tolist()
    vote_predict = vote_mdl.predict(data_to_predict).tolist()
    for i in range(len(unreleased)):
        if unreleased[i] == 0:
            revenue_predict[i] = 0
            vote_predict[i] = 0
    return (revenue_predict, vote_predict)


def create_dummies(df, col_name, dict_key, dummies_set):
    parsed_col = df[col_name].apply(
        lambda x: [d[dict_key] for d in (ast.literal_eval(x))] if pd.notna(x) else [])
    for dummy in dummies_set:
        df[dummy] = parsed_col.apply(lambda x: 1 if dummy in x else 0)


def process_data(csv_file):
    with open("dummies", mode='rb') as f:
        genres, cast_members, crew_members, budget_average, languages, keywords = pickle.load(f)
    with open("means", mode="rb") as f:
        means = pickle.load(f)
    df = pd.read_csv(csv_file)
    df["spoken_languages"] = df["spoken_languages"].apply(lambda x: len(x))
    df["has_budget"] = np.where(df["budget"] == 0, 0, 1)
    df['budget'] = df['budget'].replace([0], budget_average)
    df = (df.assign(
        release_date=lambda d: (datetime.now() - pd.to_datetime(d.release_date)).dt.days))
    create_dummies(df, "keywords", "name", keywords)
    create_dummies(df, "genres", "name", genres)
    create_dummies(df, "crew", "name", crew_members)
    create_dummies(df, "cast", "name", cast_members)
    for language in languages:
        df[language] = np.where(df.original_language == language, 1, 0)
    df.drop(columns=TO_REMOVE, inplace=True)
    df = df.fillna(value=means)
    return df


data = pd.read_csv("evaluation.csv")
true_revenue, true_vote_average = data.revenue, data.vote_average
revenue_predict, vote_predict = predict("evaluation_without_response.csv")
# print(revenue_predict, vote_predict)
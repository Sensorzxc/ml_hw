import pickle
import re

import click
import sys
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


@click.group()
def main():
    pass


@main.command()
@click.option('--data', type=click.Path(exists=True))
@click.option('--test', type=click.Path())
@click.option('--split', type=float)
@click.option('--model', type=click.Path())
def train(data, test, split, model):
    data = pd.read_csv(data)
    data['rating'] = data['rating'].apply(lambda x: 1 if x > 3 else 0)
    data['review'] = data['title'] + ' ' + data['text']
    data['review'] = data['review'].apply(lambda x: re.sub(r'[^a-zA-Z\s]', '', x) if x is not np.nan else '')

    X = data['review']
    y = data['rating']
    X_train = []
    y_train = []
    X_test = []
    y_test = []

    if test is not None:
        X_train = X
        y_train = y
        data_test = pd.read_csv(test)
        data_test['rating'] = data_test['rating'].apply(lambda x: 1 if x > 3 else 0)
        data_test['review'] = data_test['title'] + ' ' + data_test['text']
        data_test['review'] = data_test['review'].apply(
            lambda x: re.sub(r'[^a-zA-Z\s]', '', x) if x is not np.nan else '')

        X_test = data_test['review']
        y_test = data_test['rating']



    elif split is not None:
        if split == 0 or split == 1:
            click.echo('Split must be between 0 and 1')
            sys.exit(1)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split)

    pipeline = Pipeline([
        ('vectorizer', TfidfVectorizer()),
        ('classifier', LogisticRegression(max_iter=30, random_state=42))
    ])

    pipeline.fit(X_train, y_train)
    score = pipeline.score(X_test, y_test)

    with open(model, 'wb') as f:
        pickle.dump(pipeline, f)

    if test is not None or split is not None:
        click.echo(f'Test score: {score}')


@main.command()
@click.option('--model', type=click.Path(exists=True))
@click.option('--data', type=click.Path(exists=True))
def predict(model, data):
    with open(model, 'rb') as f:
        pipeline = pickle.load(f)

    data = pd.read_csv(data)
    data['rating'] = data['rating'].apply(lambda x: 1 if x > 3 else 0)
    data['review'] = data['title'] + ' ' + data['text']
    data['review'] = data['review'].apply(lambda x: re.sub(r'[^a-zA-Z\s]', '', x) if x is not np.nan else '')

    X = data['review']
    y = data['rating']

    pred = pipeline.predict(X)
    n = 0
    for p in pred:
        click.echo(f'{n}: {p}')
        n+=1


if __name__ == '__main__':
    main()

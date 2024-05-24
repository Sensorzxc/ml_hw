import pytest
import os

from click.testing import CliRunner

from main import main, train, predict
import pickle, re
import pandas as pd
import numpy as np


def test_train_good():
    runner = CliRunner()
    result = runner.invoke(train,
                           ['--data', '../data/singapore_airlines_reviews.csv', '--split', 0.2, '--model', 'test_model.pkl'])
    assert result.exit_code == 0
    assert os.path.exists('test_model.pkl')

    os.remove('test_model.pkl')



def test_train_split_bad():
    runner = CliRunner()
    result = runner.invoke(train,
                           ['--data', '../data/singapore_airlines_reviews.csv', '--split', 1, '--model', 'test_model.pkl'])
    assert result.exit_code == 1
    assert not os.path.exists('test_model.pkl')


def test_train_data_no_exist():
    runner = CliRunner()
    result = runner.invoke(train,
                           ['--data', 'zxc', '--split', 0.2, '--model', 'test_model.pkl'])
    assert result.exit_code == 2
    assert not os.path.exists('test_model.pkl')



def test_predict_good():
    runner = CliRunner()
    result = runner.invoke(train,
                           ['--data', '../data/singapore_airlines_reviews.csv', '--split', 0.2, '--model', 'test_model.pkl'])
    assert result.exit_code == 0
    assert os.path.exists('test_model.pkl')

    runner = CliRunner()
    result = runner.invoke(predict,
                           ['--model', 'test_model.pkl', '--data', '../data/singapore_airlines_reviews.csv'])

    os.remove('test_model.pkl')
    assert result.exit_code == 0




def test_predict_bad():
    runner = CliRunner()
    result = runner.invoke(train,
                           ['--data', '../data/singapore_airlines_reviews.csv', '--split', 0.2, '--model', 'test_model.pkl'])
    assert result.exit_code == 0
    assert os.path.exists('test_model.pkl')

    os.remove('test_model.pkl')

    result = runner.invoke(predict,
                           ['--model', 'test_model.pkl', '--data', '../data/singapore_airlines_reviews.csv'])
    assert result.exit_code == 2

import pytest
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris, fetch_california_housing
from automl.core import AutoML

@pytest.fixture
def iris_data():
    data = load_iris(as_frame=True)
    df = data.frame
    df['target'] = data.target
    return df, 'target', 'classification'

@pytest.fixture
def california_data():
    X = pd.DataFrame(np.random.rand(100, 8), columns=[f"feature_{i}" for i in range(8)])
    y = pd.Series(np.random.rand(100), name='target')
    df = pd.concat([X, y], axis=1)
    return df, 'target', 'regression'

def test_classification_pipeline(iris_data):
    df, target, problem_type = iris_data
    automl = AutoML(df, target, problem_type)
    best_model = automl.run()
    assert isinstance(best_model, dict)
    assert 'name' in best_model
    assert 'score' in best_model

def test_regression_pipeline(california_data):
    df, target, problem_type = california_data
    automl = AutoML(df, target, problem_type)
    best_model = automl.run()
    assert isinstance(best_model, dict)
    assert 'name' in best_model
    assert 'score' in best_model

def test_invalid_problem_type(iris_data):
    df, target, _ = iris_data
    with pytest.raises(ValueError):
        AutoML(df, target, 'clustering').run()

def test_missing_target_column(iris_data):
    df, _, problem_type = iris_data
    with pytest.raises(ValueError):
        AutoML(df.drop(columns=['target']), 'target', problem_type).run()

def test_empty_dataframe():
    df = pd.DataFrame()
    with pytest.raises(ValueError):
        AutoML(df, 'target', 'classification').run()

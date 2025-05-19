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
    y = pd.Series(np.random.rand(100))
    return X, y, 'regression'

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

def test_model_prediction_shape(iris_data):
    df, target, problem_type = iris_data
    automl = AutoML(df, target, problem_type)
    automl.run()
    preds = automl.predict(df.drop(columns=[target]))
    assert preds.shape[0] == df.shape[0]

def test_random_state_reproducibility(iris_data):
    df, target, problem_type = iris_data
    automl1 = AutoML(df, target, problem_type, random_state=42)
    automl2 = AutoML(df, target, problem_type, random_state=42)
    best1 = automl1.run()
    best2 = automl2.run()
    assert best1['name'] == best2['name']
    assert abs(best1['score'] - best2['score']) < 1e-6

def test_non_numeric_data_handling():
    df = pd.DataFrame({
        'feature': ['a', 'b', 'c', 'a', 'b'],
        'target': [0, 1, 0, 1, 0]
    })
    automl = AutoML(df, 'target', 'classification')
    result = automl.run()
    assert 'name' in result and 'score' in result

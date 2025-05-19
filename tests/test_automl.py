import pytest
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris, make_regression
from automl.core import AutoML

@pytest.fixture
def iris_data():
    data = load_iris(as_frame=True)
    df = data.frame
    df['target'] = data.target
    return df, 'target', 'classification'

@pytest.fixture
def regression_data():
    X, y = make_regression(n_samples=500, n_features=8, noise=0.3, random_state=42)
    df = pd.DataFrame(X, columns=[f"Feature_{i}" for i in range(8)])
    df["Target"] = y
    return df

def test_classification_pipeline(iris_data):
    df, target, problem_type = iris_data
    automl = AutoML(df, target, problem_type)
    best_model = automl.run()
    assert isinstance(best_model, dict)
    assert 'name' in best_model
    assert 'score' in best_model

def test_regression_pipeline(regression_data):
    from automl.core import AutoML

    X = regression_data.drop("Target", axis=1)
    y = regression_data["Target"]

    automl = AutoML(problem_type="regression", scoring="rmse", cv=3)
    automl.fit(X, y)
    result = automl.results_[0]
    assert result["score_name"] == "RMSE"
    assert result["score"] >= 0


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

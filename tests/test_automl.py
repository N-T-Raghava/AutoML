import pandas as pd
from sklearn.datasets import make_classification, make_regression
import pytest
from automl import AutoML

def test_classification():
    # Load dataset
    X, y = make_classification(n_samples=100, n_features=20, random_state=42)
    df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(20)])
    df["target"] = y
    target_column = "target"

    # Initialize AutoML
    model = AutoML(df, target=target_column)
    model.run()

    # Add assertions to validate the results
    assert model.best_model is not None
    assert model.performance_metrics is not None

def test_regression():
    # Load dataset
    X, y = make_regression(n_samples=100, n_features=20, random_state=42)
    df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(20)])
    df["target"] = y
    target_column = "target"

    # Initialize AutoML
    model = AutoML(df, target=target_column)
    model.run()

    # Add assertions to validate the results
    assert model.best_model is not None
    assert model.performance_metrics is not None

def test_classification_with_external_data():
    # Load dataset
    X, y = make_classification(n_samples=100, n_features=20, random_state=42)
    df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(20)])
    df["target"] = y
    target_column = "target"

    # Initialize AutoML
    model = AutoML(df, target=target_column)
    model.run()

    # Add assertions to validate the results
    assert model.best_model is not None
    assert model.performance_metrics is not None

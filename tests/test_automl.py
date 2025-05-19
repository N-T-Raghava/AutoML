import pandas as pd
from automl import AutoML

def test_classification():
    # Load dataset
    df = pd.read_csv('path_to_classification_dataset.csv')  # Replace with actual path
    target_column = 'target'  # Replace with actual target column name

    # Initialize AutoML
    model = AutoML(df, target=target_column)
    model.run()

    # Add assertions to validate the results
    assert model.best_model is not None
    assert model.performance_metrics is not None

def test_regression():
    # Load dataset
    df = pd.read_csv('path_to_regression_dataset.csv')  # Replace with actual path
    target_column = 'target'  # Replace with actual target column name

    # Initialize AutoML
    model = AutoML(df, target=target_column)
    model.run()

    # Add assertions to validate the results
    assert model.best_model is not None
    assert model.performance_metrics is not None

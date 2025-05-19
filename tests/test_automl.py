import pandas as pd
from sklearn.datasets import load_iris, fetch_california_housing
from automl import AutoML

def test_classification():
    # Load dataset
    iris = load_iris()
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df['target'] = iris.target
    target_column = 'target'

    # Initialize AutoML
    model = AutoML(df, target=target_column)
    model.run()

    # Add assertions to validate the results
    assert model.best_model is not None
    assert model.performance_metrics is not None

def test_regression():
    # Load dataset
    california = fetch_california_housing()
    df = pd.DataFrame(data=california.data, columns=california.feature_names)
    df['target'] = california.target
    target_column = 'target'

    # Initialize AutoML
    model = AutoML(df, target=target_column)
    model.run()

    # Add assertions to validate the results
    assert model.best_model is not None
    assert model.performance_metrics is not None

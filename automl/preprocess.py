import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class DataCleaner:
    def __init__(self, verbose=True):
        self.verbose = verbose

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # Drop duplicates
        df.drop_duplicates(inplace=True)

        # Drop constant columns
        nunique = df.nunique()
        df = df.loc[:, nunique > 1]

        # Convert numeric strings to numbers
        for col in df.select_dtypes(include="object").columns:
            try:
                df[col] = pd.to_numeric(df[col])
            except:
                pass

        # Fill missing values
        for col in df.columns:
            if df[col].dtype in [np.float64, np.int64]:
                df[col].fillna(df[col].mean(), inplace=True)
            else:
                df[col].fillna(df[col].mode()[0], inplace=True)

        # Standardize categorical text
        for col in df.select_dtypes(include="object").columns:
            df[col] = df[col].str.strip().str.lower()

        if self.verbose:
            print(f"[DataCleaner] Cleaned DataFrame with shape: {df.shape}")

        return df

def preprocess_data(df, target, problem_type):
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in dataframe.")
    if df.empty:
        raise ValueError("Input dataframe is empty.")
    
    X = df.drop(columns=[target])
    y = df[target]
    
    X = df.drop(columns=[target])
    y = df[target]

    # Guess problem type
    if problem_type is None:
        problem_type = 'classification' if y.nunique() <= 20 else 'regression'

    # Basic preprocessing
    X = pd.get_dummies(X)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test, problem_type
import pandas as pd
from src.preprocessor import preprocessor_Titanic

def sample_df():
    return pd.DataFrame({...})

def test_fit_transform_shape():
    prep = preprocessor_Titanic()
    X, y = prep.fit_transform(sample_df())
    assert len(X) == len(y)
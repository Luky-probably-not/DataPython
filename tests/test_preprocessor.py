import pandas as pd
from src.preprocessor import preprocessor_Titanic

def sample_df():
    return pd.DataFrame(pd.read_csv("../data/train.csv"))

def test_fit_transform_shape():
    prep = preprocessor_Titanic()
    df = prep.fit_transform(sample_df())
    X = df.drop('Survived', axis=1)
    y = df['Survived']
    assert len(X) == len(y)
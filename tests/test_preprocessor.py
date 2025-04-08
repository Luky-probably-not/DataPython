import pandas as pd
from src.preprocessor import preprocessor_Titanic

def sample_df():
    return pd.DataFrame(pd.read_csv("../data/train.csv"))

def test_transform_shape():
    prep = preprocessor_Titanic()
    df = prep.fit_transform(sample_df())
    X = df.drop('Survived', axis=1)
    y = df['Survived']
    assert len(X) == len(y)
    
def test_fit():
    df = sample_df()
    prep = preprocessor_Titanic()
    prep.fit(df)
    assert df["Age"].median() == prep.median_age
    
def test_fit_transform():
    dfbase = sample_df()
    prep = preprocessor_Titanic()
    df = prep.fit_transform(dfbase)
    X = df.drop('Survived', axis=1)
    y = df['Survived']
    assert len(X) == len(y)
    assert dfbase["Age"].median() == prep.median_age
    
    

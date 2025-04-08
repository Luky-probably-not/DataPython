import pandas as pd
from sklearn.model_selection import train_test_split

class preprocessor_Titanic:
    
    def __init__(self):
        self.median_age = None
        self.median_fare = None
        self.mode_embarked = None

    def fit(self, df):
        self.median_age = df["Age"].median()
        self.median_fare = df["Fare"].median()
        self.mode_embarked = df["Embarked"].mode()[0]
        
    def transform(self,df):
        df = df.drop(["Cabin", "Ticket", "Name", "PassengerId"], axis=1)
        df["Age"] = df["Age"].fillna(self.median_age)
        df["Fare"] = df["Fare"].fillna(self.median_fare)
        df["Embarked"] = df["Embarked"].fillna(self.mode_embarked)
        
        df = pd.get_dummies(df, columns=["Sex", "Embarked"], drop_first=True)
        
        return df
    
    def fit_transform(self, df):
        self.fit(df) 
        return self.transform(df)
        
    
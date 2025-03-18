import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer

train_df = pd.read_csv("../titanic/train.csv")
test_df = pd.read_csv("../titanic/test.csv")

def preprocess_data(df, train_median=None):
    df = df.copy()
    df.drop(columns=["Cabin", "Name", "Ticket", "PassengerId"], inplace=True)
    
    df["Age"].fillna(train_median["Age"] if train_median else df["Age"].median(), inplace=True)
    df["Fare"].fillna(train_median["Fare"] if train_median else df["Fare"].median(), inplace=True)

    df["isMale"] = (df["Sex"] == "male").astype(int)
    df.drop(columns=["Sex"], inplace=True)
    
    return df

train_median = {
    "Age": train_df["Age"].median(),
    "Fare": train_df["Fare"].median()
}
processed_train = preprocess_data(train_df)

processed_test = preprocess_data(test_df, train_median)

features = ['Pclass', 'isMale', 'Age', 'SibSp', 'Parch', 'Fare']

processed_train = pd.get_dummies(processed_train)
processed_test = pd.get_dummies(processed_test)

processed_test = processed_test.reindex(columns=processed_train.columns, fill_value=0)

X = processed_train[features]
y = processed_train['Survived']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

test_predictions = model.predict(processed_test[features])

results = pd.DataFrame({
    "PassengerId": test_df["PassengerId"],
    "Survived": test_predictions
})


print(results["Survived"].mean())

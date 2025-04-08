import pandas as pd
from sklearn.base import check_is_fitted
from sklearn.model_selection import train_test_split
from src.model import model_Titanic

df = pd.DataFrame(pd.read_csv("../data/train.csv"))
X = df.drop('Survived', axis=1)
y = df['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = model_Titanic()

def test_train_shape():
    model.train(X_train, y_train)
    assert check_is_fitted(model), f"Erreur: le model n'a pas été fittée"

def test_predict_shape():
    predictions = model.predict(X_test)
    assert len(predictions) == len(X_test), f"Erreur: le nombre de lignes dans les prédictions ({len(predictions)}) ne correspond pas à X_test ({len(X_test)})"

def test_evaluate_shape():
    y_pred = model.predict(X_test)
    accuracy = model_Titanic.evaluate(y_test, y_pred)
    assert 0 <= accuracy <= 1, f"Erreur: l'accuracy devrait être entre 0 et 1, mais il est {accuracy}"

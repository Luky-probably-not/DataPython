import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

class model_Titanic:
    def __init__(self):    
        self.model = LogisticRegression(max_iter=1000)
        self.scaler = StandardScaler()
    
    def train(self, X, y):
        X_train_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_train_scaled, y)
    
    def predict(self, x_test):
        X_scaled = self.scaler.transform(x_test)
        return self.model.predict(X_scaled)
    
    def evaluate(self, y_test, y_pred):
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)

        print(f"Accuracy: {acc:.4f}")
        print(f"F1-Score: {f1:.4f}")
        print("Matrice de confusion :\n", cm)
        
        
# # Entraînement du modèle
# def train_model(x_train, y_train):
#     model = LogisticRegression(max_iter=1000)
#     model.fit(x_train, y_train)
#     return model
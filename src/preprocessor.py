import pandas as pd
from sklearn.model_selection import train_test_split

# Prétraitement des données
def preprocess(df):
    
    # Gestion des valeurs manquantes
    df.drop(columns=["Cabin", "Ticket", "Name", "PassengerId"], inplace=True) #trop de valeurs manquantes, pas rentable / Ticket et Name = valeurs inutiles et qui crash le prgrame ¯\_(ツ)_/¯
    df["Age"].fillna(df["Age"].median(), inplace=True) #colonne importante, donc on la remplie comme possible avec la mediane
    df["Fare"].fillna(df["Fare"].median(), inplace=True) #on remplace les valeurs manquantes par la mediane
    df["Embarked"].fillna(df["Embarked"].mode()[0], inplace=True) #juste 2 valeurs manquantes, on peut la remplir avec la valeur la plus frequente

    # Encodage des variables catégorielles
    df = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)

    X = df.drop('Survived', axis=1) # Correction: la target est 'Survived'
    y = df['Survived']

    # Split des données
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return x_train, x_test, y_train, y_test
    
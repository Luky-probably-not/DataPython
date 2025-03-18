import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Chargement des données
df = pd.read_csv("../titanic/train.csv")

# 1. Prétraitement des données
# Gestion des valeurs manquantes
df.drop(columns=["Cabin", "Ticket", "Name", "PassengerId"], inplace=True) #trop de valeurs manquantes, pas rentable / Ticket et Name = valeurs inutiles et qui crash le prgrame ¯\_(ツ)_/¯
df["Age"].fillna(df["Age"].median(), inplace=True) #colonne importante, donc on la remplie comme possible avec la mediane
df["Embarked"].fillna(df["Embarked"].mode()[0], inplace=True) #juste 2 valeurs manquantes, on peut la remplir avec la valeur la plus frequente

# Encodage des variables catégorielles
df = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)

# 2. Préparation des données
X = df.drop('Survived', axis=1) # Correction: la target est 'Survived'
y = df['Survived']

# Split des données
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Entraînement du modèle
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# 4. Évaluation
y_pred = model.predict(X_test)

print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print(f"F1-Score: {f1_score(y_test, y_pred):.2f}")

# Visualisation de la matrice de confusion
plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(y_test, y_pred), 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=['Non-Survivant', 'Survivant'],
            yticklabels=['Non-Survivant', 'Survivant'])
plt.title('Matrice de confusion')
plt.ylabel('Vérité terrain')
plt.xlabel('Prédictions')
plt.savefig('matrice_de_confusion.png', dpi=300)
plt.show()

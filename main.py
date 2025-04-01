import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from src.preprocessor import preprocess
from src.model import train_model

# Chargement des données
df = pd.read_csv("../titanic/train.csv")

x_train, x_test, y_train, y_test = preprocess(df)

model = train_model(x_train, y_train)

# Prediction et Évaluation
y_pred = model.predict(x_test)

print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print(f"F1-Score: {f1_score(y_test, y_pred):.2f}")
print("Survivor percentage: ", pd.DataFrame({"Survived" : y_pred})["Survived"].mean())

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
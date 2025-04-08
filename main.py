import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from src.preprocessor import preprocessor_Titanic
from src.model import model_Titanic


if __name__ == "__main__":
    
    filepath = "data/train.csv"
    preprocessor = preprocessor_Titanic()
    model = model_Titanic()
    
    df = preprocessor.fit_transform(pd.read_csv(filepath))
    X = df.drop('Survived', axis=1)
    y = df['Survived']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model.train(X_train, y_train)
    y_pred = model.predict(X_test)
    
    model.evaluate(y_test, y_pred)
    

    # x_train, y_train = preprocessor_Titanic.fit_transform(train)
    # model_Titanic.train(x_train, y_train)

    # x_test, y_test = preprocessor_Titanic.transform(test)
    # y_pred = model_Titanic(x_test)
    # model_Titanic.evaluate(y_pred,y_test)


    # # Chargement des données
    # x_train, x_test, y_train, y_test = preprocess(df)

    # model = train_model(x_train, y_train)


# Visualisation de la matrice de confusion
# plt.figure(figsize=(6, 4))
# sns.heatmap(confusion_matrix(y_test, y_pred), 
#             annot=True, 
#             fmt='d', 
#             cmap='Blues',
#             xticklabels=['Non-Survivant', 'Survivant'],
#             yticklabels=['Non-Survivant', 'Survivant'])
# plt.title('Matrice de confusion')
# plt.ylabel('Vérité terrain')
# plt.xlabel('Prédictions')
# plt.savefig('matrice_de_confusion.png', dpi=300)
# plt.show()
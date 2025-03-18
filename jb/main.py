import csv
import pandas as pd

df = pd.read_csv("titanic/train.csv")

print("missing data : ", df.isnull().sum()) 

df.drop(columns=["Cabin"], inplace=True) #trop de valeurs manquantes, pas rentable
df["Age"].fillna(df["Age"].median(), inplace=True) #colonne importante, donc on la remplie comme possible avec la mediane
df["Embarked"].fillna(df["Embarked"].mode()[0], inplace=True) #juste 2 valeurs manquantes, on peut la remplir avec la valeur la plus frequente
print(df.info())

from ydata_profiling import ProfileReport
 
profile = ProfileReport(df, title="Profiling Report")
profile.to_file("your_report.html")

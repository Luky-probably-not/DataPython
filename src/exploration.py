from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
from ydata_profiling import ProfileReport

df = pd.read_csv("../data/train.csv")
print(df.head())

#print("Average age is ", df["Age"].mean())

#df["Colonne_sans_sens"]=df["Age"]/df["Pclass"]

#df["IsMale"]=(df["Sex"]=="male").astype(int)

#print("The percentage of males is ", (df["Sex"].value_counts(normalize=True)['male']) * 100)

# Creation du Profile Report de ydata_profiling
profile = ProfileReport(df, title="Profiling Report")
profile.to_file("your_report.html")

# Graphique 1 : Lien entre l'âge et le prix du billet
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Age', y='Fare', data=df, hue='Pclass', palette='deep')
plt.title('Relation entre l\'âge et le prix du billet')
plt.xlabel('Âge')
plt.ylabel('Prix du billet')

plt.savefig('titanic_graph_Age-Fare.png', dpi=300)
plt.show()

# Graphique 2 : Lien entre le sexe et la survie
plt.figure(figsize=(8, 6))
sns.countplot(x='Sex', hue='Survived', data=df)
plt.title('Survie en fonction du sexe')
plt.xlabel('Sexe')
plt.ylabel('Nombre de passagers')
plt.legend(title='Survécu', labels=['Non', 'Oui'])

plt.savefig('titanic_graph_Sex_Survived.png', dpi=300)
plt.show()
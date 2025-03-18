import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("../titanic/train.csv")


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
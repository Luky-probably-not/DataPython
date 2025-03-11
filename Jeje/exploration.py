import pandas as pd

df = pd.read_csv("../titanic/train.csv")
print(df.head())

#print("Average age is ", df["Age"].mean())

#df["Colonne_sans_sens"]=df["Age"]/df["Pclass"]

#df["IsMale"]=(df["Sex"]=="male").astype(int)

#print("The percentage of males is ", (df["Sex"].value_counts(normalize=True)['male']) * 100)

from ydata_profiling import ProfileReport

profile = ProfileReport(df, title="Profiling Report")
profile.to_file("your_report.html")
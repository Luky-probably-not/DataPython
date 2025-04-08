import sqlite3
import pandas as pd

conn = sqlite3.connect("Chinook_Sqlite.sqlite")

cursor = conn.cursor()

#cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
# print(cursor.fetchall())


# 1. Quels sont les 5 clients ayant dépensé le plus ?
print("1. Quels sont les 5 clients ayant dépensé le plus ?\n")

query = '''
SELECT FirstName, LastName, SUM(total) AS InvoiceTotalPrice FROM customer
INNER JOIN invoice ON invoice.CustomerId = customer.CustomerId
Group By LastName
ORDER BY Sum(total) DESC
LIMIT 5;
'''
df = pd.read_sql_query(query, conn)
print(df,"\n\n")

# 2. Quels genres musicaux rapportent le plus ?
print("2. Quels genres musicaux rapportent le plus ?\n")

query = '''
SELECT genre.Name, SUM(InvoiceLine.UnitPrice) AS InvoiceTotalPrice FROM Genre
Inner JOIN Track ON Track.GenreId = genre.GenreId
Inner Join InvoiceLine ON invoiceLine.TrackId = Track.TrackId
INNER JOIN invoice ON invoice.InvoiceId = InvoiceLine.InvoiceId
Group BY genre.Name
ORDER BY Sum(total) DESC;
'''
df = pd.read_sql_query(query, conn)
print(df,"\n\n")

# 3. Quelle est la durée moyenne d’un morceau de Rock ?
print("3. Quelle est la durée moyenne d’un morceau de Rock ?\n")

query = '''
SELECT genre.Name, avg(Milliseconds) AS AverageDuration FROM genre
INNER JOIN track On genre.GenreId = track.GenreId
WHERE genre.Name = 'Rock'
GROUP BY genre.GenreId
ORDER BY AverageDuration DESC;
'''
df = pd.read_sql_query(query, conn)
print(df,"\n\n")

# 4. Quel employé (Sales Support Agent) a généré le plus de revenus ?
print("4. Quel employé (Sales Support Agent) a généré le plus de revenus ?\n")

query = '''
SELECT Employee.FirstName, Employee.LastName, SUM(Invoice.Total) AS TotalRevenue FROM Invoice 
JOIN Customer ON Customer.CustomerId = Invoice.CustomerId 
JOIN Employee ON Employee.EmployeeId = Customer.SupportRepId 
WHERE Employee.Title = 'Sales Support Agent' 
GROUP BY Employee.FirstName 
ORDER BY TotalRevenue 
DESC 
LIMIT 1;
'''
df = pd.read_sql_query(query, conn)
print(df,"\n\n")

conn.close()
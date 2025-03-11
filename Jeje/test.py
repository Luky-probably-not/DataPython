import csv
# read csv file to a list of dictionaries
with open('../titanic/train.csv', 'r') as file:
    csv_reader = csv.DictReader(file)
    data = [row for row in csv_reader]
print(data[0])
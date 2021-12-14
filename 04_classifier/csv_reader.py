import csv


def csv_reader(filename):
    with open(filename, 'r') as file:
        reader = csv.reader(file, delimiter=';')

        # skip header
        next(reader)

        for row in reader:
            print(row)

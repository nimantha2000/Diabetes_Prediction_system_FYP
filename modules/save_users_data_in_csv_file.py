import csv

csv_filename = 'datasets/user_data.csv'

def save_to_csv(data):
    with open(csv_filename, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(data)
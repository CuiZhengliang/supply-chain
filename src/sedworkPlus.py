import csv

if __name__ == '__main__':
    dataset = []
    with open(r'E:\PycharmProjects\Task2\dataset\goodsale.csv',newline='') as csvfile:
        column = csvfile.readline()
        column = column.rstrip().split(',')
        data = csv.reader(csvfile)
        for string in data:
            if string[-2].find(',') > 0:
                a,b = string[-2].split(',')
                string[-2] = str(a) + str(b)
            if string[-1].find(',') > 0:
                a,b = string[-1].split(',')
                string[-1] = str(a) + str(b)
            dataset.append(string)
    with open(r'E:\PycharmProjects\Task2\dataset\goodsalePlus.csv','a',newline='') as file:
        csv_write = csv.writer(file)
        csv_write.writerow(column)
        for i in dataset:
            csv_write.writerow(i)
        print("write over!")



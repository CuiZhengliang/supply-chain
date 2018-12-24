import csv

if __name__ == '__main__':
    dataset = []
    with open(r'E:\PycharmProjects\Task2\dataset\goodsale.csv',newline='') as csvfile:
        column = csvfile.readline()
        column = column.rstrip().split(',')
        data = csv.reader(csvfile)
        for string in data:
            length = len(string)
            if length == 8:
                p1 = str(string[-4]) + str(string[-3])
                p2 = str(string[-2]) + str(string[-1])
            elif length == 7:
                p1 = str(string[-3])
                p2 = str(string[-2]) + str(string[-1])
            else:
                p1 = str(string[-2])
                p2 = str(string[-1])
            Line = string[:4]
            Line.extend([p1,p2])
            dataset.append(Line)
    with open(r'E:\PycharmProjects\Task2\dataset\goodsalePlus.csv','a',newline='') as file:
        csv_write = csv.writer(file)
        csv_write.writerow(column)
        for i in dataset:
            csv_write.writerow(i)
        print("write over!")



import sys
import pandas as pd
import numpy as np

# 读入数据
data = pd.read_csv('../data/train.csv', encoding='big5')

# 数据预处理
data = data.iloc[:, 3:]
data[data == 'NR'] = 0
raw_data = data.to_numpy()

# 按月分割数据
month_data = {}
for month in range(12):
    sample = np.empty([18, 480])
    for day in range(20):
        sample[:, day * 24: (day + 1) * 24] = raw_data[18 * (20 * month + day): 18 * (20 * month + day + 1), :]
    month_data[month] = sample

# 分割x和y
x = np.empty([12 * 471, 18 * 9 * 2], dtype=float)
y = np.empty([12 * 471, 1], dtype=float)
for month in range(12):
    for day in range(20):
        for hour in range(24):
            if day == 19 and hour > 14:
                continue
            x1 = month_data[month][:, day * 24 + hour: day * 24 + hour + 9].reshape(1,
                                                                                    -1)  # vector dim:18*9 (9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9)
            x[month * 471 + day * 24 + hour, :18 * 9] = x1
            # 在这里加入了x的二次项
            x[month * 471 + day * 24 + hour, 18 * 9: 18 * 9 * 2] = np.power(x1, 2)
            y[month * 471 + day * 24 + hour, 0] = month_data[month][9, day * 24 + hour + 9]  # value

# 对x标准化
mean_x = np.mean(x, axis=0)  # 18 * 9 * 2
std_x = np.std(x, axis=0)  # 18 * 9 * 2
for i in range(len(x)):  # 12 * 471
    for j in range(len(x[0])):  # 18 * 9 * 2
        if std_x[j] != 0:
            x[i][j] = (x[i][j] - mean_x[j]) / std_x[j]


# 随机打散X和Y
def _shuffle(X, Y):
    randomize = np.arange(len(X))
    np.random.shuffle(randomize)
    return (X[randomize], Y[randomize])


# 训练模型并保存权重
dim = 18 * 9 * 2 + 1
w = np.ones([dim, 1])
learning_rate = 2
iter_time = 5000
adagrad = np.zeros([dim, 1])
eps = 1e-7

for t in range(iter_time):
    x, y = _shuffle(x, y)
    x2 = np.concatenate((np.ones([len(x), 1]), x), axis=1).astype(float)
    gradient = 2 * np.dot(x2.transpose(), np.dot(x2, w) - y)  # dim*1
    adagrad += gradient ** 2
    w = w - learning_rate * gradient / (np.sqrt(adagrad) + eps)

    loss = np.sqrt(np.sum(np.power(np.dot(x2, w) - y, 2)) / len(x))  # rmse
    if (t % 100 == 0):
        print(str(t) + ":" + str(loss))

np.save('weight.npy', w)

# 导入测试数据test.csv
testdata = pd.read_csv('../data/test.csv', header=None, encoding='big5')
test_data = testdata.iloc[:, 2:]
test_data[test_data == 'NR'] = 0
test_data = test_data.to_numpy()
test_x1 = np.empty([240, 18 * 9], dtype=float)
test_x = np.empty([240, 18 * 9 * 2], dtype=float)
for i in range(240):
    test_x1 = test_data[18 * i: 18 * (i + 1), :].reshape(1, -1).astype(float)
    # 同样在这里加入test x的二次项
    test_x[i, : 18 * 9] = test_x1
    test_x[i, 18 * 9:] = np.power(test_x1, 2)
for i in range(len(test_x)):
    for j in range(len(test_x[0])):
        if std_x[j] != 0:
            test_x[i][j] = (test_x[i][j] - mean_x[j]) / std_x[j]
test_x = np.concatenate((np.ones([240, 1]), test_x), axis=1).astype(float)

# 对test的x进行预测，得到预测值ans_y
w = np.load('weight.npy')
ans_y = np.dot(test_x, w)
# 加一个预处理<0的都变成0
for i in range(240):
    if (ans_y[i][0] < 0):
        ans_y[i][0] = 0
    else:
        ans_y[i][0] = np.round(ans_y[i][0])

# 保存为csv文件，并提交到kaggle：https://www.kaggle.com/c/ml2020spring-hw1/submissions
import csv

with open('submit.csv', mode='w', newline='') as submit_file:
    csv_writer = csv.writer(submit_file)
    header = ['id', 'value']
    print(header)
    csv_writer.writerow(header)
    for i in range(240):
        row = ['id_' + str(i), ans_y[i][0]]
        csv_writer.writerow(row)
import numpy as np
import csv
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn import neighbors
from sklearn.metrics import accuracy_score
from sklearn import metrics

def csv_reader(file_name):
    data = []
    with open(file_name) as file:
        csv_file = csv.reader(file)
        for line in csv_file:
            data.append(line)
    return (np.array(data[1:])).astype(float)

raw_data = csv_reader('CreditCards.csv')
raw_X = raw_data[:, :-1]
raw_y = raw_data[:, -1]
min_max_scaler = preprocessing.MinMaxScaler()
processed_X = min_max_scaler.fit_transform(raw_X)
train_X = processed_X[:621, :]
test_X = processed_X[621:, :]
train_y = raw_y[:621]
test_y = raw_y[621:]


clf = neighbors.KNeighborsClassifier(2)
clf.fit(train_X, train_y.astype('int'))
prediction = clf.predict(train_X)
acc = accuracy_score(train_y.astype('int'), prediction)
print(acc)
prediction = clf.predict(test_X)
acc = accuracy_score(test_y.astype('int'), prediction)
print(acc)

recall_on = metrics.recall_score(test_y.astype('int'), prediction)
prec_on = metrics.precision_score(test_y.astype('int'), prediction)
train_auc_list = []
test_auc_list = []
for k in range(1, 31):
    clf = neighbors.KNeighborsClassifier(k)
    clf.fit(train_X, train_y.astype('int'))
    train_prediction = clf.predict_proba(train_X)
    test_prediction = clf.predict_proba(test_X)

    train_auc_list.append(metrics.roc_auc_score(train_y.astype('int'), np.array(train_prediction[:,-1])))
    test_auc_list.append(metrics.roc_auc_score(test_y.astype('int'), np.array(test_prediction[:,-1])))

opt_k = np.argmax(test_auc_list)+1
print(np.max(test_auc_list))
print(opt_k)
plt.plot(train_auc_list)
plt.show()
plt.plot(test_auc_list)
plt.show()

clf = neighbors.KNeighborsClassifier(opt_k)
clf.fit(train_X, train_y.astype('int'))
prediction = clf.predict(train_X)
recall_opt = metrics.recall_score(train_y.astype('int'), prediction)
prec_opt = metrics.precision_score(train_y.astype('int'), prediction)

print(recall_on, recall_opt)
print(prec_on, prec_opt)

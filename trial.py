import csv
import pickle
import numpy as np
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import OrdinalEncoder, Normalizer
import random
from queue import Queue
from threading import Thread
import multiprocessing

def prepareData(result_queue):
    Y = []
    list = []
    seen = {}
    cur_num = 0
    count = -1
    with open('all/train_V2.csv', 'rt') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in spamreader:
            count += 1
            if count == 0:
                continue

            if row[1] not in seen:
                seen[row[1]] = float(cur_num)
                row[1] = float(cur_num)
                cur_num += 1
            else:
                row[1] = seen[row[1]]

            if row[2] not in seen:
                seen[row[2]] = float(cur_num)
                row[2] = float(cur_num)
                cur_num += 1
            else:
                row[2] = seen[row[2]]

            if row[15] not in seen:
                seen[row[15]] = float(cur_num)
                row[15] = float(cur_num)
                cur_num += 1
            else:
                row[15] = seen[row[15]]

            try:
                Y.append(float(row[-1]))

                row = row[1:-1]
                row = [float(i) for i in row]
                list.append(row)
            except:
                print(row[-1])

        enc = OrdinalEncoder()
        X = enc.fit_transform(list, Y)

    result_queue.put([X, Y])


def prepareTest(result_queue):
    Y = []
    list = []
    seen = {}
    cur_num = 0
    count = -1
    with open('all/test_V2.csv', 'rt') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in spamreader:
            count += 1
            if count == 0:
                continue

            if row[1] not in seen:
                seen[row[1]] = float(cur_num)
                row[1] = float(cur_num)
                cur_num += 1
            else:
                row[1] = seen[row[1]]

            if row[2] not in seen:
                seen[row[2]] = float(cur_num)
                row[2] = float(cur_num)
                cur_num += 1
            else:
                row[2] = seen[row[2]]

            if row[15] not in seen:
                seen[row[15]] = float(cur_num)
                row[15] = float(cur_num)
                cur_num += 1
            else:
                row[15] = seen[row[15]]

            try:
                Y.append((row[0]))

                row = row[1:]
                row = [float(i) for i in row]
                list.append(row)
            except:
                print(row[-1])

        enc = OrdinalEncoder()
        X = enc.fit_transform(list, Y)
    result_queue.put([X, Y])


def random_partition(X, Y, size):
    partition = random.sample(range(0, len(X) - 1), size)
    partition = [int(i) for i in partition]
    partition = np.asarray(partition)

    x_new = [X[i] for i in partition]
    y_new = [Y[i] for i in partition]

    return x_new, y_new

result_queue1 = Queue()
result_queue2 = Queue()
t1 = Thread(target=prepareData, args=[result_queue1])
t2 = Thread(target=prepareTest, args=[result_queue2])

t1.start()
t2.start()
t1.join()
t2.join()

X,Y=result_queue1.get()
test,label=result_queue2.get()
print(np.shape(test))
print(np.shape(X))

partx, party = random_partition(X, Y, 600000)
print(np.shape(partx))
nrm=Normalizer()
partx = nrm.fit_transform(partx)
print(np.shape(partx))
test=nrm.transform(test)
print(np.shape(test))


parameters=[[35, 45, 36], 2.0, 0.0012044393115519438, 1.0, 3.5784753152461046e-06, 0.2751874654816516, 0.0023804489838965626, 0.0, -0.23317429215526964]
hidden_layers = parameters[0]
solver=["lbfgs", "sgd", "adam"]
learning_rate=["constant", "adaptive"]
cur_solver = solver[int(parameters[1])]
cur_learning_rate = learning_rate[int(parameters[3])]
cur_nester = False
if parameters[7] == 1:
    cur_nester = True

cur_momentum = parameters[6]
reg = MLPRegressor(hidden_layer_sizes=hidden_layers, activation="relu", solver=cur_solver, alpha=parameters[2],
                               batch_size='auto', learning_rate=cur_learning_rate, learning_rate_init=parameters[4], power_t=parameters[5],
                               max_iter=200, shuffle=True, random_state=None, tol=parameters[8], verbose=False, warm_start=False,
                               momentum=cur_momentum, nesterovs_momentum=cur_nester, early_stopping=False, validation_fraction=0.1,
                               beta_1=0.9, beta_2=0.999, epsilon=1e-08, n_iter_no_change=10)

#score = cross_val_score(reg, partx, party, cv=3,n_jobs= multiprocessing.cpu_count())
#print(np.mean(score))

reg.fit(partx,party)
predicts=reg.predict(test)
print(np.shape(predicts))

print(len(predicts))
result=[['Id','winPlacePerc']]
result+=[[i]+[j] for i,j in zip(label,predicts)]
with open("preds.csv", "w") as output:
    writer = csv.writer(output, lineterminator='\n')
    writer.writerows(result)

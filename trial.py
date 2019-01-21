import csv
import pickle
import numpy as np
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import OrdinalEncoder, Normalizer


def prepareData():
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

    return X, Y

def prepareTest():
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
                Y.append((row[0]))

                row = row[1:-1]
                row = [float(i) for i in row]
                list.append(row)
            except:
                print(row[-1])

        enc = OrdinalEncoder()
        X = enc.fit_transform(list, Y)

    return X, Y


fittest=pickle.load(open("fittest.p","rb"))
print(fittest)
parameters = fittest
hidden_layers = parameters[0]
# a,b=vars(self.solver).items()
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

print("preparing training set")
X,Y=prepareData()
print("preparing test set")
test,label=prepareTest()
print("Normalizing data")
X = Normalizer().fit_transform(X,Y)

#score = cross_val_score(reg, X, Y, cv=3)
print("Fitting network...")
reg.fit(X,Y)
predicts=reg.predict(test)
pickle.dump(predicts,open('predicts.p','wb'))
result=[[i]+[j] for i,j in zip(label,predicts)]
np.savetxt("preds.csv", result, delimiter=",")
print(result)
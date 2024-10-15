import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from keras.callbacks import EarlyStopping
from keras.layers import Dense, Dropout, LSTM
from keras.models import Sequential
from tensorflow.python.keras.layers import Activation


class SSA:
    def __init__(self, func, n_dim=None, pop_size=20, max_iter=50, lb=-512, ub=512, verbose=False):
        self.func = func
        self.n_dim = n_dim  # dimension of particles, which is the number of variables of func
        self.pop = pop_size  # number of particles
        p_percent = 0.2  # producer proportion
        d_percent = 0.1  # vigilance ratio
        self.pNum = round(self.pop * p_percent)  # producer num
        self.warn = round(self.pop * d_percent)  # vigilance number

        self.max_iter = max_iter  # max iter
        self.verbose = verbose  # print the result of each iter or not

        self.lb, self.ub = np.array(lb) * np.ones(self.n_dim), np.array(ub) * np.ones(self.n_dim)
        assert self.n_dim == len(self.lb) == len(self.ub), 'dim == len(lb) == len(ub) is not True'
        assert np.all(self.ub > self.lb), 'upper-bound must be greater than lower-bound'

        self.X = np.random.uniform(low=self.lb, high=self.ub, size=(self.pop, self.n_dim))

        self.Y = [self.func(self.X[i]) for i in range(len(self.X))]  # y = f(x) for all particles
        self.pbest_x = self.X.copy()  # personal best location of every particle in history
        self.pbest_y = [np.inf for i in range(self.pop)]  # best image of every particle in history
        self.gbest_x = self.pbest_x.mean(axis=0).reshape(1, -1)  # global best location for all particles
        self.gbest_y = np.inf  # global best y for all particles
        self.gbest_y_hist = []  # global best y of every iteration
        self.update_pbest()
        self.update_gbest()
        #
        # record verbose values
        self.record_mode = False
        self.record_value = {'X': [], 'V': [], 'Y': []}
        self.best_x, self.best_y = self.gbest_x, self.gbest_y
        self.idx_max = 0
        self.x_max = self.X[self.idx_max, :]
        self.y_max = self.Y[self.idx_max]

    def cal_y(self, start, end):
        # calculate y for every x in X
        for i in range(start, end):
            self.Y[i] = self.func(self.X[i])
        # return self.Y

    def update_pbest(self):
        '''
        personal best
        '''
        for i in range(len(self.Y)):
            if self.pbest_y[i] > self.Y[i]:
                self.pbest_x[i] = self.X[i]
                self.pbest_y[i] = self.Y[i]

    def update_gbest(self):
        idx_min = self.pbest_y.index(min(self.pbest_y))
        if self.gbest_y > self.pbest_y[idx_min]:
            self.gbest_x = self.X[idx_min, :].copy()
            self.gbest_y = self.pbest_y[idx_min]

    def find_worst(self):
        self.idx_max = self.Y.index(max(self.Y))
        self.x_max = self.X[self.idx_max, :]
        self.y_max = self.Y[self.idx_max]

    def update_finder(self):
        r2 = np.random.rand(1)  # risk level
        self.idx = sorted(enumerate(self.Y), key=lambda x: x[1])
        self.idx = [self.idx[i][0] for i in range(len(self.idx))]
        if r2 < 0.8:  # relative low risk
            for i in range(self.pNum):
                r1 = np.random.rand(1)
                self.X[self.idx[i], :] = self.X[self.idx[i], :] * np.exp(-(i) / (r1 * self.max_iter))
                self.X = np.clip(self.X, self.lb, self.ub)  # remove variables that exceed the boundary
        elif r2 >= 0.8:
            for i in range(self.pNum):
                Q = np.random.rand(1)
                self.X[self.idx[i], :] = self.X[self.idx[i], :] + Q * np.ones(
                    (1, self.n_dim))
                self.X = np.clip(self.X, self.lb, self.ub)  # remove variables that exceed the boundary
        self.cal_y(0, self.pNum)

    def update_follower(self):
        for ii in range(self.pop - self.pNum):
            i = ii + self.pNum
            A = np.floor(np.random.rand(1, self.n_dim) * 2) * 2 - 1
            best_idx = self.Y[0:self.pNum].index(min(self.Y[0:self.pNum]))
            bestXX = self.X[best_idx, :]
            if i > self.pop / 2: # hungry follower
                Q = np.random.rand(1)
                self.X[self.idx[i], :] = Q * np.exp((self.x_max - self.X[self.idx[i], :]) / np.square(i))
            else:
                self.X[self.idx[i], :] = bestXX + np.dot(np.abs(self.X[self.idx[i], :] - bestXX),
                                                         1 / (A.T * np.dot(A, A.T))) * np.ones((1, self.n_dim))
        self.X = np.clip(self.X, self.lb, self.ub)  # remove variables that exceed the boundary
        self.cal_y(self.pNum, self.pop)

    def detect(self):
        arrc = np.arange(self.pop)
        c = np.random.permutation(arrc)
        b = [self.idx[i] for i in c[0: self.warn]]
        e = 10e-10
        for j in range(len(b)):
            if self.Y[b[j]] > self.gbest_y:
                self.X[b[j], :] = self.gbest_x + np.random.rand(1, self.n_dim) * np.abs(self.X[b[j], :] - self.gbest_x)
            else:
                self.X[b[j], :] = self.X[b[j], :] + (2 * np.random.rand(1) - 1) * np.abs(
                    self.X[b[j], :] - self.x_max) / (self.func(self.X[b[j]]) - self.x_max + e)
            self.X = np.clip(self.X, self.lb, self.ub)  # remove variables that exceed the boundary
            self.Y[b[j]] = self.func(self.X[b[j]])

    def run(self, max_iter=None):
        self.max_iter = max_iter or self.max_iter
        for iter_num in range(self.max_iter):
            print("%d iteration" % iter_num)
            self.update_finder()
            self.find_worst()
            self.update_follower()
            self.update_pbest()
            self.update_gbest()
            self.detect()
            self.update_pbest()
            self.update_gbest()
            self.gbest_y_hist.append(self.gbest_y)
        return self.best_x, self.best_y


np.random.seed(666)
matplotlib.rcParams['agg.path.chunksize'] = 0
matplotlib.rcParams.update(matplotlib.rc_params())


def process_data():
    dataset = pd.read_csv("data.csv")
    dataset['flow'] = scaler.fit_transform(dataset['flow'].values.reshape(-1, 1))
    x = dataset['flow']
    y = dataset['flow']
    train_x = x.iloc[:9212]
    train_y = y.iloc[:9212]
    test_x = x.iloc[9212:]
    test_y = y.iloc[9212:]
    return train_x, train_y, test_x, test_y


def create_dataset(x, y, seq_len):
    features = []  # independent variable
    targets = []

    for i in range(0, len(x) - seq_len, 1):  # time step
        data = x.iloc[i:i + seq_len]  # independent variable
        label = y.iloc[i + seq_len]
        features.append(data) # independent variable
        targets.append(label)
    modify_x = np.array(features).astype('float64')
    modify_y = np.array(targets).reshape(-1, 1)
    return modify_x, modify_y


def build_model(neurons1, neurons2):
    train_x, train_y, test_x, test_y = process_data()
    train_x, train_y = create_dataset(train_x, train_y, steps)
    test_x, test_y = create_dataset(test_x, test_y, steps)
    model1 = Sequential()
    model1.add(LSTM(
        input_shape=(steps, 1),
        units=neurons1,
        return_sequences=True))
    model1.add(Dropout(dropout))

    model1.add(LSTM(
        units=neurons2,
        return_sequences=False))
    model1.add(Dropout(dropout))

    model1.add(Dense(units=1))
    model1.add(Activation("linear"))
    model1.compile(loss='mse', optimizer='Adam', metrics='mae')
    return model1, train_x, train_y, test_x, test_y


def training(parameters):
    neurons1 = int(parameters[0])
    neurons2 = int(parameters[1])
    batch_size = int(parameters[2])
    print(parameters)
    model, train_x, train_y, test_x, test_y = build_model(neurons1, neurons2)
    model.fit(
        train_x,
        train_y,
        batch_size=batch_size,
        epochs=100,
        validation_split=0.25,
        verbose=0,
        callbacks=[EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)])

    predict_y = model.predict(test_x)
    scaler_predict_y = scaler.inverse_transform(predict_y)
    scaler_test_y = scaler.inverse_transform(test_y)
    temp_mse = mean_squared_error(scaler_test_y, scaler_predict_y)
    temp_rmse = np.sqrt(temp_mse)
    return temp_rmse


if __name__ == '__main__':
    '''
    number of neurons in the first layer
    number of neurons in the second layer
    dropout ratio
    batch_size
    '''
    UP = [150, 15, 16]
    DOWN = [50, 5, 8]
    dropout = 0.2
    steps = 10
    scaler = MinMaxScaler(feature_range=(0, 1))

    # optimization starts
    ssa = SSA(training, n_dim=3, pop_size=10, max_iter=25, lb=DOWN, ub=UP)
    ssa.run()
    print('best_params is ', ssa.gbest_x)
    print('best_precision is', 1 - ssa.gbest_y)

    # train final model with the best hyperparameters
    best_neuron1 = int(ssa.gbest_x[0])
    best_neuron2 = int(ssa.gbest_x[1])
    best_batch_size = int(ssa.gbest_x[2])
    final_model, x_train, y_train, x_test, y_test = build_model(best_neuron1, best_neuron2)
    history1 = final_model.fit(x_train, y_train, epochs=100, batch_size=best_batch_size,
                               validation_split=0.25, verbose=1,
                               callbacks=[EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)])
    # output prediction
    y_score = final_model.predict(x_test)
    scaler_y_score = scaler.inverse_transform(y_score)
    scaler_y_test = scaler.inverse_transform(y_test)
    # draw the figure
    plt.figure(figsize=(10, 10))
    plt.plot(ssa.gbest_y_hist)
    plt.show()

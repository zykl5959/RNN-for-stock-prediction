# -*- coding: utf-8 -*-
# 11/30/2019

"""
    Our LSTM model is constructed with the reference of the following blog:
    https://blog.varunajayasiri.com/numpy_lstm.html
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

plt.style.use('seaborn-white')

"""Constants and Hyperparameters"""
input_size = 1
hidden_size = 50
seq = 3
learn_rate = 1e-1
std_weight = 0.5
concat_size = hidden_size + input_size

"""
Activation functions
"""


def tanh(x):
    return np.tanh(x)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def diff_tanh(x):
    return 1 - x * x


def diff_sigmoid(x):
    return x * (1 - x)


"""
Parameters
"""


class Unit:
    def __init__(self, name, value):
        # parameter name
        self.name = name
        # parameter value
        self.value = value
        # parameter derivative
        self.deriv = np.zeros_like(value)
        self.moment = np.zeros_like(value)


# Initialize the weights and bias
class Params:
    def __init__(self):
        self.Wi = Unit('Wi', np.random.randn(hidden_size, concat_size) * std_weight + 0.5)
        self.bi = Unit('bi', np.zeros((hidden_size, 1)))

        self.Wf = Unit('Wf', np.random.randn(hidden_size, concat_size) * std_weight + 0.5)
        self.bf = Unit('bf', np.zeros((hidden_size, 1)))

        self.Wo = Unit('Wo', np.random.randn(hidden_size, concat_size) * std_weight + 0.5)
        self.bo = Unit('bo', np.zeros((hidden_size, 1)))

        self.WC = Unit('WC', np.random.randn(hidden_size, concat_size) * std_weight)
        self.bC = Unit('bC', np.zeros((hidden_size, 1)))

        self.Wv = Unit('Wv', np.random.randn(input_size, hidden_size) * std_weight)
        self.bv = Unit('bv', np.zeros((input_size, 1)))

    def params(self):
        return [self.Wi, self.Wf, self.Wo, self.WC, self.Wv, self.bf, self.bi, self.bC, self.bo, self.bv]


"""LSTM model"""


class LSTM:
    def __init__(self, params):
        self.parameters = params

    def reset_gradients(self):
        for param in self.parameters.params():
            param.deriv.fill(0)

    def forward_propgate(self, input, prev_hidden, prev_cell):

        concat = np.row_stack((prev_hidden, input))
        forget_gate = sigmoid(np.dot(self.parameters.Wf.value, concat) + self.parameters.bf.value)
        input_gate = sigmoid(np.dot(self.parameters.Wi.value, concat) + self.parameters.bi.value)

        current_Cell = tanh(np.dot(self.parameters.WC.value, concat) + self.parameters.bC.value)
        update_Cell = forget_gate * prev_cell + input_gate * current_Cell

        output_gate = sigmoid(np.dot(self.parameters.Wo.value, concat) + self.parameters.bo.value)
        hidden = output_gate * tanh(update_Cell)

        res = np.dot(self.parameters.Wv.value, hidden) + self.parameters.bv.value
        y = res
        return concat, forget_gate, input_gate, current_Cell, update_Cell, output_gate, hidden, res, y

    def backward_propgate(self, target, d_h_next, next_dC, prev_cell,
                          concat, forget, input, C_curr, C, output, hidden, res, y):

        d_v = y - target

        self.parameters.Wv.deriv += np.dot(d_v, hidden.T)
        self.parameters.bv.deriv += d_v

        d_h = np.dot(self.parameters.Wv.value.T, d_v)
        d_h += d_h_next
        d_o = d_h * tanh(C)
        d_o = diff_sigmoid(output) * d_o
        self.parameters.Wo.deriv += np.dot(d_o, concat.T)
        self.parameters.bo.deriv += d_o

        d_C = np.copy(next_dC)
        d_C += d_h * output * diff_tanh(tanh(C))
        d_C_curr = d_C * i
        d_C_curr = diff_tanh(C_curr) * d_C_curr
        self.parameters.WC.deriv += np.dot(d_C_curr, concat.T)
        self.parameters.bC.deriv += d_C_curr

        d_f = d_C * prev_cell
        d_f = diff_sigmoid(forget) * d_f
        self.parameters.Wf.deriv += np.dot(d_f, concat.T)
        self.parameters.bf.deriv += d_f

        d_i = d_C * C_curr
        d_i = diff_sigmoid(input) * d_i
        self.parameters.Wi.deriv += np.dot(d_i, concat.T)
        self.parameters.bi.deriv += d_i

        d_z = (np.dot(self.parameters.Wf.value.T, d_f)
               + np.dot(self.parameters.Wi.value.T, d_i)
               + np.dot(self.parameters.WC.value.T, d_C_curr)
               + np.dot(self.parameters.Wo.value.T, d_o))
        d_h_prev = d_z[:hidden_size, :]
        d_C_prev = forget * d_C

        return d_h_prev, d_C_prev

    def train(self, inputs, targets, prev_hidden, prev_cell):

        # temporary parameters
        feature, concat, f, i, = {}, {}, {}, {}
        C_curr, C, o, h = {}, {}, {}, {}
        v, y = {}, {}

        # Values at t - 1
        loss = 0
        h[-1] = np.copy(prev_hidden)
        C[-1] = np.copy(prev_cell)

        # Forward pass
        for t in range(len(inputs)):
            feature[t] = inputs[t]

            (concat[t], f[t], i[t], C_curr[t], C[t], o[t], h[t], v[t], y[t]) \
                = self.forward_propgate(feature[t], h[t - 1], C[t - 1])
            loss += ((y[t] - targets[t]) ** 2) / len(inputs)

        # reset the gradients
        self.reset_gradients()

        dh_next = np.zeros_like(h[0])  # dh from the next character
        next_dC = np.zeros_like(C[0])  # dh from the next character

        # Backward pass
        for t in reversed(range(len(inputs))):
            dh_next, next_dC = self.backward_propgate(target=targets[t], d_h_next=dh_next,
                                                      next_dC=next_dC, prev_cell=C[t - 1],
                                                      concat=concat[t], forget=f[t], input=i[t], C_curr=C_curr[t],
                                                      C=C[t], output=o[t], hidden=h[t], res=v[t],
                                                      y=y[t])

        return loss, h[len(inputs) - 1], C[len(inputs) - 1]

    # update parameters
    def update(self):
        for param in self.parameters.params():
            param.moment += param.deriv * param.deriv  # Calculate sum of gradients
            param.value -= learn_rate * param.deriv / np.sqrt(param.moment + 1e-8)


if __name__ == "__main__":
    """
    data preprocessing, train : test = 8 : 2
    Use the previous 'Close' prices to predict future 'Close' prices
    """
    # load the data from the UTD web account
    df = pd.read_csv('https://personal.utdallas.edu/~qxy180009/^GSPC-3.csv')
    data = np.array(df['Close'])
    # normalize the data

    normalize_data = (data - np.mean(data)) / np.std(data)

    normal_train_set = normalize_data[:int(len(normalize_data) * 0.8)]
    normal_test_set = normalize_data[int(len(normalize_data) * 0.8):]

    # get train and test data
    train_set_x, train_set_y = [], []
    for i in range(len(normal_train_set) - seq - 1):
        x_seq = normal_train_set[i:i + seq]
        y_seq = normal_train_set[i + 1:i + seq + 1]
        train_set_x.append(x_seq.tolist())
        train_set_y.append(y_seq.tolist())

    test_set_x, test_set_y = [], []
    for i in range(len(normal_test_set) - seq - 1):
        x_seq = normal_test_set[i:i + seq]
        y_seq = normal_test_set[i + 1:i + seq + 1]
        test_set_x.append(x_seq.tolist())
        test_set_y.append(y_seq.tolist())

    """ 
    Train the LSTM model with the train_data_set
    """
    loss_ = []
    temploss = []
    epoch = 10
    curepoch = 0
    iteration, pointer = 0, 0

    # create the LSTM model
    parameters = Params()
    lstm = LSTM(parameters)

    prev_hidden = np.zeros((hidden_size, 1))
    prev_cell = np.zeros((hidden_size, 1))
    while curepoch < epoch:
        curepoch += 1
        for i in range(len(train_set_x)):

            inputs = np.array([[[j]] for j in train_set_x[i]])
            targets = np.array([[[j]] for j in train_set_y[i]])

            loss, prev_hidden, prev_cell = lstm.train(inputs, targets, prev_hidden, prev_cell)
            temploss.append(loss[0][0])
            if (i + 1) % (len(train_set_x)) == 0:
                print("curr_epoch/epoch " + str(curepoch) + "/" + str(epoch))
                print("" + str(i + 1) + "/" + str(len(train_set_x)) + " " + str(np.mean(temploss)))
                loss_.append(np.mean(temploss))
                temploss = []

            lstm.update()

    """ 
        predict the prices for the train data set based on the trained LSTM model
    """
    # Use the LSTM model to predict the prices of the test data set
    plt_test_predict = []
    plt_test_target = []

    for i in range(len(test_set_x)):
        prev_h = prev_hidden
        prev_c = prev_cell
        # input= np.array(test_set_x[i])
        predictlist = []
        for j in range(len(test_set_x[i])):
            input = np.zeros((1, 1))
            input[0][0] = test_set_x[i][j]
            _, _, _, _, _, _, _, prediction, _ = lstm.forward_propgate(input, prev_hidden, prev_cell)
            predictlist.append(prediction[0][0])
        plt_test_predict.append(predictlist[-1])
        plt_test_target.append(test_set_y[i][-1])

    x = np.arange(len(plt_test_predict))

    # predicted results and actual results
    plt.plot(x, plt_test_predict, x, plt_test_target)
    plt.show()

    # loss
    plt.plot(loss_)
    plt.show()

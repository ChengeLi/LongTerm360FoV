"""get hidden states from other users"""
import tensorflow as tf
from mycode.config import cfg
import pdb

# method 1: MLP
def multilayer_perceptron_hidden_state(others_future):
    n_hidden_1 = 128
    n_hidden_2 = 64
    # n_hidden_3 = 8
    # n_hidden_4 = 8
    n_input = 47*cfg.running_length
    weights = {
        'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1], stddev=0.001)),
        'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2], stddev=0.001)),
        # 'h3': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3], stddev=0.001)),
        # 'h4': tf.Variable(tf.random_normal([n_hidden_3, n_hidden_4], stddev=0.001)),
    }
    biases = {
        'b1': tf.Variable(tf.random_normal([n_hidden_1], stddev=0.001)),
        'b2': tf.Variable(tf.random_normal([n_hidden_2], stddev=0.001)),
        # 'b3': tf.Variable(tf.random_normal([n_hidden_3], stddev=0.001)),
        # 'b4': tf.Variable(tf.random_normal([n_hidden_4], stddev=0.001)),
    }

    # TODO: is this correct??
    others_future_reshape = tf.reshape(others_future, [batch_size, -1])

    """provide hidden state to LSTM"""
    # Hidden fully connected layer with 256 neurons
    layer_1 = tf.add(tf.matmul(others_future_reshape, weights['h1']), biases['b1'])
    # Hidden fully connected layer with 256 neurons
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    # layer_3 = tf.add(tf.matmul(layer_2, weights['h3']), biases['b3'])
    # layer_4 = tf.add(tf.matmul(layer_3, weights['h4']), biases['b4'])
    return layer_2


def multilayer_perceptron_hidden_state_series(others_future,n_hidden):
    n_hidden_1 = 128
    n_hidden_2 = n_hidden #must be the same with n_hidden(state size)
    layer_1 = tf.contrib.layers.fully_connected(others_future, n_hidden_1)
    layer_2 = tf.contrib.layers.fully_connected(layer_1, n_hidden_2, activation_fn=None)

    return layer_2

def multilayer_perceptron_hidden_state(others_future,batch_size,n_hidden):
    n_hidden_1 = 128
    n_hidden_2 = n_hidden
    others_future = tf.reshape(others_future,[batch_size,cfg.running_length*47])
    layer_1 = tf.contrib.layers.fully_connected(others_future, n_hidden_1)
    layer_2 = tf.contrib.layers.fully_connected(layer_1, n_hidden_2, activation_fn=None)

    return layer_2

# method 2: model others also in RNN
# provide the hidden states from RNN
def dynamicRNN_hidden_state(x,num_layers,n_hidden,rnn_tuple_state):
    """2 twy layer LSTM using dynamic_rnn"""
    # w/o dropout
    cells = []
    with tf.variable_scope('LSTM_others') as scope:
        for _ in range(num_layers):
          # cell = tf.contrib.rnn.GRUCell(n_hidden)
          cell = tf.contrib.rnn.LSTMCell(n_hidden,state_is_tuple=True)
          cells.append(cell)
        cell = tf.contrib.rnn.MultiRNNCell(cells)

        # Batch size x time steps x features.
        states_series, current_state = tf.nn.dynamic_rnn(cell, x, dtype=tf.float32,
                                          initial_state=rnn_tuple_state)
    return states_series, current_state

























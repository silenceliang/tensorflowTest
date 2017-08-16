import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import numpy as np

def add_layer(inputs, in_size, out_size, activation_function=None,):
    # add one more layer and return the output of this layer
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1,)
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs


_2x150_data = np.linspace(-1, 1, 1200).reshape(8, 150)
_150x2_data = np.linspace(-1, 1, 300).reshape(150, 2)
_2x2_zeroMatrix = np.zeros([2, 2]) + 0.001  # <<< 差別
_1x2_zeroMatrix = np.zeros([1, 2]) + 0.001  # <<< 差別

#print(np.matmul(_2x150_data, _150x2_data) + _2x2_zeroMatrix, "\n")
print(np.matmul(_2x150_data, _150x2_data) + _1x2_zeroMatrix, "\n")
print(np.matmul(_2x150_data, _150x2_data) + 0.001, "\n")  # add constant(0.001)
print('why,there is the same result ? ')




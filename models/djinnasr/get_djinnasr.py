import tensorflow as tf
import numpy as np
from functools import reduce

# From ClarityLab:
# https://github.com/claritylab/djinn/blob/master/common/configs/asr.prototxt

class djinnasr:

    def __init__(self):
        self.var_dict = {}

    def build(self, input):

        self.fc1 = self.fc_layer(input, 1*440, 2048, "fc1")
        self.relu1 = tf.nn.sigmoid(self.fc1)

        self.fc2 = self.fc_layer(self.relu1, 2048, 2048, "fc2")
        self.relu2 = tf.nn.sigmoid(self.fc2)
        
        self.fc3 = self.fc_layer(self.relu2, 2048, 2048, "fc3")
        self.relu3 = tf.nn.sigmoid(self.fc3)
        
        self.fc4 = self.fc_layer(self.relu3, 2048, 2048, "fc4")
        self.relu4 = tf.nn.sigmoid(self.fc4)
        
        self.fc5 = self.fc_layer(self.relu4, 2048, 2048, "fc5")
        self.relu5 = tf.nn.sigmoid(self.fc5)
        
        self.fc6 = self.fc_layer(self.relu5, 2048, 2048, "fc6")
        self.relu6 = tf.nn.sigmoid(self.fc6)
        
        self.fc7 = self.fc_layer(self.relu6, 2048, 1706, "fc7")
        self.prob = tf.nn.softmax(self.fc7, name="prob")

    def fc_layer(self, bottom, in_size, out_size, name):
        with tf.variable_scope(name):
            weights, biases = self.get_fc_var(in_size, out_size, name)

            # x = tf.reshape(bottom, [-1, in_size])
            fc = tf.nn.bias_add(tf.matmul(bottom, weights), biases)

            return fc

    def get_fc_var(self, in_size, out_size, name):
        initial_value = tf.truncated_normal([in_size, out_size], 0.0, 0.001)
        weights = self.get_var(initial_value, name, 0, name + "_weights")

        initial_value = tf.truncated_normal([out_size], .0, .001)
        biases = self.get_var(initial_value, name, 1, name + "_biases")

        return weights, biases

    def get_var(self, initial_value, name, idx, var_name):
        value = initial_value
        var = tf.Variable(value, name=var_name)
        self.var_dict[(name, idx)] = var
        assert var.get_shape() == initial_value.get_shape()
        return var

batch1 = np.zeros((1, 440))
print batch1.shape

random_result = [1 if i == 123 else 0 for i in range(1706)]

with tf.device('/cpu:0'):
    sess = tf.Session()
    audiovec = tf.placeholder(tf.float32, [1, 440])
    true_out = tf.placeholder(tf.float32, [1, 1706])
    net = djinnasr()
    net.build(audiovec)
    sess.run(tf.global_variables_initializer())
    # 1-step of training
    cost = tf.reduce_sum((net.prob - true_out) ** 2)
    train = tf.train.GradientDescentOptimizer(0.0001).minimize(cost)
    saver = tf.train.Saver()
    sess.run(train, feed_dict={audiovec: batch1, true_out: [random_result]})
    np.savetxt('./djinnasr_batch_of_1.csv',  batch1.flatten())
    saver.save(sess, './djinnasr')

print
print 'Next step:'
print ' $ python  create_inference_graph.py  checkpoint  models/djinnasr/djinnasr  prob  models/djinnasr/  djinn'
print 'Then optimize the graph using:'
print ' $ python  optimize_inference_graph.py  models/djinnasr/djinn.pb  Placeholder  prob  1,440'

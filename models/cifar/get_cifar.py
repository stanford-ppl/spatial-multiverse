import tensorflow as tf
import skimage
import skimage.io
import skimage.transform
import numpy as np
from functools import reduce

# From the TensorFlow example model:
# https://github.com/tensorflow/models/blob/master/tutorials/image/cifar10/cifar10.py
IMG_SIZE = 24

# A trainable version of Cifar10.
class Cifar10:

    def __init__(self, cifar10_npy_path=None, trainable=True):
        if cifar10_npy_path is not None:
            self.data_dict = np.load(cifar10_npy_path, encoding='latin1').item()
        else:
            self.data_dict = None

        self.var_dict = {}
        self.trainable = trainable

    # build the DNN
    def build(self, rgb, train_mode=None):

        self.conv1 = self.conv_layer(rgb, 3, 64, "conv1")
        self.pool1 = self.max_pool(self.conv1, 'pool1')
        self.conv2 = self.conv_layer(self.pool1, 64, 64, "conv2")
        self.pool2 = self.max_pool(self.conv2, 'pool2')

        self.fc6 = self.fc_layer(self.pool2, 2304, 384, "fc6")  # 6x6 image * 64 channels
        self.relu6 = tf.nn.relu(self.fc6)

        self.fc7 = self.fc_layer(self.relu6, 384, 192, "fc7")
        self.relu7 = tf.nn.relu(self.fc7)

        self.fc8 = self.fc_layer(self.relu7, 192, 10, "fc8")
        self.prob = tf.nn.softmax(self.fc8, name="prob")

        self.data_dict = None

    def max_pool(self, bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def conv_layer(self, bottom, in_channels, out_channels, name):
        with tf.variable_scope(name):
            filt, conv_biases = self.get_conv_var(5, in_channels, out_channels, name)

            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')
            bias = tf.nn.bias_add(conv, conv_biases)
            relu = tf.nn.relu(bias)

            return relu

    def fc_layer(self, bottom, in_size, out_size, name):
        with tf.variable_scope(name):
            weights, biases = self.get_fc_var(in_size, out_size, name)

            if len(bottom.shape) > 2:
              x = tf.reshape(bottom, [-1, in_size])
            else:
              x = bottom
            fc = tf.nn.bias_add(tf.matmul(x, weights), biases)

            return fc

    def get_conv_var(self, filter_size, in_channels, out_channels, name):
        initial_value = tf.truncated_normal([filter_size, filter_size, in_channels, out_channels], 0.0, 0.001)
        filters = self.get_var(initial_value, name, 0, name + "_filters")

        initial_value = tf.truncated_normal([out_channels], .0, .001)
        biases = self.get_var(initial_value, name, 1, name + "_biases")

        return filters, biases

    def get_fc_var(self, in_size, out_size, name):
        initial_value = tf.truncated_normal([in_size, out_size], 0.0, 0.001)
        weights = self.get_var(initial_value, name, 0, name + "_weights")

        initial_value = tf.truncated_normal([out_size], .0, .001)
        biases = self.get_var(initial_value, name, 1, name + "_biases")

        return weights, biases

    def get_var(self, initial_value, name, idx, var_name):
        if self.data_dict is not None and name in self.data_dict:
            value = self.data_dict[name][idx]
        else:
            value = initial_value

        if self.trainable:
            var = tf.Variable(value, name=var_name)
        else:
            var = tf.constant(value, dtype=tf.float32, name=var_name)

        self.var_dict[(name, idx)] = var

        assert var.get_shape() == initial_value.get_shape()

        return var

    def save_npy(self, sess, npy_path="./cifar10-save.npy"):
        assert isinstance(sess, tf.Session)

        data_dict = {}

        for (name, idx), var in list(self.var_dict.items()):
            var_out = sess.run(var)
            if name not in data_dict:
                data_dict[name] = {}
            data_dict[name][idx] = var_out

        np.save(npy_path, data_dict)
        print(("file saved", npy_path))
        return npy_path

    def get_var_count(self):
        count = 0
        for v in list(self.var_dict.values()):
            count += reduce(lambda x, y: x * y, v.get_shape().as_list())
        return count

# [height, width, channels]
def load_image(path):
    # load image
    img = skimage.io.imread(path)
    img = img / 255.0
    assert (0 <= img).all() and (img <= 1.0).all()
    # print "Original Image Shape: ", img.shape
    # crop image from center
    short_edge = min(img.shape[:2])
    yy = int((img.shape[0] - short_edge) / 2)
    xx = int((img.shape[1] - short_edge) / 2)
    crop_img = img[yy: yy + short_edge, xx: xx + short_edge]
    resized_img = skimage.transform.resize(crop_img, (IMG_SIZE, IMG_SIZE))
    return resized_img

# img1 = load_image("./airplane.jpg")
img1 = np.zeros((24, 24, 3))
img1_true_result = [1 if i == 3 else 0 for i in range(10)]  # 1-hot result

batch1 = img1.reshape((1, IMG_SIZE, IMG_SIZE, 3))
model_path = './cifar10'

with tf.device('/cpu:0'):
    sess = tf.Session()

    images = tf.placeholder(tf.float32, [1, IMG_SIZE, IMG_SIZE, 3])
    true_out = tf.placeholder(tf.float32, [1, 10])
    train_mode = tf.placeholder(tf.bool)

    cifar = Cifar10()
    cifar.build(images, train_mode)

    # print number of variables used:
    # print(cifar.get_var_count())

    sess.run(tf.global_variables_initializer())

    # 1-step of training
    cost = tf.reduce_sum((cifar.prob - true_out) ** 2)
    train = tf.train.GradientDescentOptimizer(0.0001).minimize(cost)
    saver = tf.train.Saver()
    sess.run(train, feed_dict={images: batch1, true_out: [img1_true_result], train_mode: True})
    np.savetxt('cifar10_batch_of_1.csv',  batch1.flatten())
    saver.save(sess, model_path)

print 'Next step:'
print ' $ python  create_inference_graph.py  checkpoint  models/cifar/cifar10  prob  models/cifar/  cifar10'
print 'Then optimize the graph using:'
print ' $ python  optimize_inference_graph.py  models/cifar/cifar10.pb  Placeholder  prob  24,24,3'

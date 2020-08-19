import tensorflow as tf
import numpy as np


def linear_layer(x, w, b, input_size, output_size):
    """ (batch, data_length, input_size) * (input_size, output_size) -> (batch, data_length, output_size) """
    batches = tf.shape(x)[0]
    return tf.reshape(tf.matmul(tf.reshape(x, [-1, input_size]), w), [batches, -1, output_size]) + b


class Network:

    def __init__(self, session, learning_rate, data_size, static_data_size, lstm_size):
        self.sess = session
        self.data_size = data_size
        self.static_data_size = static_data_size

        self.gpu_inputs = tf.placeholder(tf.float32, [None, None, data_size])
        self.gpu_labels = tf.placeholder(tf.float32, [None])

        if static_data_size > 0:
            self.gpu_static = tf.placeholder(tf.float32, [None, static_data_size])
            with tf.variable_scope("lstm"):
                total_time = tf.shape(self.gpu_inputs)[1]
                lstm = tf.contrib.rnn.LSTMCell(lstm_size, num_proj=1, forget_bias=1.0)
                self.W = tf.Variable((np.random.rand(data_size + static_data_size, lstm_size) - 0.5) * 0.01, dtype=tf.float32)
                self.b = tf.Variable(np.zeros((lstm_size)), dtype=tf.float32)
                self.stacked_lstm = tf.contrib.rnn.MultiRNNCell([lstm] * 1)
                tiled_static = tf.tile(tf.reshape(self.gpu_static, [-1, 1, static_data_size]), [1, total_time, 1])
                preLSTM = tf.tanh(linear_layer(tf.concat([self.gpu_inputs, tiled_static], axis=2), self.W, self.b, data_size + static_data_size, lstm_size))
                output, state = tf.nn.dynamic_rnn(self.stacked_lstm, preLSTM, dtype=tf.float32, time_major=False, parallel_iterations=1, swap_memory=True)
        else:
            with tf.variable_scope("lstm"):
                total_time = tf.shape(self.gpu_inputs)[1]
                lstm = tf.contrib.rnn.LSTMCell(lstm_size, num_proj=1, forget_bias=1.0)
                self.W = tf.Variable((np.random.rand(data_size, lstm_size) - 0.5) * 0.01, dtype=tf.float32)
                self.b = tf.Variable(np.zeros((lstm_size)), dtype=tf.float32)
                self.stacked_lstm = tf.contrib.rnn.MultiRNNCell([lstm] * 1)
                preLSTM = tf.tanh(linear_layer(self.gpu_inputs, self.W, self.b, data_size, lstm_size))
                output, state = tf.nn.dynamic_rnn(self.stacked_lstm, preLSTM, dtype=tf.float32, time_major=False, parallel_iterations=1, swap_memory=True)

        lstm_scope = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="lstm")

        self.y = tf.sigmoid(tf.reshape(tf.slice(output, [0, total_time - 1, 0], [-1, 1, -1]), [-1]))
        self.overall_cost = tf.reduce_sum(-tf.multiply(self.gpu_labels, tf.log(self.y)) - tf.multiply(1 - self.gpu_labels, tf.log(1 - self.y)))

        self.training_op = tf.train.AdamOptimizer(learning_rate).minimize(self.overall_cost, var_list=lstm_scope)
        self.saver = tf.train.Saver(var_list=lstm_scope, keep_checkpoint_every_n_hours=1)

    def train(self, batches, session_name, max_iteration):

        for step in range(max_iteration):
            sum_loss = 0.0
            for b in batches:
                if self.static_data_size > 0:
                    _, loss = self.sess.run((self.training_op, self.overall_cost), feed_dict={self.gpu_static: b[0], self.gpu_inputs: b[1], self.gpu_labels: b[2]})
                else:
                    _, loss = self.sess.run((self.training_op, self.overall_cost), feed_dict={self.gpu_inputs: b[0], self.gpu_labels: b[1]})
                sum_loss += loss
            print(step, sum_loss / len(batches))
            if step % 100 == 0:
                self.saver.save(self.sess, session_name)

        self.saver.save(self.sess, session_name)

    def load(self, session_name):
        print("loading from last save...")
        self.saver.restore(self.sess, session_name)

    def load_last(self, directory):
        self.saver.restore(self.sess, tf.train.latest_checkpoint(directory))

    def scan(self, data):
        if self.static_data_size > 0:
            classes = self.sess.run(self.y, feed_dict={self.gpu_static: data[0], self.gpu_inputs: data[1]})
        else:
            classes = self.sess.run(self.y, feed_dict={self.gpu_inputs: data[0]})
        return classes

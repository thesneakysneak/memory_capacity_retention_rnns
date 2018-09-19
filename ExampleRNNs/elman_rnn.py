import tensorflow as tf
import numpy as np

# https://www.data-blogger.com/2017/05/17/elman-rnn-implementation-in-tensorflow/
class ElmanRNN:
    def __init__(self, num_vocab, num_embedding, num_context, num_hidden, num_classes):
        self.num_vocab = num_vocab
        self.num_embedding = num_embedding
        self.num_context = num_context
        self.num_hidden = num_hidden
        self.num_classes = num_classes

        self.params = {}

        with tf.name_scope('input'):
            with tf.name_scope('input_data'):
                self.data = tf.placeholder(tf.int32, name='data')

            with tf.name_scope('embedding'):
                self.idxs = tf.reshape(self.data, (-1,)) + 1
                self.emb = tf.Variable(name='emb', initial_value=ElmanRNN.initialize(num_vocab + 1, num_embedding))
                self.params['emb'] = self.emb
                self.input_emb = tf.gather(self.emb, self.idxs)
                self.input_emb_fix = tf.where(tf.is_nan(self.input_emb), tf.zeros_like(self.input_emb), self.input_emb)


            with tf.name_scope('x'):
                num_inputs = tf.shape(self.data)[0]
                self.wx = tf.Variable(name='wx', initial_value=ElmanRNN.initialize(num_embedding * num_context, num_hidden))
                self.params['wx'] = self.wx
                self.x = tf.reshape(self.input_emb_fix, (num_inputs, num_embedding * num_context), name='x')

        with tf.name_scope('recurrence'):
            with tf.name_scope('recurrence_init'):
                self.h = tf.Variable(name='h', initial_value=np.zeros((1, num_hidden)))
                h_0 = tf.reshape(self.h, (1, num_hidden))
                self.params['h'] = self.h
                s_0 = tf.constant(np.matrix([0.] * num_classes))
                y_0 = tf.constant(0, 'int64')
            with tf.name_scope('hidden'):
                self.wh = tf.Variable(name='wh', initial_value=ElmanRNN.initialize(num_hidden, num_hidden))
                self.bh = tf.Variable(name='bh', initial_value=np.zeros((1, num_hidden)))
                self.params['wh'] = self.wh
                self.params['bh'] = self.bh
            with tf.name_scope('classes'):
                self.w = tf.Variable(name='w', initial_value=ElmanRNN.initialize(num_hidden, num_classes))
                self.b = tf.Variable(name='b', initial_value=np.zeros((1, num_classes)))
                self.params['w'] = self.w
                self.params['b'] = self.b
            self.h, self.s, self.y = tf.scan(self.recurrence, self.x, initializer=(h_0, s_0, y_0))

        with tf.name_scope('output'):
            with tf.name_scope('target_data'):
                self.target = tf.placeholder(tf.float64, name='target')
            with tf.name_scope('probabilities'):
                self.s = tf.squeeze(self.s)
            with tf.name_scope('outcomes'):
                self.y = tf.squeeze(self.y)
            with tf.name_scope('loss'):
                self.loss = -tf.reduce_sum(self.target * tf.log(tf.clip_by_value(self.s, 1e-20, 1.0)))

    @staticmethod
    def initialize(*shape):
        return 0.001 * np.random.uniform(-1., 1., shape)

    def recurrence(self, old_state, x_t):
        h_t, s_t, y_t = old_state
        x = tf.reshape(x_t, (1, self.num_embedding * self.num_context))
        input_layer = tf.matmul(x, self.wx)
        hidden_layer = tf.matmul(h_t, self.wh) + self.bh
        h_t_next = input_layer + hidden_layer
        s_t_next = tf.nn.softmax(tf.matmul(h_t_next, self.w) + self.b)
        y_t_next = tf.squeeze(tf.argmax(s_t_next, 1))
        return h_t_next, s_t_next, y_t_next
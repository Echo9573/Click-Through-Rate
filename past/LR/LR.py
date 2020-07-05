import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.ConfigProto(allow_soft_placement=True)
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.01)
config.gpu_options.allow_growth = True

class LR(object):
    def __init__(self, num_class, num_feat, lr=0.5):
        self.lr = lr
        self.N_CLASSES = num_class
        self.N_FEAT = num_feat
        self.defineModel()

    def defineModel(self):
        self.x = tf.placeholder(tf.float32, [None, self.N_FEAT])
        self.y = tf.placeholder(tf.float32, [None, self.N_CLASSES])

        with tf.name_scope('Weight'):
            W = tf.Variable(tf.zeros([self.N_FEAT, self.N_CLASSES]), name='W')
            tf.summary.histogram('weights', W)

        with tf.name_scope('Bias'):
            b = tf.Variable(tf.zeros([self.N_CLASSES]), name='B')
            tf.summary.histogram('bias', b)

        with tf.name_scope('logits'):
            self.yhat = tf.nn.sigmoid(tf.add(tf.matmul(self.x, W), b)) # ?矩阵乘法

        with tf.name_scope('loss'):
            self.loss = tf.reduce_mean(-tf.reduce_sum(self.y * tf.log(self.yhat), reduction_indices=[1]))
            tf.summary.scalar("loss", self.loss)

        with tf.name_scope('optimizer'):
            # self.optimizer = tf.train.GradientDescentOptimizer(self.lr).minimize(self.loss)
            self.global_step = tf.Variable(0, name='global_step', trainable=False)
            optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
            extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(extra_update_ops):
                self.optimizer = optimizer.minimize(self.loss, global_step=self.global_step)

        with tf.name_scope('accuracy'):
            correct_prediction = tf.equal(tf.argmax(self.yhat, 1), tf.argmax(self.y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            tf.summary.scalar("accuracy", self.accuracy)


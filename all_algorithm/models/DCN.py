import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
import os
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.ConfigProto(allow_soft_placement=True)
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.01)
config.gpu_options.allow_growth = True

class DCN(object):
    def __init__(self, feature_nums, field_nums, args):
        self.feature_nums = feature_nums
        self.field_nums = field_nums
        self.args = args
        self._initialize()
        self.defineModel()

    def _initialize(self):
        self.weight = {}
        input_size = self.field_nums * self.args.embedding_size
        # glorot = np.sqrt(2.0 / (input_size + self.args.hidden_units[0]))
        # self.weight["layer_0"] = tf.Variable(
        #     np.random.normal(loc=0, scale=glorot, size=(input_size, self.args.hidden_units[0])), dtype=np.float32)
        # self.weight["bias_0"] = tf.Variable(np.random.normal(loc=0, scale=glorot, size=(1, self.args.hidden_units[0])),
        #                                 dtype=np.float32)  # 1 * layers[0]
        # deep dnn
        for i in range(len(self.args.hidden_units)):
            if i == 0:
                self.weight['layer_dnn_{}'.format(i)] = tf.get_variable('w_0', shape=[input_size, self.args.hidden_units[0]],
                                                               initializer=tf.truncated_normal_initializer(mean=0,
                                                                                                           stddev=1e-2))
            else:
                self.weight['layer_dnn_{}'.format(i)] = tf.get_variable('w_{}'.format(i),
                                                               shape=[self.args.hidden_units[i - 1],
                                                                      self.args.hidden_units[i]],
                                                               initializer=tf.truncated_normal_initializer(mean=0,
                                                                                                           stddev=1e-2))
            self.weight['bias_dnn_{}'.format(i)] = tf.get_variable('b_{}'.format(i),
                                                              shape=[1, self.args.hidden_units[i]],
                                                              initializer=tf.initializers.zeros())
        # cross
        for i in range(self.args.cross_layers):
            self.weight['layer_cross_{}'.format(i)] = tf.get_variable(shape=[input_size, 1],
                                     initializer=tf.truncated_normal_initializer(), name='cross_weight{}'.format(i))
            self.weight['bias_cross_{}'.format(i)] = tf.get_variable(shape=[input_size, 1],
                                   initializer=tf.truncated_normal_initializer(), name='cross_bias{}'.format(i))

    def cross_func(self, x_0, x_l, weight, feature_size, use_better):
        if use_better:
            # interaction = x_0 * (x_l^t * w)
            transform = tf.tensordot(tf.reshape(x_l, [-1, 1, feature_size]), weight, axes=1)
            interaction = tf.multiply(x_0, transform)
        else:
            # interaction = (x_0 * x_l^t) * w
            outer_product = tf.matmul(tf.reshape(x_0, [-1, feature_size, 1]),
                                      tf.reshape(x_l, [-1, 1, feature_size]),
                                      )
            interaction = tf.tensordot(outer_product, weight, axes=1)
        return interaction

    def defineModel(self):
        self.df_i = tf.placeholder(tf.int32, [None, self.field_nums], name='df_i')
        self.df_v = tf.placeholder(tf.float32, [None, self.field_nums], name='df_v')
        self.y = tf.placeholder(tf.float32, [None, self.args.num_class]) # None * 1
        self.droupout_keep_deep = tf.placeholder(tf.float32, [None])

        with tf.variable_scope('embedding'):
            embeddings = tf.get_variable('embeddings',
                                         shape=[self.feature_nums, self.args.embedding_size],
                                         dtype=tf.float32,
                                         initializer=tf.initializers.glorot_uniform())  # f*d
            batch_embeddings = tf.nn.embedding_lookup(embeddings, self.df_i)  # None*n*d
            df_v = tf.expand_dims(self.df_v, axis=2)  # None*n*1
            self.xv = tf.multiply(df_v, batch_embeddings)  # None*n*d

        with tf.variable_scope('fm_layer'):
            sum_square = tf.square(tf.reduce_sum(self.xv, axis=1))  # None*d
            square_sum = tf.reduce_sum(tf.square(self.xv), axis=1)  # None*d
            # self.fm_output = tf.reduce_sum(0.5 * (
            #     tf.subtract(sum_square, square_sum)), axis=1, keep_dims=True)
            self.fm_output = 0.5 * (tf.subtract(sum_square, square_sum))

        with tf.variable_scope('cross_layer'):
            input_size = self.field_nums * self.args.embedding_size
            self.x_0 = tf.reshape(self.xv, shape=[-1, input_size, 1])
            self.x_l = tf.reshape(self.xv, shape=[-1, input_size, 1])
            for i in range(self.args.cross_layers):
                interation = self.cross_func(self.x_0, self.x_l, self.weight['layer_cross_{}'.format(i)],
                                             feature_size=input_size,
                                             use_better=self.args.use_better)
                self.x_l = interation + self.weight['bias_cross_{}'.format(i)] + self.x_l
            # self.x_l_shape = tf.shape(self.x_l) #  [256  24   1]
            self.cross_network_out = tf.reshape(self.x_l, [-1, input_size])
            # self.cross_network_out_shape = tf.shape(self.cross_network_out) # [256  24]

        with tf.variable_scope('deep_DNN'):
            self.deep_input = tf.reshape(self.xv,
                                         shape=[-1, self.field_nums * self.args.embedding_size])  # None * (F*K)
            for i in range(len(self.args.hidden_units)):
                with tf.name_scope('deep_dnn_{}'.format(str(i))):
                    self.deep_input = tf.add(tf.matmul(self.deep_input, self.weight['layer_dnn_{}'.format(i)]),
                                             self.weight['bias_dnn_{}'.format(i)])
                    if self.args.use_batch_normal:
                        self.deep_input = tf.layers.batch_normalization(self.deep_input)
                    self.deep_input = tf.nn.relu(self.deep_input)
                    if (i+1) != len(self.args.hidden_units):
                        self.deep_input = tf.nn.dropout(self.deep_input, self.args.dropout_keep_deep1)

        with tf.name_scope('logit'):
            if self.args.use_deep:
                concat = tf.concat([self.cross_network_out, self.deep_input], axis=1)
                # self.concat_shape = tf.shape(concat) #  [256 324]
                self.y_hat = tf.layers.dense(concat, units=1, activation=None)
            self.y_hat_prob = tf.nn.sigmoid(self.y_hat, name='y_hat_prob')

        with tf.name_scope('loss'):
            cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.y,
                logits=self.y_hat)
            self.loss = tf.reduce_mean(cross_entropy)
            # self.loss = tf.losses.log_loss(labels=self.y, predictions=self.y_hat_prob)
            tf.summary.scalar("loss", self.loss)

        with tf.name_scope('accuracy'):
            self.y_pred = tf.cast(self.y_hat_prob > self.args.threshold, tf.int32)
            self.accuracy = tf.metrics.accuracy(
                                labels=self.y,
                                predictions=self.y_pred,
                                name="accuracy")
            tf.summary.scalar("acc", self.accuracy[1])

        with tf.name_scope('auc'):
            self.auc = tf.metrics.auc(labels=self.y, predictions=self.y_hat_prob)
            tf.summary.scalar("auc", self.auc[1])

        with tf.name_scope('optimizer'):
            self.global_step = tf.Variable(0, trainable=False)
            # optimizer = tf.train.FtrlOptimizer(learning_rate=self.args.lr)
            optimizer = tf.train.AdamOptimizer(learning_rate=self.args.lr)
            extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(extra_update_ops):
                self.optimizer = optimizer.minimize(self.loss, global_step=self.global_step)



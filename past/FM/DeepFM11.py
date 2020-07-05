
import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.ConfigProto(allow_soft_placement=True)
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.01)
config.gpu_options.allow_growth = True

class DeepFM(object):
    def __init__(self, feature_nums, field_nums, args):
        self.feature_nums = feature_nums
        self.field_nums = field_nums
        self.args = args
        self.defineModel()


    def defineModel(self):
        self.df_i = tf.placeholder(tf.int32, [None, self.field_nums])
        self.df_v = tf.placeholder(tf.float32, [None, self.field_nums])
        self.y = tf.placeholder(tf.float32, [None, self.args.num_class]) # None * 1
        self.droupout_keep_deep = tf.placeholder(tf.float32, [None])

        with tf.variable_scope('linear_layer'):
            embeddings = tf.get_variable(name='emb',
                                         shape=[self.feature_nums, 1],
                                         dtype=tf.float32,
                                         initializer=tf.initializers.glorot_uniform()) # f * 1
            with tf.name_scope('Weight'):
                batch_weights = tf.nn.embedding_lookup(embeddings, self.df_i) # None * n * 1
                batch_weights = tf.squeeze(batch_weights, axis=2) # None * n

            with tf.name_scope('Bias'):
                linear_biase = tf.get_variable('linear_biase',
                                               shape=[1, 1],
                                               dtype=tf.float32,
                                               initializer=tf.initializers.glorot_uniform()) # 1 * 1
            linear_w_x = tf.multiply(self.df_v, batch_weights, name='linear_w_x')  # None * n
            self.linear_terms = tf.add(tf.reduce_sum(linear_w_x, axis=1, keep_dims=True), linear_biase)  # None * 1

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

        with tf.variable_scope('deep_layer'):
            self.deep_input = tf.reshape(self.xv, shape=[-1, self.field_nums * self.args.embedding_size])  # None * (F*K)
            self.deep_input = tf.nn.dropout(x=self.deep_input, keep_prob=self.args.dropout_keep_deep)
            for i, v in enumerate(self.args.hidden_units):
                with tf.name_scope('deep_{}'.format(str(i))):
                    self.deep_input = tf.layers.dense(self.deep_input, units=v, activation=None)
                    if self.args.use_batch_normal:
                        self.deep_input = tf.layers.batch_normalization(self.deep_input)
                    self.deep_input = tf.nn.relu(self.deep_input)
                    if (i+1) != len(self.args.hidden_units):
                        self.deep_input = tf.nn.dropout(self.deep_input, self.args.dropout_keep_deep1)

        with tf.name_scope('logit'):
            self.y_hat = tf.add(self.linear_terms, self.fm_output)
            if self.args.use_deep:
                concat = tf.concat([self.linear_terms, self.fm_output, self.deep_input], axis=1)
                self.y_hat = tf.layers.dense(concat, units=1, activation=None)
            self.y_hat_prob = tf.nn.sigmoid(self.y_hat)

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

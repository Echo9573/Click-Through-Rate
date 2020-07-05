import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.ConfigProto(allow_soft_placement=True)
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.01)
config.gpu_options.allow_growth = True

class FFM(object):
    def __init__(self, feature_nums, field_nums, args):
        self.feature_nums = feature_nums
        self.field_nums = field_nums
        self.args = args
        self.global_step = 0
        self.defineModel()

    def defineModel(self):
        self.df_i = tf.placeholder(tf.int32, [None, self.field_nums])
        self.df_v = tf.placeholder(tf.float32, [None, self.field_nums])
        self.y = tf.placeholder(tf.float32, [None, self.args.num_class]) # None * 1

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


        with tf.variable_scope('field_aware_interaction_layer'):
            embedding_interaction = tf.get_variable(name='embedding',
                                     shape=[self.field_nums, self.feature_nums, self.args.embedding_size],
                                     initializer=tf.truncated_normal_initializer(mean=0, stddev=0.01))

            self.field_aware_interaction_terms = tf.constant(0, dtype='float32')

            for i in range(self.field_nums):
                for j in range(i + 1, self.field_nums):
                    vi_fj = tf.nn.embedding_lookup(embedding_interaction[j], self.df_i[:, i]) # None * embedding_size
                    vj_fi = tf.nn.embedding_lookup(embedding_interaction[i], self.df_i[:, j]) # None * embedding_size
                    vivj = tf.multiply(vi_fj, vj_fi)
                    x_i = tf.expand_dims(self.df_v[:, i], axis=1)
                    x_j = tf.expand_dims(self.df_v[:, j], axis=1)
                    xixj = tf.multiply(x_i, x_j) # None * 1
                    # self.field_aware_interaction_terms += tf.multiply(tf.reduce_sum(vivj), xixj)
                    self.field_aware_interaction_terms += tf.reduce_sum(tf.multiply(vivj, xixj), axis=1, keepdims=True)

            self.field_aware_interaction_terms = tf.reduce_sum(self.field_aware_interaction_terms, axis=1, keepdims=True)

        with tf.name_scope('logit'):
            self.y_hat = tf.add(self.linear_terms, self.field_aware_interaction_terms)
            self.y_hat_prob = tf.nn.sigmoid(self.y_hat)

        with tf.name_scope('loss'):
            cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.cast(self.y, tf.float32),
                                                                    logits=self.y_hat)
            self.loss = tf.reduce_mean(cross_entropy)
            tf.summary.scalar("loss", self.loss)

        with tf.name_scope('accuracy'):
            # self.correct_prediction = tf.equal(tf.cast(tf.argmax(self.y_hat_prob, 1), tf.int64), self.y)
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
            optimizer = tf.train.FtrlOptimizer(learning_rate=self.args.lr)
            # optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
            extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(extra_update_ops):
                self.optimizer = optimizer.minimize(self.loss, global_step=self.global_step)

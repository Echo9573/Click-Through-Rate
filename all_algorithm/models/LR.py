import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.ConfigProto(allow_soft_placement=True)
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.01)
config.gpu_options.allow_growth = True

class LR(object):
    def __init__(self, feature_nums, field_nums, args):
        self.feature_nums = feature_nums
        self.field_nums = field_nums
        self.args = args
        self.defineModel()

    def defineModel(self):
        self.df_i = tf.placeholder(tf.int32, [None, self.field_nums], name='df_i')
        self.df_v = tf.placeholder(tf.float32, [None, self.field_nums], name='df_v')
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

        with tf.name_scope('logit'):
            linear_w_x = tf.multiply(self.df_v, batch_weights, name='linear_w_x')  # None * n
            self.linear_output = tf.add(tf.reduce_sum(linear_w_x, axis=1, keep_dims=True), linear_biase)  # None * 1
            self.yhat =  self.linear_output
            self.yhat_prob = tf.nn.sigmoid(self.yhat, name='y_hat_prob')

        with tf.name_scope('loss'):
            # cross_entropy = -tf.reduce_sum(y * tf.log(yhat), 1)
            # cross_entropy = -(y * tf.log(yhat_prob) + (1-y) * tf.log(1-yhat_prob))
            cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.y, logits=self.yhat)
            log_loss = tf.losses.log_loss(labels=self.y, predictions=self.yhat_prob)
            self.loss = tf.reduce_mean(cross_entropy)
            tf.summary.scalar("loss", self.loss)

        with tf.name_scope('optimizer'):
            self.global_step = tf.Variable(0, name='global_step', trainable=False)
            optimizer = tf.train.FtrlOptimizer(learning_rate=self.args.lr) #, global_step=self.global_step
            extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(extra_update_ops):
                self.optimizer = optimizer.minimize(self.loss, global_step=self.global_step)

        with tf.name_scope('accuracy'):
            # self.correct_prediction = tf.equal(tf.cast(tf.argmax(self.y_hat_prob, 1), tf.int64), self.y)
            self.y_pred = tf.cast(self.yhat_prob > self.args.threshold, tf.int32)
            self.accuracy = tf.metrics.accuracy(
                                labels=self.y,
                                predictions=self.y_pred,
                                name="accuracy")
            tf.summary.scalar("acc", self.accuracy[1])

        with tf.name_scope('auc'):
            self.auc = tf.metrics.auc(labels=self.y, predictions=self.yhat_prob)
            tf.summary.scalar("auc", self.auc[1])

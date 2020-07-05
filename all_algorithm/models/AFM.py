import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.ConfigProto(allow_soft_placement=True)
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.01)
config.gpu_options.allow_growth = True

class AFM(object):
    def __init__(self, feature_nums, field_nums, args):
        self.feature_nums = feature_nums
        self.field_nums = field_nums
        self.args = args

        # self._initialize()
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
            linear_w_x = tf.multiply(self.df_v, batch_weights, name='linear_w_x')  # None * n
            self.linear_terms = tf.add(tf.reduce_sum(linear_w_x, axis=1, keep_dims=True), linear_biase)  # None * 1
        #
        with tf.variable_scope('embedding_layer'):
            embeddings = tf.get_variable('embeddings',
                                         shape=[self.feature_nums, self.args.embedding_size],
                                         dtype=tf.float32,
                                         initializer=tf.initializers.glorot_uniform())  # f*d
            batch_embeddings = tf.nn.embedding_lookup(embeddings, self.df_i)  # None*n*d
            df_v = tf.expand_dims(self.df_v, axis=2)  # None*n*1
            self.xv = tf.multiply(df_v, batch_embeddings)  # None*n*d

            # sum_square = tf.square(tf.reduce_sum(self.xv, axis=1))  # None*d
            # square_sum = tf.reduce_sum(tf.square(self.xv), axis=1)  # None*d
            #
            # self.fm_output = tf.reduce_sum(0.5 * (
            #     tf.subtract(sum_square, square_sum)), axis=1, keep_dims=True)

        # element_wise
        with tf.variable_scope('element_wise'):
            element_wise_product_list = []
            for i in range(self.field_nums):
                for j in range(i+1, self.field_nums):
                    element_wise_product_list.append(tf.multiply(self.xv[:,i,:], self.xv[:,j,:])) # None*d
            self.element_wise_product = tf.stack(element_wise_product_list)
            self.element_wise_product = tf.transpose(self.element_wise_product, perm=[1, 0 ,2],
                                                name='element_wise_product') # None * (n * (n - 1)/2) /k
        # attention part
        with tf.variable_scope('attention'):
            attention_w = tf.get_variable(dtype=tf.float32, shape=[self.args.embedding_size, self.args.attention_size],
                                          initializer=tf.initializers.glorot_normal, name='attention_w')
            # tf.Variable(np.random.normal(loc=0, scale=glorot, size=(self.embedding_size, self.attention_size)),
            #             dtype=tf.float32, name='attention_w')
            attention_b = tf.get_variable(dtype=tf.float32, shape=[self.args.attention_size, ],
                                          initializer=tf.initializers.zeros(), name='attention_b')
            attention_h = tf.get_variable(dtype=tf.float32, shape=[self.args.attention_size, ],
                                          initializer=tf.initializers.glorot_normal(), name='attention_h')
            print('***********', self.args.embedding_size)
            attention_p = tf.Variable(tf.ones((self.args.embedding_size, 1)), dtype=tf.float32)

            num_interactions = int(self.field_nums * (self.field_nums -1) / 2)
            # wx + b -> relu(wx + b) -> h * relu(wx + b)
            self.attention_wx_plus_b = tf.reshape(tf.add(tf.matmul(
                tf.reshape(self.element_wise_product, [-1, self.args.embedding_size]),
                attention_w), attention_b),
                shape=[-1, num_interactions, self.args.attention_size]) # None * ( n * (n - 1) / 2) * A
            self.attention_exp = tf.exp(tf.reduce_sum(tf.multiply(tf.nn.relu(self.attention_wx_plus_b),
                                                                  attention_h),axis=2, keep_dims=True))
            # None * ( n * (n - 1) / 2) * 1,这里是广播机制，因为weight['attention_h']的size是（A,）是一种伪矩阵
            self.attention_exp_sum = tf.reduce_sum(self.attention_exp, axis=2, keep_dims=True) # n*1*1
            self.attention_out = tf.div(self.attention_exp, self.attention_exp_sum, name='attention_out')
            self.attention_x_product = tf.reduce_sum(tf.multiply(
                self.attention_out, self.element_wise_product), axis=1, name='AFM') # None * k
            self.attention_part_sum = tf.matmul(self.attention_x_product, attention_p)

        with tf.name_scope('logit'):
            self.y_hat = tf.add(self.linear_terms, self.attention_part_sum, name='out_afm')
            self.y_hat_prob = tf.nn.sigmoid(self.y_hat, name='y_hat_prob')

        with tf.name_scope('loss'):
            cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.y,
                logits=self.y_hat)
            self.loss = tf.reduce_mean(cross_entropy)
            # self.loss = tf.losses.log_loss(labels=self.y, predictions=self.y_hat_prob) # 同上两步
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
            optimizer = tf.train.FtrlOptimizer(learning_rate=self.args.lr)
            # optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
            extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(extra_update_ops):
                self.optimizer = optimizer.minimize(self.loss, global_step=self.global_step)

    def _initialize(self):
        self.weight = {}
        input_size = self.field_nums * self.args.embedding_size
        # glorot = np.sqrt(2.0 / (self.args.attention_size + self.args.embedding_size))
        # self.weight["layer_0"] = tf.Variable(
        #     np.random.normal(loc=0, scale=glorot, size=(input_size, self.args.hidden_units[0])), dtype=np.float32)
        # self.weight["bias_0"] = tf.Variable(np.random.normal(loc=0, scale=glorot, size=(1, self.args.hidden_units[0])),
        #                                 dtype=np.float32)  # 1 * layers[0]
        # for i in range(len(self.args.hidden_units)):
        #     if i == 0:
        #         self.weight['layer_{}'.format(i)] = tf.get_variable('w_0', shape=[input_size, self.args.hidden_units[0]],
        #                                                        initializer=tf.truncated_normal_initializer(mean=0,
        #                                                                                                    stddev=1e-2))
        #     else:
        #         self.weight['layer_{}'.format(i)] = tf.get_variable('w_{}'.format(i),
        #                                                        shape=[self.args.hidden_units[i - 1],
        #                                                               self.args.hidden_units[i]],
        #                                                        initializer=tf.truncated_normal_initializer(mean=0,
        #                                                                                                    stddev=1e-2))
        #     self.weight['bias_{}'.format(i)] = tf.get_variable('b_{}'.format(i),
        #                                                       shape=[1, self.args.hidden_units[i]],
        #                                                       initializer=tf.initializers.zeros())
        self.weight['attention_w'] = tf.get_Variable(shape=[self.args.embedding_size, self.args.attention_size],
                                                     initializer=tf.initializers.glorot_normal, name='attention_w')
        # tf.Variable(np.random.normal(loc=0, scale=glorot, size=(self.embedding_size, self.attention_size)),
        #             dtype=tf.float32, name='attention_w')
        self.weight['attention_b'] = tf.get_Variable(shape=[self.args.attention_size, ],
                                                     initializer=tf.initializers.zeros(), name='attention_b')
        self.weight['attention_h'] = tf.get_Variable(shape=[self.args.attention_size, ],
                                                     initializer=tf.initializers.glorot_normal(), name='attention_h')
        self.weight['attention_p'] = tf.Variable(tf.ones((self.args.embedding_size, 1)), dtype=tf.float64)
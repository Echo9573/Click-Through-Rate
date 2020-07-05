import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import roc_auc_score
from collections import Counter

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.ConfigProto(allow_soft_placement=True)
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
config.gpu_options.allow_growth = True

# load data
def load_data():
    oneHot = OneHotEncoder()
    choose_cols = ['C1', 'banner_pos', 'device_conn_type', 'C15', 'C16', 'C18', 'click']
    trainfile = 'train_df.csv'
    Train = pd.read_csv(trainfile)
    dfTrain = Train[choose_cols]
    Xtrain = dfTrain.ix[:, :-1]
    # print('**********')
    # print(Counter(dfTrain.ix[:, -1]))
    ytrain = dfTrain.ix[:, -1].values.astype(np.float32).reshape([-1, 1])
    ytrain = oneHot.fit_transform(ytrain).toarray()

    testfile = 'test_df.csv'
    Test = pd.read_csv(testfile)
    dfTest = Test[choose_cols]
    Xval = dfTest.ix[:, :-1]
    yval = dfTest.ix[:, -1].values.astype(np.float32).reshape([-1, 1])
    yval = oneHot.fit_transform(yval).toarray()

    return Xtrain, ytrain, Xval, yval

# generate feature dict
def generate_feasdict(dfTrain, dfTest, numerical_cols):
    df = pd.concat([dfTrain, dfTest])
    feat_dict = {}
    idx = 0
    for col in df.columns:
        if col in numerical_cols:
            feat_dict[col] = idx
            idx += 1
        else:
            dataunique = df[col].unique()
            feat_dict[col] = dict(zip(dataunique, range(idx, idx + len(dataunique))))
            idx += len(dataunique)
    return feat_dict, idx

def parse_data(feat_dict, df, numerical_cols):
    dfi = df.copy()
    dfv = df.copy()
    for col in dfi.columns:
        if col in numerical_cols:
            dfi[col] = feat_dict[col]
        else:
            dfi[col] = dfi[col].map(feat_dict[col])
            dfv[col] = 1.
    # list of list of feature indices of each sample in the dataset
    Xi = dfi.values#.tolist()
    # list of list of feature values of each sample in the dataset
    Xv = dfv.values#.tolist()
    return Xi, Xv


class FM(object):
    def __init__(self, feat_length, num_class, num_factors, lr, lr_l1, lr_l2):
        self.feat_length = feat_length
        self.num_class = num_class
        self.num_factors = num_factors
        self.lr = lr
        self.lr_l1 = lr_l1
        self.lr_l2 = lr_l2

    def define_model(self):
        self.X = tf.sparse_placeholder('float32', [None, self.feat_length])
        self.y = tf.placeholder('int64', [None, self.num_class])
        with tf.variable_scope('linear_layer'):
            W1 = tf.get_variable(name='w1',
                                 shape=[self.feat_length, self.num_class],
                                 initializer=tf.truncated_normal_initializer(mean=0, stddev=1e-2))
            b = tf.get_variable(name='b',
                                shape=[self.num_class],
                                initializer=tf.zeros_initializer())
            # self.linear_terms = tf.add(tf.matmul(self.X, W1), b)
            self.linear_terms = tf.add(tf.sparse_tensor_dense_matmul(self.X, W1), b)
            tf.summary.histogram('w1', W1)
            tf.summary.histogram('b', b)
        with tf.variable_scope('interaction_layer'):
            v = tf.get_variable(name='v',
                                shape=[self.feat_length, self.num_factors],
                                initializer=tf.truncated_normal_initializer(mean=0, stddev=0.01))
            # s2 = tf.pow(tf.matmul(self.X, v), 2)
            s2 = tf.pow(tf.sparse_tensor_dense_matmul(self.X, v), 2)
            # s1 = tf.matmul(tf.pow(self.X, 2), tf.pow(v, 2))
            s1 = tf.sparse_tensor_dense_matmul(tf.square(self.X), tf.pow(v, 2))
            self.interaction_terms = tf.multiply(0.5, tf.reduce_mean(tf.subtract(s2, s1), axis=1, keep_dims=True))

        with tf.name_scope('logit'):
            self.y_hat = tf.add(self.linear_terms, self.interaction_terms)
            self.y_hat_prob = tf.nn.softmax(self.y_hat)

        with tf.name_scope('loss'):
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=tf.cast(self.y, tf.float32), logits=self.y_hat)
            # cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y, logits=self.y_hat)
            ## 下面这个也可以
            # cross_entropy = - tf.cast(self.y, tf.float32) * tf.log(tf.clip_by_value(self.y_hat_prob, 1e-10, 1.0)) - \
            #                 tf.cast((1 - self.y), tf.float32) * tf.log(tf.clip_by_value(1 - self.y_hat_prob, 1e-10, 1.0))
            self.loss = tf.reduce_mean(cross_entropy)
            tf.summary.scalar("loss", self.loss)

        with tf.name_scope('accuracy'):
            correct_prediction = tf.equal(tf.argmax(self.y_hat_prob, 1), tf.argmax(self.y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            tf.summary.scalar("acc", self.accuracy)
        ## 这个有问题
        # with tf.name_scope('auc'):
        #     self.auc = tf.metrics.auc(self.y_hat_prob[0], tf.argmax(self.y, 1), num_thresholds=1000)
        #     # tf.summary.scalar("auc", self.auc)

        with tf.name_scope('optimizer'):
            self.global_step = tf.Variable(0, trainable=False)
            # optimizer = tf.train.FtrlOptimizer(self.lr, l1_regularization_strength=self.lr_l1,
            #                                    l2_regularization_strength=self.lr_l2,
            #                                    )
            optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
            # tf.train.AdagradOptimizer()
            extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(extra_update_ops):
                self.train_op = optimizer.minimize(self.loss, global_step=self.global_step)

def get_batch(Xi, Xv,  y, batch_size, index):
    start = index * batch_size
    end = (index + 1) * batch_size
    end = end if end < len(y) else len(y)
    xi_bs = []
    for k, v in enumerate(Xi[start:end]):
        for i in range(len(v)):
            xi_bs.append([k, v[i]])
    xv_bs = []
    for k, v in enumerate(Xv[start:end]):
        xv_bs.extend(v)
    y_bs = y[start:end]
    return xi_bs, xv_bs, y_bs

def shuffle_data(Xi, Xv, y):
    idx = np.arange(0, len(Xi))
    np.random.shuffle(idx)
    shuffled_Xi = Xi[idx]
    shuffled_Xv = Xv[idx]
    shuffled_y = y[idx]
    return shuffled_Xi, shuffled_Xv, shuffled_y

def train_model(sess, model, Xi_train, Xv_train, y_train, feature_length, epochs=1, batch_size=128):
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter("loss_log4/", sess.graph)
    loss_history, accuracy_history, auc_history = [], [], []
    for epoch in range(epochs):
        print('current epoch is ', epoch)
        total_batch = int(len(y_train) / batch_size)
        shuffled_Xi, shuffled_Xv, shuffled_y = shuffle_data(Xi_train, Xv_train, y_train)
        for i in range(total_batch):
            batch_index, batch_value, batch_y = get_batch(shuffled_Xi, shuffled_Xv, shuffled_y, batch_size, i)
            actual_batch = len(batch_y)
            batch_shape = np.array([actual_batch, feature_length], dtype=np.int64)
            cost, acc, rs, global_step, _ = sess.run([model.loss,
                                                      model.accuracy,
                                                      merged,
                                                      model.global_step,
                                                      model.train_op],
                                                      feed_dict={model.X: (batch_index, batch_value, batch_shape),
                                                                 model.y: batch_y}
                                                     )
            writer.add_summary(rs, global_step=global_step)
            loss_history.append(cost)
        batch_index_acc, batch_value_acc, batch_y_acc = get_batch(Xi_train, Xv_train, y_train, len(y_train), 0)
        batch_shape_acc =  np.array([len(y_train), feature_length], dtype=np.int64)
        # auc_value, auc_op = sess.run(model.auc,
        #                             feed_dict={model.X: (batch_index_acc, batch_value_acc, batch_shape_acc),
        #                                        model.y: batch_y_acc}
        #                             )
        feed_dict_all = {model.X: (batch_index_acc, batch_value_acc, batch_shape_acc),
                                  model.y: batch_y_acc}
        y_ = sess.run(model.y, feed_dict=feed_dict_all)[:, 0]
        y_prob = sess.run(model.y_hat_prob, feed_dict=feed_dict_all)[:, 0]
        print(y_.shape, y_prob.shape, y_[:100], y_prob[:100])
        # tf.metrics.auc(y_, y_prob, num_thresholds=1000)
        auc_value = roc_auc_score(y_, y_prob)
        print(auc_value)
        acc = sess.run(model.accuracy,
                       feed_dict=feed_dict_all
                       )
        accuracy_history.append(acc * 100)
        auc_history.append(auc_value)
        if epoch % 2 == 0:
            # act = sess.run([model.accuracy], feed_dict={x: tf.SparseTensorValue(indices, coo.data, coo.shape), y: y_test})
            print("Epoch " + str(epoch) + " Cost: " + str(loss_history[-1]) +
                  " Accuracy: " + str(accuracy_history[-1]) +
                  " AUC: " + str(auc_history[-1]))
            # saver.save(sess, "checkpoints/model", global_step=global_step)

def getData(filename='train_dataset.csv'):
    dataSet = pd.read_csv(filename)
    X = dataSet.ix[:, :-1].values.astype(np.float32)
    target = dataSet.ix[:, -1].values.astype(np.float32)
    target = np.reshape(target, [-1, 1])
    oneHot = OneHotEncoder()
    labels = oneHot.fit_transform(target).toarray()
    return X, labels

# def my_next_batch(data, labels, batchsize):
#     '''
#     Return a total of `num` random samples and labels.
#     '''
#     idx = np.arange(0 , len(data))
#     np.random.shuffle(idx)
#     shuffleX = data[idx]
#     shuffley = labels[idx]
#     for i in range(int(len(data) / batchsize) + 1):
#         if (i+1)*batchsize > len(data):
#             yield shuffleX[i * batchsize:], shuffley[i * batchsize:]
#         else:
#             xndarray = shuffleX[i * batchsize:(i + 1) * batchsize]
#             yndarray = shuffley[i * batchsize:(i + 1) * batchsize]
#             yield xndarray, yndarray

if __name__ == '__main__':
    num_class = 2
    num_factors = 15
    lr = 0.0003
    lr_l1 = 2e-2
    lr_l2 = 0
    BATCHSIZE = 512
    EPOCHS = 10
    all_cols = ['click', 'C1', 'banner_pos', 'device_conn_type', 'C15', 'C16', 'C18']
    numerical_cols = ['C1', 'C15', 'C16', 'C18']
    Xtrain, ytrain, Xval, yval = load_data()

    print(Xtrain.shape, ytrain.shape, Xval.shape, yval.shape)
    feat_dict, feat_length = generate_feasdict(Xtrain, Xval, numerical_cols)
    Xi_train, Xv_train = parse_data(feat_dict, Xtrain, numerical_cols)
    Xi_val, Xv_val = parse_data(feat_dict, Xval, numerical_cols)
    print(np.array(Xi_train).shape, np.array(Xv_train).shape,
          np.array(Xi_val).shape, np.array(Xv_val).shape,
          Xi_train[:5], Xv_train[:5])
    print(feat_length)
    print(feat_dict)
    # initialize FM model
    model = FM(feat_length, num_class, num_factors, lr, lr_l1, lr_l2)
    # build graph for model
    model.define_model()
    saver = tf.train.Saver(max_to_keep=5)
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        train_model(sess, model, Xi_train, Xv_train, ytrain, feature_length=feat_length, epochs=EPOCHS, batch_size=BATCHSIZE)





# print(np.array(Xi_train).shape, np.array(Xv_train).shape,
#       np.array(Xi_val).shape, np.array(Xv_val).shape,
#       Xi_train[:5], Xv_train[:5])
#
# define_model(feat_length, num_class, num_factors, lr, lr_l1, lr_l2)




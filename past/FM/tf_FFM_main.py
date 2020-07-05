import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import roc_auc_score
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.ConfigProto(allow_soft_placement=True)
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
config.gpu_options.allow_growth = True

# load data
def load_data():
    oneHot = OneHotEncoder()
    choose_cols = ['C1', 'banner_pos', 'device_conn_type', 'C15', 'C16', 'C18', 'click']
    # all_cols = ['C1', 'banner_pos', 'device_conn_type', 'C15', 'C16', 'C18', 'click', ]
    numerical_cols = ['C1', 'C15', 'C16', 'C18']
    object_cols = ['banner_pos', 'device_conn_type', 'click']#list(set(choose_cols) - set(numerical_cols))
    trainfile = '../train_df.csv'
    Train = pd.read_csv(trainfile)
    dfTrain = Train[choose_cols[:-1]]
    # Train[object_cols] = Train[object_cols].astype(str)
    Train[numerical_cols] = Train[numerical_cols].astype(np.float32)
    X_dummy = pd.concat([pd.get_dummies(Train[i], prefix=i, drop_first=False) for i in object_cols[:-1]], axis=1)
    X_train = pd.concat([Train[numerical_cols], X_dummy], axis=1)
    # ytrain = X_train.iloc[:, -2:]
    # Xtrain = X_train.iloc[:, :-1]
    ytrain = Train[choose_cols[-1]]
    print(X_train.shape)
    print('%%%%%%%%%%%%%')

    Xtrain = X_train
    # Xtrain = dfTrain.ix[:, :-1]
    # print('**********')
    # print(Counter(dfTrain.ix[:, -1]))

    # ytrain = dfTrain.ix[:, -1].values.astype(np.float32).reshape([-1, 1])
    # ytrain = oneHot.fit_transform(ytrain).toarray()

    testfile = '../test_df.csv'
    Test = pd.read_csv(testfile)
    dfTest = Test[choose_cols[:-1]]
    # Test[object_cols] = Test[object_cols].astype(str)
    Test[numerical_cols] = Test[numerical_cols].astype(np.float32)
    X_dummy = pd.concat([pd.get_dummies(Test[i], prefix=i, drop_first=False) for i in object_cols[:-1]], axis=1)
    X_test = pd.concat([Test[numerical_cols], X_dummy], axis=1)
    yval = Test[choose_cols[-1]]
    Xval = X_test#X_test.iloc[:, :-2]
    print(X_test.shape)
    print('%%%%%%%%%%%%%')
    # dfTest = Test[choose_cols]
    # Xval = dfTest.ix[:, :-1]
    # yval = dfTest.ix[:, -1].values.astype(np.float32).reshape([-1, 1])
    # yval = oneHot.fit_transform(yval).toarray()

    return Xtrain, ytrain, Xval, yval, dfTrain, dfTest

# generate feature dict
def generate_feasdict(dfTrain, dfTest, numerical_cols):

    df = pd.concat([dfTrain, dfTest])
    feat_dict = {}
    feature2field = {}
    idx = 0
    field_index = 0
    for col in df.columns:
        if col in numerical_cols:
            feat_dict[col] = idx
            feature2field[idx] = field_index
            idx += 1
        else:
            dataunique = df[col].unique()
            feat_dict[col] = dict(zip(dataunique, range(idx, idx + len(dataunique))))
            for  i in range(idx, idx + len(dataunique)):
                feature2field[i] = field_index
            idx += len(dataunique)
        field_index += 1
    return feat_dict, idx, field_index, feature2field

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


class FFM(object):
    def __init__(self, feat_length, num_class, num_factors, num_field, feature2field, lr, lr_l1, lr_l2, threshold=0.5):
        self.feat_length = feat_length
        self.num_class = num_class
        self.num_factors = num_factors
        self.feature2field = feature2field
        self.num_field = num_field
        self.lr = lr
        self.lr_l1 = lr_l1
        self.lr_l2 = lr_l2
        self.threshold = threshold

    def define_model(self):
        self.X = tf.placeholder('float32', [None, self.feat_length])
        self.y = tf.placeholder('int64',[None,self.num_class])
        with tf.variable_scope('linear_layer'):
            W1 = tf.get_variable(name='w1',
                                 shape=[self.feat_length, self.num_class],
                                 initializer=tf.truncated_normal_initializer(mean=0, stddev=1e-2))
            b = tf.get_variable(name='b',
                                shape=[self.num_class],
                                initializer=tf.zeros_initializer())
            self.linear_terms = tf.add(tf.matmul(self.X, W1), b)
            # self.linear_terms = tf.add(tf.sparse_tensor_dense_matmul(self.X, W1), b)
            tf.summary.histogram('w1', W1)
            tf.summary.histogram('b', b)

        with tf.variable_scope('field_aware_interaction_layer'):
            self.v = tf.get_variable(name='v',
                                shape=[self.feat_length, self.num_field, self.num_factors],
                                initializer=tf.truncated_normal_initializer(mean=0, stddev=0.01))
            self.field_aware_interaction_terms = tf.constant(0, dtype='float32')
            for i in range(self.feat_length):
                for j in range(i+1, self.feat_length):
                    self.field_aware_interaction_terms += tf.multiply(
                        tf.reduce_sum(tf.multiply(self.v[i, self.feature2field[j]], self.v[j,self.feature2field[i]])),
                        tf.multiply(self.X[:,i], self.X[:,j])
                    )
            self.field_aware_interaction_terms =  tf.expand_dims(self.field_aware_interaction_terms, axis=1)
            # 对每个特征求交叉项
            # for i in range(self.feat_length-1):
            #     for j in range(i+1, self.feat_length):
            #         print('i:%s, j:%s' % (i, j))
            #         vifj = self.v[i, self.feature2field[j]]
            #         print('vifj:', vifj)
            #         vjfi = self.v[j, self.feature2field[i]]
            #         print('vjfi:', vjfi)
            #         vivj = tf.reduce_sum(tf.multiply(vifj, vjfi))
            #         xixj = tf.multiply(self.X[:, i], self.X[:, j])
            #         self.field_aware_interaction_terms += tf.multiply(vivj, xixj)
            #         # self.field_aware_interaction_terms += tf.multiply(
            #         #     tf.reduce_sum(tf.multiply(v[i, self.feature2field[j]],
            #         #                               v[j, self.feature2field[i]])),
            #         #     tf.multiply(self.X[:, i], self.X[:, j])
            #         # )
        with tf.name_scope('logit'):
            # print('&&&&&&&&&&&&&&&&&&&&&')
            # print(self.linear_terms.shape, self.field_aware_interaction_terms.shape)
            # self.lshape = tf.shape(self.linear_terms)
            # self.fshape = tf.shape(self.field_aware_interaction_terms)
            self.y_hat = tf.add(self.linear_terms, self.field_aware_interaction_terms)
            # self.y_hat_shape = tf.shape(self.y_hat)
            self.y_hat_prob = tf.nn.sigmoid(self.y_hat)
            # self.y_hat = tf.add(self.linear_terms, self.interaction_terms)
            # self.y_hat_prob = tf.nn.softmax(self.y_hat)

        with tf.name_scope('loss'):
            cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.cast(self.y, tf.float32), logits=self.y_hat)
            # cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y, logits=self.y_hat)
            ## 下面这个也可以
            # cross_entropy = - tf.cast(self.y, tf.float32) * tf.log(tf.clip_by_value(self.y_hat_prob, 1e-10, 1.0)) - \
            #                 tf.cast((1 - self.y), tf.float32) * tf.log(tf.clip_by_value(1 - self.y_hat_prob, 1e-10, 1.0))
            self.loss = tf.reduce_mean(cross_entropy)
            tf.summary.scalar("loss", self.loss)

        with tf.name_scope('accuracy'):
            ## correct_prediction = tf.equal(tf.argmax(self.y_hat_prob, 1), tf.argmax(self.y, 1))
            # self.correct_prediction = tf.equal(tf.cast(tf.argmax(self.y_hat_prob, 1), tf.int64), self.y)
            # self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
            self.y_pred = tf.cast(self.y_hat_prob > self.threshold, tf.int32)
            self.accuracy = tf.metrics.accuracy(
                labels=self.y,
                predictions=self.y_pred,
                name="accuracy")
            # self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            tf.summary.scalar("acc", self.accuracy[1])
        ## 这个有问题
        # with tf.name_scope('auc'):
        #     self.auc = tf.metrics.auc(self.y_hat_prob[0], tf.argmax(self.y, 1), num_thresholds=1000)
        #     # tf.summary.scalar("auc", self.auc)

        with tf.name_scope('auc'):
            self.auc = tf.metrics.auc(labels=self.y, predictions=self.y_hat_prob)
            tf.summary.scalar("auc", self.auc[1])

        with tf.name_scope('optimizer'):
            self.global_step = tf.Variable(0, trainable=False)
            # optimizer = tf.train.FtrlOptimizer(self.lr, l1_regularization_strength=self.lr_l1,
            #                                    l2_regularization_strength=self.lr_l2,
            #                                    )
            optimizer = tf.train.FtrlOptimizer(learning_rate=self.lr)
            extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(extra_update_ops):
                self.train_op = optimizer.minimize(self.loss, global_step=self.global_step)

def get_batch(Xv,  y, batch_size, index):
    start = index * batch_size
    end = (index + 1) * batch_size
    end = end if end < len(y) else len(y)
    xv_bs = []
    for k, v in enumerate(Xv[start:end]):
        xv_bs.append(v)
    y_bs = y[start:end]
    return  xv_bs, y_bs

def shuffle_data(Xv, y):
    idx = np.arange(0, len(Xv))
    np.random.shuffle(idx)
    shuffled_Xv = Xv.values[idx]
    shuffled_y = y.values.reshape((-1,1))[idx]
    return shuffled_Xv, shuffled_y

def train_model(sess, model, Xv_train, y_train, feature_length, epochs=1, batch_size=128):
    print('*************')
    print(Xv_train.shape, y_train.shape)
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter("loss_log4/", sess.graph)
    loss_history, accuracy_history, auc_history = [], [], []
    for epoch in range(epochs):
        print('current epoch is ', epoch)
        total_batch = int(len(y_train) / batch_size)
        shuffled_Xv, shuffled_y = shuffle_data(Xv_train, y_train)
        for i in range(total_batch):
            batch_value, batch_y = get_batch(shuffled_Xv, shuffled_y, batch_size, i)
            # print(batch_value, '&&&&&', batch_y)
            # print(np.array(batch_value).shape, np.array(batch_y).shape)
            # actual_batch = len(batch_y)
            # batch_shape = np.array([actual_batch, feature_length], dtype=np.int64)
            # print('y-pred')
            # print(sess.run(model.y_pred, feed_dict={model.X: batch_value,
            #                                                      model.y: batch_y}).shape)
            # print(sess.run(model.y_pred, feed_dict={model.X: batch_value,
            #                                         model.y: batch_y}))
            # print('y')
            # print(sess.run(model.y, feed_dict={model.X: batch_value,
            #                                              model.y: batch_y}).shape)
            # print(sess.run(model.y, feed_dict={model.X: batch_value,
            #                                    model.y: batch_y}))
            # break
            # print(sess.run(model.lshape, feed_dict={model.X: batch_value,
            #                                         model.y: batch_y}))
            # print(sess.run(model.fshape, feed_dict={model.X: batch_value,
            #                                                      model.y: batch_y}))
            cost, auc, rs, global_step, acc, _ = sess.run([model.loss,
                                                      model.auc,
                                                      merged,
                                                      model.global_step,
                                                      model.accuracy,
                                                      model.train_op],
                                                      feed_dict={model.X: batch_value,
                                                                 model.y: batch_y}
                                                     )
            # acc = 0
            writer.add_summary(rs, global_step=global_step)
            loss_history.append(cost)
            auc_history.append(auc[1])
            accuracy_history.append(acc[1])

        if epoch % 2 == 0:
            print("Epoch " + str(epoch) + " Cost: " + str(loss_history[-1]) +
                  " Accuracy: " + str(accuracy_history[-1]) +
                  " Auc: " + str(auc_history[-1])
                  )
            # saver.save(sess, "checkpoints/model", global_step=global_step)

def getData(filename='../train_dataset.csv'):
    dataSet = pd.read_csv(filename)
    X = dataSet.ix[:, :-1].values.astype(np.float32)
    target = dataSet.ix[:, -1].values.astype(np.float32)
    target = np.reshape(target, [-1, 1])
    oneHot = OneHotEncoder()
    labels = oneHot.fit_transform(target).toarray()
    return X, labels

if __name__ == '__main__':
    num_class = 1
    num_factors = 15
    lr = 0.0003
    lr_l1 = 2e-2
    lr_l2 = 0
    BATCHSIZE = 512
    EPOCHS = 10
    all_cols = ['click', 'C1', 'banner_pos', 'device_conn_type', 'C15', 'C16', 'C18']
    numerical_cols = ['C1', 'C15', 'C16', 'C18']
    Xtrain, ytrain, Xval, yval, dfTrain, dfTest = load_data()

    print(Xtrain.shape, ytrain.shape, Xval.shape, yval.shape, dfTrain.shape, dfTest.shape)
    feat_dict, feat_length, field_idx, feature2field = generate_feasdict(dfTrain, dfTest, numerical_cols)
    print('*******', feat_dict, feat_length, feature2field)
    print('--- feat_length ---', feat_length)
    print('--- feat_dict ---', feat_dict)
    print('--- feature2field ---', feature2field)
    num_field = max(feature2field.values()) + 1
    print('--- num_field ---', num_field)
    # # initialize FM model
    model = FFM(feat_length, num_class, num_factors, num_field, feature2field, lr, lr_l1, lr_l2)

    # build graph for model
    model.define_model()
    saver = tf.train.Saver(max_to_keep=5)
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        train_model(sess, model, Xtrain, ytrain, feature_length=feat_length, epochs=EPOCHS, batch_size=BATCHSIZE)


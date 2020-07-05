import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.ConfigProto(allow_soft_placement=True)
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
config.gpu_options.allow_growth = True

# load data
def load_data():
    choose_cols = ['click', 'C1', 'banner_pos', 'device_conn_type', 'C15', 'C16', 'C18']
    trainfile = 'train_df.csv'
    Train = pd.read_csv(trainfile)
    dfTrain = Train[choose_cols]
    Xtrain = dfTrain.ix[:, :-1]
    ytrain = dfTrain.ix[:, -1]

    testfile = 'test_df.csv'
    Test = pd.read_csv(testfile)
    dfTest = Test[choose_cols]
    Xval = dfTest.ix[:, :-1]
    yval = dfTest.ix[:, -1]

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
    Xi = dfi.values.tolist()
    # list of list of feature values of each sample in the dataset
    Xv = dfv.values.tolist()
    return Xi, Xv


class FM(object):
    def __init__(self, feat_length, num_class, num_factors, lr, lr_l1, lr_l2):
        self.feat_length = feat_length
        self.num_class = num_class
        self.num_factor = num_factors
        self.lr = lr
        self.lr_l1 = lr_l1
        self.lr_l2 = lr_l2

    def define_model(self):
        self.X = tf.placeholder('float32', [None, self.feat_length])
        # X = tf.sparse_placeholder('float32', [None, feat_length])
        self.y = tf.placeholder('int64', [None, self.num_class])
        with tf.variable_scope('linear_layer'):
            W1 = tf.get_variable(name='w1',
                                 shape=[self.feat_length, self.num_class],
                                 initializer=tf.truncated_normal_initializer(mean=0, stddev=1e-2))
            b = tf.get_variable(name='b',
                                shape=[num_class],
                                initializer=tf.zeros_initializer())
            self.linear_terms = tf.add(tf.matmul(self.X, W1), b)
            # linear_terms = tf.add(tf.sparse_tensor_dense_matmul(X, W1), b)
            tf.summary.histogram('w1', W1)
            tf.summary.histogram('b', b)
        with tf.variable_scope('interaction_layer'):
            v = tf.get_variable(name='v',
                                shape=[feat_length, num_factors],
                                initializer=tf.truncated_normal_initializer(mean=0, stddev=0.01))
            s1 = tf.matmul(tf.pow(self.X, 2), tf.pow(v, 2))
            # s1 = tf.sparse_tensor_dense_matmul(tf.pow(X, 2), tf.pow(v, 2))
            s2 = tf.pow(tf.matmul(self.X, v), 2)
            # s2 = tf.sparse_tensor_dense_matmul(tf.matmul(X, v), 2)
            self.interaction_terms = tf.multiply(0.5, tf.reduce_mean(tf.subtract(s2, s1), axis=1, keep_dims=True))

        with tf.name_scope('logit'):
            self.y_hat = tf.add(self.linear_terms, self.interaction_terms)
            self.y_hat_prob = tf.nn.softmax(self.y_hat)

        with tf.name_scope('loss'):
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=self.y, logits=self.y_hat)
            # cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logit=y_hat)
            self.loss = tf.reduce_mean(cross_entropy)
            tf.summary.scalar("loss", self.loss)
        with tf.name_scope('accuracy'):
            correct_prediction = tf.equal(tf.argmax(self.y_hat, 1), tf.argmax(self.y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            # tf.summary.scalar("accuracy", self.accuracy)
            tf.summary.scalar("acc", self.accuracy)

        with tf.name_scope('optimizer'):
            self.global_step = tf.Variable(0, trainable=False)
            optimizer = tf.train.FtrlOptimizer(self.lr, l1_regularization_strength=self.lr_l1,
                                               l2_regularization_strength=self.lr_l2,
                                               )
            extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(extra_update_ops):
                self.train_op = optimizer.minimize(self.loss, global_step=self.global_step)


def get_batch(Xi, y, batch_size, index):
    start = index * batch_size
    end = (index + 1) * batch_size
    end = end if end < len(y) else len(y)
    return Xi[start:end], y[start:end] #[[y_] for y_ in y[start:end]]

def train_model(sess, model, X_train, y_train, epochs=1, batch_size=128):
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter("loss_log2/", sess.graph)
    loss_history, accuracy_history = [], []
    for epoch in range(epochs):
        print('current epoch is ', epoch)
        total_batch = int(len(y_train) / batch_size)
        for i in range(total_batch):
            batch_xs, batch_ys = get_batch(X_train, y_train, batch_size, i)
            cost, acc, rs, global_step, _ = sess.run([model.loss,
                                                      model.accuracy,
                                                      merged,
                                                      model.global_step,
                                                      model.train_op],
                                                      feed_dict={model.X: batch_xs, model.y: batch_ys}
                                                     )
            writer.add_summary(rs, global_step=global_step)
            loss_history.append(cost)
            accuracy_history.append(acc)
        if epoch % 2 == 0:
            print("Epoch " + str(epoch) + " Cost: " + str(loss_history[-1]) +
                  " Accuracy: " + str(accuracy_history[-1]))
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
    feat_length=33
    num_class = 2
    num_factors = 10
    lr = 0.01
    lr_l1 = 2e-2
    lr_l2 = 0
    BATCHSIZE = 128
    EPOCHS = 30
    # initialize FM model
    model = FM(feat_length, num_class, num_factors, lr, lr_l1, lr_l2)
    # build graph for model
    model.define_model()
    # 训练集
    X_train, y_train = getData(filename='train_dataset.csv')
    print(X_train.shape, y_train.shape)

    saver = tf.train.Saver(max_to_keep=5)
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        train_model(sess, model, X_train, y_train, EPOCHS, BATCHSIZE)




# EPOCHS = 30
# all_cols = ['click', 'C1', 'banner_pos', 'device_conn_type', 'C15', 'C16', 'C18']
# numerical_cols = ['C1', 'C15', 'C16', 'C18']
# Xtrain, ytrain, Xval, yval = load_data()
# print(Xtrain.shape, ytrain.shape, Xval.shape, yval.shape)
# feat_dict, feat_length = generate_feasdict(Xtrain, Xval, numerical_cols)
# Xi_train, Xv_train = parse_data(feat_dict, Xtrain, numerical_cols)
# Xi_val, Xv_val = parse_data(feat_dict, Xval, numerical_cols)
# print(np.array(Xi_train).shape, np.array(Xv_train).shape,
#       np.array(Xi_val).shape, np.array(Xv_val).shape,
#       Xi_train[:5], Xv_train[:5])
#
# define_model(feat_length, num_class, num_factors, lr, lr_l1, lr_l2)




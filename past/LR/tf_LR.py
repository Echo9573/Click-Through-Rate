import pandas as pd
import numpy as np
import tensorflow as tf
BATCH_SIZE = 128
# 训练
trainSet = pd.read_csv('train_dataset.csv')
X = trainSet.ix[:, :-1]
ylabels = trainSet.ix[:, -1]
classes = len(ylabels.value_counts())
X = X.values.astype(np.float32)
labels = pd.get_dummies(ylabels, prefix=ylabels.name).values.astype(np.float32)
# 验证
valSet = pd.read_csv('val_dataset.csv')
X_val = valSet.ix[:, :-1]
y_val = valSet.ix[:, -1]
X_val = X_val.values
y_vallabels = pd.get_dummies(y_val, prefix=y_val.name).values


# trainSet = pd.read_csv('train_dataset.csv')
# X = trainSet.ix[:, :-1].values.astype(np.float32)
# y_train = trainSet.ix[:, -1]
# labels = oneHot.fit_transform(y_train).toarray()
#
# # 验证集
# valSet = pd.read_csv('val_dataset.csv')
# X_val = valSet.ix[:, :-1].values.astype(np.float32)
# y_val = valSet.ix[:, -1]
# y_vallabels = oneHot.fit_transform(y_val).toarray()
#


def my_next_batch(data, labels, batchsize=BATCH_SIZE):
    '''
    Return a total of `num` random samples and labels.
    '''
    idx = np.arange(0 , len(data))
    np.random.shuffle(idx)
    shuffleX = data[idx]
    shuffley = labels[idx]
    for i in range(int(len(data) / batchsize) + 1):
        if (i+1)*batchsize > len(data):
            yield shuffleX[i * batchsize:], shuffley[i * batchsize:]
        else:
            xndarray = shuffleX[i * batchsize:(i + 1) * batchsize]
            yndarray = shuffley[i * batchsize:(i + 1) * batchsize]
            yield xndarray, yndarray


x = tf.placeholder(tf.float32, [None, X.shape[1]])
y = tf.placeholder(tf.float32, [None, classes])
W = tf.Variable(tf.zeros([X.shape[1], classes]), dtype=tf.float32)
b = tf.Variable(tf.zeros([classes]), dtype=tf.float32)

yhat = tf.nn.sigmoid(tf.matmul(x, W) + b)
loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(yhat), reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

for i in range(2):
    for item in my_next_batch(X, labels):
        batch_xs, batch_ys = item[0], item[1]
        train_step.run({x: batch_xs, y: batch_ys})

correct_prediction = tf.equal(tf.argmax(yhat, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(accuracy.eval({x:X_val, y:y_vallabels}))


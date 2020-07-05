import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.ConfigProto(allow_soft_placement=True)
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
config.gpu_options.allow_growth = True

N_CLASSES = 2
BATCH_SIZE = 128
EPOCHS = 10

def getData(filename='train_dataset.csv'):
    dataSet = pd.read_csv(filename)
    X = dataSet.ix[:, :-1].values.astype(np.float32)
    target = dataSet.ix[:, -1].values.astype(np.float32)
    target = np.reshape(target, [-1, 1])
    oneHot = OneHotEncoder()
    labels = oneHot.fit_transform(target).toarray()
    return X, labels

def my_next_batch(data, labels, batchsize=BATCH_SIZE):
    '''
    Return a total of `num` random samples and labels.
    '''
    idx = np.arange(0, len(data))
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


# 训练集
X, labels = getData(filename='train_dataset.csv')
print(X.shape, labels.shape)
# 验证集
X_val, labels_val = getData(filename='val_dataset.csv')
print(X_val.shape, labels_val.shape)

x = tf.placeholder(tf.float32, [None, X.shape[1]])
y = tf.placeholder(tf.float32, [None, N_CLASSES])
with tf.name_scope('Weight'):
    W = tf.Variable(tf.zeros([X.shape[1], N_CLASSES]), name='W')
    tf.summary.histogram('weights', W)
with tf.name_scope('Bias'):
    b = tf.Variable(tf.zeros([N_CLASSES]), name='B')
    tf.summary.histogram('bias', b)
with tf.name_scope('logits'):
    yhat = tf.nn.sigmoid(tf.add(tf.matmul(x, W), b))
# loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=yhat, labels=y)
with tf.name_scope('loss'):
    loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(yhat), reduction_indices=[1]))
    tf.summary.scalar("loss", loss)
with tf.name_scope('optimizer'):
    optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
with tf.name_scope('accuracy'):
    correct_prediction = tf.equal(tf.argmax(yhat, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar("accuracy", accuracy)

with tf.Session(config=config) as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter("loss_log2/", sess.graph)

    loss_history, accuracy_history = [], []
    for epoch in range(EPOCHS):
        print('current epoch is ', epoch)
        for item in my_next_batch(X, labels):
            batch_xs, batch_ys = item[0], item[1]
            opti, cost, rs = sess.run([optimizer, loss, merged], feed_dict={x: batch_xs, y: batch_ys})
            writer.add_summary(rs, epoch)
            loss_history.append(cost) #(sum(sum(cost)))
        acc = sess.run(accuracy, feed_dict={x: X, y: labels})
        accuracy_history.append(acc * 100)
        if epoch % 1 == 0:
            print("Epoch " + str(epoch) + " Cost: " + str(loss_history[-1]))
    print('current epoch is ', epoch)
correct_prediction = tf.equal(tf.argmax(yhat, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print("\nAccuracy:", accuracy_history[-1], "%")
# print(accuracy.eval({x:X_val, y:y_vallabels}))


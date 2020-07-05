import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder
import os
from LR import LR
from utils import load_data
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.ConfigProto(allow_soft_placement=True)
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
config.gpu_options.allow_growth = True


def get_batch(x, y, batch_size, index):
    start = index * batch_size
    end = (index + 1) * batch_size
    end = end if end < len(y) else len(y)
    x_bs = []
    for k, v in enumerate(x[start:end]):
        x_bs.append(v)
    y_bs = y[start:end]
    return x_bs, y_bs


def shuffle_data(x, y):
    idx = np.arange(0, len(x))
    np.random.shuffle(idx)
    shuffled_x = x[idx]
    shuffled_y = y[idx]
    return shuffled_x, shuffled_y


def train(sess, model, x_train, y_train, batch_size, epochs, shuffle=True):
        total_batchs = int(len(y_train) / batch_size)
        if shuffle:
            shuffled_x, shuffled_y = shuffle_data(x_train, y_train)
        else:
            shuffled_x, shuffled_y = x_train, y_train
        loss_history, accuracy_history = [], []
        for epoch in range(epochs):
            print('current epoch is ', epoch)
            for i in range(total_batchs):
                batch_x, batch_y = get_batch(shuffled_x, shuffled_y, batch_size, i)
                opti, cost, acc, global_step = sess.run([model.optimizer,
                                                        model.loss,
                                                        model.accuracy,
                                                        model.global_step
                                                         ],
                                            feed_dict={model.x: batch_x,
                                                          model.y: batch_y})
                loss_history.append(cost)
                accuracy_history.append(acc)
            if epoch % 2 == 0:
                print('eps: {}, loss: {}, acc: {}'.format(epoch, loss_history[-1], accuracy_history[-1]))


if __name__ == "__main__":
    N_CLASSES = 2
    BATCH_SIZE = 128
    EPOCHS = 10

    numerical_cols = ['C1', 'C15', 'C16', 'C18']
    dummy_cols = ['banner_pos', 'device_conn_type']
    target_colname = 'click'
    train_x, train_y, train_xv, train_yv, test_x, test_y, test_xv, test_yv = \
                        load_data(dummy_cols, numerical_cols, target_colname,
                                  train_file='../train_df.csv', test_file='../test_df.csv')
    num_feat = train_x.shape[1]
    model = LR(num_feat=num_feat,
               num_class=N_CLASSES)
    print(dir(model))
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        train(sess, model, train_xv, train_yv, batch_size=BATCH_SIZE, epochs=EPOCHS)


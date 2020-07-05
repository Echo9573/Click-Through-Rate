import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder
import os
from LR_embedding import LR
from utils_embedding import DataPreprocess
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.ConfigProto(allow_soft_placement=True)
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
config.gpu_options.allow_growth = True

def train_input_fn(features, labels, epochs=1, batch_size=128):
    datasets = tf.data.Dataset.from_tensor_slices((features, labels))
    return datasets.shuffle(1000).repeat(epochs).batch(batch_size)

def train(sess, model, train_features, train_labels, batch_size, epochs):
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter("loss_log0626_4/", sess.graph)

    datasets = train_input_fn(train_features, train_labels, epochs=epochs, batch_size=batch_size)
    iterator = datasets.make_one_shot_iterator()
    one_element = iterator.get_next()
    loss_history, accuracy_history, auc_history = [], [], []
    steps = 0

    try:
        while True:
            batch_feature, batch_label = sess.run(one_element)
            opti, cost, acc, auc, global_step, ms = sess.run([model.optimizer, model.loss,
                                                          model.accuracy, model.auc,
                                                          model.global_step, merged],
                                                     feed_dict={model.df_i: batch_feature['dfi'],
                                                                model.df_v: batch_feature['dfv'],
                                                                model.y: batch_label
                                                                })
            writer.add_summary(ms, global_step=global_step)
            loss_history.append(cost)
            accuracy_history.append(acc)
            auc_history.append(auc[1])
            steps += 1
            if steps % 1000 == 0:
                print('eps: {}, loss: {}, acc: {}, auc: {}'.format(steps, loss_history[-1], accuracy_history[-1], auc_history[-1]))
    except tf.errors.OutOfRangeError:
        print('Train Finished')

if __name__ == "__main__":
    N_CLASSES = 1
    BATCH_SIZE = 256
    EPOCHS = 30

    numerical_cols = []
    dummy_cols = ['banner_pos', 'device_conn_type', 'C1', 'C15', 'C16', 'C18']
    target_colname = 'click'
    trainfilename = '../train_df.csv'
    testfilename = '../test_df.csv'
    dp = DataPreprocess(dummy_cols, numerical_cols, target_colname, trainfilename, testfilename)
    train_features, train_labels = dp.parse_data(trainfilename)
    test_features, test_labels = dp.parse_data(testfilename)
    print(train_features['dfi'][:10])
    print(train_features['dfv'][:10])
    print(train_labels[:10])
    print('----------------------------------')

    feature_nums = dp.idx
    field_nums = len(dp.all_cols)

    model = LR(feature_nums, field_nums, lr=0.01, num_class=1)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        train(sess, model, train_features, train_labels, batch_size=BATCH_SIZE, epochs=EPOCHS)


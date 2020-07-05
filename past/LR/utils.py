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

def process_data(df, dummy_cols, numerical_cols, target_colname,  one_hot=False):
    x_cols = dummy_cols + numerical_cols
    print('******', x_cols + [target_colname])
    df = df[x_cols + [target_colname]]
    df[numerical_cols] = df[numerical_cols].astype(np.float32)
    x_dummy = pd.concat([pd.get_dummies(df[col], prefix=col, drop_first=False) for col in dummy_cols], axis=1)
    x = pd.concat([df[numerical_cols], x_dummy], axis=1)
    xv = x.values
    y = df[target_colname].astype(np.float32)
    yv = np.reshape(y.values, (-1, 1))
    if one_hot:
        oneHotEncoder = OneHotEncoder()
        yv = oneHotEncoder.fit_transform(yv).toarray()
    return x, y, xv, yv


def load_data(dummy_cols,
              numerical_cols,
              target_col,
              train_file='../train_df.csv',#train_dataset.csv',
              test_file='../test_df.csv',
              ):#val_dataset.csv'):
    train_df = pd.read_csv(train_file)
    print(train_df.columns)
    test_df = pd.read_csv(test_file)

    train_x, train_y, train_xv, train_yv = process_data(train_df, dummy_cols, numerical_cols, target_col, one_hot=True)
    test_x, test_y, test_xv, test_yv = process_data(test_df, dummy_cols, numerical_cols, target_col, one_hot=True)

    print(train_x.values.shape, train_xv.shape)
    print(train_y.values.shape, train_yv.shape)
    return train_x, train_y, train_xv, train_yv,\
           test_x, test_y, test_xv, test_yv

if __name__ == "__main__":
    # all_cols = ['C1', 'banner_pos', 'device_conn_type', 'C15', 'C16', 'C18', 'click']
    numerical_cols = ['C1', 'C15', 'C16', 'C18']
    dummy_cols = ['banner_pos', 'device_conn_type']
    target_colname = 'click'
    load_data(dummy_cols, numerical_cols, target_colname)

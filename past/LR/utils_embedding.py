import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.ConfigProto(allow_soft_placement=True)
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
config.gpu_options.allow_growth = True

class DataPreprocess(object):
    def __init__(self, dummy_cols, numerical_cols, target_colname, trainfilename, testfilename, one_hot = False):
        self.dummy_cols = dummy_cols
        self.numerical_cols = numerical_cols
        self.target_colname = target_colname
        self.one_hot = one_hot
        self.train_file = trainfilename
        self.test_file = testfilename
        self.all_cols = self.dummy_cols + self.numerical_cols
        self.load_data()
        self.generate_feadict()

    def _process_data(self, df):
        print('******', self.all_cols + [self.target_colname])
        df = df[self.all_cols + [self.target_colname]]
        df[self.numerical_cols] = df[self.numerical_cols].astype(np.float32)
        x_dummy = pd.concat([pd.get_dummies(df[col], prefix=col, drop_first=False) for col in self.dummy_cols], axis=1)
        x = pd.concat([df[self.numerical_cols], x_dummy], axis=1)
        xv = x.values
        y = df[self.target_colname].astype(np.float32)
        yv = np.reshape(y.values, (-1, 1))
        if self.one_hot:
            oneHotEncoder = OneHotEncoder()
            yv = oneHotEncoder.fit_transform(yv).toarray()
        return df, xv, yv

    def load_data(self):
        train_df = pd.read_csv(self.train_file)
        train_df = train_df[self.all_cols + [self.target_colname]]
        test_df = pd.read_csv(self.test_file)
        test_df = test_df[self.all_cols + [self.target_colname]]
        self.df = pd.concat([train_df, test_df])
        # self.dftrain, self.train_x, self.train_y = self._process_data(train_df)
        # self.dftest, self.test_x, self.test_y = self._process_data(test_df)
        # return dftrain, dftest, train_x, train_y, test_x, test_y

    # generate feature dict
    def generate_feadict(self):
        feat_dict = {}
        idx = 0
        for col in self.all_cols:
            if col in self.numerical_cols:
                feat_dict[col] = idx
                idx += 1
            else:
                dataunique = self.df[col].unique()
                feat_dict[col] = dict(zip(dataunique, range(idx, idx + len(dataunique))))
                idx += len(dataunique)
        self.feat_dict = feat_dict
        self.idx = idx

    def parse_data(self, filename):
        df = pd.read_csv(filename)
        df = df[self.all_cols + [self.target_colname]]
        dfi = df[self.all_cols]
        dfv = dfi.copy()
        for col in self.all_cols:
            if col in self.numerical_cols:
                dfi[col] = self.feat_dict[col]
            else:
                dfi[col] = dfi[col].map(self.feat_dict[col])
                dfv[col] = 1.
        # list of list of feature indices of each sample in the dataset
        Xi = dfi.values.astype(np.int32)
        # list of list of feature values of each sample in the dataset
        Xv = dfv.values.astype(np.float32)

        features = {
            'dfi': Xi, 'dfv':Xv
        }
        labels = df[self.target_colname].values.astype(np.float32).reshape([-1, 1])
        return features, labels

if __name__ == '__main__':
    numerical_cols = ['C1', 'C15', 'C16', 'C18']
    dummy_cols = ['banner_pos', 'device_conn_type']
    target_colname = 'click'
    trainfilename = '../train_df.csv'
    testfilename = '../test_df.csv'
    dp = DataPreprocess(dummy_cols, numerical_cols, target_colname, trainfilename, testfilename)
    train_features, train_labels = dp.parse_data(trainfilename)
    test_features, test_labels = dp.parse_data(testfilename)
    print(train_features['dfi'][:10])
    print(train_features['dfv'][:10])
    print(train_labels[:10])
    print('======================')
    embeddings = tf.get_variable(name='emb',
                                 shape=[dp.idx, 1],
                                 dtype=tf.float32,
                                 initializer=tf.initializers.glorot_uniform())

    feature = train_features['dfi'][0:3]
    batch_weights = tf.nn.embedding_lookup(embeddings, feature)

    value = train_features['dfv'][0:3]
    idxs = np.argwhere(feature >=0).tolist()
    values = feature.flatten().tolist()
    print('dfi---', feature)
    print('dfv---', value)
    print('idx---', idxs)
    print('values---', values)

    sparse_index = tf.SparseTensor(indices=idxs, values=values, dense_shape=list(feature.shape))
    batch_sparse_weights = tf.nn.embedding_lookup_sparse(embeddings, sp_ids=sparse_index, sp_weights=None, combiner='mean')

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print('embeddings---', sess.run(embeddings))
        print('batch_weights---', sess.run(batch_weights) )
        print('batch_weights.shape---', sess.run(batch_weights).shape)
        print('batch_weights.mean---', sess.run(batch_weights).mean(axis=1))

        print('sparse_index *** ', sess.run(sparse_index))
        print('batch_sparse_weights *** ', sess.run(batch_sparse_weights))


    # datasets = tf.data.Dataset.from_tensor_slices((train_features, train_labels))
    # datasets = datasets.shuffle(1000).repeat(10).batch(128)
    # iterator = datasets.make_one_shot_iterator()
    # one_element = iterator.get_next()
    # with tf.Session() as sess:
    #     sess.run(tf.global_variables_initializer())
    #     try:
    #         idx = 0
    #         while True:
    #             idx += 1
    #             print(sess.run(one_element[1]))
    #             if idx <=5:
    #                 break
    #     except tf.errors.OutOfRangeError:
    #         print('Train Finished')
import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
import os
from FFM_embedding import FFM
from utils_embedding import DataPreprocess

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.ConfigProto(allow_soft_placement=True)
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
config.gpu_options.allow_growth = True

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('train_file', '../train_df.csv', 'trainfilename')
tf.app.flags.DEFINE_string('test_file', '../test_df.csv', 'testfilename')
tf.app.flags.DEFINE_string('target_colname', 'click', 'colname of target')

tf.app.flags.DEFINE_list('numerical_cols', [], 'numerical_cols')
tf.app.flags.DEFINE_list('dummy_cols', [], 'dummy_cols')

tf.app.flags.DEFINE_integer('num_class', 1, 'N_CLASSES')
tf.app.flags.DEFINE_integer('batch_size', 256, 'BATCH_SIZE')
tf.app.flags.DEFINE_integer('epochs', 30, 'EPOCHS')

tf.app.flags.DEFINE_float('lr', 0.01, 'learning_rate')
tf.app.flags.DEFINE_float('threshold', 0.5, 'threshold')
tf.app.flags.DEFINE_integer('embedding_size', 4, 'embedding size')


def train_input_fn(features, labels, epochs=1, batch_size=128):
    datasets = tf.data.Dataset.from_tensor_slices((features, labels))
    return datasets.shuffle(1000).repeat(epochs).batch(batch_size)

def train(sess, model, train_features, train_labels, batch_size, epochs):
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter("loss_log0626_ffm_1/", sess.graph)

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
            accuracy_history.append(acc[1])
            auc_history.append(auc[1])
            steps += 1
            if steps % 1000 == 0:
                print('eps: {}, loss: {}, acc: {}, auc: {}'.format(steps, loss_history[-1], accuracy_history[-1], auc_history[-1]))
    except tf.errors.OutOfRangeError:
        print('Train Finished')

def main(unused_argv):
    FLAGS.dummy_cols = ['banner_pos', 'device_conn_type', 'C1', 'C15', 'C16', 'C18']
    dp = DataPreprocess(FLAGS.dummy_cols, FLAGS.numerical_cols,
                        FLAGS.target_colname, FLAGS.train_file, FLAGS.test_file)
    train_features, train_labels = dp.parse_data(FLAGS.train_file)
    # test_features, test_labels = dp.parse_data(FLAGS.test_file)
    print(train_features['dfi'][:10])
    print(train_features['dfv'][:10])
    print(train_labels[:10])
    print('----------------------------------')

    feature_nums = dp.feature_nums
    field_nums = len(dp.all_cols)

    model = FFM(feature_nums, field_nums, args=FLAGS)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        train(sess, model, train_features, train_labels, batch_size=FLAGS.batch_size, epochs=FLAGS.epochs)

if __name__ =='__main__':
    tf.app.run()
import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
import os
from DCN import DCN
from utils_embedding import DataPreprocess
from sklearn.metrics import roc_auc_score

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.ConfigProto(allow_soft_placement=True)
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
config.gpu_options.allow_growth = True

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('checkpoint_dir', './checkpoint', 'the path to save model checkpoint')

tf.app.flags.DEFINE_string('train_file', '../train_df.csv', 'trainfilename')
tf.app.flags.DEFINE_string('test_file', '../test_df.csv', 'testfilename')
tf.app.flags.DEFINE_string('target_colname', 'click', 'colname of target')

tf.app.flags.DEFINE_list('numerical_cols', [], 'numerical_cols')
tf.app.flags.DEFINE_list('dummy_cols', [], 'dummy_cols')

tf.app.flags.DEFINE_integer('num_class', 1, 'N_CLASSES')
tf.app.flags.DEFINE_integer('batch_size', 256, 'BATCH_SIZE')
tf.app.flags.DEFINE_integer('epochs', 5, 'EPOCHS')

tf.app.flags.DEFINE_float('lr', 0.01, 'learning_rate')
tf.app.flags.DEFINE_float('threshold', 0.5, 'threshold')
tf.app.flags.DEFINE_integer('embedding_size', 4, 'embedding size')

tf.app.flags.DEFINE_bool('use_deep', True, 'whether is deepfm')
tf.app.flags.DEFINE_list('hidden_units', [300,300,300], 'hidden_units of deep layers')
tf.app.flags.DEFINE_float('dropout_keep_deep', 0.9, 'dropout_keep_deep')
tf.app.flags.DEFINE_float('dropout_keep_deep1', 0.9, 'dropout_keep_deep1')
tf.app.flags.DEFINE_bool('use_batch_normal', True, 'use_batch_normal')

tf.app.flags.DEFINE_bool('use_better', True, 'use_better')
tf.app.flags.DEFINE_integer('cross_layers', 3, 'cross_layers')

def train_input_fn(features, labels, epochs=1, batch_size=128):
    datasets = tf.data.Dataset.from_tensor_slices((features, labels))
    return datasets.shuffle(1000).repeat(epochs).batch(batch_size)

def train(sess, model, train_features, train_labels, batch_size, epochs, checkpoint_dir='./checkpoint'):
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    saver = tf.train.Saver(tf.global_variables())

    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter("loss_log0626_fm_4/", sess.graph)

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

    print("Starting Saving Model...")
    saver.save(sess, os.path.join(checkpoint_dir, 'model_ckpt'))
    print('Saving Model Finish!')

def evaluate(test_features, test_labels,  checkpoint_dir='./checkpoint'):
    checkpoint_file = tf.train.latest_checkpoint(checkpoint_dir)
    graph = tf.Graph()
    with graph.as_default():
        with tf.Session(config=config) as sess:
            saver = tf.train.import_meta_graph('{}.meta'.format(checkpoint_file))
            saver.restore(sess, checkpoint_file)

            df_i = graph.get_tensor_by_name('df_i:0')
            df_v = graph.get_tensor_by_name('df_v:0')
            y_prob = graph.get_tensor_by_name('logit/y_hat_prob:0')

            predictions = sess.run(y_prob, feed_dict={df_i: test_features['dfi'],
                                                      df_v: test_features['dfv']})
            print('auc: {:.5f}'.format(roc_auc_score(y_true=test_labels, y_score=predictions)))
            print('mean pctr: {:.5f}'.format(predictions.mean()))
            print('mean base ctr: {:.5f}'.format(test_labels.mean()))
            print('calibration: {:.5f}'.format(predictions.mean() / test_labels.mean()))

def main(unused_argv):
    FLAGS.dummy_cols = ['banner_pos', 'device_conn_type', 'C1', 'C15', 'C16', 'C18']
    dp = DataPreprocess(FLAGS.dummy_cols, FLAGS.numerical_cols,
                        FLAGS.target_colname, FLAGS.train_file, FLAGS.test_file)
    train_features, train_labels = dp.parse_data(FLAGS.train_file)
    test_features, test_labels = dp.parse_data(FLAGS.test_file)
    print(train_features['dfi'][:10])
    print(train_features['dfv'][:10])
    print(train_labels[:10])
    print('----------------------------------')

    feature_nums = dp.feature_nums
    field_nums = len(dp.all_cols)

    model = DCN(feature_nums, field_nums, args=FLAGS)

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        train(sess, model, train_features, train_labels,
              batch_size=FLAGS.batch_size,
              epochs=FLAGS.epochs,
              checkpoint_dir=FLAGS.checkpoint_dir)

    evaluate(test_features, test_labels, checkpoint_dir=FLAGS.checkpoint_dir)

if __name__ =='__main__':
    tf.app.run()

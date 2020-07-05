import tensorflow as tf
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('train_file', '../train_df', 'trainfilename')
print(FLAGS.train_file)
print('1')
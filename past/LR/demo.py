
import tensorflow as tf

labels = [[0.2 ,0.3 ,0.5],
          [0.1 ,0.6 ,0.3]]
logits = [[2 ,0.5 ,1],
          [0.1 ,1 ,3]]
logits_scaled = tf.nn.softmax(logits)

result1 = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)
result2 = -tf.reduce_sum(labels *tf.log(logits_scaled),1)
result3 = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits_scaled)

yhat = tf.constant([2, 1])
# yhat_prob =  tf.nn.softmax(yhat)
# log_loss = tf.losses.log_loss(labels=y, predictions=yhat_prob)
with tf.Session() as sess:
    print(sess.run(logits_scaled))
    print(sess.run(result1))
    print(sess.run(result2))
    print(sess.run(result3))

# labels = tf.constant([0, 1, 1, 1], dtype=tf.float32)
# pred = tf.constant([0.22, 0.4, 0.67, 0.88], dtype=tf.float32)
# labels_ = tf.reshape(labels, shape=[-1, 1])
# pred_ =

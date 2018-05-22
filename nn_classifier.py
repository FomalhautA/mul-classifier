import numpy as np
import redis
import json
import tensorflow as tf

_redis = redis.Redis(host='192.168.0.100', port=6379, db=0)


def forward_calc(X):
    """

    :param X:
    :return:
    """


def loss_calc(X, Y):
    """

    :param X:
    :param Y:
    :return:
    """


def load_data():
    pass


def backword_calc():
    pass


def evaluate(sess, X, Y):
    pass


saver = tf.train.Saver()


with tf.Session() as sess:
    tf.initialize_all_variables().run()

    X, Y = load_data()

    total_loss = loss_calc(X, Y)
    train_op = backword_calc(total_loss)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    training_steps = 1000
    for step in range(training_steps):
        sess.run([train_op])

        if step % 10 == 0:
            print("loss: {}".format(sess.run(total_loss)))

        if step % 1000 == 0:
            saver.save(sess, 'my-model', global_step=step)

    evaluate(sess, X, Y)

    coord.request_stop()
    coord.join(threads)

    saver.save(sess, 'my-model', global_step=training_steps)

    sess.close()


def NN_classifier():
    idx_mat = _redis.hget("classification_dataset", "index_mat")

    feature_mat = _redis.hget("classification_dataset", "feature_mat")

    label_mat = _redis.hget("classification_dataset", "label_mat")

    print("index matrix: {}".format(json.loads(idx_mat)))
    print("feature matrix: {}".format(json.loads(feature_mat)))
    print("label matrix: {}".format(json.loads(label_mat)))

    np.array(feature_mat).shape

    X = tf.placeholder(tf.float16, name="features")
    W = tf.placeholder_with_default(tf.float16, name="weights")


if __name__ == '__main__':
    NN_classifier()

#!/usr/bin/env python3
'''evaluate output'''

import tensorflow as tf


def evaluate(X, Y, save_path):
    '''eval output'''
    sess = tf.session()
    saved = tf.train.import_meta_graph(save_path + '.meta')
    saved.restore(sess, save_path)
    graph = tf.get_default_graph()
    x = graph.get_tensor_by_name('x:0')
    y = graph.get_tensor_by_name('y:0')
    y_pred = tf.get_collection('tensors')[0]
    loss = tf.get_collection('loss')[0]
    accuracy = tf.get_collection('accuracy')[0]
    return tuple(sess.run([y_pred, accuracy, loss], feed_dict={x: X, y: Y}))


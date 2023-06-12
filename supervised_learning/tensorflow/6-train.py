#!/usr/bin/env python3
'''train time'''
import tensorflow as tf


def train(X_train, Y_train, X_valid, Y_valid, layer_sizes, activations, alpha, iterations, save_path="/tmp/model.ckpt"):
    '''trians model'''
    tf.reset_default_graph()
    
    x = tf.placeholder(tf.float32, shape=(None, X_train.shape[1]), name='x')
    y = tf.placeholder(tf.float32, shape=(None, Y_train.shape[1]), name='y')
    
    prev = x
    for size, activation in zip(layer_sizes, activations):
        initializer = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
        layer = tf.layers.Dense(size, activation=activation, kernel_initializer=initializer, name="layer")
        prev = layer(prev)
    y_pred = prev
    
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_pred))
    correct_predictions = tf.equal(tf.argmax(y_pred, axis=1), tf.argmax(y, axis=1))
    accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
    
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=alpha)
    train_op = optimizer.minimize(loss)
    
    tf.add_to_collection('placeholders', x)
    tf.add_to_collection('placeholders', y)
    tf.add_to_collection('tensors', y_pred)
    tf.add_to_collection('tensors', loss)
    tf.add_to_collection('tensors', accuracy)
    tf.add_to_collection('operations', train_op)
    
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for i in range(iterations + 1):
            _, train_cost, train_accuracy = sess.run([train_op, loss, accuracy], feed_dict={x: X_train, y: Y_train})
            if i % 100 == 0 or i == 0 or i == iterations:
                valid_cost, valid_accuracy = sess.run([loss, accuracy], feed_dict={x: X_valid, y: Y_valid})
                print(f"After {i} iterations:")
                print(f"\tTraining Cost: {train_cost}")
                print(f"\tTraining Accuracy: {train_accuracy}")
                print(f"\tValidation Cost: {valid_cost}")
                print(f"\tValidation Accuracy: {valid_accuracy}")

        saver.save(sess, save_path)

    return save_path

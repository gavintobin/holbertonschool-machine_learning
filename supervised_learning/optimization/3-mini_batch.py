#!/usr/bin/env python3
'''task 3'''
import tensorflow as tf

shuffle_data = __import__('2-shuffle_data').shuffle_data
def train_mini_batch(X_train, Y_train, X_valid, Y_valid, batch_size=32, epochs=5, load_path="/tmp/model.ckpt", save_path="/tmp/model.ckpt"):
    '''trains loaded nn'''
    with tf.session as sesh:
        saver = tf.train.import_meta_graph(load_path + '.meta')
        saver.restore(sesh, load_path)

        graph = tf.get_default_graph()
        x = graph.get_collection('x')[0]
        y = graph.get_collection('y')[0]
        accuracy = graph.get_collection('accuracy')[0]
        loss = graph.get_collection('loss')[0]
        train = graph.get_collection("train")
        batchs = len(X_train)// batch_size

        while batchs % batch_size != 0:
            batchs += 1

        for i in range(epochs + 1):
            train_cost, accuracy = sesh.run(
                [loss, accuracy], feed_dict={x: X_train, y: Y_train})
            valid_cost, accuracy = sesh.run(
                [loss, accuracy], feed_dict={x: X_valid, y: Y_valid})

            print("After {} epochs:".format(i))
            print("\tTraining Cost: {}".format(train_cost))
            print("\tTraining Accuracy: {}".format(accuracy))
            print("\tValidation Cost: {}".format(valid_cost))
            print("\tValidation Accuracy: {}".format(accuracy))

            if i < epochs:
                X_Shuffled, Y_Shuffled = shuffle_data(X_train, Y_train)

                for j in range(batchs):
                    batch_dict = {
                        x: X_Shuffled[batch_size * j: batch_size * (j + 1)],
                        y: Y_Shuffled[batch_size * j: batch_size * (j + 1)]
                    }
                    sesh.run((train), feed_dict=batch_dict)

                if (j + 1) % 100 == 0 and j != 0:

                    batch_cost = loss.eval(batch_dict)
                    batch_accuracy = accuracy.eval(batch_dict)
                    print("\tStep {}:".format(j + 1))
                    print("\t\tCost: {}".format(batch_cost))
                    print("\t\tAccuracy: {}".format(batch_accuracy))

        save_path = saver.save(sesh, save_path)

    return save_path



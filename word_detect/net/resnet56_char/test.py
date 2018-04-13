import cPickle
import numpy as np
import tensorflow as tf
import os
from net.resnet_train import resnet
from net.data import test_inputs

def main():
    data_list = "dataset/train_char2.txt"
    batch_size = 1024
    X_test, Y_test, num_classes, num_examples = test_inputs(data_list, batch_size)

    # Construct Network
    fc = resnet(X_test, 20, num_classes)
    saver = tf.train.Saver()
    
    pred = tf.nn.softmax(fc)
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(Y_test, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    
    sess = tf.Session()
    saver.restore(sess, 'model_char/')

    correct = 0.0
    try:
        while True:
            corr = np.sum(sess.run(correct_prediction))
            print corr
            correct += corr
    except tf.errors.OutOfRangeError:
        print correct / num_examples

    sess.close()

if __name__ == '__main__':
    main()

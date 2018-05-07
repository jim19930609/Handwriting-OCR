import cPickle
import numpy as np
import tensorflow as tf
import os
from net.resnet_train import resnet
from net.data import train_inputs

def main():
    data_list = "dataset/train_char2.txt"
    epoch = 20
    batch_size = 1024
    X_train, Y_train, num_classes, num_examples = train_inputs(data_list, batch_size, epoch)
    
    print "Total Number of Classes: ", num_classes, " ", "Number of Examples: ", num_examples
    
    # Construct Network
    fc = resnet(X_train, 20, num_classes)
    
    pred = tf.nn.softmax(fc) 
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(Y_train, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    
    saver = tf.train.Saver()

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=fc, labels=Y_train, name='xentropy'),name='loss')
    
    opt = tf.train.AdamOptimizer(0.001)
    train_op = opt.minimize(loss)

    sess = tf.Session()
    sess.run(tf.initialize_all_variables())

    try:
        while True:
            _, tot_loss, acc = sess.run([train_op, loss, accuracy])
            print "Tot Loss: ", tot_loss, " Accuracy: ", acc
    except tf.errors.OutOfRangeError:
        print "End of Training, Save Model"
        saver.save(sess, 'model_char/')
    
    sess.close()

if __name__ == '__main__':
    main()

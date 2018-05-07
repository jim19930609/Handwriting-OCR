import cPickle
import numpy as np
import tensorflow as tf
import os
import cv2
from net.data import test_inputs
from net.resnet_train import resnet

def main():
    image = tf.placeholder(tf.float32, [None, 32, 32, 1])
    
    fc = resnet(image, 20, 62)
    
    saver = tf.train.Saver()
    
    pred = tf.nn.softmax(fc)
    prediction = tf.argmax(pred, 1)

    sess = tf.Session()
    saver.restore(sess, 'model_char/')

    # Input Image for Prediction
    for name in os.listdir("dataset/train_char/Sample004/"):
        img_dir = "dataset/train_char/Sample004/" + name
        file_contents = tf.read_file(img_dir)
        img = tf.image.decode_jpeg(file_contents, channels=1)
        img = tf.image.convert_image_dtype(img, dtype=tf.float32)
        img = tf.image.per_image_standardization(img)
        img = tf.expand_dims(img, 0)
        
        img = sess.run(img)

        print sess.run(prediction, feed_dict={image: img}), np.max(sess.run(pred, feed_dict={image:img}), 1)
    
    sess.close()

if __name__ == '__main__':
    main()

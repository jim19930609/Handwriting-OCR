import cPickle
import numpy as np
import tensorflow as tf
import os
import net.readin as readin
from net.resnet_train import resnet

def Batch_Norm(img):
    # Batch Norm Input Images
    std = np.maximum(np.std(img, axis=(1,2,3)), 1.0/np.sqrt(img[0].size))
    mean = np.mean(img, axis=(1,2,3))
    img = ( (img.T - mean) / std).T
    
    return img

def shuffle(X, Y):
    ind = np.random.permutation(X.shape[0])
    X[:] = X[ind]
    Y[:] = Y[ind]

def main():
    epoch = 10
    batch_size = 1024
    learn_rate = 0.001
    reg = 0.0001
    # Readin Dataset
    X_train, Y_train, X_test, Y_test = readin.read_emnist()
    #X_train, Y_train, X_test, Y_test = readin.read_mnist()
    num_classes = 62

    # Perform Batch Norm
    X_train = Batch_Norm(X_train)
    X_test = Batch_Norm(X_test)

    # Construct Network
    image = tf.placeholder(tf.float32, [None, 28, 28, 1])
    label = tf.placeholder(tf.int32, [None, num_classes])
    fc = resnet(image, 56, num_classes)
    
    # Further Computations    
    pred = tf.nn.softmax(fc) 
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(label, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    
    cross_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=fc, labels=label, name='xentropy'),name='loss')
    var = tf.trainable_variables() 
    reg_loss = tf.add_n([ tf.nn.l2_loss(v) for v in var ]) * reg
    
    loss = reg_loss + cross_loss
    opt = tf.train.AdamOptimizer(learn_rate)
    train_op = opt.minimize(loss)
    
    saver = tf.train.Saver()
    sess = tf.Session()
    sess.run(tf.initialize_all_variables())


    # Start Training
    num_train = X_train.shape[0]
    num_iter_train = num_train // batch_size
    for epc in range(epoch):
        shuffle(X_train, Y_train)
        for i in range(num_iter_train + 1):
            if i == num_iter_train:
                img = X_train[i*batch_size:,:,:,:]
                lab = Y_train[i*batch_size:,:]
            else:
                img = X_train[i*batch_size:(i+1)*batch_size,:,:,:]
                lab = Y_train[i*batch_size:(i+1)*batch_size,:]
            
            _, l_t, l_r, l_c = sess.run([train_op, loss, reg_loss, cross_loss], feed_dict={image: img, label: lab})
            print "Step: ", i * batch_size, "/", num_train
            print " Loss: ", l_t
            print " Reg Loss: ", l_r
            print " Cross Loss: ", l_c
        
        if epc % 5 == 0:
            batch_size_test = 512
            num_test = X_test.shape[0]
            num_iter_test = num_test // batch_size_test
            tot_accuracy = 0.0
            for i in range(num_iter_test + 1):
                if i == num_iter_test:
                    img = X_test[i*batch_size_test:,:,:,:]
                    lab = Y_test[i*batch_size_test:,:]
                else:
                    img = X_test[i*batch_size_test:(i+1)*batch_size_test,:,:,:]
                    lab = Y_test[i*batch_size_test:(i+1)*batch_size_test,:]
                
                correct = sess.run(correct_prediction, feed_dict={image: img, label:lab})
                correct = np.sum(correct)
                tot_accuracy += correct
            print "Accuracy = ", tot_accuracy/num_test
        
    saver.save(sess, 'model_char/')
    sess.close()

if __name__ == '__main__':
    main()

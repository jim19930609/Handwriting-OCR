import numpy as np
from mnist import MNIST

def read_emnist():
    def randomize(dataset, labels):
        permutation = np.random.permutation(labels.shape[0])
        shuffled_dataset = dataset[permutation, :, :]
        shuffled_labels = labels[permutation]
        return shuffled_dataset, shuffled_labels
    
    def one_hot_encode(np_array):
        return (np.arange(62) == np_array[:,None]).astype(np.float32)

    def reformat_data(dataset, labels, image_width, image_height, image_depth):
        np_dataset_ = np.array([np.array(image_data).reshape(image_width, image_height, image_depth) for image_data in dataset])
        np_labels_ = one_hot_encode(np.array(labels, dtype=np.float32))
        np_dataset, np_labels = randomize(np_dataset_, np_labels_)
        return np_dataset, np_labels

    mndata = MNIST('dataset/emnist')
    mnist_image_size = 28
    mnist_image_depth = 1
    mnist_num_labels = 62
    mndata.select_emnist('byclass')
    mndata.gz = False

    mnist_train_dataset_, mnist_train_labels_ = mndata.load_training()
    mnist_test_dataset_, mnist_test_labels_ = mndata.load_testing()

    X_train, Y_train = reformat_data(mnist_train_dataset_, mnist_train_labels_, mnist_image_size, mnist_image_size, mnist_image_depth)
    X_test, Y_test = reformat_data(mnist_test_dataset_, mnist_test_labels_, mnist_image_size, mnist_image_size, mnist_image_depth)
    
    print "Training Dataset: ", X_train.shape, Y_train.shape
    print "Testing Dataset: ", X_test.shape, Y_test.shape
    
    return X_train, Y_train, X_test, Y_test

def read_mnist():
    def randomize(dataset, labels):
        permutation = np.random.permutation(labels.shape[0])
        shuffled_dataset = dataset[permutation, :, :]
        shuffled_labels = labels[permutation]
        return shuffled_dataset, shuffled_labels
    
    def one_hot_encode(np_array):
        return (np.arange(10) == np_array[:,None]).astype(np.float32)

    def reformat_data(dataset, labels, image_width, image_height, image_depth):
        np_dataset_ = np.array([np.array(image_data).reshape(image_width, image_height, image_depth) for image_data in dataset])
        np_labels_ = one_hot_encode(np.array(labels, dtype=np.float32))
        np_dataset, np_labels = randomize(np_dataset_, np_labels_)
        return np_dataset, np_labels

    mndata = MNIST('dataset/mnist')
    mnist_image_size = 28
    mnist_image_depth = 1
    mnist_num_labels = 10

    mnist_train_dataset_, mnist_train_labels_ = mndata.load_training()
    mnist_test_dataset_, mnist_test_labels_ = mndata.load_testing()

    X_train, Y_train = reformat_data(mnist_train_dataset_, mnist_train_labels_, mnist_image_size, mnist_image_size, mnist_image_depth)
    X_test, Y_test = reformat_data(mnist_test_dataset_, mnist_test_labels_, mnist_image_size, mnist_image_size, mnist_image_depth)
    
    print "Training Dataset: ", X_train.shape, Y_train.shape
    print "Testing Dataset: ", X_test.shape, Y_test.shape
    
    return X_train, Y_train, X_test, Y_test

# infer_averages.py - A script to perform naive Bayesian inference on the MNIST test images 
#                       with associative memory using a (pixel-wise) hamming distance metric.
# Wesley Chavez, 06-21-2016
# Portland State University
#                      
# The stored patterns, P(X|Y), here are the averages of each of the ten digits over the 
# whole training set, binarized with a threshold of 127.  

import numpy as np
import mnist_numpy as mnist
import CDI_modules as cdi

# Load test vectors and labels
test_images, test_labels = mnist.load_mnist('testing')

# I pre-allocate these matrices
train_mat = np.zeros((10,784))
train_lab = np.zeros((10,1))

# Get all of the training vectors for i digit, average and threshold them to produce P(X|Y)
for i in range (0,10):
    train_images, train_labels = mnist.load_mnist('training', digits=[i])
    p=train_images.mean(axis=0)
    y=np.uint8(p)
    y[y<=127]=0
    y[y>127]=1
	
    train_mat[i,:] = y.flatten()
    train_lab[i] = i

# Load a test pattern (X), flatten it, threshold it, and compute hamming distance between
# X and P(X|Y). Then compute a binary error for this test pattern --Y are the digit labels--
err=np.zeros((10000,1))
for i in range(len(test_images)):
    q=test_images[i].flatten()
    q[q<=127]=0
    q[q>127]=1
    error, Winner1, winning_label = cdi.BM_ham(train_mat,q,test_labels[i],train_lab)
    err[i]=error

# Print error over the whole test set (lower is better)
print("Error:")
print(sum(err)/len(test_images))

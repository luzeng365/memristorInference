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
train_mat_jittered = np.zeros((90,784))
train_lab_jittered = np.zeros((90,1))
# Get all of the training vectors for i digit, average and threshold them to produce P(X|Y)
for i in range (0,10):
    train_images, train_labels = mnist.load_mnist('training', digits=[i])
    p=train_images.mean(axis=0)
    y=np.uint8(p)
    y[y<=127]=0
    y[y>127]=1
	
    train_mat[i,:] = y.flatten()
    train_lab[i] = i

sizeY = 28
sizeX = 28

img0 = np.zeros((sizeY,sizeX))
img1 = np.zeros((sizeY,sizeX))
img2 = np.zeros((sizeY,sizeX))
img3 = np.zeros((sizeY,sizeX))
img4 = np.zeros((sizeY,sizeX))
img5 = np.zeros((sizeY,sizeX))
img6 = np.zeros((sizeY,sizeX))
img7 = np.zeros((sizeY,sizeX))
img8 = np.zeros((sizeY,sizeX))

jitteredVec0 = np.zeros(784)
jitteredVec1 = np.zeros(784)
jitteredVec2 = np.zeros(784)
jitteredVec3 = np.zeros(784)
jitteredVec4 = np.zeros(784)
jitteredVec5 = np.zeros(784)
jitteredVec6 = np.zeros(784)
jitteredVec7 = np.zeros(784)
jitteredVec8 = np.zeros(784)

for i in range (0,10):
    imgs = train_mat[i]
    label = train_lab[i]
    img = np.reshape(imgs,(sizeY,sizeX),order='C')
    img0[0:sizeY-1,0:sizeX-1] = img[1:sizeY,1:sizeX]
    img0[sizeY-1,:] = 0
    img0[:,sizeX-1] = 0
    img1[0:sizeY-1,0:sizeX] = img[1:sizeY,0:sizeX]
    img1[sizeY-1,:] = 0
    img2[0:sizeY-1,1:sizeX] = img[1:sizeY,0:sizeX-1]
    img2[sizeY-1,:] = 0
    img2[:,0] = 0
    img3[0:sizeY,0:sizeX-1] = img[0:sizeY,1:sizeX]
    img3[:,sizeX-1] = 0
    img4[0:sizeY,0:sizeX] = img[0:sizeY,0:sizeX]
    img5[0:sizeY,1:sizeX] = img[0:sizeY,0:sizeX-1]
    img5[:,0] = 0
    img6[1:sizeY,0:sizeX-1] = img[0:sizeY-1,1:sizeX]
    img6[0,:] = 0
    img6[:,sizeX-1] = 0
    img7[1:sizeY,0:sizeX] = img[0:sizeY-1,0:sizeX]
    img7[0,:] = 0
    img8[1:sizeY,1:sizeX] = img[0:sizeY-1,0:sizeX-1]
    img8[:,0] = 0
    img8[0,:] = 0

    jitteredVec0 = img0.flatten()
    jitteredVec1 = img1.flatten()
    jitteredVec2 = img2.flatten()
    jitteredVec3 = img3.flatten()
    jitteredVec4 = img4.flatten()
    jitteredVec5 = img5.flatten()
    jitteredVec6 = img6.flatten()
    jitteredVec7 = img7.flatten()
    jitteredVec8 = img8.flatten()

    train_lab_jittered[i*9:(i+1)*9] = label
    train_mat_jittered[i*9] = jitteredVec0       
    train_mat_jittered[i*9+1] = jitteredVec1       
    train_mat_jittered[i*9+2] = jitteredVec2    
    train_mat_jittered[i*9+3] = jitteredVec3       
    train_mat_jittered[i*9+4] = jitteredVec4       
    train_mat_jittered[i*9+5] = jitteredVec5       
    train_mat_jittered[i*9+6] = jitteredVec6       
    train_mat_jittered[i*9+7] = jitteredVec7       
    train_mat_jittered[i*9+8] = jitteredVec8      




# Load a test pattern (X), flatten it, threshold it, and compute hamming distance between
# X and P(X|Y). Then compute a binary error for this test pattern --Y are the digit labels--
err=np.zeros((10000,1))
for i in range(len(test_images)):
    q=test_images[i].flatten()
    q[q<=127]=0
    q[q>127]=1
    error, Winner1, winning_label = cdi.BM_ham(train_mat_jittered,q,test_labels[i],train_lab_jittered)
    err[i]=error

# Print error over the whole test set (lower is better)
print("Error:")
print(sum(err)/len(test_images))

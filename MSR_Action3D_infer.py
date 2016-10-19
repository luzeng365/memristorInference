## MSR_Action3D_infer.py, a function to perform naive Bayesian inference
##                        on the MSR-Action3D dataset with associative memory
##                        using a (pixel-wise) error metric, and save the 
##                        accuracy result in a text file.
## Wesley Chavez 07-05-2016
## Portland State University
##
## TODO:  make it a function, arguments of: size of input images, random seed,
##            error metric (dot product or Hamming distance), percent of
##            dataset to test on, and temporal window size
##        add temporal window

import numpy as np
import CDI_modules as cdi
import scipy.misc as sm
import math
import random
import time
from joblib import Parallel, delayed
import multiprocessing

# For shuffling the dataset
randomSeed = 12345

# Number of pixels in input images
vecSize = 300

# 20 MSR-Action3D actions
classes = ['a01', 'a02', 'a03', 'a04', 'a05', 'a06', 'a07', 'a08', 'a09', 'a10', 
'a11', 'a12', 'a13', 'a14', 'a15', 'a16', 'a17', 'a18', 'a19', 'a20']

# Read list of .pngs
f = open('/stash/tlab/datasets/MSR-Action3D/SkeletonFilenames_15x20.txt', 'r')
pngList = f.read().splitlines()

# Allocate your stuff
testSetSize = int(math.floor(len(pngList)*.25))
dataset = np.zeros((len(pngList),vecSize))
labels = np.zeros((len(pngList)))
dataset_shuf = np.zeros((len(pngList),vecSize))
labels_shuf = np.zeros((len(pngList),1))
err = np.zeros((testSetSize))


# .pngs to np array, whole dataset
print('IMREADING THE DATASET')
for i in np.arange(len(pngList)):
    img = sm.imread(pngList[i])
    dataset[i,:] = img.flatten()
    for j in np.arange(20):
        if (pngList[i].find(classes[j]) != -1):
            labels[i] = j+1;

# Binary data
dataset[dataset > 0] = 1;
dataset[dataset <= 0] = 0;

# Shuffle indices with the random seed
shufindex = np.arange(len(pngList))
random.shuffle(shufindex,random.seed(randomSeed))

# Shuffle the dataset and labels, correspondingly
print('SHUFFLING')
for i in np.arange(len(pngList)):
    dataset_shuf[i] = dataset[shufindex[i]]
    labels_shuf[i] = labels[shufindex[i]]

# Inference with x% of the dataset as testing, and the rest as stored patterns, using the Bayesian Module (BM)
def infer_MSR(i):
    error = cdi.BM_dp(dataset_shuf[testSetSize+1:len(pngList)],dataset_shuf[i],labels_shuf[i],labels_shuf[testSetSize+1:len(pngList)])
    #error, dontcare1, dontcare2 = cdi.BM_ham(dataset_shuf[testSetSize+1:len(pngList)],dataset_shuf[i],labels_shuf[i],labels_shuf[testSetSize+1:len(pngList)])
    return error

# Call infer_MSR using all cores on the machine
print('PERFORMING INFERENCE')
inferindex = np.arange(testSetSize)
num_cores = multiprocessing.cpu_count()
err = Parallel(n_jobs = num_cores)(delayed(infer_MSR)(i) for i in inferindex)

accuracy = 1-sum(err)/float(testSetSize)

# Print accuracy over the whole test set
print("Accuracy_15x20: ")
print(accuracy)

# Save results and parameters to file
fid = open('MSR_results.txt','a')
fid.write('Size: ')
fid.write('15x20\n')
fid.write('Metric: ')
fid.write('dp\n')
fid.write('Test Set Length: ')
fid.write(str(testSetSize))
fid.write('\n')
fid.write('Temporal Window Size: 1\n')
fid.write('Random Seed: ')
fid.write(str(randomSeed))
fid.write('\n')
fid.write('Accuracy: ')
fid.write(str(accuracy))
fid.write('\n\n')
fid.close()

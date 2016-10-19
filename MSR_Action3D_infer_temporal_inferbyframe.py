## MSR_Action3D_infer_temporal.py, a function to perform naive Bayesian spatiotemporal 
##                        inference on the MSR-Action3D dataset with associative memory
##                        using a (pixel-wise) error metric, and save the accuracy
##                        result in a text file.
## Wesley Chavez 07-12-2016
## Portland State University
##
## TODO:  make it a function, arguments of: size of input images, random seed,
##            error metric (dot product or Hamming distance), percent of
##            dataset to test on, and temporal window size

import numpy as np
import CDI_modules as cdi
import scipy.misc as sm
import math
import random
import time
from joblib import Parallel, delayed
import multiprocessing

# Number of pixels in input images
sizeY=60
sizeX=80
temporalWindowSize = 4
metric = 'dp'
# For shuffling the dataset to split into testing and stored pattern sets
randomSeed = 34567
actionSet = 'AS3'
testSize = .66666667

vecSize = sizeY*sizeX

# 20 MSR-Action3D actions
classes = ['a01', 'a02', 'a03', 'a04', 'a05', 'a06', 'a07', 'a08', 'a09', 'a10', 
'a11', 'a12', 'a13', 'a14', 'a15', 'a16', 'a17', 'a18', 'a19', 'a20']

# Read list of .pngs
f = open('/stash/tlab/datasets/MSR-Action3D/SkeletonFilenames_' + str(sizeY) + 'x' + str(sizeX) + '_' + actionSet + '.txt', 'r')
pngList = f.read().splitlines()
f.close()

# Allocate your stuff
testSetSize = int(math.floor((len(pngList)-temporalWindowSize+1)*testSize))
trainSetSize = len(pngList)-temporalWindowSize+1-testSetSize
dataset = np.zeros((len(pngList),vecSize))
labels = np.zeros((len(pngList),1))
dataset_shuf = np.zeros((len(pngList),vecSize))
labels_shuf = np.zeros((len(pngList),1))
dataset_temporal = np.zeros((len(pngList)-temporalWindowSize+1,vecSize*temporalWindowSize))
err = np.zeros((testSetSize))
vidStartIndices = []

# .pngs to np array, whole dataset
print('IMREADING THE DATASET')
for i in np.arange(len(pngList)):
    frameNum = int(pngList[i][65:68])
    if (frameNum == 1):
         vidStartIndices.append(i)
    img = sm.imread(pngList[i])
    dataset[i,:] = img.flatten()
    for j in np.arange(20):
        if (pngList[i].find(classes[j]) != -1):
            labels[i] = j+1;

# Binary data
dataset[dataset > 0] = 1;
dataset[dataset <= 0] = 0;

# Shuffle indices with the random seed
shufindex = np.arange(len(vidStartIndices))
random.shuffle(shufindex,random.seed(randomSeed))

# Shuffle the dataset and labels, correspondingly
print('SHUFFLING')
framecount = 0
for i in np.arange(len(vidStartIndices)):
    if(shufindex[i] == len(vidStartIndices)-1):
        vidLength = len(dataset) - vidStartIndices[shufindex[i]]        
        dataset_shuf[framecount:framecount+vidLength] = dataset[vidStartIndices[shufindex[i]]:len(dataset)]
        labels_shuf[framecount:framecount+vidLength] = labels[vidStartIndices[shufindex[i]]:len(dataset)]
    else:
        vidLength = vidStartIndices[shufindex[i]+1] - vidStartIndices[shufindex[i]]
        dataset_shuf[framecount:framecount+vidLength] = dataset[vidStartIndices[shufindex[i]]:vidStartIndices[shufindex[i]+1]]
        labels_shuf[framecount:framecount+vidLength] = labels[vidStartIndices[shufindex[i]]:vidStartIndices[shufindex[i]+1]]
    framecount = framecount + vidLength

# Convert to temporal format, whole dataset
if(temporalWindowSize == 1):
	dataset_temporal = dataset_shuf
elif(temporalWindowSize == 2):
    print('CONVERTING TO TEMPORAL FORMAT')
    for i in np.arange(len(pngList)-temporalWindowSize+1):
        dataset_temporal[i] = np.concatenate((dataset_shuf[i],dataset_shuf[i+1]),axis=0)
elif(temporalWindowSize == 4):
    print('CONVERTING TO TEMPORAL FORMAT')
    for i in np.arange(len(pngList)-temporalWindowSize+1):
        dataset_temporal[i] = np.concatenate((dataset_shuf[i],dataset_shuf[i+1],dataset_shuf[i+2],dataset_shuf[i+3]),axis=0)
elif(temporalWindowSize == 6):
    print('CONVERTING TO TEMPORAL FORMAT')
    for i in np.arange(len(pngList)-temporalWindowSize+1):
        dataset_temporal[i] = np.concatenate((dataset_shuf[i],dataset_shuf[i+1],dataset_shuf[i+2],dataset_shuf[i+3],dataset_shuf[i+4],dataset_shuf[i+5]),axis=0)
elif(temporalWindowSize == 8):
    print('CONVERTING TO TEMPORAL FORMAT')
    for i in np.arange(len(pngList)-temporalWindowSize+1):
        dataset_temporal[i] = np.concatenate((dataset_shuf[i],dataset_shuf[i+1],dataset_shuf[i+2],dataset_shuf[i+3],dataset_shuf[i+4],dataset_shuf[i+5],dataset_shuf[i+6],dataset_shuf[i+7]),axis=0)


# Inference with x% of the dataset as testing, and the rest as stored patterns, using the Bayesian Module (BM)
def infer_MSR(i):
    if(metric == 'dp'):
        error, dontcare1, dontcare2, dontcare3 = cdi.BM_dp(dataset_temporal[testSetSize:testSetSize+trainSetSize],dataset_temporal[i],[labels_shuf[i]],labels_shuf[testSetSize:testSetSize+trainSetSize])
    elif(metric == 'ham'):
        error, dontcare1, dontcare2, dontcare3 = cdi.BM_ham(dataset_temporal[testSetSize:testSetSize+trainSetSize],dataset_temporal[i],[labels_shuf[i]],labels_shuf[testSetSize:testSetSize+trainSetSize])
    return error

# Call infer_MSR using all cores on the machine
print('PERFORMING INFERENCE')
inferindex = np.arange(testSetSize)
num_cores = 24
#num_cores = multiprocessing.cpu_count()
err = Parallel(n_jobs = num_cores)(delayed(infer_MSR)(i) for i in inferindex)

accuracy = 1-sum(err)/float(testSetSize)

# Print accuracy over the whole test set
print('Accuracy_' + str(sizeY) + 'x' + str(sizeX) + ': ')
print(accuracy)

# Save results and parameters to file
fid = open('MSR_results_nojitter_splits.txt','a')
fid.write('Size: ')
fid.write(str(sizeY) + 'x' + str(sizeX) + '\n')
fid.write('Metric: ')
fid.write(metric + '\n')
fid.write('Test Set Length: ')
fid.write(str(testSetSize))
fid.write('\n')
fid.write(str(testSize))
fid.write('\n')
fid.write('Action Set: ')
fid.write(actionSet)
fid.write('\n')
fid.write('Temporal Window Size: ' + str(temporalWindowSize) + '\n')
fid.write('Random Seed: ')
fid.write(str(randomSeed))
fid.write('\n')
fid.write('Accuracy: ')
fid.write(str(accuracy))
fid.write('\n\n')
fid.close()

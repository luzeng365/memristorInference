## MSR_Action3D_infer_temporal_jitter.py, a function to perform naive Bayesian spatiotemporal 
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
sizeY=15
sizeX=20
temporalWindowSize = 4
metric = 'dp'
# For shuffling the dataset to split into testing and stored pattern sets
randomSeed = 12345


vecSize = sizeY*sizeX

# 20 MSR-Action3D actions
classes = ['a01', 'a02', 'a03', 'a04', 'a05', 'a06', 'a07', 'a08', 'a09', 'a10', 
'a11', 'a12', 'a13', 'a14', 'a15', 'a16', 'a17', 'a18', 'a19', 'a20']

# Read list of .pngs
f = open('/stash/tlab/datasets/MSR-Action3D/SkeletonFilenames_' + str(sizeY) + 'x' + str(sizeX) + '.txt', 'r')
pngList = f.read().splitlines()
f.close()

# Allocate your stuff
testSetSize = int(math.floor((len(pngList)-temporalWindowSize+1)*.25))
testSetSizeJittered = testSetSize*9
trainSetSize = len(pngList)-temporalWindowSize+1-testSetSize
dataset = np.zeros((len(pngList),vecSize))
labels = np.zeros((len(pngList),1))
dataset_shuf = np.zeros((len(pngList),vecSize))
labels_shuf = np.zeros((len(pngList),1))
dataset_temporal = np.zeros((len(pngList)-temporalWindowSize+1,vecSize*temporalWindowSize))
dataset_jittered = np.zeros((testSetSize*9+trainSetSize,vecSize*temporalWindowSize))
labels_jittered = np.zeros((testSetSize*9+trainSetSize,1))
actualerror = np.zeros((testSetSize))
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

# Convert to jittered format, stored patterns only
print('CONVERTING TO JITTERED FORMAT')
dataset_jittered[testSetSize*9:testSetSize*9+trainSetSize] = dataset_temporal[testSetSize:testSetSize+trainSetSize]
labels_jittered[testSetSize*9:testSetSize*9+trainSetSize] = labels_shuf[testSetSize:testSetSize+trainSetSize]

img0 = np.zeros((sizeY,sizeX))
img1 = np.zeros((sizeY,sizeX))
img2 = np.zeros((sizeY,sizeX))
img3 = np.zeros((sizeY,sizeX))
img4 = np.zeros((sizeY,sizeX))
img5 = np.zeros((sizeY,sizeX))
img6 = np.zeros((sizeY,sizeX))
img7 = np.zeros((sizeY,sizeX))
img8 = np.zeros((sizeY,sizeX))

jitteredVec0 = np.zeros(vecSize*temporalWindowSize)
jitteredVec1 = np.zeros(vecSize*temporalWindowSize)
jitteredVec2 = np.zeros(vecSize*temporalWindowSize)
jitteredVec3 = np.zeros(vecSize*temporalWindowSize)
jitteredVec4 = np.zeros(vecSize*temporalWindowSize)
jitteredVec5 = np.zeros(vecSize*temporalWindowSize)
jitteredVec6 = np.zeros(vecSize*temporalWindowSize)
jitteredVec7 = np.zeros(vecSize*temporalWindowSize)
jitteredVec8 = np.zeros(vecSize*temporalWindowSize)

for i in np.arange(testSetSize):
    imgs = dataset_temporal[i]
    label = labels_shuf[i]
    for j in np.arange(temporalWindowSize):
        img = np.reshape(imgs[j*vecSize:(j+1)*vecSize],(sizeY,sizeX),order='C')
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

        jitteredVec0[j*vecSize:(j+1)*vecSize] = img4.flatten()
        jitteredVec1[j*vecSize:(j+1)*vecSize] = img3.flatten()
        jitteredVec2[j*vecSize:(j+1)*vecSize] = img5.flatten()
        jitteredVec3[j*vecSize:(j+1)*vecSize] = img1.flatten()
        jitteredVec4[j*vecSize:(j+1)*vecSize] = img7.flatten()
        jitteredVec5[j*vecSize:(j+1)*vecSize] = img0.flatten()
        jitteredVec6[j*vecSize:(j+1)*vecSize] = img2.flatten()
        jitteredVec7[j*vecSize:(j+1)*vecSize] = img6.flatten()
        jitteredVec8[j*vecSize:(j+1)*vecSize] = img8.flatten()

    labels_jittered[i*9:(i+1)*9] = label
    dataset_jittered[i*9] = jitteredVec0       
    dataset_jittered[i*9+1] = jitteredVec1       
    dataset_jittered[i*9+2] = jitteredVec2    
    dataset_jittered[i*9+3] = jitteredVec3       
    dataset_jittered[i*9+4] = jitteredVec4       
    dataset_jittered[i*9+5] = jitteredVec5       
    dataset_jittered[i*9+6] = jitteredVec6       
    dataset_jittered[i*9+7] = jitteredVec7       
    dataset_jittered[i*9+8] = jitteredVec8      

# Inference with x% of the dataset as testing, and the rest as stored patterns, using the Bayesian Module (BM)
def infer_MSR(i):
    if(metric == 'dp'):
        notcorrect, dontcare1, dontcare2, error = cdi.BM_dp(dataset_jittered[testSetSizeJittered:testSetSizeJittered+trainSetSize],dataset_jittered[i],[labels_jittered[i]],labels_jittered[testSetSizeJittered:testSetSizeJittered+trainSetSize])
    elif(metric == 'ham'):
        notcorrect, dontcare1, dontcare2, error = cdi.BM_ham(dataset_jittered[testSetSizeJittered:testSetSizeJittered+trainSetSize],dataset_jittered[i],[labels_jittered[i]],labels_jittered[testSetSizeJittered:testSetSizeJittered+trainSetSize])
    return notcorrect, error

# Call infer_MSR using all cores on the machine
print('PERFORMING INFERENCE')
inferindex = np.arange(testSetSizeJittered)
num_cores = multiprocessing.cpu_count()
returnvals = Parallel(n_jobs = num_cores)(delayed(infer_MSR)(i) for i in inferindex)
notcorr, err = zip(*returnvals)

for i in np.arange(testSetSize):
    notcorrr = notcorr[i*9:(i+1)*9]
    errr = err[i*9:(i+1)*9]
    if(metric == 'dp'):
        idx = np.argmax(errr) 
        actualerror[i] = notcorrr[idx]
    elif(metric == 'ham'):
        idx = np.argmin(errr) 
        actualerror[i] = notcorrr[idx]

accuracy = 1-sum(actualerror)/float(testSetSize)

# Print accuracy over the whole test set
print('Accuracy_' + str(sizeY) + 'x' + str(sizeX) + ': ')
print(accuracy)

# Save results and parameters to file
fid = open('MSR_results_shufflebyvideo_jittered_testingframes.txt','a')
fid.write('Size: ')
fid.write(str(sizeY) + 'x' + str(sizeX) + '\n')
fid.write('Metric: ')
fid.write(metric + '\n')
fid.write('Test Set Length: ')
fid.write(str(testSetSize))
fid.write('\n')
fid.write('Number of Jitter Positions: 9\n')
fid.write('Temporal Window Size: ' + str(temporalWindowSize) + '\n')
fid.write('Random Seed: ')
fid.write(str(randomSeed))
fid.write('\n')
fid.write('Accuracy: ')
fid.write(str(accuracy))
fid.write('\n\n')
fid.close()

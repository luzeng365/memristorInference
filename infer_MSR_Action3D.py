import numpy as np
import CDI_modules as cdi
import scipy.misc as sm
import math
import random
import time
from joblib import Parallel, delayed
import multiprocessing

randomSeed = 12345
vecSize = 300

f = open('SkeletonFilenames_15x20.txt', 'r')
pngList = f.read().splitlines()
classes = ['a01', 'a02', 'a03', 'a04', 'a05', 'a06', 'a07', 'a08', 'a09', 'a10', 
'a11', 'a12', 'a13', 'a14', 'a15', 'a16', 'a17', 'a18', 'a19', 'a20']

dataset = np.zeros((len(pngList),vecSize))
labels = np.zeros((len(pngList)))
dataset_shuf = np.zeros((len(pngList),vecSize))
labels_shuf = np.zeros((len(pngList),1))
err = np.zeros((int(math.floor(len(pngList)*.25))))



print('IMREADING THE DATASET')
for i in np.arange(len(pngList)):
    img = sm.imread(pngList[i])
    dataset[i,:] = img.flatten()
    for j in np.arange(20):
        if (pngList[i].find(classes[j]) != -1):
            labels[i] = j+1;


dataset[dataset > 0] = 1;
dataset[dataset <= 0] = 0;

shufindex = np.arange(len(pngList))
random.shuffle(shufindex,random.seed(randomSeed))

print('SHUFFLING')
for i in np.arange(len(pngList)):
    dataset_shuf[i] = dataset[shufindex[i]]
    labels_shuf[i] = labels[shufindex[i]]

print('PERFORMING INFERENCE')
def infer_MSR(i):
    print(i)
    #for i in np.arange(int(math.floor(len(pngList)*.25))):
    q=dataset_shuf[i]
    error, dontcare1, dontcare2 = cdi.BM_ham(dataset_shuf[int(math.floor(len(pngList)*.25))+1:len(pngList)],q,labels_shuf[i],labels_shuf[int(math.floor(len(pngList)*.25))+1:len(pngList)])
        #err[i]=error
    if(i==0):
        print("--- %s seconds for first iteration---" % (time.time() - start_time))
    return error

inferindex = np.arange(int(math.floor(len(pngList)*.25)))
num_cores = multiprocessing.cpu_count()
start_time = time.time()
err = Parallel(n_jobs = num_cores)(delayed(infer_MSR)(i) for i in inferindex)
print('err:')
print(err)
print('size of err')
print(err.shape)
# Print error over the whole test set (lower is better)
print("Error:")
print(sum(err)/math.floor(len(pngList)*.25))

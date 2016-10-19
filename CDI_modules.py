# !usr/bin/env python
####################################################################
#Author: Mohammad MA Taha
#email: motaha@pdx.edu
#Portland State University
#Date Created: 06/16/2016
#Date Modified: 08/28/2016
#####################################################################
'''
Log::New sftp
Mohammad Taha (06/30/2016):
-> Added two functions each construct a 3 levels hierarchy with a generic number of children and parents. However, the number of children and parents depend on the number of bits of the pattern.
-> The first hierarchy called ****_N slices the input pattern and distributes the slices over the children and parents. While the other function ,****_F, patches the 2D array into patches and then distributes them over the children and parents.
-> Also, added another function "lower_level_ham_infer" to perform the Hamming distance for the lowest level nodes without needing the labels, to save time
-> Obtained results (error percentage in classification)  as of the date :
****_N:
Averaging of classes new features for 10000 test version#1=40.03%
Averaging of classes new features for 10000 test version#2=43.42%
****_F:
Averaging of classes new features for 10000 test version#1=42.75%
Averaging of classes new features for 10000 test version#2=42.54%
<<<<<<< .mine
##NNB
Averaging over classes new features for 1000 NNB Hierarchy_F version#1=[ 0.4324]
Averaging over classes new features for 1000 NNB Hierarchy_N version#1=[ 0.4119]
Averaging over classes new features for 1000 (10%) noise 1-node =[ 0.2892]

Averaging over classes new features for 1000 NNB Hierarchy_F version#2=[ 0.4374]
Averaging over classes new features for 1000 NNB Hierarchy_N version#2=[ 0.4465]
Averaging over classes new features for 1000 NNB 1-node =[ 0.2892]



||||||| .r4350
=======

Wesley Chavez (07/02/2016):
- Changed the hamming distance calculation, is faster

Wesley Chavez (07/05/2016):
- Changed the hamming distance calculation, is even faster
>>>>>>> .r4383
Mohammad Taha (08/01/2016):
-> Added functions for the dot product calculation in the Hierarchy.

'''



################################################################################

import numpy as np
from termcolor import colored, cprint
import sys

#############
# NB Usage example:
	# ham_sw_classify_NB=BM_ham(trn,tst,tst_label,trn_labels)
# NNB Usage example:
	# ham_NNB_classify_ver1,length_new_trn=NNB_ver1(trn,trn_labels,tst[tst_index],tst_labels[tst_index])
	# ham_NNB_classify_ver2=NNB_ver2(trn1,trn_labels1,trn2,trn_labels2,tst_sw_noise,tst_labels[tst_index],iteration,tst_index) # I am using tst_index because I parallelize the simulation.
#############


def blockshaped(arr, nrows, ncols):
    """
    Return an array of shape (n, nrows, ncols) where
    n * nrows * ncols = arr.size

    If arr is a 2D array, the returned array should look like n subblocks with
    each subblock preserving the "physical" layout of arr.
    """
    h, w = arr.shape
    return (arr.reshape(h//nrows, nrows, -1, ncols)
               .swapaxes(1,2)
               .reshape(-1, nrows, ncols))

def hamming_distance(s1, s2):
	#Return the Hamming distance between equal-length sequences
	if len(s1) != len(s2):
		raise ValueError("Undefined for sequences of unequal length")
	x = np.count_nonzero(s1!=s2)
        #x = sum(s1 != s2)
	#x = np.uint8(sum(ch1 != ch2 for ch1, ch2 in zip(s1, s2)))
	#y = np.uint8(x)
	return x
##############
#BMs for MNIST classification
##############
def ham_infer (test_vector,trn, labels): # Passing the test vector, train vectors, and train labels (inference phase only), returns the closest vector (Hamming distance) and its label
	Distance=np.empty(len(trn),dtype=np.uint16)
	for i in range(len(trn)):
		x=hamming_distance(test_vector, trn[i])
		Distance[i]=x
	error = min(Distance)
	closest_vector_index=np.argmin(Distance)
	closest_vector = trn[closest_vector_index]
	winning_label=labels[closest_vector_index]
	Winner = closest_vector
	return Winner, winning_label

def lower_level_ham_infer (test_vector,trn): # Passing the test vector, train vectors, and train labels (inference phase only), returns the closest vector (Hamming distance)  so it can act as a lower level node that doesn't benefit from the labels
	Distance=np.empty(len(trn),dtype=np.uint16)
	for i in range(len(trn)):
		x=hamming_distance(test_vector, trn[i])
		Distance[i]=x
	error = min(Distance)
	closest_vector_index=np.argmin(Distance)
	closest_vector = trn[closest_vector_index]
	# winning_label=labels[closest_vector_index]
	Winner = closest_vector
	return Winner

def lower_level_BM_ham(trn_vectors,tst1):
	Winner = []
	error = []
	Winner1= lower_level_ham_infer(tst1,trn_vectors)
	# Winner.append((Winner1))
	return Winner1

def dp_infer (test_vector,trn,labels):# Passing the test vector, train vectors, and train labels (inference phase only), returns the closest vector (Dot product) and its label
	Distance=np.empty(len(trn),dtype=np.uint16)
	for i in range(len(trn)):
		x=np.dot(test_vector, trn[i])
		Distance[i]=x
	error = min(Distance)
	closest_vector_index=Distance.argmax(axis=0)
	closest_vector = trn[closest_vector_index]
	winning_label=labels[closest_vector_index]
	Winner = closest_vector
	return Winner, winning_label

def lower_level_dp_infer (test_vector,trn): # Passing the test vector, train vectors, and train labels (inference phase only), returns the closest vector (Hamming distance)  so it can act as a lower level node that doesn't benefit from the labels
	Distance=np.empty(len(trn),dtype=np.uint16)
	for i in range(len(trn)):
		x=np.dot(test_vector, trn[i])
		Distance[i]=x
	error = min(Distance)
	closest_vector_index=np.argmax(Distance)
	closest_vector = trn[closest_vector_index]
	# winning_label=labels[closest_vector_index]
	Winner = closest_vector
	return Winner

def lower_level_BM_dp(trn_vectors,tst1):
	Winner = []
	error = []
	Winner1= lower_level_dp_infer(tst1,trn_vectors)
	# Winner.append((Winner1))
	return Winner1

def BM_dp(trn_vectors,tst1,tst_labels,labels): # NB BM module, call this function a number "X" of times to generate a hierarchy with size "X". The arguments should be processed according to "X"
	Winner = []
	error = []
	Winner1, winning_label = dp_infer(tst1, trn_vectors,labels)
	Winner.append((Winner1))
	#error.append(hamming_distance(trn_vectors[j], Winner[j]))
	error=hamming_distance(tst_labels, winning_label)
	return error,Winner1,winning_label

def NNB_dp(trn_vectors,tst,tst_label,trn_labels,prior,NNB_version):
	final_prior=np.empty((len(trn_vectors)))
	# print colored (min(trn_labels),'green')
	for i in enumerate(trn_labels):
		final_prior[i[0]]=prior[np.int8(i[1][0]-1)]
	DP=np.empty(len(trn_vectors),dtype=np.uint16)
	for i in range(len(trn_vectors)):
		x=np.dot(tst,trn_vectors[i])
		DP[i]=x
	if len(DP) != len(final_prior):
		raise ValueError("motaha:length of DP %d is not equal to length of final_prior %d" % (len(DP), len(final_prior)))
	max_distance=max(DP)
	new_DP_list=[]
	new_DP_list_index=[]
	for i in range(len(DP)):
		if DP[i] == max_distance:
			new_DP_list.append(DP[i])
			new_DP_list_index.append(i)
	new_prior=np.take(final_prior,new_DP_list_index)
	new_trn_lbl=np.take(trn_labels,new_DP_list_index)
	if NNB_version == '1-step':
		MAP_table=np.array([1.*a*b for a,b in zip(DP,final_prior)])
		winner_index=np.argmax(MAP_table)
	elif NNB_version == '2-step':
		MAP_table=np.array([1.*a*b for a,b in zip(new_DP_list,new_prior)])
		winner_index=new_DP_list_index[np.argmax(MAP_table)]
	else:
	    raise NotImplementedError(NNB_version)
	Winner1=trn_vectors[winner_index]
	winner_label=trn_labels[winner_index]
	classification=hamming_distance(tst_label,winner_label)
	return classification,Winner1, winner_label

def BM_ham(trn_vectors,tst1,tst_labels,labels): # Same as BM_dp but this uses the Hamming distance
	Winner = []
	error = []
	Winner1, winning_label = ham_infer(tst1,trn_vectors,labels)
	# Winner.append((Winner1))
	#error.append(hamming_distance(trn_vectors[j], Winner[j]))
	error=hamming_distance(tst_labels, winning_label)
	return error,Winner1,winning_label

def NNB_ham(trn_vectors,tst,tst_label,trn_labels,prior,NNB_version):
	final_prior=np.empty((len(trn_vectors)))
	for i in enumerate(trn_labels):
		final_prior[i[0]]=prior[np.int8(i[1][0]-1)]
	HD=np.empty(len(trn_vectors),dtype=np.uint16)
	for i in range(len(trn_vectors)):
		x=hamming_distance(tst,trn_vectors[i])
		HD[i]=x
	if len(HD) != len(final_prior):
		raise ValueError("motaha:length of HD %d is not equal to length of final_prior %d" % (len(HD), len(final_prior)))
	min_distance=min(HD)
	new_HD_list=[]
	new_HD_list_index=[]
	for i in range(len(HD)):
		if HD[i] == min_distance:
			new_HD_list.append(HD[i])
			new_HD_list_index.append(i)
	new_prior=np.take(final_prior,new_HD_list_index)
	new_trn_lbl=np.take(trn_labels,new_HD_list_index)
	if NNB_version == '1-step':
		MAP_table=np.array([1.*a/b for a,b in zip(HD,final_prior)])
		winner_index=np.argmin(MAP_table)
	elif NNB_version == '2-step':
		MAP_table=np.array([1.*a/b for a,b in zip(new_HD_list,new_prior)])
		winner_index=new_HD_list_index[np.argmin(MAP_table)]
	else:
	    raise NotImplementedError(NNB_version)
	Winner1=trn_vectors[winner_index]
	winner_label=trn_labels[winner_index]
	classification=hamming_distance(tst_label,winner_label)
	return classification,Winner1, winner_label

def remove_duplicate(item): # Set() can used on a numpy array to remove the duplicates much more efficient than this function, but returns a type Set
	new_item=[]
	new_item_index=[0]
	new_item=[item[0]]
	for i in range(1,len(item)):
		if item[i] not in new_item:
			new_item.append(item[i])
			new_item_index.append(i)
	return new_item,new_item_index

def NNB_ver1(trn,trn_labels,tst,tst_label): # MNIST NNB classification with Hamming distance measure using normal prior probability counting (accordning to the labels not the image pattern) with NNB training version#1
	lbl=np.empty(len(trn_labels),dtype=np.uint16)
	for i in range(len(trn_labels)):
		lbl[i]=np.argmax(trn_labels[i])

	max_lbl=np.max(lbl)
	lbls=[ i for i in range(max_lbl+1)]
	prior1=np.empty(len(lbl),dtype=np.uint16)
	for i in lbls:
		count=np.count_nonzero(lbl==i)
		for j in range(len(lbl)):
			if lbl[j] == i:
				prior1[j] = count

	NNB_lbl,NNB_lbl_index=remove_duplicate(lbl) #removing duplicate labels while keeping the main's labels
	new_trn=np.empty([len(NNB_lbl),256],dtype=np.uint16)
	NNB_prior=np.empty(len(NNB_lbl),dtype=np.uint16)
	for i in range(len(NNB_lbl_index)):
		new_trn[i]=trn[NNB_lbl_index[i]]
		NNB_prior[i]=prior1[NNB_lbl_index[i]]

	HD=np.empty(len(new_trn),dtype=np.uint16)
	for i in range(len(new_trn)):
		x=BM.hamming_distance(tst,new_trn[i])
		HD[i]=x
	MAP_table=np.array([1.*a/b for a,b in zip(HD,NNB_prior)])
	winner_index=np.argmin(MAP_table)
	winner=trn[winner_index]
	winner_label=trn_labels[winner_index]
	classification=BM.hamming_distance(tst_label,winner_label)
	length_new_trn=len(new_trn)
	return classification#,Winner,winning_label

def NNB_ver2(trn1,trn_labels1,trn2,trn_labels2,tst,tst_label,iteration,tst_index): # MNIST NNB classification with Hamming distance measure using normal prior probability counting (accordning to the labels not the image pattern) with NNB training version#2
	prior1=NNB_trn(trn1,trn_labels1,trn2,trn_labels2)

	HD=np.empty(len(trn1),dtype=np.uint16)
	for i in range(len(trn1)):
		x=BM.hamming_distance(tst,trn1[i])
		HD[i]=x
	HD_final=256-HD

	if np.any(prior1==0):
		raise ValueError('motaha: I counted zero number of occurrences, which is wrong. Recheck the code with the trn patterns')
	else:
		MAP_table=np.array([1.*a/b for a,b in zip(HD,prior1)])
		winner_index=np.argmin(MAP_table)
	winner=trn1[winner_index]
	winner_label=trn_labels1[winner_index]
	classification=BM.hamming_distance(tst_label,winner_label)
	return classification#,Winner,winning_label

###Hierarchy Function for normally flattening the image and splitting the array
def hierarcy_3_levels_ham_N(children,parent,train,trn_labels,tst,tst_label,version,prior,NNB_version,priors): #train is a 2D array and test is a 1D array i.e. one image only
	ntrain=len(train)
	nbits_level1_train=(len(train[0])/children)
	nbits_level2_train=(len(train[0])/parent)
	level1_mat_tmp1=blockshaped(train,ntrain,nbits_level1_train)
	#print level1_mat_tmp1.shape
	if version == "version#1":
		level1_mat=level1_mat_tmp1
		#print level1_mat.shape
	elif version == "version#2":
		level1_mat_tmp2=np.array([np.vstack(level1_mat_tmp1)])
		level1_mat=np.empty((children,children*len(level1_mat_tmp1[0]),nbits_level1_train))
		for i in range(len(level1_mat)):
			level1_mat[i]=level1_mat_tmp2[0]
	else:
		raise NotImplementedError(version)

	level2_mat_train=blockshaped(train,ntrain,nbits_level2_train)
	test=np.split(tst,children)
	#winner would be saved in a 2D array as well and then concatenate by the number of parents available
	level1_winner=np.empty((children,nbits_level1_train),dtype=np.uint16)
	for i in range(len(level1_mat)):
		level1_winner[i]=lower_level_BM_ham(level1_mat[i],test[i])
	#merge level1 and split them according tothe number of the parents
	level1_winner_tmp=level1_winner.ravel()
	# Winner1_tmp=level1_winner_tmp.reshape(28,28)
	level2_input=np.split(level1_winner_tmp,parent)
	nbits_level1_2=len(level2_input[0])
	level2_output_len=len(level2_input)
	level2_winner=np.empty((level2_output_len,nbits_level1_2),dtype=np.uint16)
	for i in range(len(level2_input)):
		level2_winner[i]=lower_level_BM_ham(level2_mat_train[i],level2_input[i])
	#Final node classification
	level2_winner_tmp=level2_winner.ravel()
	if prior == 'NB':
		classification_error,Winner1,Winning_label1=BM_ham(train,level2_winner_tmp,tst_label,trn_labels)
	elif prior == 'NNB':
		classification_error,Winner1,Winning_label1=NNB_ham(train,level2_winner_tmp,tst_label,trn_labels,priors,NNB_version)
	else:
		raise NotImplementedError(prior)
	# Winner=Winner1.reshape(28,28)
	# Winner2_tmp=level2_winner_tmp.reshape(28,28)
	return classification_error
#####################################################################################
###Hierarchy function Hamming distance as well but with new kind of splitting the image, the one I intended doing from the beginning (splitting by windows)
####
def hierarcy_3_levels_ham_F(children,parent,train,trn_labels,tst,tst_label,version,prior,windowsize_r,windowsize_c,rows,columns,NNB_version,priors): #train is a 2D array and test is a 1D array i.e. one image only (takes a 2D array for the training set(each subarray in the main array is a flattened train image) and a flattened test image.)

	ntrain=len(train.reshape(len(train),rows*columns))
	ftrain=train.reshape(len(train),rows*columns)
	nbits_level1_train=(len(train[0])/children)
	nbits_level2_train=(len(train[0])/parent)
	length_train_tst=len(train[0])
	# print colored(length_train_tst,'red')
	'''
	Training the first two levels of the hierarchy
	'''
	row=[0,0,windowsize_r,windowsize_r]
	column=[0,windowsize_c,windowsize_c,0]
	j=0
	window=np.empty((4*len(train),windowsize_r,windowsize_c))
	images=np.empty((4*len(train),length_train_tst/children))
	train_2D=train.reshape((len(train),rows,columns))
	for i in range(len(train)):
		for r,c in zip(row,column):
			window[j] = train_2D[i][r:r+windowsize_r,c:c+windowsize_c]
			j+=1
	images=np.uint16(np.split(window,len(train)))
	images1=images.reshape(len(train),children,length_train_tst/children)
	fimage=np.swapaxes(images1,1,0)
	#####################
	if version == "version#1":
		level1_mat=fimage

	elif version == "version#2":
		level1_mat_tmp2=np.array([np.vstack(fimage)])
		level1_mat=np.empty((children,children*len(fimage[0]),nbits_level1_train))
		for i in range(len(level1_mat)):
			level1_mat[i]=level1_mat_tmp2[0]
	else:
		raise NotImplementedError(version)

	row=[0,windowsize_r]
	column=[0,0]
	j=0
	window_lvl2=np.empty((2*len(train),windowsize_r,columns))
	images_lvl2=np.empty((2*len(train),length_train_tst/parent))
	for i in range(len(train)):
		for r,c in zip(row,column):
			window_lvl2[j] = train_2D[i][r:r+windowsize_r,c:c+columns]
			j+=1
	images_lvl2=np.uint16(np.split(window_lvl2,len(train)))
	images1_lvl2=images_lvl2.reshape(len(train),parent,length_train_tst/parent)
	fimage_lvl2=np.swapaxes(images1_lvl2,1,0)
	level2_mat_train=fimage_lvl2
	'''
	End Training of the first two levels of the hierarchy
	'''
	####################################
	'''
	Start testing
	'''

	#winner would be saved in a 2D array as well and then concatenate by the number of parents available
	level1_winner=np.empty((children,nbits_level1_train),dtype=np.uint16)
	row=[0,0,windowsize_r,windowsize_r]
	column=[0,windowsize_c,windowsize_c,0]
	j=0
	ttest1=np.empty((children,windowsize_r,windowsize_c))
	test_2D=tst.reshape(rows,columns)
	for r,c in zip(row,column):
		ttest1[j] = test_2D[r:r+windowsize_r,c:c+windowsize_c]
		j+=1
	test1=ttest1.reshape(children,length_train_tst/children)
	for i in range(len(level1_mat)):
		level1_winner[i] = lower_level_BM_ham(level1_mat[i],test1[i])
	level1_winner_tmp=level1_winner.ravel()
	Winner1_tmp=level1_winner_tmp.reshape(rows,columns)
	ttst=blockshaped(Winner1_tmp,windowsize_r,columns)
	level2_input=ttst.reshape(parent,length_train_tst/parent)


	level2_input_tmp = np.array([np.concatenate((level1_winner[0].reshape(windowsize_r,windowsize_c), level1_winner[1].reshape(windowsize_r,windowsize_c)),axis=1), np.concatenate((level1_winner[3].reshape(windowsize_r,windowsize_c), level1_winner[2].reshape(windowsize_r,windowsize_c)),axis=1)])

	level2_input=level2_input_tmp.reshape(parent,length_train_tst/parent)
	Winner1_tmp=level2_input_tmp.reshape(parent,length_train_tst/parent)
	###########################
	nbits_level1_2=len(level2_input[0])
	level2_output_len=len(level2_input)
	level2_winner=np.empty((level2_output_len,nbits_level1_2),dtype=np.uint16)
	for i in range(len(level2_input)):
		level2_winner[i]=lower_level_BM_ham(level2_mat_train[i],level2_input[i])
	level2_winner_tmp=level2_winner.ravel()
	if prior == 'NB':
		classification_error,Winner1,Winning_label1=BM_ham(ftrain,level2_winner_tmp,tst_label,trn_labels)
	elif prior == 'NNB':
		classification_error,Winner1,Winning_label1=NNB_ham(ftrain,level2_winner_tmp,tst_label,trn_labels,priors,NNB_version)
	else:
		raise NotImplementedError(prior)
	Winner=Winner1.reshape(rows,columns)
	Winner2_tmp=level2_winner_tmp.reshape(rows,columns)
	# print level1_winner[2]
	return classification_error

#####################################################################################
###Hierarchy Function for normally flattening the image and splitting the array using the dot product
def hierarcy_3_levels_dp_N(children,parent,train,trn_labels,tst,tst_label,version,prior,NNB_version,priors): #train is a 2D array and test is a 1D array i.e. one image only
	ntrain=len(train)
	nbits_level1_train=(len(train[0])/children)
	nbits_level2_train=(len(train[0])/parent)
	level1_mat_tmp1=blockshaped(train,ntrain,nbits_level1_train)
	#print level1_mat_tmp1.shape
	if version == "version#1":
		level1_mat=level1_mat_tmp1
		#print level1_mat.shape
	elif version == "version#2":
		level1_mat_tmp2=np.array([np.vstack(level1_mat_tmp1)])
		level1_mat=np.empty((children,children*len(level1_mat_tmp1[0]),nbits_level1_train))
		for i in range(len(level1_mat)):
			level1_mat[i]=level1_mat_tmp2[0]
	else:
		raise NotImplementedError(version)

	level2_mat_train=blockshaped(train,ntrain,nbits_level2_train)
	test=np.split(tst,children)
	#winner would be saved in a 2D array as well and then concatenate by the number of parents available
	level1_winner=np.empty((children,nbits_level1_train),dtype=np.uint16)
	for i in range(len(level1_mat)):
		level1_winner[i]=lower_level_BM_dp(level1_mat[i],test[i])
	#merge level1 and split them according tothe number of the parents
	level1_winner_tmp=level1_winner.ravel()
	#Winner1_tmp=level1_winner_tmp.reshape(28,28)
	level2_input=np.split(level1_winner_tmp,parent)
	nbits_level1_2=len(level2_input[0])
	level2_output_len=len(level2_input)
	level2_winner=np.empty((level2_output_len,nbits_level1_2),dtype=np.uint16)
	for i in range(len(level2_input)):
		level2_winner[i]=lower_level_BM_dp(level2_mat_train[i],level2_input[i])
	#Final node classification
	level2_winner_tmp=level2_winner.ravel()
	if prior == 'NB':
		classification_error,Winner1,Winning_label1=BM_dp(train,level2_winner_tmp,tst_label,trn_labels)
	elif prior == 'NNB':
		classification_error,Winner1,Winning_label1=NNB_dp(train,level2_winner_tmp,tst_label,trn_labels,priors,NNB_version)
	else:
		raise NotImplementedError(prior)
	# Winner=Winner1.reshape(28,28)
	# Winner2_tmp=level2_winner_tmp.reshape(28,28)
	return classification_error#,level1_winner,Winner1_tmp,level2_winner,Winner2_tmp,Winner,Winning_label1
#####################################################################################
#####################################################################################
###Hierarchy function as well but with new kind of splitting the image, the one I intended doing from the beginning (splitting by windows)
####
def hierarcy_3_levels_dp_F(children,parent,train,trn_labels,tst,tst_label,version,prior,windowsize_r,windowsize_c,rows,columns,NNB_version,priors): #train is a 2D array and test is a 1D array i.e. one image only
	ntrain=len(train.reshape(len(train),rows*columns))
	ftrain=train.reshape(len(train),rows*columns)
	nbits_level1_train=(len(train[0])/children)
	nbits_level2_train=(len(train[0])/parent)
	length_train_tst=len(train[0])
	# print colored(length_train_tst,'red')
	'''
	Training the first two levels of the hierarchy
	'''
	row=[0,0,windowsize_r,windowsize_r]
	column=[0,windowsize_c,windowsize_c,0]
	j=0
	window=np.empty((4*len(train),windowsize_r,windowsize_c))
	images=np.empty((4*len(train),length_train_tst/children))
	train_2D=train.reshape((len(train),rows,columns))
	for i in range(len(train)):
		for r,c in zip(row,column):
			window[j] = train_2D[i][r:r+windowsize_r,c:c+windowsize_c]
			j+=1
	images=np.uint16(np.split(window,len(train)))
	images1=images.reshape(len(train),children,length_train_tst/children)
	fimage=np.swapaxes(images1,1,0)
	#####################
	if version == "version#1":
		level1_mat=fimage

	elif version == "version#2":
		level1_mat_tmp2=np.array([np.vstack(fimage)])
		level1_mat=np.empty((children,children*len(fimage[0]),nbits_level1_train))
		for i in range(len(level1_mat)):
			level1_mat[i]=level1_mat_tmp2[0]
	else:
		raise NotImplementedError(version)

	row=[0,windowsize_r]
	column=[0,0]
	j=0
	window_lvl2=np.empty((2*len(train),windowsize_r,columns))
	images_lvl2=np.empty((2*len(train),length_train_tst/parent))
	for i in range(len(train)):
		for r,c in zip(row,column):
			window_lvl2[j] = train_2D[i][r:r+windowsize_r,c:c+columns]
			j+=1
	images_lvl2=np.uint16(np.split(window_lvl2,len(train)))
	images1_lvl2=images_lvl2.reshape(len(train),parent,length_train_tst/parent)
	fimage_lvl2=np.swapaxes(images1_lvl2,1,0)
	level2_mat_train=fimage_lvl2
	'''
	End Training of the first two levels of the hierarchy
	'''
	####################################
	'''
	Start testing
	'''

	#winner would be saved in a 2D array as well and then concatenate by the number of parents available
	level1_winner=np.empty((children,nbits_level1_train),dtype=np.uint16)
	row=[0,0,windowsize_r,windowsize_r]
	column=[0,windowsize_c,windowsize_c,0]
	j=0
	ttest1=np.empty((children,windowsize_r,windowsize_c))
	test_2D=tst.reshape(rows,columns)
	for r,c in zip(row,column):
		ttest1[j] = test_2D[r:r+windowsize_r,c:c+windowsize_c]
		j+=1
	test1=ttest1.reshape(children,length_train_tst/children)
	for i in range(len(level1_mat)):
		level1_winner[i] = lower_level_BM_dp(level1_mat[i],test1[i])
	level1_winner_tmp=level1_winner.ravel()
	Winner1_tmp=level1_winner_tmp.reshape(rows,columns)
	ttst=blockshaped(Winner1_tmp,windowsize_r,columns)
	level2_input=ttst.reshape(parent,length_train_tst/parent)


	level2_input_tmp = np.array([np.concatenate((level1_winner[0].reshape(windowsize_r,windowsize_c), level1_winner[1].reshape(windowsize_r,windowsize_c)),axis=1), np.concatenate((level1_winner[3].reshape(windowsize_r,windowsize_c), level1_winner[2].reshape(windowsize_r,windowsize_c)),axis=1)])

	level2_input=level2_input_tmp.reshape(parent,length_train_tst/parent)
	Winner1_tmp=level2_input_tmp.reshape(parent,length_train_tst/parent)
	###########################
	nbits_level1_2=len(level2_input[0])
	level2_output_len=len(level2_input)
	level2_winner=np.empty((level2_output_len,nbits_level1_2),dtype=np.uint16)
	for i in range(len(level2_input)):
		level2_winner[i]=lower_level_BM_dp(level2_mat_train[i],level2_input[i])
	level2_winner_tmp=level2_winner.ravel()
	if prior == 'NB':
		classification_error,Winner1,Winning_label1=BM_dp(ftrain,level2_winner_tmp,tst_label,trn_labels)
	elif prior == 'NNB':
		classification_error,Winner1,Winning_label1=NNB_dp(ftrain,level2_winner_tmp,tst_label,trn_labels,priors,NNB_version)
	else:
		raise NotImplementedError(prior)
	Winner=Winner1.reshape(rows,columns)
	Winner2_tmp=level2_winner_tmp.reshape(rows,columns)
	# print level1_winner[2]
	return classification_error

#####################################################################################
def prior_cal(trn_labels):
	# lbl=np.empty(len(trn_labels),dtype=np.uint16)
	# for i in range(len(trn_labels)):
	# 	lbl[i]=np.argmax(trn_labels[i])
	'''
	count=np.empty((10))
	for i in range (10):
		train_images, train_labels = avg.load_mnist('training', digits=[i])
		count[i]=len(train_images)
	# prior=count
	prior=1.*count/60000
	print sum(prior)
	return prior
	prior=np.array([ 0.09871667,  0.11236667,  0.0993    ,  0.10218333,  0.09736667,
        0.09035   ,  0.09863333,  0.10441667,  0.09751667,  0.09915   ])# calculated previously from the previous function no need to compute it with every iteration.
	'''
	# print colored(count,'cyan')
	# lbl=np.asarray([i for x in trn_labels for i in x])# For changing a list of lists (2D) to a list(1D).
	# max_lbl=np.max(lbl)
	# lbls=[ i for i in range(max_lbl+1)]
	# prior1=np.empty(len(lbl),dtype=np.uint16)#TAKE CARE ALWAYS DOING THIS MISTAKE OF USING A LOW UNSIGNED INTEGER AND CANNOT SAVE MORE THAN THE NUMBER GIVEN.
	# print colored(lbls,'red')
	# for i in lbls:
	# 	count=np.count_nonzero(lbl==i)
	# 	print colored(count,'yellow')
	# 	for j in range(len(lbl)):
	# 		if lbl[j] == i:
	# 			prior1[j] = count
	# print colored(prior1,'green')
	# prior=1.*prior1/60000 #Normalizing
	# # prior=prior1 #No normalizing
	# return prior

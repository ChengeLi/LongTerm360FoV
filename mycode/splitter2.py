import os
import sys
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from datetime import datetime as dt
import _pickle as pickle
from sklearn.cluster import KMeans
import random
#data = pd.read_csv(path)

#expNumber = sys.argv[1]

file_Name = "/scratch/wz1219/FoV/data/tsinghua_merged_dataset.p"
fileObject = open(file_Name,'r')
data = pickle.load(fileObject)
print(type(data))
fileObject.close()

#combined_recon = np.array(np.rec.fromarrays((completed1SI, completed2SI, completed3SI)))

def hard_easy(matrix1, matrix2, matrix3):
	rows, columns = matrix1.shape
	#compute velocity
	acceleration1 = np.diff(np.diff(matrix1))
	acceleration2 = np.diff(np.diff(matrix2))
	acceleration3 = np.diff(np.diff(matrix3))
	#
	combined_acc = np.array(np.rec.fromarrays([acceleration1, acceleration2, acceleration3]))
	newmat = np.zeros((combined_acc.shape[0], combined_acc.shape[1]))
	for row in range(combined_acc.shape[0]):
		for record in range(combined_acc.shape[1]):
			newmat[row, record] = np.linalg.norm(np.array(combined_acc[row,record].tolist()))
	# newmat = newmat/time_mat
	

        
	train_users = np.random.choice(48,33,replace=False)#train1 + train2
        #!/bin/bash
	test_users = [x for x in range(48) if x not in train_users]
	print("Train_users are: ",train_users)
	print("Test_users are: ", test_users)
	return train_users, test_users

directory = "/scratch/wz1219/FoV/data/"

all_video_train = {}
all_video_test = {}
#split
for video in data.keys():
	print("Video number " + str(video))
	xmat = data[video]['x']
	ymat = data[video]['y']
	zmat = data[video]['z']
	# time = data[video]['playback']
	train, test = hard_easy(xmat, ymat, zmat)
	np.save(directory + "train_video" + str(video) + ".npy", train)
	np.save(directory + "test_video" + str(video) + ".npy", test)
        video_train = {}
        video_test = {}
        video_train['x'] = np.zeros((len(train),xmat.shape[1]))
        video_train['y'] = np.zeros((len(train),ymat.shape[1]))
        video_train['z'] = np.zeros((len(train),zmat.shape[1]))
        video_test['x'] = np.zeros((len(test),xmat.shape[1]))
        video_test['y'] = np.zeros((len(test),ymat.shape[1]))
        video_test['z'] = np.zeros((len(test),zmat.shape[1]))
        for i in range(len(train)):
            video_train['x'][i,:] = xmat[train[i],:]
            video_train['y'][i,:] = ymat[train[i],:]
            video_train['z'][i,:] = zmat[train[i],:]
        for j in range(len(test)):
            video_test['x'][j,:] = xmat[test[j],:]
            video_test['y'][j,:] = ymat[test[j],:]
            video_test['z'][j,:] = zmat[test[j],:]
        all_video_train[video] = video_train
        all_video_test[video] = video_test
pickle.dump(all_video_train,open(directory + 'tsinghua_train_video_data.p','wb'))
pickle.dump(all_video_test,open(directory + 'tsinghua_test_video_data.p','wb'))




# np.linalg.norm(x, axis=1)

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

expNumber = sys.argv[1]

file_Name = "/scratch/ag6925/intern_apurv/NYU/FoV-master/data/modified_exp_" + expNumber + "_xyz.p"
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
	kmeans = KMeans(random_state=0).fit(newmat)
	max_element = max(set(kmeans.labels_), key=kmeans.labels_.tolist().count)
	#now take 70% 1's and 70% 0's for train
	users1 = np.where(kmeans.labels_ == max_element)[0]
	users2 = np.where(kmeans.labels_ != max_element)[0]
	#
	train1 = random.sample(users1, int(0.7*len(users1)))
	test1 = [x for x in users1 if x not in train1]
	#
	train2 = random.sample(users2, int(0.7*len(users2)))
	test2 = [x for x in users2 if x not in train2]
	#
	train_users = train1 + train2
	test_users = test1 + test2
	print("Train_users are: ",train_users)
	print("Test_users are: ", test_users)
	return train_users, test_users

directory = "/scratch/ag6925/intern_apurv/NYU/FoV-master/data/"

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
# np.linalg.norm(x, axis=1)
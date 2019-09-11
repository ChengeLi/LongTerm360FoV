import pandas as pd
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import numpy as np
from copy import deepcopy
import _pickle as pickle
import sys

expNumber = sys.argv[1]

file_Name = "/scratch/ag6925/intern_apurv/NYU/FoV-master/data/modified_exp_" + expNumber + "_xyz.p"
fileObject = open(file_Name,'r')
data = pickle.load(fileObject)
print(type(data))
fileObject.close()

directory = "/scratch/ag6925/intern_apurv/NYU/FoV-master/data/"

train_dict = deepcopy(data)
test_dict = deepcopy(data)

for video in range(9):
	print("Video number " + str(video))
	train = np.load(directory + "train_video" + str(video) + ".npy")
	test = np.load(directory + "test_video" + str(video) + ".npy")	
	test_dict[video]['x'] = np.delete(test_dict[video]['x'], train, axis = 0)
	test_dict[video]['y'] = np.delete(test_dict[video]['y'], train, axis = 0)
	test_dict[video]['z'] = np.delete(test_dict[video]['z'], train, axis = 0)
	train_dict[video]['x'] = np.delete(train_dict[video]['x'], test, axis = 0)
	train_dict[video]['y'] = np.delete(train_dict[video]['y'], test, axis = 0)
	train_dict[video]['z'] = np.delete(train_dict[video]['z'], test, axis = 0)


pickle.dump(train_dict,open(directory + 'train_exp_'+expNumber+'_xyz.p','wb'))
pickle.dump(test_dict,open(directory + 'test_exp_'+expNumber+'_xyz.p','wb'))


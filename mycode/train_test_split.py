#train-test split of original pickle file

import os
import sys
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from datetime import datetime as dt
import _pickle as pickle
from copy import deepcopy
import random

exp_number = sys.argv[1]

file_Name = "/scratch/ag6925/intern_apurv/NYU/FoV-master/data/DATA_exp_" + exp_number + "_xyz.p"
fileObject = open(file_Name,'r')
data1 = pickle.load(fileObject)
data2 = {}
print(type(data1))
fileObject.close()

user_list =  random.sample(range(48), 5)
video_list = [2,5]

def delete_users(diction):
	for vid in diction.keys():
		diction[vid]['y'] = deepcopy(np.delete(diction[vid]['y'], user_list, axis = 0))
		diction[vid]['x'] = deepcopy(np.delete(diction[vid]['x'], user_list, axis = 0))
		diction[vid]['z'] = deepcopy(np.delete(diction[vid]['z'], user_list, axis = 0))
		print("Video " + str(vid) + " user deletion completed.")


#remove videos
for video in data1.keys():
	if video in video_list:
		data2[video] = deepcopy(data1[video])
		del data1[video]
		print("Video " + str(video) + " removed from training, added to test.")

#remove users
delete_users(data1)
delete_users(data2)

pickle.dump(data1, open("/scratch/ag6925/intern_apurv/NYU/FoV-master/data/TRAIN_exp_" + exp_number + "_xyz.p", wb))
pickle.dump(data2, open("/scratch/ag6925/intern_apurv/NYU/FoV-master/data/TEST_exp_" + exp_number + "_xyz.p", wb))

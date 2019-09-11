#anomaly detection

import os
import sys
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from datetime import datetime as dt
import _pickle as pickle

#data = pd.read_csv(path)

expNumber = sys.argv[1]
video = sys.argv[2]

experiment = 'Experiment_' + expNumber + '/'
directory = os.path.join('/home/apurvji/Desktop/apurv/NYU/dataset/Formated_Data/',experiment)
user_num = 48

f = open("/home/apurvji/Desktop/apurv/NYU/exp_" + expNumber + "/video_" + video + ".out", 'w')
sys.stdout = f

for user_ind in range(1,user_num + 1):
	print("\nUser " + str(user_ind) + " anomalies:")
	filepath = directory + str(user_ind) + '/new_video_' + video + '.csv'
	data = pd.read_csv(filepath)
	frame_vals = data['Frame Index'].values
	# print(frame_vals.shape)
	anomalies = frame_vals[:-1] <= frame_vals[1:]
	indices = np.where(anomalies == False)
	for idx in indices[0]:
		print("Frame Index changed from " + str(frame_vals[idx]) + " to " + str(frame_vals[idx + 1]) + " at index number " + str(idx) + ".")

f.close()

import glob as glob
import os
import numpy as np
from matplotlib import pyplot as plt
# from sklearn.gaussian_process import GaussianProcessRegressor
# from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as C
np.random.seed(1)
import pdb
import sys
import _pickle as pickle
if '/scratch/wz1219/FoV/' not in sys.path:
    sys.path.insert(0, '/scratch/wz1219/FoV/')
from mycode.config import cfg
import math


dataset1_exp1 = pickle.load(open('/scratch/wz1219/FoV/data/exp_1_xyz.p','rb'))
dataset1_exp2 = pickle.load(open('/scratch/wz1219/FoV/data/exp_2_xyz.p','rb'))
#dataset2 = pickle.load(open('/scratch/wz1219/FoV/data/video_data_2018.p','rb'))

new_dataset = {}

video_num1 = 9
tester_num1 = 48
video_num2 = 19
tester_num2 = 57


video_num_new = 37

for video_ind in range(video_num1):
    new_dataset[video_ind] = dataset1_exp1[video_ind]

for video_ind in range(video_num1):
    new_dataset[video_ind+9] = dataset1_exp2[video_ind]

#for video_ind in range(video_num2):
#    new_dataset[video_ind+18] = dataset2[video_ind]


pickle.dump(new_dataset,open('./data/tsinghua_merged_dataset.p','wb'))

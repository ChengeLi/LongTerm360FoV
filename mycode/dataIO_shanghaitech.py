"""
dataIO for Shanghaitech dataset, convert (latitude,longitude) into xyz
"""
import os
import sys
import numpy as np
from matplotlib import pyplot as plt
import _pickle as pickle
import sys,glob,io,random
if './360video/' not in sys.path:
    sys.path.insert(0, './360video/')
from mycode.config import cfg
import pandas as pd


def lat_long2xyz(lat,lon,R=1):
    lat = lat-np.pi/2 #phi [-pi/2,pi/2]
    lon = lon+np.pi #theta [0,2*pi]
    x = R *np.cos(lat)*np.cos(lon)
    y = R *np.cos(lat)*np.sin(lon)
    z = R *np.sin(lat)
    return x,y,z


def _split_train_test_onvideos(videoind):
    #split shanghaitech into train and test videos
    np.random.seed(0)
    np.random.shuffle(videoind)
    training_ind = videoind[:int(len(videoind)*0.8)]
    testing_ind = videoind[int(len(videoind)*0.8):]
    return training_ind,testing_ind



def others_user_num():
    user_num = []
    for key in training_ind:
        user_num.append(shanghaitech[key]['latitude'].shape[0])
    user_num = max(user_num)


if __name__ == '__main__':
    
    shanghaitech = pickle.load(open('./data/shanghai_dataset_theta_phi.p','rb'))


    ### convert latitude and longitude into xyz (right hand sys)
    for key in shanghaitech.keys():
        lat = shanghaitech[key]['latitude']
        lon = shanghaitech[key]['longitude']
        x,y,z = lat_long2xyz(lat,lon)
        shanghaitech[key]['x'] = x
        shanghaitech[key]['y'] = y
        shanghaitech[key]['z'] = z
    ### 1) for all videos
    pickle.dump(shanghaitech,open('./data/shanghai_dataset_theta_phi_xyz.p','wb'))



    #split into train/test
    videoind = shanghaitech.keys()
    training_ind,testing_ind = _split_train_test_onvideos(videoind)

    ### convert latitude and longitude into xyz (right hand sys)
    temp = {}
    # for key in training_ind:
    for key in testing_ind:
        lat = shanghaitech[key]['latitude']
        lon = shanghaitech[key]['longitude']
        x,y,z = lat_long2xyz(lat,lon)
        temp[key] = {}
        temp[key]['x'] = x
        temp[key]['y'] = y
        temp[key]['z'] = z

    ### 2) separately save for train and test
    pickle.dump(temp,open('./data/shanghai_dataset_xyz_test.p','wb'))







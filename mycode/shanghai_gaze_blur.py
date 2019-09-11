import os
import sys
import numpy as np
import _pickle as pickle
import sys,glob,io,random
import pandas as pd
import math
import matplotlib.pyplot as plt
import random
fps = 30
import h5py
data = pickle.load(open('data/shanghai_dataset_gaze_theta_phi.p','rb'), encoding='latin1')
def gauss_2d(one_hot):
    # generate a heatmap for each frame within one second(30 fps)
    IMAGE_WIDTH = 72
    IMAGE_HEIGHT = 36

    center = np.where(one_hot==1)
    #print(center[0][0])
    #print(center[1][0])
    hor = np.linspace(1,IMAGE_WIDTH,IMAGE_WIDTH)
    ver = np.linspace(1,IMAGE_HEIGHT,IMAGE_HEIGHT)
    #print(ver)
    v_dis = abs(ver-center[0][0])
    h_dis = abs(hor-center[1][0])
    for i in range(v_dis.shape[0]):
        v_dis[i] = min(v_dis[i],IMAGE_HEIGHT-v_dis[i])
    for i in range(h_dis.shape[0]):
        h_dis[i] = min(h_dis[i],IMAGE_WIDTH-h_dis[i])
    #v_dis = min(abs(ver-center[0][0]),IMAGE_HEIGHT-abs(ver-center[0][0]))
    #h_dis = min(abs(hor-center[1][0]),IMAGE_WIDTH-abs(hor-center[1][0]))
    ver_gau = np.exp(-(v_dis**2/8))
    hor_gau = np.exp(-(h_dis**2/8))
    gau = np.outer(ver_gau,hor_gau)
    return gau
    
    

def save2hdf5(path_name,key,data_to_store):
    hf = h5py.File(path_name, 'a')
    hf.create_dataset(key,data=data_to_store)
    hf.close()

def load_h5(path_name,key):
    #load
    hf = h5py.File(path_name, 'r')
    #print(list(hf.keys()))
    data = hf.get(key)
    data = np.array(data)
    return data

def discretization(lat,lon):
    lat = lat#-np.pi/2
    lon = lon+np.pi
    n = lat.shape[0]
    bin_size = 5
    one_hot_code_matrix = np.zeros((n,int(180/bin_size),int(360/bin_size)))
    for i in range(n):
        theta = lon[i]
        phi = lat[i]
        #print(phi)
        theta = theta/np.pi*180   # 0 to 360
        phi = phi/np.pi*180       # 0 to 180
        col = math.floor(theta/bin_size)
        row = math.floor(phi/bin_size)
        #print(col,row)
        #if phi == 180:
        #    row = 17
        #if theta == 360:
        #    col = 35
        if col == 72:
            col = 71
        if row == 36:
            row = 35

        one_hot_code_matrix[i,int(row),int(col)] = 1
        #print("row is:",row)
        #print("col is:",col)
    return one_hot_code_matrix






data_heatmap = {}
for video_idx in data.keys():
        lat = data[video_idx]['latitude']
        lon = data[video_idx]['longitude']
        user_num = lat.shape[0]
        video_length = lat.shape[1]

        sec = lat.shape[1]/30
        data_heatmap[video_idx] = np.zeros((user_num,int(sec),36,72,30))
        for user_idx in range(user_num):
            temp_lat = np.array(lat[user_idx])
            temp_long = np.array(lon[user_idx])

            one_hot = discretization(temp_lat,temp_long)
            for i in range(int(sec)):
                for j in range(30):
                    data_heatmap[video_idx][user_idx,i,:,:,j] = gauss_2d(one_hot[i*30+j,:])


for key in data_heatmap.keys():
    path = 'data/shanghai_heatmap_72/shanghai_gauss_tile'+key+'.hdf5'
    data = data_heatmap[key].astype('f4')
    key = 'gauss_tile'
    save2hdf5(path,key,data)


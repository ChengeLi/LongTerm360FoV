import os
import sys
import numpy as np
import cPickle as pickle
import sys,glob,io,random
import pandas as pd
import math
import random
fps = 30

data = pickle.load(open('/scratch/wz1219/FoV/data/shanghai_dataset_theta_phi.p','rb'))



def gauss_2d(one_hot):
    # generate a heatmap for each frame within one second(30 fps)
    #heatmap = np.zeros((30,18,36))
    #mu_lat = np.mean(latitude)
    #mu_long = np.mean(longitude)
    #var_lat = np.mean(latitude)
    #var_long = np.mean(longitude)
    #mean=np.array([mu_lat,mu_long])
    #X = np.stack((long, lat), axis=0)
    #cov = np.cov(X)
    #print(cov)
    #heatmap = np.random.multivariate_normal(mean, cov, (18, 36))
    IMAGE_WIDTH = 36
    IMAGE_HEIGHT = 18
    center = np.where(one_hot == 1)

    R = 1#np.sqrt(center[0]**2 + center[1]**2)/10
    Gauss_map = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH))
    for i in range(IMAGE_HEIGHT):
        for j in range(IMAGE_WIDTH):
            dis = np.sqrt((i-center[1])**2+(j-center[0])**2)
            Gauss_map[i, j] = np.exp(-0.5*dis/R)
    #plt.figure()
    #plt.imshow(Gauss_map, plt.cm.gray)
    #plt.imsave('out_2.jpg', Gauss_map, cmap=plt.cm.gray)
    #plt.show()


    return Gauss_map

def lat_long2xyz(lat,lon,R=1):
    #require lat \belong [-pi/2,pi/2], lon \belong [0,2*pi],
    lat = lat-np.pi/2
    lon = lon+np.pi
    x = R *np.cos(lat)*np.cos(lon)
    y = R *np.cos(lat)*np.sin(lon)
    z = R *np.sin(lat)
    return x,y,z

def discretization(lat,lon):
    lat = lat-np.pi/2
    lon = lon+np.pi
    n = lat.shape[0]
    bin_size = 10
    one_hot_code_matrix = np.zeros((n,int(180/bin_size),int(360/bin_size)))
    for i in range(n):
        theta = lon[i]
        phi = lat[i]
        theta = theta/np.pi*180   # 0 to 360
        phi = phi/np.pi*180       # 90 to -90
        col = math.floor(theta/bin_size)
        row = math.floor(phi/bin_size)
        if phi == 180:
            row = 17
        if theta == 180:
            col = 17
        if theta == 360:
            col = 0

        one_hot_code_matrix[i,int(row),int(col)] = 1

    return one_hot_code_matrix


data_heatmap = {}
for video_idx in data.keys():
    lat = data[video_idx]['latitude']
    lon = data[video_idx]['longitude']
    user_num = lat.shape[0]
    video_length = lat.shape[1]
    data_heatmap[video_idx] = {}
    sec = int(lat.shape[1]/30)
    data_heatmap[video_idx]['one_hot'] = np.zeros((user_num,sec,18,36))

    for user_idx in range(user_num):
        temp_lat = np.array(lat[user_idx])
        temp_long = np.array(lon[user_idx])

        one_hot = discretization(temp_lat,temp_long)
        new_one_hot = np.zeros((sec,18,36))
        for i in range(sec):
            temp = np.zeros((18,36))
            for j in range(30):
                temp = np.add(temp,one_hot[i*30+j,:])
            new_one_hot[i,:] = temp

        #for frame_idx in range(one_hot.shape[0]):
        #    heatmap = gauss_2d(one_hot[frame_idx,:])
        #    data_heatmap[video_idx]['heatmap'][user_idx,frame_idx,:] = heatmap
        data_heatmap[video_idx]['one_hot'][user_idx,:] = new_one_hot
    filename = '/scratch/wz1219/FoV/data/shanghaitech_one_hot_second/shanghaitech_onehot'+str(video_idx)+'.p'
    pickle.dump(data_heatmap[video_idx],open(filename,'wb'))

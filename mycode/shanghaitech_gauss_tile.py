import os
import sys
import numpy as np
import _pickle as pickle
import sys,glob,io,random
import pandas as pd
import math
import matplotlib.pyplot as plt
import random
%matplotlib inline
fps = 30
import h5py

data = pickle.load(open('C:/Users/weixi/Desktop/temp/shanghai_dataset_theta_phi.p','rb'), encoding='latin1')\

def fovCentroidCropping(labelarray):
    counter=0
    iteration=0
    while True:

        inputarray = np.ones((labelarray.shape[0],18,36))

        inputarraynew=inputarray.copy()
        cccc=0
        while cccc<inputarray.shape[0]:
            #x,z= phi,theta
            x=labelarray[cccc,0]
            z=labelarray[cccc,1]
            xd=x*18
            zd=z*36
            xi=int(xd)
            zi=int(zd)
            iterative=0
            row=(xi-4+18)
            inputarraynew[cccc,:,:]=0
            while iterative<9:
                row+=1
                row%=18
                rowx=row/float(18)
                #print(rowx)
                longitude=int(6/(math.cos(math.pi*(abs(rowx-0.5)))+0.00001))
                mu,sigma = 0,0.01+0.008*(math.pi*(abs(rowx-0.5)))
                X = np.linspace(0,35,36)
                G = np.exp(-(X-zi)**2/2.0*sigma**2)
                #print(longitude)
                if longitude>=17:
                    longitude=17
                zlow=zi-longitude
                zhigh=zi+longitude
                if zlow<0:
                    #print(cccc)
                    #print(row)
                    #print(zlow)
                    rr2=inputarray[cccc,row:row+1,zlow+36:36].copy()

                    inputarraynew[cccc,row:row+1,zlow+36:36]=rr2
                    zlow=0
                if zhigh>36:
                    rr2=inputarray[cccc,row:row+1,0:zhigh%36].copy()
                    inputarraynew[cccc,row:row+1,0:zhigh%36]=rr2
                    zhigh=35
                rr=inputarray[cccc,row:row+1,zlow:zhigh].copy()
                inputarraynew[cccc,row:row+1,zlow:zhigh]=rr
                #for i in range(60):
                inputarraynew[cccc,row:row+1,:] = inputarraynew[cccc,row:row+1,:]*G
                iterative+=1


            cccc+=1
      
        counter+=1
        if counter%150==0:
            iteration+=1
        return inputarraynew, labelarray


def lat_long2xyz(lat,lon,R=1):
    #require lat \belong [-pi/2,pi/2], lon \belong [0,2*pi],
    lat = lat-np.pi/2
    lon = lon+np.pi
    x = R *np.cos(lat)*np.cos(lon)
    y = R *np.cos(lat)*np.sin(lon)
    z = R *np.sin(lat)
    return x,y,z
    
    def discretization(lat,lon):
    lat = lat#-np.pi/2
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
        if col == 36:
            col = 0
        if row == 18:
            row = 0

        one_hot_code_matrix[i,int(row),int(col)] = 1

    return one_hot_code_matrix
    
data_heatmap = {}
for video_idx in data.keys():
    print(video_idx)
    lat = data[video_idx]['latitude']
    long = data[video_idx]['longitude']
    user_num = lat.shape[0]
    video_length = lat.shape[1]
    data_heatmap[video_idx] = {}
    sec = lat.shape[1]/30
    #print(lat.shape)
    data_heatmap[video_idx]['one_hot'] = np.zeros((user_num,int(sec),18,36,30))
    for user_idx in range(user_num):
        temp_lat = np.array(lat[user_idx])
        temp_long = np.array(long[user_idx])

        one_hot = discretization(temp_lat,temp_long)
        #print(one_hot.shape)
        labelarray = np.ones((one_hot.shape[0],2))
        labelarray[:,0] = temp_lat
        labelarray[:,1] = temp_long
        distorted_gaussian,label =  fovCentroidCropping(labelarray)
        for i in range(int(sec)):
            for j in range(30):
                data_heatmap[video_idx]['one_hot'][user_idx,i,:,:,j] = distorted_gaussian[i*30+j,:]
                
                
                
hf = h5py.File('C:/Users/weixi/Desktop/temp/shanghaitech_gauss_second/shanghaitech_gauss_tile.hdf5', 'w')
for k1 in data_heatmap.keys():
    
    hf.create_dataset(k1,data=data_heatmap[k1]['one_hot'])
hf.close()       

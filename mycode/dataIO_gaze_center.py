import os
import sys
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import _pickle as pickle
import sys,glob,io,random
if '/scratch/wz1219/FoV/' not in sys.path:
    sys.path.insert(0, '/scratch/wz1219/FoV/')
#from code.config import cfg
#from code.dataIO import xyz2thetaphi

#returns list of x,y,z coordinates for each time block for one particular video
def _get_data(filepath, user_ind, video_ind):
    """for Tsinghua dataset"""
    # coord_list = {}
    coord_list_x = np.zeros((1,sframes))
    coord_list_y = np.zeros((1,sframes))
    coord_list_z = np.zeros((1,sframes))

    data = pd.read_csv(filepath)
    data['Frame Index'] = data['Frame Index'].astype(int)
    if any(data['Frame Index'] == 1) == False:
        raise ValueError('User ' + str(user_ind + 1) + ' has no frame with index 1 for video ' + str(video_ind) + '.')

    ## delete the 'bad' head (not starting from frame 1)
    list1 = []
    count = 0
    while data['Frame Index'][count] != 1:
        list1.append(count)
        count = count + 1
    data = data.drop(data.index[list1])

    ## get the mean (x,y,z) from the same frames
    for fr in range(sframes):
        cutdata = data[data['Frame Index'] == (fr + 1)]
        temp = cutdata[['xCoordinate', 'yCoordinate', 'zCoordinate']].mean(axis=0)
        # coord_list[fr + 1] = np.array(temp).tolist()
        coord_list_x[0,fr] = temp['xCoordinate']
        coord_list_y[0,fr] = temp['yCoordinate']
        coord_list_z[0,fr] = temp['zCoordinate']

    coord_list_x = _fill_in_the_nan(coord_list_x)
    coord_list_y = _fill_in_the_nan(coord_list_y)
    coord_list_z = _fill_in_the_nan(coord_list_z)
    return coord_list_x,coord_list_y,coord_list_z
    
    
    
    
    
    
    
    
def _fill_in_the_nan(coord_list):
    """user neighbours to fill in the nan values"""
    nanindex = np.where(np.isnan(coord_list[0,:])==1)[0]
    print('has '+str(nanindex.shape[0])+' nan values!')
    if nanindex.shape[0]==0:
        return coord_list
    for ii in range(nanindex.shape[0]):
        candidate_list = []
        count = 0

        for ind in range(max(0,nanindex[ii]-5),min(coord_list.shape[1],nanindex[ii]+5)):
            candidate = coord_list[0,ind]
            if not np.isnan(candidate):
                candidate_list.append(candidate)
                count+=1.0
        coord_list[0,nanindex[ii]] = sum(candidate_list)/count

    nanindex = np.where(np.isnan(coord_list[0,:])==1)[0]
    print('Now has '+str(nanindex.shape[0])+' nan values!')
    return coord_list





def _get_minframes(video_ind):
    # get the rough length, cut the tail
    frames = 1e8
    for user_ind in range(1,user_num+1):
        filepath = directory+str(user_ind)+'/new_video_'+str(video_ind)+'.csv'
        file = pd.read_csv(filepath)
        frames = min(frames, int(file['Frame Index'].iloc[-1]))
    return int(frames)



def thetaphi2xyz(longitude, latitude):
    longitude = longitude+np.pi
    latitude = latitude-np.pi/2
    x = np.cos(longitude)*np.cos(latitude)  #z
    y = np.sin(longitude)*np.cos(latitude)
    z = np.sin(latitude)     #x
    return x, y, z


def normalize_data(df):
    from sklearn import preprocessing
    min_max_scaler = preprocessing.MinMaxScaler()
    df['open'] = min_max_scaler.fit_transform(df.open.values.reshape(-1,1))
    df['high'] = min_max_scaler.fit_transform(df.high.values.reshape(-1,1))
    df['low'] = min_max_scaler.fit_transform(df.low.values.reshape(-1,1))
    df['volume'] = min_max_scaler.fit_transform(df.volume.values.reshape(-1,1))
    df['adj close'] = min_max_scaler.fit_transform(df['adj close'].values.reshape(-1,1))
    return df
# df = normalize_data(df)

def denormalize(df, normalized_value):
    """Denormalize the data"""
    df = df['adj close'].values.reshape(-1,1)
    normalized_value = normalized_value.reshape(-1,1)

    #return df.shape, p.shape
    min_max_scaler = preprocessing.MinMaxScaler()
    a = min_max_scaler.fit_transform(df)
    new = min_max_scaler.inverse_transform(normalized_value)
    return new



directory = os.path.join('/scratch/wz1219/FoV/data/Gaze_txt_files/')
user_num = 45
video_list = [0 for i in range(208)]
c = 0
#get video_list
for filename in os.listdir('/scratch/wz1219/FoV/data/Gaze_txt_files/p042'):
    video_list[c] = filename[:3]
    c = c+1
all_video_data = {}
for video_idx in range(208):
    all_video_data[video_list[video_idx]] = {}
    all_video_data[video_list[video_idx]]['longitude'] = np.array([])
    all_video_data[video_list[video_idx]]['latitude'] = np.array([])
    all_video_data[video_list[video_idx]]['user_id'] = np.array([])
    all_video_data[video_list[video_idx]]['x'] = np.array([])
    all_video_data[video_list[video_idx]]['y'] = np.array([])
    all_video_data[video_list[video_idx]]['z'] = np.array([])
for user_idx in range(1,user_num+1):
    sub_dir = "p0{0:0=2d}".format(user_idx)
    sub_path = os.path.join(directory,sub_dir)
    for filename in os.listdir(sub_path):
        print(sub_path)
        print(filename)
        txt = open(os.path.join(sub_path,filename),'r')#txt
        lines = txt.readlines()
        video_idx = filename[:3]
        longitude = np.array([])
        latitude = np.array([])
        x = np.array([])
        y = np.array([])
        z = np.array([])
        for line in lines:
            #line = str(line)
            line_list = line.split(",")
            #print(line_list)
            #print(line_list)
            longitude = np.append(longitude,float(line_list[6])*np.pi*2-np.pi)
            latitude = np.append(latitude,float(line_list[7])*np.pi)
        #print(longitude.shape)
        #print(all_video_data[video_idx]['longitude'].shape)
        if all_video_data[video_idx]['longitude'].shape == (0,):
            all_video_data[video_idx]['longitude'] = np.reshape(longitude,(1,len(longitude)))
        else :
            if len(longitude) < all_video_data[video_idx]['longitude'].shape[1]:
                all_video_data[video_idx]['longitude'] = all_video_data[video_idx]['longitude'][:,:len(longitude)]
                all_video_data[video_idx]['longitude'] = np.vstack([all_video_data[video_idx]['longitude'],longitude])
            else:
                longitude = longitude[:all_video_data[video_idx]['longitude'].shape[1]]
                all_video_data[video_idx]['longitude'] = np.vstack([all_video_data[video_idx]['longitude'],longitude])

        if all_video_data[video_idx]['latitude'].shape == (0,):
            all_video_data[video_idx]['latitude'] = np.reshape(latitude,(1,len(latitude)))
        else :
            if len(longitude) < all_video_data[video_idx]['latitude'].shape[1]:
                all_video_data[video_idx]['latitude'] = all_video_data[video_idx]['latitude'][:,:len(latitude)]
                all_video_data[video_idx]['latitude'] = np.vstack([all_video_data[video_idx]['latitude'],latitude])
            else:
                latitude = latitude[:all_video_data[video_idx]['latitude'].shape[1]]
                all_video_data[video_idx]['latitude'] = np.vstack([all_video_data[video_idx]['latitude'],latitude])


        all_video_data[video_idx]['user_id'] = np.append(all_video_data[video_idx]['user_id'],user_idx)
        #print(longitude.shape)
        x,y,z = thetaphi2xyz(longitude,latitude)

        if all_video_data[video_idx]['x'].shape == (0,):
            all_video_data[video_idx]['x'] = np.reshape(x,(1,len(x)))
        else :
            if len(x) < all_video_data[video_idx]['x'].shape[1]:
                all_video_data[video_idx]['x'] = all_video_data[video_idx]['x'][:,:len(x)]
                all_video_data[video_idx]['x'] = np.vstack([all_video_data[video_idx]['x'],x])
            else:
                x = x[:all_video_data[video_idx]['x'].shape[1]]
                all_video_data[video_idx]['x'] = np.vstack([all_video_data[video_idx]['x'],x])

        if all_video_data[video_idx]['y'].shape == (0,):
            all_video_data[video_idx]['y'] = np.reshape(y,(1,len(y)))
        else :
            if len(y) < all_video_data[video_idx]['y'].shape[1]:
                all_video_data[video_idx]['y'] = all_video_data[video_idx]['y'][:,:len(y)]
                all_video_data[video_idx]['y'] = np.vstack([all_video_data[video_idx]['y'],y])
            else:
                y = y[:all_video_data[video_idx]['y'].shape[1]]
                all_video_data[video_idx]['y'] = np.vstack([all_video_data[video_idx]['y'],y])



        if all_video_data[video_idx]['z'].shape == (0,):
            all_video_data[video_idx]['z'] = np.reshape(z,(1,len(z)))
        else :
            if len(z) < all_video_data[video_idx]['z'].shape[1]:
                all_video_data[video_idx]['z'] = all_video_data[video_idx]['z'][:,:len(z)]
                all_video_data[video_idx]['z'] = np.vstack([all_video_data[video_idx]['z'],z])
            else:
                z = z[:all_video_data[video_idx]['z'].shape[1]]
                all_video_data[video_idx]['z'] = np.vstack([all_video_data[video_idx]['z'],z])


        #all_video_data[video_idx]['x'] = x
        #all_video_data[video_idx]['y'] = y
        #all_video_data[video_idx]['z'] = z


pickle.dump(all_video_data,open('./data/shanghai_dataset_gaze_center.p','wb'))



"""
dataIO for tsinghua dataset after cleaning using dataModder.py 
(user gazes are aligned using frame index (1-based))
"""
import os
import sys
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import _pickle as pickle
from mycode.config import cfg
from mycode.dataIO import xyz2thetaphi

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



if __name__ == '__main__':
    expNumber = str(1)
    experiment = 'Experiment_' + expNumber + '/'
    directory = os.path.join('./360video/tsinghua/Formated_Data/',experiment)
    user_num = 48
    all_video_data_xyz = {}
    all_video_data_phi_theta = {}
    for video_ind in range(1,9):
        #shortest video length for this video among all users
        sframes = _get_minframes(video_ind)
        print("shortest frames for video #",video_ind,"among all users is",sframes)
        if cfg.use_xyz:
            all_video_data_xyz[video_ind] = {}
            all_video_data_xyz[video_ind]['x'] = np.zeros((user_num,sframes))
            all_video_data_xyz[video_ind]['y'] = np.zeros((user_num,sframes))
            all_video_data_xyz[video_ind]['z'] = np.zeros((user_num,sframes))

        if cfg.use_phi_theta:
            all_video_data_phi_theta[video_ind] = {}
            all_video_data_phi_theta[video_ind]['phi'] = np.zeros((user_num,sframes))
            all_video_data_phi_theta[video_ind]['theta'] = np.zeros((user_num,sframes))

        #stores all the data for every user, for this video
        for user_ind in range(user_num):
            filepath = directory+str(user_ind + 1)+'/new_video_'+str(video_ind)+'.csv'            
            coord_list_x,coord_list_y,coord_list_z = _get_data(filepath, user_ind, video_ind)            
            if cfg.use_xyz:
                all_video_data_xyz[video_ind]['x'][user_ind,:] = coord_list_x
                all_video_data_xyz[video_ind]['y'][user_ind,:] = coord_list_y
                all_video_data_xyz[video_ind]['z'][user_ind,:] = coord_list_z
            
            elif cfg.use_phi_theta:
                theta_list, phi_list = xyz2thetaphi(np.array(coord_list_x),np.array(coord_list_y),np.array(coord_list_z))
                all_video_data_phi_theta[video_ind]['phi'][user_ind,:] = phi_list
                all_video_data_phi_theta[video_ind]['theta'][user_ind,:] = theta_list


            print("user",user_ind,"complete.")#" Length of vector appended:", sframes,len(user_points),len(np.unique(np.array(user_points.keys()))))
        
        if cfg.use_xyz:
            print("Matrix shape",np.shape(all_video_data_xyz[video_ind]['x']))
            print(video_ind,'has null at: ',np.where(pd.isnull(all_video_data_xyz[video_ind]['x'])))
            if len(np.where(pd.isnull(all_video_data_xyz[video_ind]['x']))[0])>0:
                pdb.set_trace()
        if cfg.use_phi_theta:
            print("Matrix shape",np.shape(all_video_data_phi_theta[video_ind]['phi']))
            print(video_ind,'has null at: ',np.where(pd.isnull(all_video_data_phi_theta[video_ind]['phi'])))
            if len(np.where(pd.isnull(all_video_data_phi_theta[video_ind]['phi']))[0])>0:
                pdb.set_trace()


    if cfg.use_xyz:
        pickle.dump(all_video_data_xyz,open('./360video/data/new_exp_'+expNumber+'_xyz.p','wb'))
    if cfg.use_phi_theta:
        pickle.dump(all_video_data_phi_theta,open('./data/new_exp_'+str(expNumber)+'_phi_theta.p','wb'))













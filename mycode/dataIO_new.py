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
import warnings

expNumber = sys.argv[1]
experiment = 'Experiment_' + expNumber + '/'
directory = os.path.join('/scratch/ag6925/intern_apurv/NYU/FoV-master/data/Formated_Data/',experiment)

def non_decreasing(L):
    return all(x<=y for x, y in zip(L, L[1:]))

def nan_helper(y):
    """Helper to handle indices and logical indices of NaNs.
    #
    Input:
        - y, 1d numpy array with possible NaNs
    Output:
        - nans, logical indices of NaNs
        - index, a function, with signature indices= index(logical_indices),
          to convert logical indices of NaNs to 'equivalent' indices
    Example:
        >>> # linear interpolation of NaNs
        >>> nans, x= nan_helper(y)
        >>> y[nans]= np.interp(x(nans), x(~nans), y[~nans])
    """
    #
    return np.isnan(y), lambda z: z.nonzero()[0]

def anomaly(filepath, video_no, user_ind):
    data = pd.read_csv(filepath)
    frame_vals = data['Frame Index'].values
    # print(frame_vals.shape)
    anomalies = frame_vals[:-1] <= frame_vals[1:]
    indices = np.where(anomalies == False)
    #
    return indices[0]

#returns list of x,y,z coordinates for each time block for one particular video
def _get_data(filepath, user_ind, video_ind):
    """for Tsinghua dataset"""
    #
    data = pd.read_csv(filepath)
    data['Frame Index'] = data['Frame Index'].astype(int)
    #
    if any(data['Frame Index'].values <= 3) == False:
        warnings.warn('User ' + str(user_ind + 1) + ' has no frame with index less than equal to 3 for video ' + str(video_ind) + '.')
    #
    indices = anomaly(filepath, user_ind, video_ind)
    print("indices are", indices)
    #
    if indices.size:
        del_range = []
        idx = 0
        jump = data['Frame Index'].loc[indices[idx]] - data['Frame Index'].loc[indices[idx] + 1]
        if jump > 5:
            del_range = del_range + range(indices[0] + 1)
            idx = 1
        while idx < len(indices):
            print(indices[idx])
            curr_frame = data['Frame Index'].loc[indices[idx]]
            next_frame = data['Frame Index'].loc[indices[idx] + 1]
            # if (curr_frame - next_frame) > 5:
            #go back till you find the next frame and delete the range till frame of current anomaly
            ptr = indices[idx]
            temp_frame = data['Frame Index'].loc[ptr]
            while temp_frame > next_frame:
                ptr = ptr - 1
                temp_frame = data['Frame Index'].loc[ptr]
            #
            del_range = del_range + range(ptr + 1, indices[idx] + 1)
            idx = idx + 1
            # data = data.drop(data.index[range(ptr + 1, indices[idx] + 1)])
            # data.reset_index(drop=True, inplace = True)
        #remove second anomaly only if worse than 10 frames
        data = data.drop(data.index[del_range])
    
    if non_decreasing(data['Frame Index'].values.tolist()):
        print("the modified frame numbers are all non non decreasing.")
    else:
        print("The list contains anomalies.")
        print(anomaly(filepath, user_ind, video_ind))
    #
    #
    coord_list_x = np.zeros((1,sframes))
    coord_list_y = np.zeros((1,sframes))
    coord_list_z = np.zeros((1,sframes))
    #
    for fr in range(sframes):
        cutdata = data[data['Frame Index'] == (fr + 1)]
        temp = cutdata[['xCoordinate', 'yCoordinate', 'zCoordinate']].mean(axis=0)
        # coord_list[fr + 1] = np.array(temp).tolist()
        coord_list_x[0,fr] = temp['xCoordinate']
        coord_list_y[0,fr] = temp['yCoordinate']
        coord_list_z[0,fr] = temp['zCoordinate']
    #
    #fill_in_the_nan takes the number of neighbours as a parameter which is not feasible everytime
    # coord_list_x = _fill_in_the_nan(coord_list_x)
    # coord_list_y = _fill_in_the_nan(coord_list_y)
    # coord_list_z = _fill_in_the_nan(coord_list_z)
    #
    nans, x = nan_helper(coord_list_x[0])
    coord_list_x[0][nans]= np.interp(x(nans), x(~nans), coord_list_x[0][~nans])
    #
    nans, x = nan_helper(coord_list_y[0])
    coord_list_y[0][nans]= np.interp(x(nans), x(~nans), coord_list_y[0][~nans])
    #
    nans, x = nan_helper(coord_list_z[0])
    coord_list_z[0][nans]= np.interp(x(nans), x(~nans), coord_list_z[0][~nans])
    #
    return coord_list_x,coord_list_y,coord_list_z

def _fill_in_the_nan(coord_list):
    """user 10 neighbours to fill in the nan values"""
    nanindex = np.where(np.isnan(coord_list[0,:])==1)[0]
    print(nanindex)
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
        filepath = directory+str(user_ind)+'/NEW_video_'+str(video_ind)+'.csv'
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
    #    
    #return df.shape, p.shape
    min_max_scaler = preprocessing.MinMaxScaler()
    a = min_max_scaler.fit_transform(df)
    new = min_max_scaler.inverse_transform(normalized_value)
    return new

#do not use this for now
def vid_max_stretch(vid_no, user_no):
    # stretch = [1, _get_minframes(vid_no)]
    anomalies = anomaly(vid_no, user_no)
    stretch = np.argmax(np.diff(anomalies))
    start = anomalies[stretch] + 1
    end = anomalies[stretch + 1] - 1
    return start, end

if __name__ == '__main__':
    user_num = 48
    all_video_data_xyz = {}
    
    for video_ind in range(user_num):
        #shortest video length for this video among all users
        sframes = _get_minframes(video_ind)
        print("shortest frames for video #",video_ind,"among all users is",sframes)
        all_video_data_xyz[video_ind] = {}
        all_video_data_xyz[video_ind]['x'] = np.zeros((user_num,sframes))
        all_video_data_xyz[video_ind]['y'] = np.zeros((user_num,sframes))
        all_video_data_xyz[video_ind]['z'] = np.zeros((user_num,sframes))

        #stores all the data for every user, for this video
        for user_ind in range(user_num):
        # for user_ind in [37]:
            filepath = directory+str(user_ind + 1)+'/NEW_video_'+str(video_ind)+'.csv'            
            coord_list_x,coord_list_y,coord_list_z = _get_data(filepath, user_ind, video_ind)            
            all_video_data_xyz[video_ind]['x'][user_ind,:] = coord_list_x
            all_video_data_xyz[video_ind]['y'][user_ind,:] = coord_list_y
            all_video_data_xyz[video_ind]['z'][user_ind,:] = coord_list_z
            
            print("user",user_ind,"complete.")#" Length of vector appended:", sframes,len(user_points),len(np.unique(np.array(user_points.keys()))))
        
        print("Matrix shape",np.shape(all_video_data_xyz[video_ind]['x']))
        print(video_ind,'has null at: ',np.where(pd.isnull(all_video_data_xyz[video_ind]['x'])))
        if len(np.where(pd.isnull(all_video_data_xyz[video_ind]['x']))[0])>0:
            pdb.set_trace()

    pickle.dump(all_video_data_xyz,open('/scratch/ag6925/intern_apurv/NYU/FoV-master/data/modified_exp_'+expNumber+'_xyz.p','wb'))


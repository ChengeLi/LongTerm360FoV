
# coding: utf-8

# In[3]:


import glob as glob
import os
import numpy as np
from matplotlib import pyplot as plt
# from sklearn.gaussian_process import GaussianProcessRegressor
# from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as C
np.random.seed(1)
import pdb
import _pickle as pickle              ############# import cPickle asa pickle
#from mycode.config import cfg
import math
from scipy import signal


# In[4]:


def clip_xyz(x):
    """clip xyz to [-1,1]"""
    # for ii in range(len(x.keys())):
    for ii in x.keys():
        x[ii]['x'][x[ii]['x']<-1]=-1
        x[ii]['x'][x[ii]['x']>1]=1
        x[ii]['y'][x[ii]['y']<-1]=-1
        x[ii]['y'][x[ii]['y']>1]=1        
        x[ii]['z'][x[ii]['z']<-1]=-1
        x[ii]['z'][x[ii]['z']>1]=1
    return x

def quaternion2euler2(q0,q1,q2,q3):
    q0 = np.float(q0)
    q1 = np.float(q1)
    q2 = np.float(q2)
    q3 = np.float(q3)
    """convert quaternion tuple to euler angles"""
    roll = np.arctan2(2*(q0*q1+q2*q3),(1-2*(q1**2+q2**2)))
    # confine to [-1,1] to avoid nan from arcsin
    sintemp = min(1,2*(q0*q2-q3*q1))
    sintemp = max(-1,sintemp)
    pitch = np.arcsin(sintemp)
    yaw = np.arctan2(2*(q0*q3+q1*q2),(1-2*(q2**2+q3**2)))
    return roll,pitch,yaw


def quaternion2euler3(qx,qy,qz,qw):
    """from tsinghua paper"""
    qx = np.float(qx)
    qy = np.float(qy)
    qz = np.float(qz)
    qw = np.float(qw)
    """convert quaternion tuple to euler angles"""
    x = 2*qx*qz+2*qy*qw
    y = 2*qy*qz-2*qx*qw
    z = 1-2*(qx**2)-2*(qy**2)
    return x,y,z

# def quaternion2euler4(yaw,pitch):
#     # from kanishk's report 
#     x = np.cos(yaw)*np.cos(pitch)
#     y = np.sin(yaw)*np.cos(pitch)
#     z = np.sin(pitch)
#     return x,y,z

def quaternion2euler4(yaw,pitch):
    # from CUB360 paper 
    z = np.cos(yaw)*np.cos(pitch)
    x = np.sin(yaw)*np.cos(pitch)
    y = np.sin(pitch)
    return x,y,z


def xyz2thetaphi_thu(x,y,z):
    # convert x,y,z to theta and phi
    # theta = arctan(y/x)
    theta = np.arctan2(y,x)
    phi = np.arctan2(np.sqrt(x**2+y**2),z)
    return theta, phi

def get_anglelist_from_file_HDM(temp):
    """for HMD dataset"""
    """get euler angle list from the file lines"""
    roll_list = []
    pitch_list = []
    yaw_list = []
    for ii in range(len(temp)):
        [q0,q1,q2,q3] = np.array(temp[ii].split(' '))[-4:]
        roll,pitch,yaw = quaternion2euler2(q0,q1,q2,q3)
        roll_list.append(roll)
        pitch_list.append(pitch)
        yaw_list.append(yaw)
    return roll_list,pitch_list,yaw_list



def get_anglelist_from_file_tsinghua(temp):
    """for Tsinghua dataset"""
    """get euler angle list from the file lines"""
    roll_list = []
    pitch_list = []
    yaw_list = []
    # HmdPosition_x_list = []
    # HmdPosition_y_list = []
    # HmdPosition_z_list = []
    for ii in range(1,len(temp)):
        [qx,qy,qz,qw] = np.array(temp[ii].split(','))[2:6]
        #!!!order not one to one! https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles
        q0,q1,q2,q3 = qw,qx,qy,qz 
        roll,pitch,yaw = quaternion2euler2(q0,q1,q2,q3)
        roll_list.append(roll)
        pitch_list.append(pitch)
        yaw_list.append(yaw)
        # [x,y,z] = np.array(temp[ii].split(','))[6:]
        # HmdPosition_x_list.append(x)
        # HmdPosition_y_list.append(y)
        # HmdPosition_z_list.append(z)
    return roll_list,pitch_list,yaw_list


## interpolation done here/////////////

def get_xyz_from_file_tsinghua(temp,shortest_len):
    """for Tsinghua dataset"""
    x_list = []
    y_list = []
    z_list = []
    t_list = []
    jump_list = []
    frame_list = []
    time = 0
    #print(len(temp))
    
   
    # find smallest playback time as the beginning
    t_begin = 100
    begin_row = 0
    for ii in range(1,len(temp)):
        t = np.array(temp[ii].split(','))[1]
        t = float(t)
        if t_begin > t:
            t_begin = t
            begin_row = ii
    # find largest playback time as the end
    t_end = 0
    end_row = 0
    for ii in range(1,len(temp)):
        t = np.array(temp[ii].split(','))[1]
        t = float(t)
        if t_end < t:
            t_end = t
            end_row = ii
   # print(begin_row)
   # print(end_row)
    
    # average on playback time to get rid of jump-backs
    for ii in range(begin_row,end_row):
        [qx,qy,qz,qw] = np.array(temp[ii].split(','))[2:6]
        x,y,z = quaternion2euler3(qx,qy,qz,qw)
        t = np.array(temp[ii].split(','))[1]
        t = float(t)
        flag = 0
        #print(len(t_list))
        if len(t_list) == 0:
            t_list.append(t)
            x_list.append(x)
            y_list.append(y)
            z_list.append(z) 
            #print(t_list)
            #print(x_list)
            #print(x)
        else:
            if t<t_list[-1]:
                for jj in range(len(t_list)):
                    if abs(t-t_list[-1-jj])<=0.0095:
                        x_list[jj] = x
                        y_list[jj] = y
                        z_list[jj] = z        
                        print("one jumpback found!")
            else:
                x_list.append(x)
                y_list.append(y)
                z_list.append(z)           
                t_list.append(t)
    tt = 0
    temp_list_x = []
    temp_list_y = []
    temp_list_z = []
    final_x = []
    final_y = []
    final_z = []
    print(len(t_list))
    for ii in range(len(t_list)):

        if t_list[ii] >= tt+1:
            tt = tt+1
            temp_list_x = signal.resample(temp_list_x, 30)
            temp_list_y = signal.resample(temp_list_y, 30)
            temp_list_z = signal.resample(temp_list_z, 30)
            
            #print(type(temp_list_x))
            final_x=final_x+temp_list_x.tolist()
            final_y=final_y+temp_list_y.tolist()
            final_z=final_z+temp_list_z.tolist()
            temp_list_x = []
            temp_list_y = []
            temp_list_z = []
        else:
            temp_list_x.append(x_list[ii])
            temp_list_y.append(y_list[ii])
            temp_list_z.append(z_list[ii])

    print(len(final_x))
    
    return final_x,final_y,final_z


def raw_2_sincos(angle_array):
    """convert raw angles (yaw,pitch,roll) into (sin, cos) pairs"""
    cos_val = np.cos(angle_array)
    sin_val = np.sin(angle_array)
    return (cos_val,sin_val)


def _get_length(video_ind):
    # get the rough length, cut the tail
    length = 1e8
    for user_ind in range(1,user_num+1):
        filepath = directory+str(user_ind)+'/video_'+str(video_ind)+'.csv'
        temp = open(filepath,'r').readlines()
        length = min(len(temp)-1,length)
    # print('video_ind', video_ind, 'length ',length)
    return int(length)


def _get_time_duration(watch_time_start,watch_time_end):
    hr,mt,sd = np.float32(watch_time_start.split(':'))
    hr1,mt1,sd1 = np.float32(watch_time_end.split(':'))
    duration = (hr1-hr)*3600+(mt1-mt)*60+(sd1-sd)
    return duration


def discretize360(x_list,y_list,z_list,bin_size=30):
    """quantize the sphere into one-hot vector, each block is (bin_size,bin_size)"""
    n = len(x_list)
    one_hot_code_matrix = np.zeros((n,180/bin_size,360/bin_size))
    for i in range(n):
        theta,phi = xyz2thetaphi(x_list[i],y_list[i],z_list[i])
        theta = theta/np.pi*180
        phi = phi/np.pi*180
        col = math.floor(theta/bin_size)
        row = math.floor(phi/bin_size)
        if phi == 180:
            row = 5
        if theta == 180:
            col = 5
        #print(theta)
        #print(phi)

        #one_hot_code_matrix[i,0] = int(row)
        #one_hot_code_matrix[i,1] = int(col+6)
        #one_hot_code = np.zeros((6,12))
        one_hot_code_matrix[i,int(row),int(col+6)] = 1
        #one_hot_code_matrix[i,:] = one_hot_code.flatten()  
    return one_hot_code_matrix

def reshape_one_user(one_hot_code_matrix):
    """reshape the data of one user for one video to n-by-30-by-6-by-12 matrix"""
    #n is secs
    #input size is numframes-by-6-by-12
    #output size is numsec-by-30-by-6-by-12
    nframes = one_hot_code_matrix.shape[0]
    nsecs = nframes//30
    new_representation = np.zeros((nsecs,30,6,12)) #ignore the remainder here
    for i in range(nsecs):
        new_representation[i,:,:,:] = one_hot_code_matrix[i*30:(i+1)*30,:,:]
    return new_representation


def reshape_multi_users(one_hot_code_matrix_of_multiusers):
    """reshape the data of multiple users for one video to one single n-by-30-by-6-by-12 matrix"""
    #n is secs
    #input size is numusers-by-numframes-by-6-by-12
    #output size is numsec-by-30-by-6-by-12
    nusers = one_hot_code_matrix_of_multiusers.shape[0]
    nframes = one_hot_code_matrix_of_multiusers.shape[1]
    nsecs = nframes//30
    new_representation = np.zeros((nsecs,30,6,12)) #ignore the remainder
    for i in range(nsecs):
        
        temp = np.sum(one_hot_code_matrix_of_multiusers[:,i*30:(i+1)*30,:,:],0)
        new_representation[i,:,:,:] = temp
    return new_representation

def gen_heatmap_like_matrix(one_hot_code_matrix):
    """generate a heatmap-like matrix for one user"""
    #input size is numframes-by-6-by-12
    #output size is 6-by-12

    heatmap = np.zeros((6,12))
    heatmap = np.sum(one_hot_code_matrix,0)
    return heatmap

def gen_heatmap_like_matrix_for_multiusers(one_hot_code_matrix_of_multiusers):
    """generate a heatmap-like matrix for one user"""
    #input size is numusers-by-numframes-by-6-by-12
    #output size is 6-by-12
    heatmap = np.zeros((6,12))
    heatmap = np.sum(np.sum(one_hot_code_matrix_of_multiusers,0),0)
    return heatmap


# In[ ]:


if __name__ == '__main__':
    # Tsinghua Dataset
    experiment = 2    # 1
    experimentpath = 'Experiment_'+str(experiment)+'/'
    directory = os.path.join(experimentpath)
    user_num = 48
    # video info
    videoMetaFile = os.path.join(experimentpath+'/videoMeta.csv')
    videoMeta = open(videoMetaFile,'r').readlines()

    all_video_data = {} #from all videos of one experiment
    all_video_data_pair = {}
    all_video_data_xyz = {}
    duration_dic = {}
    all_video_data_phi_theta = {}

    for video_ind in range(9):
        video_info = videoMeta[video_ind+1].split(',')
        fps = int(video_info[3])
        vid_duration = int(video_info[2].split(':')[0])*60+int(video_info[2].split(':')[1])
        print('video ',video_ind,' has duration of: ', vid_duration,' seconds')
        duration_dic[video_ind] = {}
        shortest_len = vid_duration*30
        if shortest_len==5610:
            shortest_len =5580
        if shortest_len==5160:
            shortest_len =5130
        if shortest_len==6780:
            shortest_len =6750
        #shortest_len = _get_length(video_ind)
        #print(shortest_len)
        if 0:#cfg.use_yaw_pitch_roll:
            all_video_data[video_ind] = {}
            all_video_data[video_ind]['raw_roll'] = np.zeros((user_num,shortest_len))
            all_video_data[video_ind]['raw_pitch'] = np.zeros((user_num,shortest_len))
            all_video_data[video_ind]['raw_yaw'] = np.zeros((user_num,shortest_len))
        if 1:#cfg.use_xyz:
            all_video_data_xyz[video_ind] = {}
            all_video_data_xyz[video_ind]['x'] = np.zeros((user_num,shortest_len))
            all_video_data_xyz[video_ind]['y'] = np.zeros((user_num,shortest_len))
            all_video_data_xyz[video_ind]['z'] = np.zeros((user_num,shortest_len))
        if 0:#cfg.use_phi_theta:
            all_video_data_phi_theta[video_ind] = {}
            all_video_data_phi_theta[video_ind]['phi'] = np.zeros((user_num,shortest_len))
            all_video_data_phi_theta[video_ind]['theta'] = np.zeros((user_num,shortest_len))


        for user_ind in range(1,user_num+1):
            filepath = directory+str(user_ind)+'/video_'+str(video_ind)+'.csv'
            temp = open(filepath,'r').readlines()
            # watch_time_start = temp[1].split(',')[0].split(' ')[1]
            # watch_time_end = temp[-1].split(',')[0].split(' ')[1]
            # duration_dic[video_ind][user_ind] = _get_time_duration(watch_time_start,watch_time_end)
            # print('user ',user_ind,' watched: ', duration_dic[video_ind][user_ind])
            if 1:#cfg.use_xyz:
                x_list,y_list,z_list = get_xyz_from_file_tsinghua(temp,shortest_len)
                all_video_data_xyz[video_ind]['x'][user_ind-1,:] = x_list[:shortest_len]
                all_video_data_xyz[video_ind]['y'][user_ind-1,:] = y_list[:shortest_len]
                all_video_data_xyz[video_ind]['z'][user_ind-1,:] = z_list[:shortest_len]

            elif 0:#cfg.use_yaw_pitch_roll:
                roll_list,pitch_list,yaw_list = get_anglelist_from_file_tsinghua(temp)
                all_video_data[video_ind]['raw_roll'][user_ind-1,:] = roll_list[:shortest_len]
                all_video_data[video_ind]['raw_pitch'][user_ind-1,:] = pitch_list[:shortest_len]
                all_video_data[video_ind]['raw_yaw'][user_ind-1,:] = yaw_list[:shortest_len]
            elif 0:#cfg.use_phi_theta:
                x_list,y_list,z_list = get_xyz_from_file_tsinghua(temp,shortest_len)
                theta_list, phi_list = xyz2thetaphi(np.array(x_list),np.array(y_list),np.array(z_list))
                all_video_data_phi_theta[video_ind]['phi'][user_ind-1,:] = phi_list[:shortest_len]
                all_video_data_phi_theta[video_ind]['theta'][user_ind-1,:] = theta_list[:shortest_len]

                pdb.set_trace()
                # TODO: test whether the same with tsinghua paper
                x_list2,y_list2,z_list2 = quaternion2euler4(yaw_list[:shortest_len],pitch_list[:shortest_len])

                # print('video_ind ',video_ind,'len(yaw_list) ',len(yaw_list))
                ###lengths are different, why?
                # print user_ind,len(yaw_list) 
                # print(temp[1].split(',')[0].split(' ')[1], temp[-1].split(',')[0].split(' ')[1])

        if 0:#cfg.use_cos_sin:
            all_video_data_pair[video_ind] = {}
            # (cos,sin) pair    
            all_video_data_pair[video_ind]['cos_yaw'] = np.zeros((user_num,shortest_len))
            all_video_data_pair[video_ind]['sin_yaw'] = np.zeros((user_num,shortest_len))
            all_video_data_pair[video_ind]['cos_pitch'] = np.zeros((user_num,shortest_len))
            all_video_data_pair[video_ind]['sin_pitch'] = np.zeros((user_num,shortest_len))

            cos_yaw,sin_yaw = raw_2_sincos(all_video_data[video_ind]['raw_yaw'])
            cos_pitch,sin_pitch = raw_2_sincos(all_video_data[video_ind]['raw_pitch'])
            # cos_roll,sin_roll = raw_2_sincos(all_video_data[video_ind]['raw_roll'])

            all_video_data_pair[video_ind]['cos_yaw'] = cos_yaw
            all_video_data_pair[video_ind]['sin_yaw'] = sin_yaw
            all_video_data_pair[video_ind]['cos_pitch'] = cos_pitch
            all_video_data_pair[video_ind]['sin_pitch'] = sin_pitch
    
    if 0:#cfg.use_yaw_pitch_roll:
        pickle.dump(all_video_data,open('./data/exp_'+str(experiment)+'_raw.p','wb'))
    if 0:#cfg.use_cos_sin:
        pickle.dump(all_video_data_pair,open('./data/exp_'+str(experiment)+'_raw_pair.p','wb'))
    if 1:#cfg.use_xyz:
        pickle.dump(all_video_data_xyz,open('./exp_'+str(experiment)+'_xyz.p','wb'))
    if 0:#cfg.use_phi_theta:
        pickle.dump(all_video_data_phi_theta,open('./data/exp_'+str(experiment)+'_phi_theta.p','wb'))


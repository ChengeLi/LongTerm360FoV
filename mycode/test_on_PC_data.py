"""
This script is used to test the trained model on PC data collected by our lab
"""
# LOCAL
from numpy import genfromtxt
import glob as glob
from scipy.io import loadmat
import _pickle as pickle

def lat_long2xyz(lat,lon,R=1):
    #require lat \belong [-pi/2,pi/2], lon \belong [0,2*pi],
    x = R *np.cos(lat)*np.cos(lon)
    y = R *np.cos(lat)*np.sin(lon)
    z = R *np.sin(lat)
    return x,y,z

parent_folder = './Downloads/FOV_yixiang/'
folder_list = sorted(glob.glob(parent_folder+'*'))
pc_data = {}

for folder in folder_list:
    video_name = folder.split('/')[-1]
    matlist = sorted(glob.glob(folder+'/*.mat'))
    theta_array = []
    phi_array = []
    x_array = []
    y_array = []
    z_array = []

    for user_ind in range(len(matlist)):
        matfile = matlist[user_ind]
        my_data = loadmat(matfile)
        theta = my_data['data'][:,1]
        phi = my_data['data'][:,2]
        theta_array.append(theta)
        phi_array.append(phi)
        x,y,z= lat_long2xyz(phi,theta)
        x_array.append(x)
        y_array.append(y)
        z_array.append(z)

    pc_data[video_name]={}
    pc_data[video_name]['theta'] = np.array(theta_array)
    pc_data[video_name]['phi'] = np.array(phi_array)
    pc_data[video_name]['x'] = np.array(x_array)
    pc_data[video_name]['y'] = np.array(y_array)
    pc_data[video_name]['z'] = np.array(z_array)

pickle.dump(pc_data,open(parent_folder+'resampled.p','wb')) #all others videos yixiang sent me, except for the ski video


#SuperMario has some users with more samples




### on HPC
import numpy as np
from scipy import interpolate
import _pickle as pickle
# pc_data = pickle.load(open('./data/pc_data.p','rb'),encoding='latin')
pc_data = pickle.load(open('../../data/resampled.p','rb'),encoding='latin')
#resample from 25 FPS to 30 FPS

pc_data_interp = {}
def resample(key,video_ind):
    xnews=[]
    x = np.linspace(0,1,pc_data[video_ind]['x'].shape[1])
    for user_ind in range(pc_data[video_ind]['x'].shape[0]):
        f = interpolate.interp1d(x, pc_data[video_ind][key][user_ind,:])
        xnew = np.linspace(0,1,int(pc_data[video_ind][key].shape[1]/25.*30))
        xnews.append(f(xnew))
    return np.array(xnews)

for video_ind in pc_data.keys():
    if video_ind in ['SuperMario']:
        continue
    pc_data_interp[video_ind] = {}
    pc_data_interp[video_ind]['x'] = resample('x',video_ind)
    pc_data_interp[video_ind]['y'] = resample('y',video_ind)
    pc_data_interp[video_ind]['z'] = resample('z',video_ind)




# datadb = {}
# datadb['ski_video'] = pc_data_interp
datadb = pc_data_interp
_video_db_tar, _video_db_future_tar, _video_db_future_input_tar, \
_video_db_oth,_video_db_future_oth,_video_db_future_input_oth = util.get_data(datadb,pick_user=True,num_user=47)  #ski trace has 40 users



_video_db_future_tar = _video_db_future_tar.reshape((_video_db_tar.shape[0],_video_db_tar.shape[1],cfg.fps,3))
_video_db_future_input_tar = _video_db_future_input_tar.reshape((_video_db_tar.shape[0],_video_db_tar.shape[1],cfg.fps,3))
_video_db_future_oth = _reshape_others_data(_video_db_future_oth)
select_k_neighbours = True
if select_k_neighbours:
    _video_db_future_oth = util.get_random_k_other_users(_video_db_future_oth,47,34)




# assert _video_db_future_tar.shape==(2040, 10, 30, 3)
# chunks = 2040/40 #predicted 51 seconds, stride=1

# ...
# build the graph using given_others_gt_mean_var_seq2seq.py
# ...
model.load_weights('fctar_seqseq_mlpmixing_shanghai_traintest_split_predmeanvar_Aug9_epoch12-0.0903.h5')

# test....





### build a simple FoV reliability predictor
# given the last 

# baseline
def exponential_weight(temp):
    # print(len(temp))
    weighted=0
    denorm=0
    for time_ind in range(len(temp)):
        print(temp[time_ind])
        weighted +=np.exp(-time_ind)*(np.mean(temp[time_ind])) #this second's mean acc
        denorm += np.exp(-time_ind)
    return weighted/denorm

# baseline
def given_10_gt_history_fov_acc(history):
    """history""" #(300,)(270,)(240,)(210,)(180,)(150,)(120,)(90,)(60,)(30,)
    future_fovacc_prediction = [0]*10
    for future_ind in range(10):
        print(future_ind)
        temp = []
        for ii in range(len(history)-future_ind-1,-1,-1):
            temp.append(history[ii][future_ind:(future_ind+1)]) #from bottom to up

        future_fovacc_prediction[future_ind] = exponential_weight(temp)
    return future_fovacc_prediction

def get_mean_in_sec(frame_hitrate):
    second_mean = []
    for ii in range(0,int(len(frame_hitrate)/fps)):
        second_mean.append(frame_hitrate[ii*fps:(ii+1)*fps].mean())
    return second_mean



fig = plt.figure()
ax = fig.subplots(2,1)
fps = 30
for user_ind in range(num_user):
    # use the last 10 seconds' gt hitrate acc to predict the future 10-sec fov curve (1*10 vector)
    for start_time_index in range(300,hitrate2[user_ind].shape[0]-10,1):
        history = []
        for time_ind in range(10):
            temp = hitrate2[user_ind][start_time_index+time_ind,:300-fps*time_ind]
            temp = get_mean_in_sec(temp) #only use second mean
            history.append(temp) ##from up to bottom
        future_fovacc_prediction = given_10_gt_history_fov_acc(history)
        future_fovacc_gt = hitrate2[user_ind][start_time_index+10]

        ax[0].plot(future_fovacc_prediction)
        ax[0].set_ylim([0.5,1.1])
        ax[1].plot(future_fovacc_gt)
        ax[1].set_ylim([0.5,1.1])
        plt.draw()
        plt.show()
        pdb.set_trace()
        ax[0].cla()
        ax[1].cla()






# Naive MLP: flatten history, output 1*10 vector to match the future_fovacc_gt
def flatten_history(history):
    temp = []
    for ii in range(len(history)):
        temp = temp + list(history[ii])
    return temp



# create model
model = Sequential()
model.add(Dense(128, input_dim=55, kernel_initializer='normal', activation='relu'))
model.add(Dense(128, input_dim=128, kernel_initializer='normal', activation='relu'))
model.add(Dense(64, input_dim=128, kernel_initializer='normal', activation='relu'))
model.add(Dense(64, input_dim=10, kernel_initializer='normal', activation='relu'))
model.compile(loss='mean_squared_error', optimizer='adam')


flatten_history(history)




















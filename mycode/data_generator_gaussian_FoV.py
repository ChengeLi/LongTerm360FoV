"""
use method similar to ./saliency/fovCentroidCropping.py
create a distorted FoV on equirectangular, summing up
"""
from mycode.config import cfg
import mycode.data_generator_including_saliency as sal_generator
import mycode.utility as util
import numpy as np
from mycode.dataIO import xyz2thetaphi
import math
import pdb
import scipy as spy 
from random import shuffle
import _pickle as pickle

fps = cfg.fps
target_user_only = cfg.target_user_only
use_larger_batch_size = False


def get_theta_phi_array(per_video_db):
    # per_video_db_tar.shape=(num_chunks,10,90)
    # per_video_db_oth.shape=(num_users-1,num_chunks,10,90)
    if len(per_video_db.shape)==4:#others
        theta,phi = xyz2thetaphi(per_video_db[:,:,:,0::3],per_video_db[:,:,:,1::3],per_video_db[:,:,:,2::3])
        labelarray = np.zeros((theta.shape[0],theta.shape[1],theta.shape[2]*fps,2))
        for user_ind in range(per_video_db.shape[0]):
            labelarray[user_ind,:] = get_theta_phi_array_per_user(theta[user_ind,:],phi[user_ind,:])

    else:#tar user
        theta,phi = xyz2thetaphi(per_video_db[:,:,0::3],per_video_db[:,:,1::3],per_video_db[:,:,2::3])
        labelarray = get_theta_phi_array_per_user(theta,phi)
    return labelarray

def get_theta_phi_array_per_user(theta,phi):
    # second-level theta_phi_array
    # labelarray = np.zeros((inputarray.shape[0],2))
    # for time_ind in range(phi.shape[0]):
    #     labelarray[time_ind,0] = np.mean(phi[time_ind,0,:])
    #     labelarray[time_ind,1] = np.mean(theta[time_ind,0,:])

    # labelarray[:,0] = labelarray[:,0]/np.pi
    # labelarray[:,1] = (labelarray[:,1]+np.pi)/2/np.pi


    # frame-level theta_phi_array
    labelarray = np.zeros((phi.shape[0],phi.shape[1]*fps,2))
    for time_ind in range(phi.shape[1]):
        labelarray[:,time_ind*fps:(time_ind+1)*fps,0] = phi[:,time_ind,:] #0-pi
        labelarray[:,time_ind*fps:(time_ind+1)*fps,1] = theta[:,time_ind,:] #-pi,pi

    labelarray[:,:,0] = labelarray[:,:,0]/np.pi
    labelarray[:,:,1] = (labelarray[:,:,1]+np.pi)/2/np.pi

    return labelarray

def crop_FoV_from_equirect(inputarray,labelarray,img_h=256,img_w=512):
    # originally inputarray is saliency maps: inputarray.shape=(21, 256, 512, 60)
    # labelarray labelarray.shape=(21,2): theta, phi in 0-1, 
    inputarraynew=inputarray.copy()
    shrink_factor = img_h/256
    for cccc in range(inputarray.shape[0]):
        #x,z= phi,theta
        x=labelarray[cccc,0] #phi: 0-1,  theta: 0-1
        z=labelarray[cccc,1]
        xd=x*img_h
        zd=z*img_w
        xi=int(xd)
        zi=int(zd)
        iterative=0
        row=int(xi-img_h/4+img_h)
        inputarraynew[cccc,:,:,:]=0
        while iterative<img_h/2: #vertical 90 degree
            row+=1
            row%=img_h
            rowx=row/float(img_h)
            # half_span = int(img_w/2/math.pi)#80 #what is this？
            half_span = int(img_w/6)  #horizontal span=2*1/6=1/3*360=120 degree
            longitude=int(half_span/(math.cos(math.pi*(abs(rowx-0.5)))+0.00001))
            if longitude>img_w/2-1:
                longitude=img_w/2-1
            zlow=zi-longitude
            zhigh=zi+longitude
            if zlow<0:
                rr2=inputarray[cccc,row:row+1,zlow+img_w:img_w,:].copy()
                inputarraynew[cccc,row:row+1,zlow+img_w:img_w,:]=rr2
                zlow=0
            if zhigh>img_w:
                rr2=inputarray[cccc,row:row+1,0:zhigh%img_w,:].copy()
                inputarraynew[cccc,row:row+1,0:zhigh%img_w,:]=rr2
                zhigh=img_w-1
            rr=inputarray[cccc,row:row+1,zlow:zhigh,:].copy()
            inputarraynew[cccc,row:row+1,zlow:zhigh,:]=rr
            iterative+=1
        X,Y=np.meshgrid(np.linspace(0,img_w-1,img_w),np.linspace(0,img_h-1,img_h));
        sigma=0.01+0.008*(math.pi*(abs(rowx-0.5))); #suppose that mux=muy=mu=0 and sigmax=sigmay=sigma
        sigma = sigma/shrink_factor
        # print(sigma)
        G=np.exp(-((X-zi)**2+(Y-xi)**2)/2.0*sigma**2)
        xoo=xi
        zoo=zi
        margin = int(img_w*0.3)#150 what is this？
        if zi+margin>img_w:
            zoo=zi-img_w
        if zi-margin<0:
            zoo=zi+img_w
        if zi+margin>img_w or zi-margin<0:
            G1=np.exp(-((X-zoo)**2+(Y-xoo)**2)/2.0*sigma**2)
        else:
            G1=0.00001*np.exp(-((X-zoo)**2+(Y-xoo)**2)/2.0*sigma**2)
        for channel in range(inputarraynew.shape[-1]):
            inputarraynew[cccc,:,:,channel]=inputarraynew[cccc,:,:,channel]*(G+G1)

    inputarraynew = inputarraynew.astype(np.float32, copy=False)
    inputarraynew=inputarraynew/np.max(inputarraynew)
    return inputarraynew


def get_gaussian_FoV(phi_theta_array):
    img_h=180
    img_w=360
    inputarray = np.ones((phi_theta_array.shape[0],img_h,img_w,1))
    inputarraynew =  crop_FoV_from_equirect(inputarray,phi_theta_array,img_h=img_h,img_w=img_w)
    inputarraynew = inputarraynew[:,0::10,0::10,:] #18x36
    return inputarraynew


def get_gaussianFoV_per_vid_per_target(per_video_db):
    phi_theta_array = get_theta_phi_array(per_video_db)
    phi_theta_array = phi_theta_array.reshape(-1,2)
    fov_per_video = get_gaussian_FoV(phi_theta_array)
    fov_per_video = fov_per_video.reshape(per_video_db.shape[0],per_video_db.shape[1],fps,18,36)
    fov_per_video = fov_per_video.transpose((0,1,3,4,2))
    return fov_per_video


def get_gaussianFoV_per_vid_per_target_giventhetaphi(phi_theta_array):
    num_user = phi_theta_array.shape[0]
    num_sec = phi_theta_array.shape[1]//fps
    phi_theta_array = phi_theta_array.reshape(-1,2)
    fov_per_video = get_gaussian_FoV(phi_theta_array)
    fov_per_video = fov_per_video.reshape(num_user,num_sec,fps,18,36)
    fov_per_video = fov_per_video.transpose((0,1,3,4,2))
    return fov_per_video





def blur_head_direction_equirect(inputarray,labelarray,img_h=256,img_w=512):
    # originally inputarray is saliency maps: inputarray.shape=(21, 256, 512, 60)
    # labelarray labelarray.shape=(21,2): theta, phi in 0-1, 
    inputarraynew=inputarray.copy()
    shrink_factor = img_h/256
    for cccc in range(inputarray.shape[0]):
        #x,z= phi,theta
        x=labelarray[cccc,0] #phi: 0-1,  theta: 0-1
        z=labelarray[cccc,1]
        xi=int(x*img_h) #phi center
        zi=int(z*img_w) #theta center
        blur_h = 5/18*img_h
        row=int(xi-blur_h/2)
        inputarraynew[cccc,:,:,:]=0
        iterative=0
        while iterative<blur_h:
            if row<=0:
                row=0 #just ignore the upper region over the north pole. no extending the whole pole area.
            row+=1
            # row%=img_h
            rowx=row/float(img_h)
            # half_span = int(img_w/2/math.pi) #use radius as half of the horizontal span, whole span w/np.pi ~115 degree
            half_span = 0.5*(5/36)*img_w
            longitude=int(half_span/(math.cos(math.pi*(abs(rowx-0.5)))+0.00001)) #half of the horizontal span: i.e. r/cos(phi)
            if longitude>=img_h-1:
                longitude=img_h-1
            zlow=zi-longitude
            zhigh=zi+longitude
            if zlow<0:
                rr2=inputarray[cccc,row:row+1,zlow+img_w:img_w,:].copy()
                inputarraynew[cccc,row:row+1,zlow+img_w:img_w,:]=rr2
                zlow=0
            if zhigh>img_w:
                rr2=inputarray[cccc,row:row+1,0:zhigh%img_w,:].copy()
                inputarraynew[cccc,row:row+1,0:zhigh%img_w,:]=rr2
                zhigh=img_w-1
            rr=inputarray[cccc,row:row+1,zlow:zhigh,:].copy()
            inputarraynew[cccc,row:row+1,zlow:zhigh,:]=rr
            iterative+=1
        X,Y=np.meshgrid(np.linspace(0,img_w-1,img_w),np.linspace(0,img_h-1,img_h));
        sigma=0.05+0.008*(math.pi*(abs(rowx-0.5)));#actually it's 1/sigma
        sigma = sigma/shrink_factor
        # print(sigma)
        G=np.exp(-((X-zi)**2+(Y-xi)**2)/2.0*sigma**2)
        xoo=xi
        zoo=zi
        margin = int(img_w*0.3)#150 what is this？
        if zi+margin>img_w:
            zoo=zi-img_w #OOB??
        if zi-margin<0:
            zoo=zi+img_w  #OOB?
        if zi+margin>img_w or zi-margin<0:
            G1=np.exp(-((X-zoo)**2+(Y-xoo)**2)/2.0*sigma**2)
        else:
            G1=0.00001*np.exp(-((X-zoo)**2+(Y-xoo)**2)/2.0*sigma**2)
        for channel in range(inputarraynew.shape[-1]):
            inputarraynew[cccc,:,:,channel]=inputarraynew[cccc,:,:,channel]*(G+G1)
        # plt.imshow(inputarraynew[cccc,:,:,0])
        # plt.show()
        # pdb.set_trace()

    inputarraynew = inputarraynew.astype(np.float32, copy=False)
    inputarraynew=inputarraynew/np.max(inputarraynew)
    return inputarraynew


def get_head_direction(phi_theta_array):
    img_h=180
    img_w=360
    inputarray = np.ones((phi_theta_array.shape[0],img_h,img_w,1))
    inputarraynew =  blur_head_direction_equirect(inputarray,phi_theta_array,img_h=img_h,img_w=img_w)
    inputarraynew = inputarraynew[:,0::10,0::10,:] #18x36
    return inputarraynew


def get_headdirection_per_vid_per_target_giventhetaphi(phi_theta_array):
    num_user = phi_theta_array.shape[0]
    num_sec = phi_theta_array.shape[1]//fps
    phi_theta_array = phi_theta_array.reshape(-1,2)
    head_per_video = get_head_direction(phi_theta_array)
    head_per_video = head_per_video.reshape(num_user,num_sec,fps,18,36)
    head_per_video = head_per_video.transpose((0,1,3,4,2))
    return head_per_video









def heatmap_sum(heat_frame):
    """summing all heatmaps in one second"""
    if len(heat_frame.shape)==5:
        heat_sec = np.sum(heat_frame,axis=-1)[:,:,:,:,np.newaxis]
    elif len(heat_frame.shape)==6: #for others
        heat_sec = np.sum(heat_frame,axis=-1)[:,:,:,:,:,np.newaxis]
    # heat_sec = heat_sec/np.max(heat_sec)
    heat_sec = normalize_to_distribution(heat_sec)
    return heat_sec

def normalize_to_distribution(data):
    # instead of normalize s.t. the largest val=1, normalize the sum to be 1
    for ii in range(data.shape[0]):
        for jj in range(data.shape[1]):
            data[ii,jj,:] = data[ii,jj,:]/np.sum(data[ii,jj,:])
    return data

def _prepare_data_gaussian_FoV(fov_tar, fov_future_tar, fov_future_input_tar,
                            fov_oth, fov_future_oth, fov_future_input_oth,phase='train'):
    # fov_tar.shape=N*10*19440 
    # fov_oth.shape=33*N*10*19440
    max_encoder_seq_length = cfg.running_length
    num_chunks = fov_tar.shape[0]
    num_other_user = fov_oth.shape[0]

    if cfg.process_in_seconds:
        num_channel = fps
    else:
        num_channel = 1
    fov_tar = fov_tar.reshape(num_chunks,max_encoder_seq_length,18,36,num_channel)
    fov_future_tar = fov_future_tar.reshape(num_chunks,max_encoder_seq_length,18,36,num_channel)
    fov_future_input_tar = fov_future_input_tar.reshape(num_chunks,max_encoder_seq_length,18,36,num_channel)
    fov_oth = fov_oth.reshape(num_other_user,num_chunks,max_encoder_seq_length,18,36,num_channel)
    # fov_future_oth = fov_future_oth.reshape(num_other_user,num_chunks,max_encoder_seq_length,18,36,num_channel)
    # fov_future_input_oth = fov_future_input_oth.reshape(num_other_user,num_chunks,max_encoder_seq_length,18,36,num_channel)

    if cfg.use_heat_sum:
        encoder_input_data = heatmap_sum(fov_tar)
        decoder_target_data = heatmap_sum(fov_future_tar)
        decoder_input_data = heatmap_sum(fov_future_input_tar[:,0,:,:,:][:,np.newaxis,:])
        others_fut_input_data = heatmap_sum(fov_oth)

    else:
        encoder_input_data = fov_tar
        decoder_target_data = fov_future_tar
        decoder_input_data = fov_future_input_tar[:,0,:,:,:][:,np.newaxis,:]
        others_fut_input_data = fov_oth


    others_fut_input_data = others_fut_input_data.transpose(1,2,3,4,0,5) #(1, 10, 18, 36, 33, 30)
    others_fut_input_data = others_fut_input_data.reshape((others_fut_input_data.shape[0],
                                                            others_fut_input_data.shape[1],
                                                            others_fut_input_data.shape[2],
                                                            others_fut_input_data.shape[3],-1))

    if target_user_only:
        return encoder_input_data,decoder_input_data,decoder_target_data
    else:
        return encoder_input_data,others_fut_input_data,decoder_input_data,decoder_target_data


def crop_saliency(future_saliency,encoder_input_data):
    # using the generated gaussian FoV to mask the saliency map
    # encoder_input_data.shape = (num_chunks, 10, 18, 36, 1)
    assert encoder_input_data.shape[0]==future_saliency.shape[0]
    cropped_saliency = np.zeros_like(future_saliency)
    for ii in range(future_saliency.shape[0]):
        last_known_fov = encoder_input_data[ii,-1,:,:,0]
        for jj in range(future_saliency.shape[1]):
            for chnn in range(future_saliency.shape[-1]): #num_channel
                cropped_saliency[ii,jj,:,:,chnn] = future_saliency[ii,jj,:,:,chnn]*last_known_fov
    return cropped_saliency


def generator_train_GaussianFoV(datadb,phase='train'):
    # need to Generate Gaussian FoV on the fly
    max_encoder_seq_length = cfg.running_length
    num_encoder_tokens = 3*fps
    batch_size = cfg.batch_size
    stride=cfg.data_chunk_stride

    if  phase in ['val','test']:
        stride=cfg.data_chunk_stride #test stride=10
        datadb_keys_longerthan20 = sal_generator.test_val_videoind
    else:
        datadb_keys_longerthan20 = sal_generator.train_video_ind
    ii=0
    thetaphi_allvideo = pickle.load(open('../../data/shanghai_dataset_theta_phi.p','rb'),encoding='latin1')

    # from collections import OrderedDict
    # phi_theta_array_dict=OrderedDict()
    for ii in range(len(datadb_keys_longerthan20)):
        _video_ind = datadb_keys_longerthan20[ii%len(datadb_keys_longerthan20)]
    # for _video_ind in thetaphi_allvideo.keys(): #all keys
        print('_video_ind',_video_ind)

        # method 1: directly read theta,phi to generate heatmap
        thetaphi = thetaphi_allvideo[_video_ind]
        phi_theta_array = np.stack([thetaphi['latitude']/np.pi,(thetaphi['longitude']+np.pi)/2./np.pi],axis=-1) #norm to 0-1
        phi_theta_array = util.cut_head_or_tail_less_than_1sec(phi_theta_array)
        # phi_theta_array_dict[_video_ind]=phi_theta_array

    # pickle.dump(phi_theta_array_dict,open('shanghaitech_phi_theta_array_dict_test.p','wb'))

        gaussian_fov_db = get_gaussianFoV_per_vid_per_target_giventhetaphi(phi_theta_array)
        util.save2hdf5('../../data/gaussian_fov_db_directlythetaphi.h5',_video_ind, gaussian_fov_db)

        # method 2: convert xyz back to theta to generate heatmap (result==method1)
        # per_video_db = np.stack((datadb[_video_ind]['x'],datadb[_video_ind]['y'],datadb[_video_ind]['z']),axis=-1)
        # per_video_db = util.cut_head_or_tail_less_than_1sec(per_video_db)
        # per_video_db = np.reshape(per_video_db,(per_video_db.shape[0],per_video_db.shape[1]//fps,num_encoder_tokens))
        # if per_video_db.shape[1]<max_encoder_seq_length*2:
        #     print('video %s only has %d seconds. skip...'%(_video_ind,per_video_db.shape[1]))
        #     continue

        # gaussian_fov_db2 = get_gaussianFoV_per_vid_per_target(per_video_db)


        #### generate head direction db
        head_db = get_headdirection_per_vid_per_target_giventhetaphi(phi_theta_array)
        util.save2hdf5('../../data/head_direction_db_directlythetaphi.h5',_video_ind, head_db)




        # gaussian_fov_db = gaussian_fov_db.reshape(gaussian_fov_db.shape[0],gaussian_fov_db.shape[1],-1)
        # for _target_user in range(datadb[_video_ind]['x'].shape[0]):
        #     # print('target user:',_target_user)
        #     fov_tar, fov_future_tar, fov_future_input_tar,\
        #         fov_oth, fov_future_oth, fov_future_input_oth = sal_generator.get_data_per_vid_per_target(gaussian_fov_db,_target_user,num_user=34,stride=stride)

        #     # encoder_input_data,others_fut_input_data,decoder_input_data,decoder_target_data = \
        #     #                 _prepare_data_gaussian_FoV(fov_tar, fov_future_tar, fov_future_input_tar,
        #     #                 fov_oth, fov_future_oth, fov_future_input_oth,phase)
        #     encoder_input_data,decoder_input_data,decoder_target_data= \
        #                     _prepare_data_gaussian_FoV(fov_tar, fov_future_tar, fov_future_input_tar,
        #                     fov_oth, fov_future_oth, fov_future_input_oth,phase)

        #     local_counter=0
        #     while local_counter<encoder_input_data.shape[0]:
        #         # print(local_counter,local_counter+batch_size)
        #         # print(encoder_input_data[local_counter:local_counter+batch_size,:].shape)
        #         # yield [encoder_input_data[local_counter:local_counter+batch_size,:],\
        #         #       others_fut_input_data[local_counter:local_counter+batch_size,:],\
        #         #       decoder_input_data[local_counter:local_counter+batch_size,:]],\
        #         #       decoder_target_data[local_counter:local_counter+batch_size,:]  
        #         yield [encoder_input_data[local_counter:local_counter+batch_size,:],\
        #                 decoder_input_data[local_counter:local_counter+batch_size,:]],\
        #                 decoder_target_data[local_counter:local_counter+batch_size,:]                    
        #         local_counter+=batch_size

        # ii+=1#next video


def generator_train_GaussianFoV2(datadb,segment_index_tar=None,segment_index_tar_future=None,phase='train'):
    # given shanghaitech_gauss_tile.hdf5 for whole dataset, don't have to generate Gaussian FoV on the fly
    max_encoder_seq_length = cfg.running_length
    num_encoder_tokens = 3*fps
    batch_size = cfg.batch_size
    stride=cfg.data_chunk_stride

    if  phase in ['val','test']:
        if phase=='test':
            stride=10
        datadb_keys_longerthan20 = sal_generator.test_val_videoind
        if cfg.thu_tag!='':
            datadb_keys_longerthan20=['5','7','9','15']
    else:
        datadb_keys_longerthan20 = sal_generator.train_video_ind
        if cfg.thu_tag!='':
            datadb_keys_longerthan20=['0','1','2','3','4','6','8','10','11','12','13','14','16','17']
            


    if cfg.shuffle_data:
        shuffle(datadb_keys_longerthan20)

    video_ii=0
    while True:
        _video_ind = datadb_keys_longerthan20[video_ii%len(datadb_keys_longerthan20)]
        print('_video_ind=',_video_ind)
        if cfg.thu_tag!='':
            gaussian_fov_db = util.load_h5('../../data/tsinghua_heatmap_wx/tsingua_gauss_tile'+str(_video_ind)+'.hdf5','gauss_tile')#THU fov data from weixi
        else:
            # gaussian_fov_db = util.load_h5('../../data/shanghaitech_gauss_tile.hdf5',key=_video_ind)
            # gaussian_fov_db = util.load_h5('../../data/gaussian_fov_db.h5',key=_video_ind) #not matched with original theta, phi
            # gaussian_fov_db = util.load_h5('../../data/shanghai_heatmap_verticalGau_wz/shanghaitech_gauss_tile'+str(_video_ind)+'.hdf5',key='gauss_tile') #from weixi, buggy
            gaussian_fov_db = util.load_h5('../../data/gaussian_fov_db_directlythetaphi.h5',key=_video_ind) 
            # gaussian_fov_db = util.load_h5('../../data/gaussian_fov_db_directlythetaphi_shifted.h5',key=_video_ind) 
            # gaussian_fov_db = util.load_h5('../../data/head_direction_db_directlythetaphi.h5',key=_video_ind) 
            # gaussian_fov_db = util.load_h5('../../data/shanghai_blurred_gaze_wz/shanghai_gauss_tile'+str(_video_ind)+'.hdf5',key='gauss_tile')
            # gaussian_fov_db = gaussian_fov_db[:,:,0::2,0::2,:]#downsample to 18*36 for now

        if cfg.process_in_seconds:
            gaussian_fov_db = gaussian_fov_db.reshape(gaussian_fov_db.shape[0],gaussian_fov_db.shape[1],-1) #(34, N_seconds, 19440): 19400=18*36*30
        else:
            gaussian_fov_db = gaussian_fov_db.transpose((0,2,3,1,4))
            gaussian_fov_db = gaussian_fov_db.reshape(gaussian_fov_db.shape[0],gaussian_fov_db.shape[1],gaussian_fov_db.shape[2],-1) #(34,18,36,N_frames)
            gaussian_fov_db = gaussian_fov_db.transpose((0,3,1,2))
            gaussian_fov_db = gaussian_fov_db.reshape(gaussian_fov_db.shape[0],gaussian_fov_db.shape[1],-1) #(34, N_frames, 648) 648=18*36
            if cfg.skip_frame_gap>0:
                gaussian_fov_db = gaussian_fov_db[:,0::cfg.skip_frame_gap,:]

        # print(ii%len(datadb_keys_longerthan20),'_video_ind',_video_ind)
        for _target_user in range(datadb[_video_ind]['x'].shape[0]):
            # print('target user:',_target_user)
            fov_tar, fov_future_tar, fov_future_input_tar,\
            fov_oth, fov_future_oth, fov_future_input_oth = sal_generator.get_data_per_vid_per_target(gaussian_fov_db,_target_user,num_user=34,stride=stride)

            if _target_user==0 and cfg.use_saliency:
                #only need to run 1 time
                ###visual saliency
                num_chunks = fov_tar.shape[0]
                real_num_users = gaussian_fov_db.shape[0]
                if len(segment_index_tar[_video_ind])/real_num_users != num_chunks:
                    pdb.set_trace()
                # saliency for the whole video
                saliency_data = util.load_h5('./360video/temp/saliency_input_pad0_fulllen/input'+_video_ind+'.hdf5','input')
                # saliency for the prediction time
                future_saliency = saliency_data[segment_index_tar_future[_video_ind][:num_chunks],:].copy()
                del saliency_data #to save memory
                future_saliency = 1.*future_saliency/future_saliency.max() 
                #reduce to 2 channels
                future_saliency = np.concatenate((np.mean(future_saliency[:,:,:,:,:30],axis=-1)[:,:,:,:,np.newaxis],
                                                 np.mean(future_saliency[:,:,:,:,30:],axis=-1)[:,:,:,:,np.newaxis]),axis=-1)

                # downsample
                # future_saliency = future_saliency[:,:,::4,::4,:]
                future_saliency_small = np.zeros((num_chunks,cfg.predict_step,18,36,2))
                for ii in range(num_chunks):
                    for jj in range(cfg.predict_step):
                        for chnn in range(future_saliency_small.shape[-1]): #num_channel
                            future_saliency_small[ii,jj,:,:,chnn] = spy.misc.imresize(future_saliency[ii,jj,:,:,chnn],size=(18,36),interp='bicubic')
                            # normalize each channel into a 2d distribution
                            future_saliency_small[ii,jj,:,:,chnn] = future_saliency_small[ii,jj,:,:,chnn]/np.sum(future_saliency_small[ii,jj,:,:,chnn])

                if cfg.only_use_sal_local:
                    #1 channel
                    future_saliency = future_saliency_small[:,:,:,:,1][:,:,:,:,np.newaxis]
                else:
                    #2 channels
                    future_saliency = future_saliency_small


            if target_user_only:
                encoder_input_data,decoder_input_data,decoder_target_data= \
                                _prepare_data_gaussian_FoV(fov_tar, fov_future_tar, fov_future_input_tar,
                                fov_oth, fov_future_oth, fov_future_input_oth,phase)
            else:
                encoder_input_data,others_fut_input_data,decoder_input_data,decoder_target_data = \
                                _prepare_data_gaussian_FoV(fov_tar, fov_future_tar, fov_future_input_tar,
                                                    fov_oth, fov_future_oth, fov_future_input_oth,phase)
            if cfg.use_saliency:
                if cfg.use_cropped_sal:
                    target_user_saliency = crop_saliency(future_saliency,encoder_input_data)
                else:
                    target_user_saliency = future_saliency #all same for all users
                # normalize the (cropped) target user saliency, otherwise value is too small
                for ii in range(target_user_saliency.shape[0]):
                    for jj in range(target_user_saliency.shape[1]):
                        for chnn in range(target_user_saliency.shape[-1]): #num_channel
                            target_user_saliency[ii,jj,:,:,chnn] = target_user_saliency[ii,jj,:,:,chnn]/np.sum(target_user_saliency[ii,jj,:,:,chnn])


            #stack all users' data together to allow larger batch size!
            if use_larger_batch_size:
                if _target_user==0:
                    across_user_encoder_input_data = encoder_input_data
                    across_user_decoder_input_data = decoder_input_data
                    across_user_decoder_target_data = decoder_target_data
                    if not target_user_only:
                        across_user_others_fut_input_data = others_fut_input_data
                else:
                    across_user_encoder_input_data = np.vstack([across_user_encoder_input_data,encoder_input_data])
                    across_user_decoder_input_data = np.vstack([across_user_decoder_input_data,decoder_input_data])
                    across_user_decoder_target_data = np.vstack([across_user_decoder_target_data,decoder_target_data])
                    if not target_user_only:
                        across_user_others_fut_input_data = np.vstack([across_user_others_fut_input_data,others_fut_input_data])

                local_counter=0
                while local_counter<across_user_encoder_input_data.shape[0]:
                    # print('local_counter=',local_counter)
                    # print(across_user_encoder_input_data[local_counter:local_counter+batch_size,:].shape)
                    if not target_user_only:
                        if cfg.use_saliency:
                            yield [across_user_encoder_input_data[local_counter:local_counter+batch_size,:],\
                                  across_user_others_fut_input_data[local_counter:local_counter+batch_size,:],\
                                  across_user_decoder_input_data[local_counter:local_counter+batch_size,:],\
                                  target_user_saliency[local_counter:local_counter+batch_size,:]],\
                                  across_user_decoder_target_data[local_counter:local_counter+batch_size,:]
                        else:
                            yield [across_user_encoder_input_data[local_counter:local_counter+batch_size,:],\
                                  across_user_others_fut_input_data[local_counter:local_counter+batch_size,:],\
                                  across_user_decoder_input_data[local_counter:local_counter+batch_size,:]],\
                                  across_user_decoder_target_data[local_counter:local_counter+batch_size,:]
                    else:
                        if cfg.use_saliency:
                            yield [across_user_encoder_input_data[local_counter:local_counter+batch_size,:],\
                                    across_user_decoder_input_data[local_counter:local_counter+batch_size,:],
                                    target_user_saliency[local_counter:local_counter+batch_size,:]],\
                                    across_user_decoder_target_data[local_counter:local_counter+batch_size,:]  
                        else:
                            yield [across_user_encoder_input_data[local_counter:local_counter+batch_size,:],\
                                    across_user_decoder_input_data[local_counter:local_counter+batch_size,:]],\
                                    across_user_decoder_target_data[local_counter:local_counter+batch_size,:]                  
                    real_batch_size = across_user_encoder_input_data[local_counter:local_counter+batch_size,:].shape[0]
                    local_counter+=real_batch_size
            else: #batch_size=1
                local_counter=0
                while local_counter<encoder_input_data.shape[0]:
                    # print(local_counter)
                    # print(encoder_input_data[local_counter:local_counter+batch_size,:].shape)
                    if not target_user_only:
                        if cfg.use_saliency:
                            if cfg.include_time_ind:
                                time_ind_input_data = np.repeat(np.repeat(np.arange(cfg.predict_step)[:,np.newaxis],1, axis=1)[np.newaxis,:],
                                                             encoder_input_data[local_counter:local_counter+batch_size,:].shape[0],axis=0)
                                yield [encoder_input_data[local_counter:local_counter+batch_size,:],\
                                      others_fut_input_data[local_counter:local_counter+batch_size,:],\
                                      decoder_input_data[local_counter:local_counter+batch_size,:],\
                                      target_user_saliency[local_counter:local_counter+batch_size,:],
                                      time_ind_input_data],\
                                      decoder_target_data[local_counter:local_counter+batch_size,:]
                            else:
                                yield [encoder_input_data[local_counter:local_counter+batch_size,:],\
                                      others_fut_input_data[local_counter:local_counter+batch_size,:],\
                                      decoder_input_data[local_counter:local_counter+batch_size,:],\
                                      target_user_saliency[local_counter:local_counter+batch_size,:]],\
                                      decoder_target_data[local_counter:local_counter+batch_size,:]
                        else:
                            yield [encoder_input_data[local_counter:local_counter+batch_size,:],\
                                  others_fut_input_data[local_counter:local_counter+batch_size,:],\
                                  decoder_input_data[local_counter:local_counter+batch_size,:]],\
                                  decoder_target_data[local_counter:local_counter+batch_size,:]
                    else:
                        if cfg.use_saliency:
                            yield [encoder_input_data[local_counter:local_counter+batch_size,:],\
                                    decoder_input_data[local_counter:local_counter+batch_size,:],
                                    target_user_saliency[local_counter:local_counter+batch_size,:]],\
                                    decoder_target_data[local_counter:local_counter+batch_size,:]  
                        else:
                            yield [encoder_input_data[local_counter:local_counter+batch_size,:],\
                                    decoder_input_data[local_counter:local_counter+batch_size,:]],\
                                    decoder_target_data[local_counter:local_counter+batch_size,:]                  
                    real_batch_size = encoder_input_data[local_counter:local_counter+batch_size,:].shape[0]
                    local_counter+=real_batch_size

        video_ii+=1#next video





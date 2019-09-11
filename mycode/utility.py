import tensorflow as tf
from mycode.config import cfg
import numpy as np
import os
from keras.layers import Lambda
from keras.utils import to_categorical
import pdb
import random
import h5py
from collections import OrderedDict

batch_size = cfg.batch_size
fps = cfg.fps
def _reshape_batchsize(tensor):
    if cfg.predict_len>1:
        return tf.reshape(tensor,[batch_size,cfg.predict_len,1])
    else:
        return tf.reshape(tensor,[batch_size,1])


# def _filter_stuffed_fake_future_val(data_further):
#     tf.equal(data_further,1.123)
#     # TODO

def tf_get_gt_target_xyz(y):
    """get gt mean var"""
    target_x = y[:,:,0::3]
    target_y = y[:,:,1::3]
    target_z = y[:,:,2::3]
    gt_mean_x, gt_var_x = tf.nn.moments(target_x, axes=[-1])
    gt_mean_y, gt_var_y = tf.nn.moments(target_y, axes=[-1])
    gt_mean_z, gt_var_z = tf.nn.moments(target_z, axes=[-1])

    target = (_reshape_batchsize(gt_mean_x),_reshape_batchsize(gt_mean_y),_reshape_batchsize(gt_mean_z),
                _reshape_batchsize(gt_var_x),_reshape_batchsize(gt_var_y),_reshape_batchsize(gt_var_z))
    return target

def change_input_format(batch_x,data_dim=3):
    """change [xyzxyz...]90*1 into [xxx...yyy...zzz..]"""
    assert batch_x.shape[0]==batch_size
    assert batch_x.shape[2]==data_dim
    temp = np.concatenate((batch_x[:,:,0::3],batch_x[:,:,1::3],batch_x[:,:,2::3]),axis=-1)
    return temp


def get_gt_target_xyz_pop(y):
    """get pop mean var"""
    target_x = tf.contrib.layers.flatten(y[:,:,:fps*1,0])
    target_y = tf.contrib.layers.flatten(y[:,:,:fps*1,1])
    target_z = tf.contrib.layers.flatten(y[:,:,:fps*1,2])
    pop_mean_x, pop_var_x = tf.nn.moments(target_x, axes=[1])
    pop_mean_y, pop_var_y = tf.nn.moments(target_y, axes=[1])
    pop_mean_z, pop_var_z = tf.nn.moments(target_z, axes=[1])

    target = (_reshape_batchsize(pop_mean_x),_reshape_batchsize(pop_mean_y),_reshape_batchsize(pop_mean_z),
            _reshape_batchsize(pop_var_x),_reshape_batchsize(pop_var_y),_reshape_batchsize(pop_var_z))
    return target

    
def get_gt_target_phi_theta(y):
    """get gt mean var"""
    target_phi = y[:,0,0::3]
    target_theta = y[:,0,1::3]
    gt_mean_phi, gt_var_phi = tf.nn.moments(target_phi, axes=[1])
    gt_mean_theta, gt_var_theta = tf.nn.moments(target_theta, axes=[1])

    target = (_reshape_batchsize(gt_mean_phi),_reshape_batchsize(gt_mean_theta),
                _reshape_batchsize(gt_var_phi),_reshape_batchsize(gt_var_theta))
    return target



def generate_fake_batch_numpy(mu,var,batch_size):
    """generate new data for 1 second using predicted mean and variance"""
    # print('there are %s variance less than zero.'%str((var<0).sum()))
    var[var<0]=1e-3
    temp = []
    for ii in range(batch_size):
        temp.append(np.random.normal(mu[ii], np.sqrt(var[ii]), fps*1))
    return temp


def generate_fake_batch_tf(mu,var):
    """for tf"""
    temp = []
    batch_size = mu.shape[0].value
    for ii in range(batch_size):
        temp.append(tf.random_normal(np.array([fps,1]),mean=mu[ii], stddev=tf.sqrt(var[ii])))
    return temp


def generate_fake_batch_multivariate_normal_numpy(mu_theta,mu_phi,sigma_theta,sigma_phi,rho,batch_size):
    """generate new data for 1 second using predicted mean and covariance matrix using multivariate_normal"""
    # print('there are %s variance less than zero.'%str((var<0).sum()))
    # var[var<0]=1e-3
    if cfg.process_in_seconds:
        size = fps*1
    else:
        size = [int(0.5*cfg.running_length),2]#for theta,phi
    covmat = np.array([[sigma_theta**2,rho*sigma_theta*sigma_phi],[rho*sigma_theta*sigma_phi,sigma_phi**2]])
    temp = []
    for ii in range(batch_size):
        mu = [mu_theta[ii][0],mu_phi[ii][0]]
        temp.append(np.random.multivariate_normal(mu, cov=covmat[:,:,ii,0], size=size))
    temp = np.array(temp).reshape((batch_size,1,-1))
    return temp


def generate_fake_batch_mixture(mixture_pi,us,sigmas,rho):
    """using GMM"""
    mu_theta = us[:,0::2]
    mu_phi = us[:,1::2]
    sigma_theta = sigmas[:,0::2]
    sigma_phi = sigmas[:,1::2]
    covmat = np.array([[sigma_theta**2,rho*sigma_theta*sigma_phi],[rho*sigma_theta*sigma_phi,sigma_phi**2]])
    samples = []
    if cfg.process_in_seconds:
        size = fps*1
    else:
        size = int(0.5*cfg.running_length)
    for ii in range(batch_size):
        temp = np.zeros((size,2))
        for mixture_ind in range(mu_theta.shape[1]):
            temp+=mixture_pi[ii,mixture_ind]*np.random.multivariate_normal([mu_theta[ii,mixture_ind],mu_phi[ii,mixture_ind]], cov=covmat[:,:,ii,mixture_ind], size=size)
        samples.append(temp)
    return np.array(samples)


def sample_mixture(predictions):
    """randomly one sample from GMM"""
    if cfg.predict_eos:
        end_of_stroke,mixture_pi,us,sigmas,rho = predictions
        samples = np.zeros([batch_size, 1, 3], np.float32)
    else:
        mixture_pi,us,sigmas,rho = predictions
        samples = np.zeros([batch_size, 1, 2], np.float32)        
    mu_theta = us[:,0::2]
    mu_phi = us[:,1::2]
    sigma_theta = sigmas[:,0::2]
    sigma_phi = sigmas[:,1::2]
    r = np.random.rand()
    for ii in range(batch_size):
        accu = 0
        for m in range(mu_theta.shape[1]):
            # accu += mixture_pi[ii, m]
            # if accu > r:
            #     samples[ii, 0, 0:2] = np.random.multivariate_normal(
            #         [mu_theta[ii, m], mu_phi[ii, m]],
            #         [[np.square(sigma_theta[ii, m]), rho[ii, m] * sigma_theta[ii, m] * sigma_phi[ii, m]],
            #         [rho[ii, m] * sigma_theta[ii, m] * sigma_phi[ii, m], np.square(sigma_phi[ii, m])]])
            #     break

            convariance_mat = [[np.square(sigma_theta[ii, m]), rho[ii, m] * sigma_theta[ii, m] * sigma_phi[ii, m]],
                    [rho[ii, m] * sigma_theta[ii, m] * sigma_phi[ii, m], np.square(sigma_phi[ii, m])]]
            convariance_mat = _make_positive_semidefinite(convariance_mat)
            samples[ii, 0, 0:2] += mixture_pi[ii, m]*np.random.multivariate_normal(
                    [mu_theta[ii, m], mu_phi[ii, m]],
                     convariance_mat)
            if accu > r:
                break

        if cfg.predict_eos:
            # e = np.random.rand()
            # if e < end_of_stroke[ii]:
            #     samples[ii, 0, 2] = 1
            # else:
            #     samples[ii, 0, 2] = 0
            if end_of_stroke[ii]>0.7:
                samples[ii, 0, 2] = 1
            else:
                samples[ii, 0, 2] = 0

    return np.array(samples)




def sample_mixture_3D(predictions):
    """randomly one sample from 3D GMM"""
    pdb.set_trace()
    mixture_pi,us,sigmas,rho = predictions
    samples = np.zeros([batch_size, 1, 2], np.float32)        
    mu_theta = us[:,0::2]
    mu_phi = us[:,1::2]
    sigma_theta = sigmas[:,0::2]
    sigma_phi = sigmas[:,1::2]
    r = np.random.rand()
    for ii in range(batch_size):
        accu = 0
        for m in range(mu_theta.shape[1]):
            # accu += mixture_pi[ii, m]
            # if accu > r:
            #     samples[ii, 0, 0:2] = np.random.multivariate_normal(
            #         [mu_theta[ii, m], mu_phi[ii, m]],
            #         [[np.square(sigma_theta[ii, m]), rho[ii, m] * sigma_theta[ii, m] * sigma_phi[ii, m]],
            #         [rho[ii, m] * sigma_theta[ii, m] * sigma_phi[ii, m], np.square(sigma_phi[ii, m])]])
            #     break

            convariance_mat = [[np.square(sigma_theta[ii, m]), rho[ii, m] * sigma_theta[ii, m] * sigma_phi[ii, m]],
                    [rho[ii, m] * sigma_theta[ii, m] * sigma_phi[ii, m], np.square(sigma_phi[ii, m])]]
            convariance_mat = _make_positive_semidefinite(convariance_mat)
            samples[ii, 0, 0:2] += mixture_pi[ii, m]*np.random.multivariate_normal(
                    [mu_theta[ii, m], mu_phi[ii, m]],
                     convariance_mat)
            if accu > r:
                break

    return np.array(samples)



def _make_positive_semidefinite(convariance_mat):
    min_eig = np.min(np.real(np.linalg.eigvals(convariance_mat)))
    if min_eig < 0:
        print('need to make the covariance matrix SPD, min_eig =',min_eig)
        convariance_mat -= 10*min_eig * np.eye(*convariance_mat.shape)
    return convariance_mat


def snapshot(sess, epoch, saver,netname='',tag=''):
    output_dir = cfg.OUTPUT_DIR
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Store the model snapshot
    filename = netname+tag+ '_epoch_{:d}'.format(epoch) + '.ckpt'
    filename = os.path.join(output_dir, filename)
    saver.save(sess, filename)
    print('Wrote snapshot to: {:s}'.format(filename))


def split_into_u_var(pred):
    ux = tf.slice(pred,[0,0],[-1,1])
    uy = tf.slice(pred,[0,1],[-1,1])
    uz = tf.slice(pred,[0,2],[-1,1])
    # variance must >0
    varx = tf.abs(tf.slice(pred,[0,3],[-1,1]))
    vary = tf.abs(tf.slice(pred,[0,4],[-1,1]))
    varz = tf.abs(tf.slice(pred,[0,5],[-1,1]))
    return ux,uy,uz,varx,vary,varz





def slice_layer(dimension, start, end):
    # Crops (or slices) a Tensor on a given dimension from start to end
    # example : to crop tensor x[:, :, 5:10]
    # call slice(2, 5, 10) as you want to crop on the second dimension
    def func(x):
        if dimension == 0:
            return x[start: end]
        if dimension == 1:
            return x[:, start: end]
        if dimension == 2:
            return x[:, :, start: end]
        if dimension == 3:
            return x[:, :, :, start: end]
        if dimension == 4:
            return x[:, :, :, :, start: end]
    return Lambda(func)


def reshape2second_stacks(per_video_db,collapse_user=False,stride=cfg.data_chunk_stride,purelly_testing=cfg.purelly_testing):
    """reshape from N*T*90 into M*max_encoder_seq_length*90"""
    # split into chunks of max_encoder_seq_length seconds
    #per_video_db = np.array(np.split(per_video_db,per_video_db.shape[1]/max_encoder_seq_length,axis=1))
    max_encoder_seq_length = cfg.running_length
    # num_encoder_tokens = 3*fps
    num_encoder_tokens=per_video_db.shape[-1]
    assert per_video_db.shape[1]>=max_encoder_seq_length*2
    # if stride=2, shift by 5 2 seconds to make per_video_db_future 10 seconds away
    shift = max_encoder_seq_length//stride
    if purelly_testing:
        # append fake zero future as the tail
        per_video_db = np.concatenate((per_video_db,
                                    np.zeros((per_video_db.shape[0],shift,per_video_db.shape[2]))),axis=1)
    nrows = ((per_video_db.shape[1]-max_encoder_seq_length)//stride)+1
    new_idx = stride*np.arange(nrows)[:,None] + np.arange(max_encoder_seq_length)
    temp = np.zeros((new_idx.shape[0],per_video_db.shape[0],new_idx.shape[1],per_video_db.shape[2]))
    for i in range(new_idx.shape[0]):
        for j in range(new_idx.shape[1]):
            temp[i,:,j,:] = per_video_db[:,new_idx[i,j],:]
    per_video_db = temp

    # decoder target
    per_video_db_future = per_video_db[shift:,:,:,:]
    per_video_db = per_video_db[:-shift,:]#discard last part, since no future
    last_location = per_video_db[:,:,-1,:][:,:,np.newaxis,:]
    # decoder input
    per_video_db_future_input = np.concatenate((last_location,per_video_db_future[:,:,:-1,:]),axis=2)

    if collapse_user:
        #mix user and time
        #reshape, collapse user dimension
        per_video_db = np.reshape(per_video_db,(-1,max_encoder_seq_length,num_encoder_tokens))
        per_video_db_future = np.reshape(per_video_db_future,(-1,max_encoder_seq_length,num_encoder_tokens))
        per_video_db_future_input = np.reshape(per_video_db_future_input,(-1,max_encoder_seq_length,num_encoder_tokens))
    else:
        #keep the user dimension in dim 0
        #userful for other users' data
        per_video_db = per_video_db.transpose((1,0,2,3))
        per_video_db_future = per_video_db_future.transpose((1,0,2,3))
        per_video_db_future_input = per_video_db_future_input.transpose((1,0,2,3))
    return per_video_db,per_video_db_future,per_video_db_future_input


def cut_head_or_tail_less_than_1sec(per_video_db):
    if cfg.cut_data_head:
        #cut head to make dividable by seconds
        per_video_db = per_video_db[:,per_video_db.shape[1]-per_video_db.shape[1]//fps*fps:,:] 
    else: #cut tail
        per_video_db = per_video_db[:,:per_video_db.shape[1]//fps*fps,:] 
    return per_video_db


def cut_head_or_tail(per_video_db):
    max_encoder_seq_length = cfg.running_length
    if cfg.cut_data_head:
        #cut head to make dividable by TEN seconds
        per_video_db = per_video_db[:,per_video_db.shape[1]-per_video_db.shape[1]//fps//max_encoder_seq_length*fps*max_encoder_seq_length:,:] 
    else: #cut tail
        per_video_db = per_video_db[:,:per_video_db.shape[1]//fps//max_encoder_seq_length*fps*max_encoder_seq_length,:] 
    return per_video_db


def get_time_for_visual(datadb,stride=cfg.data_chunk_stride):
    """get the exact time for visual input
    The time must be matched for both visual and trj inputs."""
    max_encoder_seq_length = cfg.running_length
    num_encoder_tokens = 3*fps
    segment_index_tar = OrderedDict()
    temp =0
    for _video_ind in datadb.keys():
        segment_index_tar[_video_ind] = []
        num_user_this_vid = datadb[_video_ind]['x'].shape[0]
        
        per_video_db = np.stack((datadb[_video_ind]['x'],datadb[_video_ind]['y'],datadb[_video_ind]['z']),axis=-1)
        # per_video_db = cut_head_or_tail(per_video_db)
        per_video_db = cut_head_or_tail_less_than_1sec(per_video_db)
        per_video_db = np.reshape(per_video_db,(per_video_db.shape[0],per_video_db.shape[1]//fps,num_encoder_tokens))
        if per_video_db.shape[1]<max_encoder_seq_length*2:
            print('video %s only has %d seconds. skip in get_time_for_visual()...'%(_video_ind,per_video_db.shape[1]))
            continue
    
        nrows = ((per_video_db.shape[1]-max_encoder_seq_length)//stride)+1
        new_idx = stride*np.arange(nrows)[:,None] + np.arange(max_encoder_seq_length)
        shift = int(max_encoder_seq_length/cfg.data_chunk_stride)
        new_idx = new_idx[:-shift,:]#discard last part, since no future

        segment_index_tar[_video_ind] +=list(new_idx)*num_user_this_vid #record how many seconds are there in this video, in order to match the time with viusal input
        temp=temp+(new_idx.shape[0])*num_user_this_vid
    
    # pickle.dump(segment_index_tar,open('./cache/format4/test/stride1_cut_tail/sec_segment_index_tar_test.p','wb'))
    # pickle.dump(segment_index_tar,open('./cache/format4/train/stride1_cut_tail/sec_segment_index_tar_train.p','wb'))
    return segment_index_tar


def get_data(datadb,pick_user=False,num_user=48):
    max_encoder_seq_length = cfg.running_length
    num_encoder_tokens = 3*fps
    if not pick_user:
        ### concat all 9 videos and all users
        # don't distinguish users or videos during training
        _video_db = np.zeros((1,max_encoder_seq_length,num_encoder_tokens))
        _video_db_future = np.zeros((1,max_encoder_seq_length,num_encoder_tokens))
        _video_db_future_input = np.zeros((1,max_encoder_seq_length,num_encoder_tokens))
        for _video_ind in datadb.keys():
            per_video_db = np.stack((datadb[_video_ind]['x'],datadb[_video_ind]['y'],datadb[_video_ind]['z']),axis=-1)
            # per_video_db = cut_head_or_tail(per_video_db)
            per_video_db = cut_head_or_tail_less_than_1sec(per_video_db)
            per_video_db = np.reshape(per_video_db,(per_video_db.shape[0],per_video_db.shape[1]//fps,num_encoder_tokens))
            if per_video_db.shape[1]<max_encoder_seq_length*2:
                print('video %s only has %d seconds. skip...'%(_video_ind,per_video_db.shape[1]))
                continue
            per_video_db, per_video_db_future, per_video_db_future_input = reshape2second_stacks(per_video_db,collapse_user=True)
            print('_video_ind = ',_video_ind,'per_video_db.shape = ',per_video_db.shape)            
            _video_db = np.concatenate((_video_db, per_video_db),axis=0)
            _video_db_future = np.concatenate((_video_db_future, per_video_db_future),axis=0)
            _video_db_future_input = np.concatenate((_video_db_future_input, per_video_db_future_input),axis=0)
        
        if cfg.time_shift:
            return _video_db[1:,:-1,:], _video_db_future[1:,:,:], _video_db_future_input[1:,:,:]
        else:
            return _video_db[1:,:,:], _video_db_future[1:,:,:], _video_db_future_input[1:,:,:]
    else:
        _video_db_tar = np.zeros((1,max_encoder_seq_length,num_encoder_tokens))
        _video_db_future_tar = np.zeros((1,max_encoder_seq_length,num_encoder_tokens))
        _video_db_future_input_tar = np.zeros((1,max_encoder_seq_length,num_encoder_tokens))

        _video_db_oth = np.zeros((num_user-1,1,max_encoder_seq_length,num_encoder_tokens))
        _video_db_future_oth = np.zeros((num_user-1,1,max_encoder_seq_length,num_encoder_tokens))
        _video_db_future_input_oth = np.zeros((num_user-1,1,max_encoder_seq_length,num_encoder_tokens))

        for _video_ind in datadb.keys(): #every user will be selected as target user and roll over whole dataset
            print('_video_ind',_video_ind)
            for _target_user in range(datadb[_video_ind]['x'].shape[0]):
                print('target user:',_target_user)
                # for each video, pick out a target user, split target user and other users
                per_video_db = np.stack((datadb[_video_ind]['x'],datadb[_video_ind]['y'],datadb[_video_ind]['z']),axis=-1)
                # per_video_db = cut_head_or_tail(per_video_db)
                per_video_db = cut_head_or_tail_less_than_1sec(per_video_db)
                per_video_db = np.reshape(per_video_db,(per_video_db.shape[0],per_video_db.shape[1]//fps,num_encoder_tokens))
                if per_video_db.shape[1]<max_encoder_seq_length*2:
                    print('video %s only has %d seconds. skip...'%(_video_ind,per_video_db.shape[1]))
                    continue
                per_video_db_tar = per_video_db[_target_user,:][np.newaxis,:,:]
                per_video_db_oth = np.delete(per_video_db,_target_user,axis=0)
                if per_video_db_oth.shape[0]<num_user-1:
                    for concat_ind in range(per_video_db_oth.shape[0],num_user-1):
                        duplicate_ind = np.random.randint(per_video_db_oth.shape[0])
                        # print('duplicate_ind:',duplicate_ind)
                        temp = per_video_db_oth[duplicate_ind,:][np.newaxis,:]
                        per_video_db_oth = np.concatenate([per_video_db_oth,temp],axis=0)

                elif per_video_db_oth.shape[0]>num_user-1:
                    # delete the last one, currently only happens in the pc daaset
                    per_video_db_oth = per_video_db_oth[:num_user-1,:]

                try:
                    assert per_video_db_oth.shape[0]==num_user-1
                except:
                    pdb.set_trace()
                print('has %d other users'%(num_user-1))

                per_video_db_tar, per_video_db_future_tar, per_video_db_future_input_tar = reshape2second_stacks(per_video_db_tar,collapse_user=True)
                per_video_db_oth, per_video_db_future_oth, per_video_db_future_input_oth = reshape2second_stacks(per_video_db_oth,collapse_user=False)
                # print('_video_ind = ',_video_ind,'per_video_db.shape = ',per_video_db.shape)            
                # print('_video_ind = ',_video_ind,'per_video_db_tar.shape = ',per_video_db_tar.shape)            
                # print('_video_ind = ',_video_ind,'per_video_db_oth.shape = ',per_video_db_oth.shape)            
                _video_db_tar = np.concatenate((_video_db_tar, per_video_db_tar),axis=0)
                _video_db_future_tar = np.concatenate((_video_db_future_tar, per_video_db_future_tar),axis=0)
                _video_db_future_input_tar = np.concatenate((_video_db_future_input_tar, per_video_db_future_input_tar),axis=0)

                _video_db_oth = np.concatenate((_video_db_oth, per_video_db_oth),axis=1)
                _video_db_future_oth = np.concatenate((_video_db_future_oth, per_video_db_future_oth),axis=1)
                _video_db_future_input_oth = np.concatenate((_video_db_future_input_oth, per_video_db_future_input_oth),axis=1)


        if cfg.time_shift:
            pdb.set_trace()
            return _video_db_tar[1:,:-1,:], _video_db_future_tar[1:,:,:], _video_db_future_input_tar[1:,:,:], \
                    _video_db_oth[:,1:,:-1,:], _video_db_future_oth[:,1:,:,:], _video_db_future_input_oth[:,1:,:,:]
        else:
            return _video_db_tar[1:,:,:], _video_db_future_tar[1:,:,:], _video_db_future_input_tar[1:,:,:], \
                    _video_db_oth[:,1:,:,:], _video_db_future_oth[:,1:,:,:], _video_db_future_input_oth[:,1:,:,:]



# get target user data and other user's data
# _video_db_tar, _video_db_future_tar, _video_db_future_input_tar, \
# _video_db_oth,_video_db_future_oth,_video_db_future_input_oth = get_data(datadb,pick_user=True)
# # # cache data
# pickle.dump(_video_db_tar,open('./cache/format5_tsinghua_by_sec_interp/train/_video_db_tar.p','wb'))
# pickle.dump(_video_db_future_tar,open('./cache/format5_tsinghua_by_sec_interp/train/_video_db_future_tar.p','wb'))
# pickle.dump(_video_db_future_input_tar,open('./cache/format5_tsinghua_by_sec_interp/train/_video_db_future_input_tar.p','wb'))
# pickle.dump(_video_db_oth,open('./cache/format5_tsinghua_by_sec_interp/train/_video_db_oth.p','wb'))
# pickle.dump(_video_db_future_oth,open('./cache/format5_tsinghua_by_sec_interp/train/_video_db_future_oth.p','wb'))
# pickle.dump(_video_db_future_input_oth,open('./cache/format5_tsinghua_by_sec_interp/train/_video_db_future_input_oth.p','wb'))



# #TODO!!!!!!!!!!!!!!
# save2hdf5('./cache/'+dataformat+'/train/stride1_cut_tail/_video_db_tar.h5','_video_db_tar',_video_db_tar)
# save2hdf5('./cache/'+dataformat+'/train/stride1_cut_tail/_video_db_future_tar.h5','_video_db_future_tar',_video_db_future_tar)
# save2hdf5('./cache/'+dataformat+'/train/stride1_cut_tail/_video_db_future_input_tar.h5','_video_db_future_input_tar',_video_db_future_input_tar)
# save2hdf5('./cache/'+dataformat+'/train/stride1_cut_tail/_video_db_oth.h5','_video_db_oth',_video_db_oth)
# save2hdf5('./cache/'+dataformat+'/train/stride1_cut_tail/_video_db_future_oth.h5','_video_db_future_oth',_video_db_future_oth)
# save2hdf5('./cache/'+dataformat+'/train/stride1_cut_tail/_video_db_future_input_oth.h5','_video_db_future_input_oth',_video_db_future_input_oth)




def get_shuffle_index(data_length):
    index_shuf = range(data_length)
    shuffle(index_shuf)
    return index_shuf

def shuffle_data(index_shuf,list1):
    list1_shuf = [list1[i] for i in index_shuf]
    return np.array(list1_shuf)

def get_gt_target_xyz(y):
    if y.shape[-1]==3:
        assert len(y.shape)==4
        target_x = y[:,:,:,0]
        target_y = y[:,:,:,1]
        target_z = y[:,:,:,2]
    elif y.shape[-1]==90:
        assert len(y.shape)==3
        target_x = y[:,:,0::3]
        target_y = y[:,:,1::3]
        target_z = y[:,:,2::3]
    gt_mean_x = np.mean(target_x, axis=-1)[:,:,np.newaxis]
    gt_var_x = np.var(target_x, axis=-1)[:,:,np.newaxis]
    gt_mean_y = np.mean(target_y, axis=-1)[:,:,np.newaxis]
    gt_var_y = np.var(target_y, axis=-1)[:,:,np.newaxis]
    gt_mean_z = np.mean(target_z, axis=-1)[:,:,np.newaxis]
    gt_var_z = np.var(target_z, axis=-1)[:,:,np.newaxis]
    return np.concatenate((gt_mean_x,gt_mean_y,gt_mean_z,gt_var_x,gt_var_y,gt_var_z),axis=-1)




def get_gt_target_xyz_oth(y):
    # for others data
    # assert y.shape==(11040, 20, 47, 30, 3)
    target_x = y[:,:,:,:,0]
    target_y = y[:,:,:,:,1]
    target_z = y[:,:,:,:,2]
    gt_mean_x = np.mean(target_x, axis=-1)[:,:,:,np.newaxis]
    gt_var_x = np.var(target_x, axis=-1)[:,:,:,np.newaxis]
    gt_mean_y = np.mean(target_y, axis=-1)[:,:,:,np.newaxis]
    gt_var_y = np.var(target_y, axis=-1)[:,:,:,np.newaxis]
    gt_mean_z = np.mean(target_z, axis=-1)[:,:,:,np.newaxis]
    gt_var_z = np.var(target_z, axis=-1)[:,:,:,np.newaxis]
    return np.concatenate((gt_mean_x,gt_mean_y,gt_mean_z,gt_var_x,gt_var_y,gt_var_z),axis=-1)




def _save_theta_phi_index(reshape_datadb):
    def _theta_phi_index_for_onehot(_video_db,name='_video_db',data_format=1):
        #shape: (11040, 10, 1, 30, 3)
        if len(_video_db.shape)==5:
            x_list = _video_db[:,:,0,:,0]
            y_list = _video_db[:,:,0,:,1]
            z_list = _video_db[:,:,0,:,2]
        else:
            x_list = _video_db[:,:,:,0]
            y_list = _video_db[:,:,:,1]
            z_list = _video_db[:,:,:,2]

        theta_list, phi_list = xyz2thetaphi(x_list,y_list,z_list)

        bin_size=10
        theta_list2 = theta_list+np.pi
        theta_index = np.floor(theta_list2/np.pi*180/bin_size)
        theta_index[theta_index==360/bin_size]-=1

        phi_index = np.floor(phi_list/np.pi*180/bin_size)
        phi_index[phi_index==180/bin_size]-=1
        pickle.dump(theta_index,open('./cache/format'+str(data_format)+'/'+name+'_theta_index_exp'+str(experiment)+'.p','wb'))    
        pickle.dump(phi_index,open('./cache/format'+str(data_format)+'/'+name+'_phi_index_exp'+str(experiment)+'.p','wb'))    

    # format 1
    _theta_phi_index_for_onehot(_video_db,name='_video_db',data_format=1)
    _theta_phi_index_for_onehot(_video_db_future_input,name='_video_db_future_input',data_format=1)
    _theta_phi_index_for_onehot(_video_db_future,name='_video_db_future',data_format=1)
    # format 2
    _theta_phi_index_for_onehot(_video_db,name='_video_db_tar',data_format=2)
    _theta_phi_index_for_onehot(_video_db_future_input,name='_video_db_future_input_tar',data_format=2)
    _theta_phi_index_for_onehot(_video_db_future,name='_video_db_future_tar',data_format=2)



def _create_one_hot(theta_index,phi_index,bin_size=10,vector=False):
    #one-hot matrix -> reshape into vector
    one_hot = np.zeros((theta_index.shape[0],theta_index.shape[1],theta_index.shape[2],(360/bin_size),(180/bin_size)))
    #any faster ways?
    for ii in range(theta_index.shape[0]):
        for jj in range(theta_index.shape[1]):
            for kk in range(theta_index.shape[2]):
                one_hot[ii,jj,kk,int(theta_index[ii,jj,kk]),int(phi_index[ii,jj,kk])]=1
    if vector:
        one_hot = one_hot.reshape((theta_index.shape[0],theta_index.shape[1],theta_index.shape[2],-1))

    # # one hot encode using keras
    # encoded = to_categorical(data)
    # pickle.dump(one_hot,open('./cache/'+name+'_one_hot_exp'+str(experiment)+'.p','wb')) ##too large, should save as sparse
    return one_hot



def rand_sample_ind(total_num_samples,num_testing_sample,batch_size,validation_ratio=0.1):
    """ensure dividable by batch size"""
    # the num of training samples should be dividable by batch_size (sometimes error msg)
    # should use fit_generator() instead of fit() in the future
    x2 = int((total_num_samples-num_testing_sample)/batch_size*validation_ratio)
    m1 = int((1-validation_ratio)/validation_ratio*x2*batch_size)
    m2 = int(x2*batch_size)
    # num_validation = (total_num_samples-num_testing_sample-num_training)/batch_size*batch_size
    # sample_ind = random.sample(xrange(total_num_samples-num_testing_sample),num_training+num_validation)
    sample_ind = random.sample(range(total_num_samples-num_testing_sample),m1+m2)
    return sample_ind


def rand_sample(data,sample_ind):
    data = [ data[i] for i in sorted(sample_ind)]
    return np.array(data)



def _rescale_data(data):
    """normalize data from [-1,1] to [0,1]"""
    return (data+1)/2




def _normalize_data(data):
    """normalize data into N(0,1)"""
    # if data has mulitple dimensions, np.mean() will mean across all dims
    return (data-np.mean(data))/np.std(data)



def _insert_end_of_stroke(per_video_db,original_per_video_db):
    """create fake end of stroke signals
    eos will be set to 1 if:
    1) in the beginning and end of the whole trj
    2) whenever the theta has larger than pi changes 
       (~2pi change from pi to -pi or vice versa)
    """
    eos = np.zeros((per_video_db['theta'].shape[0],per_video_db['theta'].shape[1]))
    eos[:,0]=1
    eos[:,-1]=1
    #note that there are near 2pi changes in theta
    #we used +-pi as the thereshold
    if cfg.use_residual_input and not cfg.normalize_residual:
        eos[:,1:][per_video_db['theta'][:,1:]>np.pi]=1
        eos[:,1:][per_video_db['theta'][:,1:]<-np.pi]=1
        ##change from diff to real location at the eos==1 
        # per_video_db['theta'][:,1:][eos[:,1:]==1] = original_per_video_db['theta'][:,1:][eos[:,1:]==1]
        # per_video_db['phi'][:,1:][eos[:,1:]==1] = original_per_video_db['phi'][:,1:][eos[:,1:]==1]
    else:
        raise NotImplementedError
    per_video_db['eos'] = eos

def data_to_step_residual(datadb):
    """convert the input data into step-residual: 
    x_t-x_{t-1}, x_{t+1}-x_t ..."""
    # in handwritten sequence generation, the first location is the real location
    datadb_res = {}
    for ii in datadb.keys():
        datadb_res[ii]={}
        for key in datadb[ii].keys():
            datadb_res[ii][key] = np.hstack([np.expand_dims(datadb[ii][key][:,0],1),datadb[ii][key][:,1:]-datadb[ii][key][:,:-1]])
        if cfg.normalize_residual:
            for key in datadb_res[ii].keys():
                datadb_res[ii][key] = _normalize_data(datadb_res[ii][key])
        
        if cfg.predict_eos:
            #insert eos
            _insert_end_of_stroke(datadb_res[ii],datadb[ii]) #seems dict doesn't have to be returned...
    return datadb_res


def _insert_end_of_stroke2(datadb):
    """
       for datadb that is not residual! 
    """
    assert cfg.use_residual_input==False
    for ii in datadb.keys():
        per_video_db = datadb[ii]
        eos = np.zeros((per_video_db['theta'].shape[0],per_video_db['theta'].shape[1]))
        eos[:,0]=1
        eos[:,-1]=1
        diff = per_video_db['theta'][:,1:]-per_video_db['theta'][:,:-1]
        eos[:,1:][diff>np.pi]=1
        eos[:,1:][diff<-np.pi]=1 
        datadb[ii]['eos'] = eos
    return datadb

def subsample_datadb(datadb,subsample_factor=5):
    """subsample datadb into lower FPS"""
    # to introduce more variations
    for ii in datadb.keys():
        for kk in datadb[ii].keys():
            datadb[ii][kk] = datadb[ii][kk][:,0::subsample_factor]
    return datadb





def get_mu_std(datadb,vid_ind=cfg.test_video_ind):
    """for denormalization"""
    std1 = np.std(datadb[vid_ind]['theta'])
    mu1 = np.mean(datadb[vid_ind]['theta'])

    std2 = np.std(datadb[vid_ind]['phi'])
    mu2 = np.mean(datadb[vid_ind]['phi'])
    return mu1,std1,mu2,std2


def _denormalize_data(output,mu1,std1,mu2,std2):
    """de-normalize the predicted output"""
    out1 = output[:,:,0]
    out2 = output[:,:,1]
    #note that there are near 2pi changes in theta
    out = np.stack([out1*std1+mu1,
                    out2*std2+mu2],-1)
    return out




##simplied from https://github.com/snowkylin/rnn-handwriting-generation/blob/master/model.py
def sample(pi, mu1, mu2, sigma1, sigma2, rho, num_mixtures=20):
    """cg: sample from predicted mixture density"""
    # deleted the 3rd dimension since we didn't predict the End of Stroke
    x = np.zeros([cfg.batch_size, 1, 2], np.float32)
    strokes = np.zeros([cfg.batch_size, 2], dtype=np.float32)
    r = np.random.rand()
    accu = 0
    for batch_ind in range(cfg.batch_size):
        for m in range(num_mixtures):
            accu += pi[batch_ind, m]
            if accu > r:
                x[batch_ind, 0, 0:2] = np.random.multivariate_normal(
                    [mu1[batch_ind, m], mu2[batch_ind, m]],
                    [[np.square(sigma1[batch_ind, m]), rho[batch_ind, m] * sigma1[batch_ind, m] * sigma2[batch_ind, m]],
                     [rho[batch_ind, m] * sigma1[batch_ind, m] * sigma2[batch_ind, m], np.square(sigma2[batch_ind, m])]]
                )
                break

        strokes[batch_ind, :] = x[batch_ind, 0, :]
    return strokes



def data_visualization(datadb):
    def plot_xyz(datadb,video_ind,user_ind,start=0,end=-1):
        plt.plot(datadb[video_ind]['x'][user_ind,start:end])
        plt.plot(datadb[video_ind]['y'][user_ind,start:end])
        plt.plot(datadb[video_ind]['z'][user_ind,start:end])
        plt.show()

    def plot_phi_theta(datadb,video_ind,user_ind,start=0,end=-1):
        plt.plot(datadb[video_ind]['theta'][user_ind,start:end])
        plt.plot(datadb[video_ind]['phi'][user_ind,start:end])
        plt.show()


    def plot_all_users(datadb,video_ind,key,start=0,end=-1):
        for ii in range(datadb[video_ind][key].shape[0]):
            plt.plot(datadb[video_ind][key][ii,start:end])
        plt.show()


    def get_random_ind():
        video_ind = np.random.randint(9)
        key = datadb[video_ind].keys()[0]
        user_ind = np.random.randint(datadb[video_ind][key].shape[0])
        rand_s = np.random.randint(datadb[video_ind][key].shape[1])
        print('video_ind',video_ind,'user_ind',user_ind,'random start',rand_s)
        return video_ind,user_ind,rand_s

    # for video_ind in datadb.keys():
    #     for user_ind in range(datadb[video_ind]['x'].shape[0]):
    #         plot_xyz(video_ind,user_ind)
    #         pdb.set_trace()
    #         plt.cla()

    while True:
        #random select
        video_ind,user_ind,rand_s = get_random_ind()
        plt.figure('xyz')
        plot_xyz(video_db_xyz,video_ind,user_ind,start=rand_s,end=rand_s+500)
        plt.figure('theta_phi')
        plot_phi_theta(video_db_thetaphi,video_ind,user_ind,start=rand_s,end=rand_s+500)
        pdb.set_trace()
        plt.cla()


    while True:
        video_ind,user_ind,rand_s = get_random_ind()
        # plot_all_users(video_db_thetaphi,video_ind,'phi',start=rand_s,end=rand_s+500)
        plot_all_users(video_db_xyz,video_ind,'y',start=rand_s,end=rand_s+500)
        pdb.set_trace()
        plt.cla()





from statsmodels.tsa.vector_ar.var_model import VAR
def _VAR(train,test=None):
    model = VAR(train)
    model_fit = model.fit() #maxlags=299, ic='aic')
    print('Lag: %s' % model_fit.k_ar)
    if test!=None:
        predictions = model_fit.forecast(train[-10:,:],len(test))
        error = mean_squared_error(test, predictions)
        print('Test MSE: %.3f' % error)
    else:
        predictions = model_fit.forecast(train[-10:,:],len(train))
    return predictions



def _linear_model_residual_input_(data,data_future,mode='presistence'):
    def _get_it(data,data_future,mode):   
        residual_input = np.zeros_like(data_future)
        prediction_future = np.zeros_like(data_future) #predictions using mode
        data = data.reshape(-1,10,30,3)
        data_future = data_future.reshape(-1,10,30,3)
        for ii in range(residual_input.shape[0]):
            train = data[ii,:,:,:].reshape(-1,3)
            gt = data_future[ii,:,:,:].reshape(-1,3)
            # if np.sum(train,0)[2]==-300: var will have problem!!!
            #     predictions = np.repeat(data[ii,-1,-1,:].reshape(1,3),300,axis=0)
            #     continue
            if mode=='var':
                try:
                    predictions = _VAR(train=train,test=None)
                except:
                    predictions = np.repeat(data[ii,-1,-1,:].reshape(1,3),300,axis=0)                    
            elif mode=='presistence':
                predictions = np.repeat(data[ii,-1,-1,:].reshape(1,3),300,axis=0)
            residual = (gt-predictions).reshape(1,10,90)
            residual_input[ii,:] = residual
            prediction_future[ii,:] = predictions.reshape(1,10,90)
        return residual_input,prediction_future

    # if data.shape[0]==num_user-1: #oth
    if data.shape[0]==47:
        residual_input = np.zeros_like(data_future)
        prediction_future = np.zeros_like(data_future)
        for user_ind in range(data.shape[0]):
            print('user ind,',user_ind)
            data_temp = data[user_ind,:]
            data_future_temp = data_future[user_ind,:]
            residual_input_temp,prediction_future_temp = _get_it(data_temp,data_future_temp,mode)
            residual_input[user_ind,:] = residual_input_temp
            prediction_future[user_ind,:] = prediction_future_temp
    else:#tar
        residual_input,prediction_future = _get_it(data,data_future,mode)
    return residual_input,prediction_future





def find_k_neighbours(target_user_last_sec, other_users_this_sec, k):
    """return all neighbours for the target user"""
    num_others =  other_users.shape[0] #num_othersx90
    fps = 30
    distance = (other_users_this_sec - target_user_last_sec).reshape(num_others,fps,3)
    distance = np.sqrt(np.sum(distance**2,axis=-1))
    ave_distance = np.mean(distance,axis=1)
    neighbor_ind = ave_distance.argsort()[:k]
    return neighbor_ind


def find_k_neighbours_TF(target_user_last_sec, other_users_this_sec, k):
    # tensorflow 
    # on mean and variance batchx1x6, batchx1xnum_othersx6
    """return all neighbours for the target user"""
    num_others =  other_users_this_sec.shape[-2].value #num_othersx90
    fps = 30
    distance = tf.reshape((other_users_this_sec[:,0,:,:] - target_user_last_sec[:,0,:]),[-1,num_others,6])
    ave_distance = tf.reduce_sum(distance,axis=-1)
    _,neighbor_ind = tf.nn.top_k(ave_distance,k=k,sorted=True,name=None)

    # # nearest k points
    # _, top_k_indices = tf.nn.top_k(tf.negative(distance), k=k)
    # top_k_label = tf.gather(y_data_train, top_k_indices)

    # sum_up_predictions = tf.reduce_sum(top_k_label, axis=1)
    # prediction = tf.argmax(sum_up_predictions, axis=1)
    return neighbor_ind





def save2hdf5(path_name,key,data_to_store):
    hf = h5py.File(path_name, 'a')
    hf.create_dataset(key,data=data_to_store)
    hf.close()

def load_h5(path_name,key):
    #load
    hf = h5py.File(path_name, 'r')
    # print(list(hf.keys()))
    data = hf.get(key)
    data = np.array(data)
    return data





def saliency2h5():
    #Prepare saliency data (one pickle for each video) (cutted tail less than one second) into h5. 
    filelist = glob.glob('./360video/temp/saliency_input/input*p')
    for ii in range(1,len(filelist)):
        temp = pickle.load(open(filelist[ii],'rb'))
        save2hdf5(filelist[ii][:-2]+'.h5','source_surround',temp)


def get_random_k_other_users(_video_db_oth,from_user_num=48,to_user_num=34):
    """pretrained from shanghai and test it on tsinghua"""
    # since tsinghua dataset has 47 other user, shanghai has 33 other users. 
    # we randomly select 33 other users from the 47
    new = np.zeros((_video_db_oth.shape[0],_video_db_oth.shape[1],to_user_num-1,30,3))
    for ii in range(_video_db_oth.shape[0]):
        index = np.arange(from_user_num-1)
        random.shuffle(index)
        index = index[:to_user_num-1]
        new[ii,:] = _video_db_oth[ii,:][:,index,:,:].copy()
    return new


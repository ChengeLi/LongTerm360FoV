from __future__ import print_function
import tensorflow as tf
import random
import sys,glob
if './360video/' not in sys.path:
    sys.path.insert(0, './360video/')
# from mycode.dataLayer import DataLayer
from mycode.dataLayer2 import DataLayer
import mycode.cost as costfunc
from mycode.provide_hidden_state import multilayer_perceptron_hidden_state_series,\
                            multilayer_perceptron_hidden_state,dynamicRNN_hidden_state
from mycode.config import cfg
from mycode.dataIO import clip_xyz
import mycode.utility as util
import _pickle as pickle
import numpy as np
import pdb

is_test = True
test_epoch=1
# tag = "july11"
# tag = "pred_10_july11"
# tag = "meanvar2fc_pred_10_july12"
# tag = "raw_pred_10_july13"
# tag='5-5meanvar_july13'
# tag='3-3meanvar_july13'
# tag='1-1meanvar_july14'
# tag = "raw_pred_10_TV_july13"
# tag = "raw_pred_10_TV_sum1reg_july13"
# tag='phi_theta_2dGaussian_july15'
# tag='mixture_2dGaussian_july16'
# tag='1dlikelihood_xyz_july17'
# tag='1dlikelihood_xyz_july17_noreg_5-5'
# tag='oneD_gaussian_loss_5-5'
# tag='1dlikelihood_xyz_noreg_5-5_mixed_july18'
# tag='1dlikelihood_xyz_noreg_5-5_mixed_stride1_july18'
# tag='mse_mixed_stride1_july19'
# tag='mixture_framelvl_july19' #overlapping 15
# tag='pred_end_of_stroke_noresidualinput_july23'
# tag='pred_end_of_stroke_300-300_subsampled_july25'
# tag='300-300_subsampled_residual_july25'
# tag='300-300_subsampled_residual_10bernolli_july25'
# tag='60-60_subsampled_residual_xyz_july26'
# tag='60-60_subsampled_residual_xyz_mapfunc_july26'
tag='shanghai_split_aug22'
# use_reg = True

# training_epochs = cfg.training_epochs
training_epochs = 5
batch_size = cfg.batch_size #need to be smaller than (one video length)/running_length
display_step = 200
fps = cfg.fps
experiment = 1

# Network Parameters
num_layers = 2
# num_user = 48
truncated_backprop_length = cfg.running_length 
n_hidden = 400 # hidden layer num of features
# n_output = cfg.running_length

if cfg.use_xyz:
    if cfg.use_mixed_dataset:
        all_video_data = pickle.load(open('./data/merged_dataset.p','rb'))
    else:
        # Tsinghua dataset
        # all_video_data = pickle.load(open('./data/new_exp_'+str(experiment)+'_xyz.p','rb'))
        # Shanghai dataset
        if not is_test:
            all_video_data = pickle.load(open('./360video/data/shanghai_dataset_xyz_train.p','rb'))    
        else:
            all_video_data = pickle.load(open('./360video/data/shanghai_dataset_xyz_test.p','rb'))    
    data_dim = 3
    datadb = clip_xyz(all_video_data)
elif cfg.use_yaw_pitch_roll:
    all_video_data = pickle.load(open('./data/exp_2_raw.p','rb'))
    data_dim = 2 #only use yaw and pitch
    datadb = all_video_data
elif cfg.use_cos_sin:
    all_video_data = pickle.load(open('./data/exp_2_raw_pair.p','rb'))
    data_dim = 2
    datadb = all_video_data
elif cfg.use_phi_theta:
    all_video_data = pickle.load(open('./data/new_exp_'+str(experiment)+'_phi_theta.p','rb'))
    data_dim = 2
    if cfg.predict_eos:
        data_dim += 1        
    datadb = all_video_data

if cfg.process_in_seconds:
    data_dim = data_dim*fps

if cfg.subsample_datadb:
    datadb = util.subsample_datadb(datadb)
if cfg.use_residual_input:
    datadb_original = datadb
    # mu1,std1,mu2,std2 = util.get_mu_std(datadb_original,vid_ind=cfg.test_video_ind)
    datadb = util.data_to_step_residual(datadb)
else:
    if cfg.predict_eos:
        datadb = util._insert_end_of_stroke2(datadb)

if not is_test:
    # tf Graph input
    if cfg.own_history_only:
        x = tf.placeholder("float", [None, truncated_backprop_length, data_dim])
    else:
        x = tf.placeholder("float", [None, truncated_backprop_length, 48*data_dim])
    if cfg.has_reconstruct_loss:
        y = tf.placeholder("float", [None, cfg.running_length, data_dim])
    else:
        y = tf.placeholder("float", [None, cfg.predict_len, data_dim])
else:
    x = tf.placeholder(dtype=tf.float32, shape=[None, 1, data_dim])
    y = tf.placeholder(dtype=tf.float32, shape=[None, 1, data_dim])


# TODO: embedding
# can only handle int inputs???
# embeddings = tf.Variable(tf.random_uniform([200, 64], -1.0, 1.0), dtype=tf.float32)
# x = tf.nn.embedding_lookup(embeddings, x)
# decoder_inputs_embedded = tf.nn.embedding_lookup(embeddings, decoder_inputs)


# for population distribution
# others_future_further = tf.placeholder("float", [None, 47, cfg.predict_step*fps, data_dim/fps])
# states c,h
init_state = tf.placeholder(tf.float32, [num_layers, 2, batch_size, n_hidden])
state_per_layer_list = tf.unstack(init_state, axis=0)
rnn_tuple_state = tuple([tf.contrib.rnn.LSTMStateTuple(state_per_layer_list[idx][0],
                        state_per_layer_list[idx][1])
                        for idx in range(num_layers)])

# A placeholder for indicating each sequence length
# seqlen = tf.placeholder(tf.int32, [None])
dropout = tf.placeholder(tf.float32)

# Define weights
# weights = {
#     'out': tf.Variable(tf.random_normal([n_hidden, n_output]))
# }
# biases = {
#     'out': tf.Variable(tf.random_normal([n_output]))
# }


def pred_cnn_model_fn(inputs):
    """use cnn to predict raw trj"""
    # inputs.shape (batch_size, time, latent_dim) e.g. 8*1*64
    inputs = tf.expand_dims(inputs,1)
    # if input timporal dimension=1, it's equivalent to 3 fc layers
    conv1 = tf.layers.conv1d(
      inputs=inputs,
      filters=128,
      kernel_size=5,
      padding="same",
      activation=tf.nn.relu)

    conv2 = tf.layers.conv1d(
      inputs=conv1,
      filters=256,
      kernel_size=5,
      padding="same",
      activation=tf.nn.relu)

    conv3 = tf.layers.conv1d(
      inputs=conv2,
      filters=fps*3,
      kernel_size=5,
      padding="same",
      activation=tf.nn.tanh)

    prediction = conv3
    return prediction




def manual_dynamicRNN(x, seqlen, weights, biases):
    """CG: this func use static_rnn and manually change it into dynamic (?)"""
    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, n_steps, n_input)
    # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)
    
    # Unstack to get a list of 'n_steps' tensors of shape (batch_size, n_input)
    x = tf.unstack(x, truncated_backprop_length, 1)

    # Define a lstm cell with tensorflow
    lstm_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden)

    # Get lstm cell output, providing 'sequence_length' will perform dynamic
    # calculation.
    outputs, states = tf.contrib.rnn.static_rnn(lstm_cell, x, dtype=tf.float32,
                                sequence_length=seqlen)
    # When performing dynamic calculation, we must retrieve the last
    # dynamically computed output, i.e., if a sequence length is 10, we need
    # to retrieve the 10th output.
    # However TensorFlow doesn't support advanced indexing yet, so we build
    # a custom op that for each sample in batch size, get its length and
    # get the corresponding relevant output.

    # 'outputs' is a list of output at every timestep, we pack them in a Tensor
    # and change back dimension to [batch_size, n_step, n_input]
    outputs = tf.stack(outputs)
    outputs = tf.transpose(outputs, [1, 0, 2])

    # Hack to build the indexing and retrieve the right output.
    batch_size = tf.shape(outputs)[0]
    # Start indices for each sample
    index = tf.range(0, batch_size) * truncated_backprop_length + (seqlen - 1)
    # Indexing
    outputs = tf.gather(tf.reshape(outputs, [-1, n_hidden]), index)

    # Linear activation, using outputs computed above
    return tf.matmul(outputs, weights['out']) + biases['out']


def dynamicRNN(x,dropout):
    """two layer LSTM using dynamic_rnn"""
    cells = []
    for _ in range(num_layers):
      # cell = tf.contrib.rnn.GRUCell(n_hidden)
      cell = tf.contrib.rnn.LSTMCell(n_hidden,state_is_tuple=True)
      cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=1.0 - dropout)
      cells.append(cell)
    cell = tf.contrib.rnn.MultiRNNCell(cells)

    if cfg.predict_len>1:
        with tf.variable_scope("dynamicRNN", reuse=tf.AUTO_REUSE):
            # Batch size x time steps x features.
            states_series, current_state = tf.nn.dynamic_rnn(cell, x, dtype=tf.float32,
                                              initial_state=rnn_tuple_state)

            # # Batch size x time steps x 1 (1 output at 1 timestamp)
            # pred_output = tf.contrib.layers.fully_connected(
            #                 states_series, 1, activation_fn=None) 
    else:
        states_series, current_state = tf.nn.dynamic_rnn(cell, x, dtype=tf.float32,
                                  initial_state=rnn_tuple_state)
    return states_series, current_state

def concat_current_state(current_state,mlp_state):
    """concat last hidden and memery states from LSTM and MLP, 
    predict using another FC"""
    if len(current_state)>1: 
        # multiple layers
        # only keep the last [c,h]
        current_state = current_state[-1]
    c,h = current_state[0],current_state[1]
    concat_state = tf.reshape(tf.stack((c,h,mlp_state),axis=-1),[batch_size,-1])
    # CAUTION: directly predicting whole length!!
    finalpred = tf.contrib.layers.fully_connected(
            concat_state, cfg.running_length, activation_fn=None)
    return finalpred


def concat_states_series(states_series,states_series_others):
    """concat state_series from LSTM and MLP, 
    predict using another FC"""
    concat_state = tf.reshape(tf.stack((states_series,states_series_others),axis=-1),[batch_size,cfg.running_length,n_hidden*2])
    # Batch size x time steps x 1 (1 output at 1 timestamp)
    finalpred = tf.contrib.layers.fully_connected(
                concat_state, 2, activation_fn=None)

    return finalpred


def _pred_mean_var_xyz(state):
    # predict mean and variance for x,y,z
    pred = tf.contrib.layers.fully_connected(state, 3*2, activation_fn=None)
    ux = tf.slice(pred,[0,0],[-1,1])
    uy = tf.slice(pred,[0,1],[-1,1])
    uz = tf.slice(pred,[0,2],[-1,1])
    # variance must >0
    varx = tf.abs(tf.slice(pred,[0,3],[-1,1]))
    vary = tf.abs(tf.slice(pred,[0,4],[-1,1]))
    varz = tf.abs(tf.slice(pred,[0,5],[-1,1]))
    return ux,uy,uz,varx,vary,varz

def _pred_mean_var_xyz2(state):
    # predict mean and variance for x,y,z
    # using 2 fc layers
    pred = tf.contrib.layers.fully_connected(state, 3, activation_fn=tf.nn.tanh)
    ux = tf.slice(pred,[0,0],[-1,1])
    uy = tf.slice(pred,[0,1],[-1,1])
    uz = tf.slice(pred,[0,2],[-1,1])

    # variance must >0
    pred2 = tf.contrib.layers.fully_connected(state, 3, activation_fn=tf.nn.relu)
    varx = tf.slice(pred2,[0,0],[-1,1])
    vary = tf.slice(pred2,[0,1],[-1,1])
    varz = tf.slice(pred2,[0,2],[-1,1])
    return ux,uy,uz,varx,vary,varz

def _pred_mean_var_phi_theta(state):
    # predict mean and variance for phi,theta
    pred = tf.contrib.layers.fully_connected(state, 2*2, activation_fn=None)
    u1 = tf.slice(pred,[0,0],[-1,1])
    u2 = tf.slice(pred,[0,1],[-1,1])
    # variance must >0
    var1 = tf.abs(tf.slice(pred,[0,2],[-1,1]))
    var2 = tf.abs(tf.slice(pred,[0,3],[-1,1]))
    return u1,u2,var1,var2

def _pred_mean_var_phi_theta_2dgassian(state):
    # predict mean and variance for phi,theta in 2D gaussian
    pred = tf.contrib.layers.fully_connected(internal, 5, activation_fn=None)
    utheta = tf.slice(pred,[0,0],[-1,1])
    uphi = tf.slice(pred,[0,1],[-1,1])
    sigma_theta = tf.slice(pred,[0,2],[-1,1])
    sigma_phi = tf.slice(pred,[0,3],[-1,1])
    rho = tf.slice(pred,[0,4],[-1,1])

    sigma_theta = tf.abs(sigma_theta)
    sigma_phi = tf.abs(sigma_phi)
    rho = tf.clip_by_value(rho,-1,1)

    return utheta,uphi,sigma_theta,sigma_phi,rho


def _pred_mean_var_xyz2_new(state):
    # predict mean and variance for x,y,z
    # using 2 fc layers
    pred = tf.contrib.layers.fully_connected(state, 32, activation_fn=tf.nn.relu)
    pred = tf.contrib.layers.fully_connected(pred, 3, activation_fn=tf.nn.tanh)
    ux = tf.slice(pred,[0,0],[-1,1])
    uy = tf.slice(pred,[0,1],[-1,1])
    uz = tf.slice(pred,[0,2],[-1,1])

    # variance must >0
    pred2 = tf.contrib.layers.fully_connected(state, 32, activation_fn=tf.nn.relu)
    pred2 = tf.contrib.layers.fully_connected(pred2, 3, activation_fn=None)
    pred2 = tf.exp(pred2)
    varx = tf.slice(pred2,[0,0],[-1,1])
    vary = tf.slice(pred2,[0,1],[-1,1])
    varz = tf.slice(pred2,[0,2],[-1,1])
    return ux,uy,uz,varx,vary,varz


def _GMM_2dgassian(state):
    #4 layer MLP
    internal = tf.contrib.layers.fully_connected(state, 64, activation_fn=tf.nn.relu)
    internal = tf.layers.dropout(internal,rate=0.2)
    internal = tf.contrib.layers.fully_connected(internal, 128, activation_fn=tf.nn.relu)
    internal = tf.layers.dropout(internal,rate=0.2)
    internal = tf.contrib.layers.fully_connected(internal, 256, activation_fn=tf.nn.relu)
    if cfg.predict_eos:
        pred = tf.contrib.layers.fully_connected(internal, 121, activation_fn=None)
    else:
        pred = tf.contrib.layers.fully_connected(internal, 120, activation_fn=None)

    #use 20 mixtures by default
    #20 mixture weights, 20*2 means, 20*2 stds, 20 correlation rhos
    #1 param to model "end-of-stroke" behavior (in our case, changing from 2pi to -2pi)
    mixture_pi = tf.slice(pred,[0,0],[-1,20])
    us = tf.slice(pred,[0,20],[-1,40])
    sigmas = tf.slice(pred,[0,60],[-1,40])
    rhos = tf.slice(pred,[0,100],[-1,20])
    if cfg.predict_eos:
        end_stroke = tf.slice(pred,[0,120],[-1,1])
        end_stroke = 1/(1+tf.exp(end_stroke)) #(0,1)
    
    # mixture_pi = tf.nn.softmax(mixture_pi) #(0,1)
    pi_exp = tf.exp(mixture_pi)
    pi_exp_sum = tf.reduce_sum(pi_exp, 1)
    mixture_pi = pi_exp / tf.concat([tf.expand_dims(pi_exp_sum, 1) for _ in range(20)],1)

    sigmas = tf.exp(sigmas)
    rhos = tf.tanh(rhos) #(-1,1)

    if cfg.predict_eos:
        return end_stroke,mixture_pi,us,sigmas,rhos        
    else:
        return mixture_pi,us,sigmas,rhos


def _GMM_3dgassian(state):
    #4 layer MLP
    internal = tf.contrib.layers.fully_connected(state, 64, activation_fn=tf.nn.relu)
    internal = tf.layers.dropout(internal,rate=0.2)
    internal = tf.contrib.layers.fully_connected(internal, 128, activation_fn=tf.nn.relu)
    internal = tf.layers.dropout(internal,rate=0.2)
    internal = tf.contrib.layers.fully_connected(internal, 256, activation_fn=tf.nn.relu)
    pred = tf.contrib.layers.fully_connected(internal, 200, activation_fn=None)

    #use 20 mixtures by default
    #20 mixture weights, 20*3 means, 20*3 stds, 60 correlation rhos
    mixture_pi = tf.slice(pred,[0,0],[-1,20])
    us = tf.slice(pred,[0,20],[-1,60])
    sigmas = tf.slice(pred,[0,80],[-1,60])
    rhos = tf.slice(pred,[0,140],[-1,60])

    # mixture_pi = tf.nn.softmax(mixture_pi) #(0,1)
    pi_exp = tf.exp(mixture_pi)
    pi_exp_sum = tf.reduce_sum(pi_exp, 1)
    mixture_pi = pi_exp / tf.concat([tf.expand_dims(pi_exp_sum, 1) for _ in range(20)],1)

    sigmas = tf.exp(sigmas)
    rhos = tf.tanh(rhos) #(-1,1)
    return mixture_pi,us,sigmas,rhos





# pred1 = manual_dynamicRNN(x, seqlen, weights, biases)
states_series, current_state = dynamicRNN(x,dropout)


if not cfg.own_history_only:
    # others_future = tf.placeholder("float", [None, 47, cfg.predict_len, data_dim])
    others = tf.placeholder("float", [None, 47, cfg.running_length, data_dim])
    states_series_others, current_state_others = dynamicRNN(x_others,dropout)


if cfg.concat_state:
    # mlp_state_series = multilayer_perceptron_hidden_state_series(others_future,n_hidden)
    # mlp_state = multilayer_perceptron_hidden_state(others_future,batch_size,n_hidden)
    # pred1 = concat_current_state(current_state,mlp_state)
    # pred = concat_states_series(states_series,mlp_state_series)
    states_series_others, current_state_others = dynamicRNN_hidden_state(others_future,num_layers,n_hidden,rnn_tuple_state)
    pred = concat_states_series(states_series,states_series_others)  
else:
    # pred = tf.contrib.layers.fully_connected(current_state[1][1], data_dim, activation_fn=None)
    if cfg.use_xyz:
        if cfg.predict_mean_var:
            ux,uy,uz,varx,vary,varz = _pred_mean_var_xyz2_new(current_state[1][1])
            # target = util.tf_get_gt_target_xyz(y)

            if cfg.predict_len==1 or is_test:
                # population_target = util.tf_get_gt_target_xyz_pop(others_future_further)
                # cost = costfunc._mean_var_cost_xyz(ux,uy,uz,varx,vary,varz,target,population_target)
                # cost = costfunc._mean_var_cost_xyz(ux,uy,uz,varx,vary,varz,target)
                cost = costfunc.likelihood_loss_tf([ux,uy,uz,varx,vary,varz],y)
                # cost = costfunc.oneD_gaussian_loss([ux,uy,uz,varx,vary,varz],y)

                # # cost,cost_x,cost_y,cost_z = costfunc._mean_var_cost_xyz_metric(ux,uy,uz,varx,vary,varz,target,costfunc.Bhattacharyya_distance)
                # # cost,cost_x,cost_y,cost_z = costfunc._mean_var_cost_xyz_metric(ux,uy,uz,varx,vary,varz,target,costfunc.Wasserstein_distance)
                # # cost,cost_x,cost_y,cost_z = costfunc._mean_var_cost_xyz_metric(ux,uy,uz,varx,vary,varz,target,costfunc.Kullback_Leibler_divergence_Gaussian)
            
                # directly predict new samples
                # predict_sample = tf.contrib.layers.fully_connected(current_state[1][1], fps*1*3, activation_fn=None)
                # predict_sample = tf.reshape(predict_sample,[batch_size,fps*1,3])
                # cost,inspect1,inspect2 = costfunc.conditional_prob_loss(predict_sample,target,population_target)

            elif cfg.predict_len>1 and not is_test:
                #compute the first step loss
                input_temp = x
                # target_now = [tf.squeeze(tf.slice(target[0],[0,0,0],[-1,1,-1]),axis=-1),
                #         tf.squeeze(tf.slice(target[1],[0,0,0],[-1,1,-1]),axis=-1),
                #         tf.squeeze(tf.slice(target[2],[0,0,0],[-1,1,-1]),axis=-1),
                #         tf.squeeze(tf.slice(target[3],[0,0,0],[-1,1,-1]),axis=-1),
                #         tf.squeeze(tf.slice(target[4],[0,0,0],[-1,1,-1]),axis=-1),
                #         tf.squeeze(tf.slice(target[5],[0,0,0],[-1,1,-1]),axis=-1)]
                # cost = costfunc._mean_var_cost_xyz(ux,uy,uz,varx,vary,varz,target_now)
                # cost = costfunc.oneD_gaussian_loss([ux,uy,uz,varx,vary,varz],tf.expand_dims(y[:,0,:],1))
                cost = costfunc.likelihood_loss_tf([ux,uy,uz,varx,vary,varz],tf.expand_dims(y[:,0,:],1))

                #compute the sequence loss after the first step in a loop: cost=\sum cost_{t=0,T}
                for time_ind in range(1,cfg.predict_len):#already computed the first future step cost
                    #generate fake batch        
                    this_input_temp = tf.stack((util.generate_fake_batch_tf(ux,varx),
                                util.generate_fake_batch_tf(uy,vary),
                                util.generate_fake_batch_tf(uz,varz)),axis=-1)
                    this_input_temp = tf.reshape(this_input_temp,[batch_size,1,fps*3])
                    #shift the input_temp
                    history_input = tf.slice(input_temp,[0,1,0],[-1,-1,-1])
                    input_temp = tf.concat((history_input,this_input_temp),axis=1)

                    states_series, current_state = dynamicRNN(input_temp,dropout)
                    ux,uy,uz,varx,vary,varz = _pred_mean_var_xyz2_new(current_state[1][1])
                    # target_now = [tf.squeeze(tf.slice(target[0],[0,time_ind,0],[-1,1,-1]),axis=-1),
                    #         tf.squeeze(tf.slice(target[1],[0,time_ind,0],[-1,1,-1]),axis=-1),
                    #         tf.squeeze(tf.slice(target[2],[0,time_ind,0],[-1,1,-1]),axis=-1),
                    #         tf.squeeze(tf.slice(target[3],[0,time_ind,0],[-1,1,-1]),axis=-1),
                    #         tf.squeeze(tf.slice(target[4],[0,time_ind,0],[-1,1,-1]),axis=-1),
                    #         tf.squeeze(tf.slice(target[5],[0,time_ind,0],[-1,1,-1]),axis=-1)]
                    # cost += costfunc._mean_var_cost_xyz(ux,uy,uz,varx,vary,varz,target_now)
                    # cost += costfunc.oneD_gaussian_loss([ux,uy,uz,varx,vary,varz],tf.expand_dims(y[:,time_ind,:],1))
                    cost += costfunc.likelihood_loss_tf([ux,uy,uz,varx,vary,varz],tf.expand_dims(y[:,time_ind,:],1))

        elif cfg.use_GMM:
            mixture_pi,us,sigmas,rhos = _GMM_3dgassian(current_state[1][1])
            y_pred = [mixture_pi,us,sigmas,rhos]
            cost = costfunc.mixture_3d_gaussian_loss(y,y_pred)

        else:#directly predict raw
            predict_sample = pred_cnn_model_fn(current_state[1][1])  
            this_input_temp = predict_sample
            if cfg.predict_len==1:
                cost = tf.losses.mean_squared_error(y,predict_sample)
            elif cfg.predict_len>1:
                input_temp = x
                #compute the first step loss
                this_y = tf.slice(y,[0,0,0],[-1,1,-1])
                # cost = tf.losses.mean_squared_error(this_y,predict_sample)
                cost = costfunc.pred_raw_loss_tf(this_y,predict_sample,use_reg=use_reg)

                #compute the sequence loss after the first step
                for time_ind in range(1,cfg.predict_len):#already computed the first future step cost
                    #shift the input_temp
                    history_input = tf.slice(input_temp,[0,1,0],[-1,-1,-1])
                    input_temp = tf.concat((history_input,this_input_temp),axis=1)
                    states_series, current_state = dynamicRNN(input_temp,dropout)
                    predict_sample = pred_cnn_model_fn(current_state[1][1]) 
                    this_input_temp = predict_sample
                    this_y = tf.slice(y,[0,time_ind,0],[-1,1,-1])
                    # cost += tf.losses.mean_squared_error(this_y,predict_sample)
                    cost += costfunc.pred_raw_loss_tf(this_y,predict_sample,use_reg=use_reg)


    elif cfg.use_phi_theta:
        # target = get_gt_target_phi_theta(y)
        # u1,u2,var1,var2 = _pred_mean_var_phi_theta(current_state[1][1])
        # cost = costfunc._mean_var_cost_phi_theta(u1,u2,var1,var2,target)

        if cfg.use_GMM:
            if cfg.predict_eos:
                end_stroke,mixture_pi,us,sigmas,rhos = _GMM_2dgassian(current_state[1][1])
                y_pred = [end_stroke,mixture_pi,us,sigmas,rhos]
            else:
                mixture_pi,us,sigmas,rhos = _GMM_2dgassian(current_state[1][1])
                y_pred = [mixture_pi,us,sigmas,rhos]
            # cost = costfunc.mixture_likelihood_loss_phi_theta_tf(y,y_pred)
            cost = costfunc.mixture_bivariate_gaussian_loss(y,y_pred)
        else:
            utheta,uphi,sigma_theta,sigma_phi,rho = _pred_mean_var_phi_theta_2dgassian(current_state[1][1])
            cost = costfunc.likelihood_loss_phi_theta_tf(y,[utheta,uphi,sigma_theta,sigma_phi,rho])


# Define loss and optimizer
# cost = tf.losses.mean_squared_error(pred,y)
# cost = costfunc._modified_mse(pred1,y)
# cost = costfunc._modified_mse(pred,y)
# cost = costfunc._modified_mse(pred[:,:int(1/2*cfg.running_length)],y[:,:int(1/2*cfg.running_length)])\
#     + 10*_modified_mse(pred[:,int(1/2*cfg.running_length):],y[:,int(1/2*cfg.running_length):])

# cost = costfunc._modified_mse(pred,y[:,0,:])

# summary
all_losses_dict = {}
all_losses_dict['MSE_loss'] = cost
# all_losses_dict['modified_MSE_loss'] = cost
# all_losses_dict['modified_MSE_loss_staticRNN'] = cost1
event_summaries = {}
event_summaries.update(all_losses_dict)
summaries = []
for key, var in event_summaries.items():
    summaries.append(tf.summary.scalar(key, var))
summary_op = tf.summary.merge(summaries)

saver = tf.train.Saver()
model_path = "./model/LSTM_"+tag+".ckpt"
lr = tf.Variable(cfg.LEARNING_RATE, trainable=False)
# optimizer = tf.train.AdamOptimizer(learning_rate=lr)
optimizer = tf.train.RMSPropOptimizer(learning_rate=lr)
gvs = optimizer.compute_gradients(cost)

def ClipIfNotNone(grad):
    if grad is None:
        return grad
    return tf.clip_by_value(grad, -1, 1)
if cfg.clip_gradient:
    clipped_gradients = [(ClipIfNotNone(grad), var) for grad, var in gvs]
    train_op = optimizer.apply_gradients(clipped_gradients)
else:
    train_op = optimizer.apply_gradients(gvs)

init = tf.global_variables_initializer()

batch_nums = []
for ii in datadb.keys():  
    if ii == cfg.test_video_ind:
        continue
    batch_nums.append(datadb[ii]['x'].shape[1])
if cfg.process_in_seconds:
    total_batch = (np.sum(batch_nums)-cfg.running_length*fps)/cfg.data_chunk_stride/batch_size
else:
    total_batch = (np.sum(batch_nums)-cfg.running_length)/cfg.data_chunk_stride/batch_size
total_batch*=datadb[ii]['x'].shape[0]

# Start training
if not is_test:
    data_io = DataLayer(datadb, random=False, is_test=False)
    with tf.Session() as sess:
        sess.run(init)
        starting_epoch = 2
        summary_writer = tf.summary.FileWriter('./tfsummary/'+tag, sess.graph)
        if len(glob.glob(model_path[:-5]+'epoch'+str(starting_epoch-1)+".ckpt.meta"))>0:
            saver.restore(sess, model_path[:-5]+'epoch'+str(starting_epoch-1)+".ckpt")
            print("Model restored.")
            starting_epoch = training_epochs

        _current_state = np.zeros((num_layers, 2, batch_size, n_hidden))
        for epoch in range(starting_epoch,starting_epoch+training_epochs,1):
            if epoch>0 and epoch%2==0:
                save_path = saver.save(sess, model_path[:-5]+'epoch'+str(epoch)+'.ckpt')
                print("Model saved: %s" % save_path)
                lr_temp = cfg.LEARNING_RATE*(0.5**(epoch/cfg.lr_epoch_step))
                print('epoch: ',epoch, ', change lr=lr*0.5, lr=', lr_temp)
                sess.run(tf.assign(lr, lr_temp))
            for step in range(total_batch):
                # TODO: zero before every new minibatch?
                # _current_state = np.zeros((num_layers, 2, batch_size, n_hidden))
    
                # batch_x, batch_y, batch_seqlen = trainset.next(batch_size)
                batch_x, batch_y, batch_x_others,_,_,batch_x_others_further = data_io._get_next_minibatch(datadb,batch_size) 
                # ## collapse last dimension, batch*len*(num_user*2)
                # batch_x = np.reshape(batch_x,[batch_size,batch_x.shape[1],-1])
                # batch_seqlen = np.array([cfg.running_length]*batch_size) #fixed length
                if cfg.change_xyz2xxxyyyzzz:
                    batch_x = util.change_input_format(batch_x)#even worse, why
                # Run optimization op (backprop)
                _, _current_state = sess.run([train_op,current_state],
                                             feed_dict={x: batch_x, y: batch_y,
                                             dropout: 0.1,
                                             # others_future_further:batch_x_others_further,
                                             # seqlen: batch_seqlen,
                                             # others_future: batch_x_others,
                                             init_state: _current_state})
                count = (step+1)*batch_size+epoch*total_batch*batch_size
                if count<200:
                    display_step=10
                else:
                    display_step=200                    
                if count% display_step == 0 or count==0:         
                    if cfg.predict_mean_var:
                        if cfg.use_xyz:
                            loss,summary,_clipped_gradients,_gv,_current_state,_states_series,ux_temp,uy_temp,uz_temp,varx_temp,vary_temp,varz_temp = sess.run(
                                            [cost,summary_op,clipped_gradients[2:],gvs[2:],current_state,states_series,ux,uy,uz,varx,vary,varz], 
                                            feed_dict={x: batch_x, y: batch_y,
                                                        dropout: 0.1,
                                                        # others_future_further:batch_x_others_further,
                                                        # seqlen: batch_seqlen,
                                                        # others_future: batch_x_others,
                                                        init_state: _current_state})      
                        elif cfg.use_phi_theta:
                            loss,summary,_clipped_gradients,_gv,_current_state,_states_series = sess.run(
                                            [cost,summary_op,clipped_gradients[2:],gvs[2:],current_state,states_series], 
                                            feed_dict={x: batch_x, y: batch_y,
                                                        dropout: 0.1,
                                                        init_state: _current_state})
                    if not cfg.predict_mean_var: 
                        #predict raw
                        loss,summary = sess.run(
                                        [cost,summary_op], 
                                        feed_dict={x: batch_x, y: batch_y,
                                                    dropout: 0.1,
                                                    # others_future_further:batch_x_others_further,
                                                    # seqlen: batch_seqlen,
                                                    # others_future: batch_x_others,
                                                    init_state: _current_state})                    
                    
                    summary_writer.add_summary(summary, float(count))
                    print("Step " + str(count) + ", Minibatch Loss= " + \
                          "{:.6f}".format(loss))

        print("Optimization Finished!")
        save_path = saver.save(sess, model_path[:-5]+'epoch'+str(epoch)+'.ckpt')
        print("Model saved: %s" % save_path)  


if is_test:
    # test
    with tf.Session() as sess:
        # Run the initializer
        sess.run(init) 
        if test_epoch==None:
            saver.restore(sess, model_path)
        else:
            saver.restore(sess, model_path[:-5]+'epoch'+str(test_epoch)+'.ckpt')
        print("Model restored.")
        data_io_test = DataLayer(datadb, random=False, is_test=True)
        # if cfg.use_GMM:
        #     data_io_test_original = DataLayer(datadb_original, random=False, is_test=True)
        # batch_seqlen = np.array([cfg.running_length]*batch_size) #fixed length
        test_out = []
        gt_out = []
        unnormalized_input = []
        num_trials = 2393/batch_size+1
        strokes=np.zeros((num_trials,cfg.predict_step,cfg.batch_size,2))
        for ii in range(num_trials):
            # every test time feed in zero state?
            _current_state = np.zeros((num_layers, 2, batch_size, n_hidden))
            batch_x, batch_y, batch_x_others,batch_y_further,db_index,batch_x_others_further = data_io_test._get_next_minibatch(datadb,batch_size) 
            ###caution!!!: only feed one frame as batch_x
            batch_x = np.expand_dims(batch_x[:,-1,:],1)
            # if cfg.predict_eos:  batch_x[:,:,2]=1 
            batch_y = np.expand_dims(batch_y[:,-1,:],1)
            if cfg.change_xyz2xxxyyyzzz:
                batch_x = util.change_input_format(batch_x)
            # batch_x, batch_y, batch_x_others = data_io_test._get_next_minibatch(datadb,batch_size)
            # collapse last dimension, batch*len*(num_user*2)
            # batch_x = np.reshape(batch_x,[batch_size,batch_x.shape[1],-1])
            
            if cfg.use_GMM:
                # batch_x111, batch_y111, _,batch_y_further111,_,_ = data_io_test_original._get_next_minibatch(datadb_original,batch_size)
                # gt_out.append(batch_y_further111)
                # unnormalized_input.append(batch_x111)
                gt_out.append(batch_y_further)                
                # gt_out.append(util._denormalize_data(batch_y_further,mu1,std1,mu2,std2))
            else:
                gt_out.append(batch_y_further)

            # for predict_step in range(cfg.predict_step):
            #     # feed prediction back as input!
            #     pred_temp,_current_state = sess.run([pred,current_state], feed_dict={x: batch_x,
            #                                 # y: batch_y,
            #                                 dropout: 0.0,
            #                                 # seqlen: batch_seqlen,
            #                                 # others_future: batch_x_others,
            #                                 init_state: _current_state})
            #     batch_x = np.concatenate((batch_x[:,1:,:],pred_temp[:,np.newaxis,:]),axis=1)
            #     test_out.append(pred_temp)
            for predict_step in range(cfg.predict_step):
                # feed prediction back as input!
                if cfg.use_xyz:
                    if cfg.predict_mean_var:
                        loss,ux_temp,uy_temp,uz_temp,varx_temp,vary_temp,varz_temp,_current_state = sess.run(
                                                [cost,ux,uy,uz,varx,vary,varz,current_state],
                                                feed_dict={x: batch_x, 
                                                y: batch_y,
                                                dropout: 0.0,
                                                # seqlen: batch_seqlen,
                                                # others_future: batch_x_others,
                                                # others_future_further:batch_x_others_further,
                                                init_state: _current_state})
                        test_out.append([ux_temp,uy_temp,uz_temp,varx_temp,vary_temp,varz_temp])
                        
                        temp_newdata = np.stack((util.generate_fake_batch_numpy(ux_temp,varx_temp,batch_size),
                                        util.generate_fake_batch_numpy(uy_temp,vary_temp,batch_size),
                                        util.generate_fake_batch_numpy(uz_temp,varz_temp,batch_size)),axis=-1)[:,np.newaxis,:,:].reshape((batch_size,1,-1))
                    

                    elif cfg.use_GMM:
                        mixture_pi_temp,us_temp,sigmas_temp,rhos_temp,_current_state = sess.run(
                                                                    [mixture_pi,us,sigmas,rhos,current_state], 
                                                                    feed_dict={x: batch_x,
                                                                    y: batch_y,
                                                                    dropout: 0.0,
                                                                    init_state: _current_state})
                        predictions = [mixture_pi_temp,us_temp,sigmas_temp,rhos_temp]
                        temp_newdata = util.sample_mixture_3D(predictions)
                        test_out.append(temp_newdata)

                    if not cfg.predict_mean_var:
                        loss,pred_temp,_current_state = sess.run(
                                                [cost,predict_sample,current_state],
                                                feed_dict={x: batch_x,
                                                        y: batch_y,
                                                        dropout: 0.0,
                                                        # others_future_further:batch_x_others_further,
                                                        init_state: _current_state})
                        test_out.append([pred_temp])
                        temp_newdata = pred_temp.reshape(batch_size,1,-1)

                    print ('loss ',loss)
                elif cfg.use_phi_theta:
                    ## mse on (u,var)
                    # uphi_temp,utheta_temp,varphi_temp,vartheta_temp,_current_state = sess.run([u1,u2,var1,var2,current_state], 
                    #                                                                     feed_dict={x: batch_x,
                    #                                                                     # y: batch_y,
                    #                                                                     dropout: 0.0,
                    #                                                                     # seqlen: batch_seqlen,
                    #                                                                     # others_future: batch_x_others,
                    #                                                                     init_state: _current_state})
                    
                    # test_out.append([uphi_temp,utheta_temp,varphi_temp,vartheta_temp])
                    # temp_newdata = np.stack((util.generate_fake_batch_numpy(uphi_temp,varphi_temp,batch_size),
                    #                     util.generate_fake_batch_numpy(utheta_temp,vartheta_temp,batch_size)),axis=-1)[:,np.newaxis,:,:].reshape((batch_size,1,-1))
    

                    ## likelihood_loss_phi_theta_tf (NLL)
                    if cfg.use_GMM:
                        if cfg.predict_eos:
                            end_of_stroke_temp,mixture_pi_temp,us_temp,sigmas_temp,rhos_temp,_current_state = sess.run(
                                                                        [end_stroke,mixture_pi,us,sigmas,rhos,current_state], 
                                                                        feed_dict={x: batch_x,
                                                                        y: batch_y,
                                                                        dropout: 0.0,
                                                                        init_state: _current_state})                       
                            predictions = [end_of_stroke_temp,mixture_pi_temp,us_temp,sigmas_temp,rhos_temp]
                        else:
                            mixture_pi_temp,us_temp,sigmas_temp,rhos_temp,_current_state = sess.run(
                                                                        [mixture_pi,us,sigmas,rhos,current_state], 
                                                                        feed_dict={x: batch_x,
                                                                        y: batch_y,
                                                                        dropout: 0.0,
                                                                        init_state: _current_state})
                            # strokes[ii,predict_step,:,:] = util.sample(mixture_pi_temp, us_temp[:,:20], us_temp[:,20:],
                            #                         sigmas_temp[:,:20], sigmas_temp[:,20:],
                            #                         rhos_temp,num_mixtures=20)

                            # test_out.append([mixture_pi_temp,us_temp,sigmas_temp,rhos_temp])
                            # temp_newdata = util.generate_fake_batch_mixture(mixture_pi_temp,us_temp,sigmas_temp,rhos_temp)
                            predictions = [mixture_pi_temp,us_temp,sigmas_temp,rhos_temp]
                        
                        temp_newdata = util.sample_mixture(predictions)
                        test_out.append(temp_newdata)
                        # test_out.append(util._denormalize_data(temp_newdata,mu1,std1,mu2,std2))                    
                        if cfg.process_in_seconds:
                            temp_newdata = temp_newdata.reshape(batch_size,1,-1)

                    else:
                        utheta_temp,uphi_temp,sigma_theta_temp,sigma_phi_temp,rho_temp,_current_state = sess.run([utheta,uphi,sigma_theta,sigma_phi,rho,current_state], 
                                                                                            feed_dict={x: batch_x,
                                                                                            y: batch_y,
                                                                                            dropout: 0.0,
                                                                                            init_state: _current_state})
                        # test_out.append([utheta_temp,uphi_temp,sigma_theta_temp,sigma_phi_temp,rho_temp])
                        temp_newdata = util.generate_fake_batch_multivariate_normal_numpy(utheta_temp,uphi_temp,sigma_theta_temp,sigma_phi_temp,rho_temp,batch_size)
                        test_out.append(temp_newdata)
    
                if cfg.change_xyz2xxxyyyzzz:
                    temp_newdata = np.concatenate((temp_newdata[:,:,0::3],temp_newdata[:,:,1::3],temp_newdata[:,:,2::3]),axis=-1)
                if cfg.process_in_seconds:
                    batch_x = np.concatenate((batch_x[:,1:,:],temp_newdata),axis=1)
                else:
                    #shift by half of the running_length
                    # batch_x = np.concatenate((batch_x[:,int(0.5*cfg.running_length):,:],temp_newdata),axis=1)
                    batch_x = temp_newdata

        pickle.dump(test_out,open('LSTM_test_out'+tag+'.p','wb'))
        pickle.dump(gt_out,open('LSTM_gt_out'+tag+'.p','wb'))
        if cfg.use_GMM:
            pickle.dump(unnormalized_input,open('LSTM_gt_input'+tag+'.p','wb'))
            pickle.dump(strokes,open('strokes_'+tag+'.p','wb'))
        print("Test finished!")

        if cfg.concat_state:
            # get activation 
            mlp_trained_w_series,lstm_states_series = sess.run([mlp_state_series,states_series],feed_dict={x: batch_x, y: batch_y,
                                            dropout: 0.0,
                                            # others_future: batch_x_others,
                                            init_state: _current_state})

            pickle.dump(mlp_trained_w_series,open('mlp_trained_w_series.p','wb'))
            pickle.dump(lstm_states_series,open('lstm_states_series.p','wb'))
            print("weight finished!")











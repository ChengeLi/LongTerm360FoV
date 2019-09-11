
# coding: utf-8

# In[1]:

from __future__ import print_function
import tensorflow as tf
import random
import sys,glob
if './360video/' not in sys.path:
    sys.path.insert(0, './360video/')
from mycode.dataLayer import DataLayer
import mycode.cost as costfunc
from mycode.provide_hidden_state import multilayer_perceptron_hidden_state_series, multilayer_perceptron_hidden_state,dynamicRNN_hidden_state
from mycode.config import cfg
from mycode.dataIO import clip_xyz
import _pickle as pickle
import numpy as np
import pdb
import matplotlib.pyplot as plt


# In[2]:

tf.__version__


# In[1]:

is_test = False
training_epochs = 300
batch_size = 8 #need to be smaller than (one video length)/running_length/fps
display_step = 10
fps = 30
# Network Parameters
num_layers = 2
num_user = 48
truncated_backprop_length = cfg.running_length 
n_hidden = 64 # hidden layer num of features
n_output = cfg.running_length
# data_dim = 90


# In[3]:
experiment = 1
if cfg.use_xyz:
    all_video_data = pickle.load(open('./data/new_exp_'+str(experiment)+'_xyz.p','rb'))
    # all_video_data = pickle.load(open('./data/exp_'+str(experiment)+'_xyz.p','rb'))
    data_dim = 3
elif cfg.use_yaw_pitch_roll:
    all_video_data = pickle.load(open('./data/exp_2_raw.p','rb'))
    data_dim = 2 #only use yaw and pitch
elif cfg.use_cos_sin:
    all_video_data = pickle.load(open('./data/exp_2_raw_pair.p','rb'))
    data_dim = 2
elif cfg.use_phi_theta:
    all_video_data = pickle.load(open('./data/exp_2_phi_theta.p','rb'))
    data_dim = 2
if cfg.process_in_seconds:
    data_dim = data_dim*fps
all_video_data = clip_xyz(all_video_data)
datadb = all_video_data.copy()


# ## start to train
# Graph def

# In[2]:

# tf Graph input
# if cfg.own_history_only:
#     x = tf.placeholder("float", [None, truncated_backprop_length, data_dim])
# else:
#     x = tf.placeholder("float", [None, truncated_backprop_length, 48*data_dim])
x = tf.placeholder("float", [None, truncated_backprop_length, data_dim])
if cfg.has_reconstruct_loss:
    y = tf.placeholder("float", [None, cfg.running_length, data_dim])
else:
    y = tf.placeholder("float", [None, cfg.predict_len, data_dim])

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

# # Define weights
# weights = {
#     'out': tf.Variable(tf.random_normal([n_hidden, n_output]))
# }
# biases = {
#     'out': tf.Variable(tf.random_normal([n_output]))
# }


# ## Model Choices

# In[5]:

# -------------- ConvLSTM ---------------

# import keras
# keras.layers.ConvLSTM2D(filters, kernel_size, strides=(1, 1), padding='valid', data_format=None, dilation_rate=(1, 1), activation='tanh', recurrent_activation='hard_sigmoid', use_bias=True, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros', unit_forget_bias=True, kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, return_sequences=False, go_backwards=False, stateful=False, dropout=0.0, recurrent_dropout=0.0)
# m = Split(25)(m)
# for i in range(3):
#     m = keras.layers.ConvLSTM2D(2 ** (i + 3), (3, 3), padding='same', return_sequences=True)(m)
#     m = keras.layers.BatchNormalization()(m)

def convlstm(x):
    """2 two layer conv LSTM using dynamic_rnn"""
    cells = []
    for _ in range(num_layers):
        cell= tf.contrib.rnn.ConvLSTMCell(
                conv_ndims=2,
                input_shape=[num_user-1, fps, 3],
                output_channels=32,
                kernel_shape=[num_user-1, 3],
                use_bias=True,
                skip_connection=False,
                forget_bias=1.0,
                initializers=None,
                name="conv_lstm_cell")
        cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=1.0 - dropout)
        cells.append(cell)
    cell = tf.contrib.rnn.MultiRNNCell(cells)
    
    initial_state = cell.zero_state(batch_size, dtype=tf.float32)
    states_series, current_state =tf.nn.dynamic_rnn(cell,x,initial_state=initial_state,time_major=False,dtype="float32")
    return states_series, current_state


# In[6]:

def dynamicRNN(x,dropout):
    """2 two layer LSTM using dynamic_rnn"""
    cells = []
    for _ in range(num_layers):
        # cell = tf.contrib.rnn.GRUCell(n_hidden)
        cell = tf.contrib.rnn.LSTMCell(n_hidden,state_is_tuple=True)
        cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=1.0 - dropout)
        cells.append(cell)
    cell = tf.contrib.rnn.MultiRNNCell(cells)

    # Batch size x time steps x features.
    states_series, current_state = tf.nn.dynamic_rnn(cell, x, dtype=tf.float32,
                                      initial_state=rnn_tuple_state)

    # # Batch size x time steps x 1 (1 output at 1 timestamp)
    # pred_output = tf.contrib.layers.fully_connected(
    #                 states_series, 1, activation_fn=None) 
    return states_series, current_state


# In[7]:


def dynamicRNN_decoder(last_x,dropout,init_state_decoder):
    """decoder"""
    """2 two layer LSTM using dynamic_rnn"""

    # states c,h
    # init_state_decoder = tf.placeholder(tf.float32, [num_layers, 2, batch_size, n_hidden])
    state_per_layer_list_decoder = tf.unstack(init_state_decoder, axis=0)
    rnn_tuple_state_decoder = tuple([tf.contrib.rnn.LSTMStateTuple(state_per_layer_list_decoder[idx][0],
                    state_per_layer_list_decoder[idx][1])
                    for idx in range(num_layers)])

    cells = []
    n_hidden = 64*2 #concat
    for _ in range(num_layers):
        # cell = tf.contrib.rnn.GRUCell(n_hidden)
        cell = tf.contrib.rnn.LSTMCell(n_hidden,state_is_tuple=True,name='decoder_cell')
        cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=1.0 - dropout)
        cells.append(cell)
    cell = tf.contrib.rnn.MultiRNNCell(cells)

    # Batch size x time steps x features.
    states_series, last_state = tf.nn.dynamic_rnn(cell, last_x, dtype=tf.float32,
                                      initial_state=rnn_tuple_state_decoder)
    return states_series, last_state


# In[8]:

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



def concat_current_state_convLSTM(current_state,current_state_others,mode=None):
    if mode=='init_decoder':
        """concat each layer's hidden and memery states from LSTM and others' conv LSTM, 
        initialize decoder's memory and hidden states"""
        
        c0,h0 = current_state[0][0],current_state[0][1]
        c1,h1 = current_state[1][0],current_state[1][1]
        c_others0,h_others0 = current_state_others[0][0],current_state_others[0][1]
        c_others1,h_others1 = current_state_others[1][0],current_state_others[1][1]
        
        # change 32*47*30*32 into 32*64
        c_others0 = tf.contrib.layers.fully_connected(
                    tf.contrib.layers.flatten(c_others0,0), 64, activation_fn=None)
        h_others0 = tf.contrib.layers.fully_connected(
                    tf.contrib.layers.flatten(h_others0,0), 64, activation_fn=None)
        c_others1 = tf.contrib.layers.fully_connected(
                    tf.contrib.layers.flatten(c_others1,0), 64, activation_fn=None)
        h_others1 = tf.contrib.layers.fully_connected(
                    tf.contrib.layers.flatten(h_others1,0), 64, activation_fn=None)
        
        concat_state0 = tf.reshape(tf.stack((h0,h_others0),axis=-1),[batch_size,-1])
        concat_state1 = tf.reshape(tf.stack((h1,h_others1),axis=-1),[batch_size,-1])
        concat_memo0 = tf.reshape(tf.stack((c0,c_others0),axis=-1),[batch_size,-1])
        concat_memo1 = tf.reshape(tf.stack((c1,c_others1),axis=-1),[batch_size,-1])

        return ((concat_memo0,concat_state0),(concat_memo1,concat_state1))

    else:
        """concat last hidden and memery states from LSTM and others' conv LSTM, 
        predict using another FC"""
        if len(current_state)>1: 
            # multiple layers
            # only keep the last [c,h]
            current_state = current_state[-1]
        if len(current_state_others)>1: 
            current_state_others = current_state_others[-1]
        
        c,h = current_state[0],current_state[1]
        c_others,h_others = current_state_others[0],current_state_others[1]
        
        # change 32*47*30*32 into 32*64
        c_others = tf.contrib.layers.fully_connected(
                    tf.contrib.layers.flatten(c_others,0), 64, activation_fn=None)
        h_others = tf.contrib.layers.fully_connected(
                    tf.contrib.layers.flatten(h_others,0), 64, activation_fn=None)

        concat_state = tf.reshape(tf.stack((h,h_others),axis=-1),[batch_size,-1])
        concat_memo = tf.reshape(tf.stack((c,c_others),axis=-1),[batch_size,-1])

        return concat_state,concat_memo


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
def _pred_mean_var_phi_theta(state):
    # predict mean and variance for x,y,z
    pred = tf.contrib.layers.fully_connected(state, 2*2, activation_fn=None)
    u1 = tf.slice(pred,[0,0],[-1,1])
    u2 = tf.slice(pred,[0,1],[-1,1])
    # variance must >0
    var1 = tf.abs(tf.slice(pred,[0,2],[-1,1]))
    var2 = tf.abs(tf.slice(pred,[0,3],[-1,1]))
    return u1,u2,var1,var2

def _reshape_batchsize(tensor):
    return tf.reshape(tensor,[batch_size,1])


def get_gt_target_xyz(y):
    """get gt mean var"""
    target_x = y[:,0,0:data_dim:3]
    target_y = y[:,0,1:data_dim:3]
    target_z = y[:,0,2:data_dim:3]
    gt_mean_x, gt_var_x = tf.nn.moments(target_x, axes=[1])
    gt_mean_y, gt_var_y = tf.nn.moments(target_y, axes=[1])
    gt_mean_z, gt_var_z = tf.nn.moments(target_z, axes=[1])

    target = (_reshape_batchsize(gt_mean_x),_reshape_batchsize(gt_mean_y),_reshape_batchsize(gt_mean_z),
                _reshape_batchsize(gt_var_x),_reshape_batchsize(gt_var_y),_reshape_batchsize(gt_var_z))
    return target


# def _filter_stuffed_fake_future_val(data_further):
#     tf.equal(data_further,1.123)
#     # TODO




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
    target_phi = y[:,0,0:data_dim:3]
    target_theta = y[:,0,1:data_dim:3]
    gt_mean_phi, gt_var_phi = tf.nn.moments(target_phi, axes=[1])
    gt_mean_theta, gt_var_theta = tf.nn.moments(target_theta, axes=[1])

    target = (_reshape_batchsize(gt_mean_phi),_reshape_batchsize(gt_mean_theta),
                _reshape_batchsize(gt_var_phi),_reshape_batchsize(gt_var_theta))
    return target



def generate_fake_batch(mu,var):
    """generate new data for 1 second using predicted mean and variance"""
    temp = []
    for ii in range(batch_size):
        temp.append(np.random.normal(mu[ii,0], np.sqrt(var[ii,0]), fps*1))
    return temp


# ### target user LSTM

# In[9]:

states_series, current_state = dynamicRNN(x,dropout)


# ### other users' LSTM

# In[10]:

cfg.concat_state = True
if cfg.concat_state:
    ## fc lstm
    #states_series_others, current_state_others = dynamicRNN_hidden_state(others_future,num_layers,n_hidden,rnn_tuple_state)

    ## conv lstm
    # 5-D tensor
    x_others= tf.placeholder(tf.float32, [None, cfg.running_length, num_user-1, fps, 3])
    # states_series_others, current_state_others = dynamicRNN(x_others,dropout)
    states_series_others, current_state_others = convlstm(x_others)

    if cfg.use_decoder:
        init_state_decoder = concat_current_state_convLSTM(current_state,current_state_others,mode='init_decoder')  
        last_x= tf.placeholder(tf.float32, [None, 1, data_dim])
        states_series_decoder,last_state_decoder = dynamicRNN_decoder(last_x,dropout,init_state_decoder)
        ux,uy,uz,varx,vary,varz = _pred_mean_var_xyz(last_state_decoder[1][1])
    else:
        # use MLP to predict one step further
        concat_state, concat_memo = concat_current_state_convLSTM (current_state,current_state_others)  
        ux,uy,uz,varx,vary,varz = _pred_mean_var_xyz(concat_state)
 
else:    
    ux,uy,uz,varx,vary,varz = _pred_mean_var_xyz(current_state[1][1])






## loss op
target = get_gt_target_xyz(y)
# population_target = get_gt_target_xyz_pop(others_future_further)
# cost = costfunc._mean_var_cost_xyz(ux,uy,uz,varx,vary,varz,target,population_target)
cost = costfunc._mean_var_cost_xyz(ux,uy,uz,varx,vary,varz,target)
# # cost,cost_x,cost_y,cost_z = costfunc._mean_var_cost_xyz_metric(ux,uy,uz,varx,vary,varz,target,costfunc.Bhattacharyya_distance)
# # cost,cost_x,cost_y,cost_z = costfunc._mean_var_cost_xyz_metric(ux,uy,uz,varx,vary,varz,target,costfunc.Wasserstein_distance)
# # cost,cost_x,cost_y,cost_z = costfunc._mean_var_cost_xyz_metric(ux,uy,uz,varx,vary,varz,target,costfunc.Kullback_Leibler_divergence_Gaussian)

# directly predict new samples
# predict_sample = tf.contrib.layers.fully_connected(current_state[1][1], fps*1*3, activation_fn=None)
# predict_sample = tf.reshape(predict_sample,[batch_size,fps*1,3])
# cost,inspect1,inspect2 = costfunc.conditional_prob_loss(predict_sample,target,population_target)







# In[11]:

# summary
all_losses_dict = {}
# all_losses_dict['MSE_loss'] = cost1
all_losses_dict['modified_MSE_loss'] = cost
# all_losses_dict['modified_MSE_loss_staticRNN'] = cost1
event_summaries = {}
event_summaries.update(all_losses_dict)
summaries = []
for key, var in event_summaries.items():
    summaries.append(tf.summary.scalar(key, var))
summary_op = tf.summary.merge(summaries)

saver = tf.train.Saver()
lr = tf.Variable(cfg.LEARNING_RATE, trainable=False)
optimizer = tf.train.AdamOptimizer(learning_rate=lr)
gvs = optimizer.compute_gradients(cost)

def ClipIfNotNone(grad):
    if grad is None:
        return grad
    return tf.clip_by_value(grad, -10, 10)
if cfg.clip_gradient:
    clipped_gradients = [(ClipIfNotNone(grad), var) for grad, var in gvs]
    train_op = optimizer.apply_gradients(clipped_gradients)
else:
    train_op = optimizer.apply_gradients(gvs)


# In[13]:

# tag = "concat_convLSTM_decoder_newdata_june9"
tag = "concat_convLSTM_fc_newdata_june9"
model_path = "./model/LSTM_"+tag+".ckpt"
print(tag)

# In[14]:

init = tf.global_variables_initializer()


# ## training

# In[24]:

is_test = False
# Start training
if not is_test:
    data_io = DataLayer(datadb, random=False, is_test=False)
    with tf.Session() as sess:
        sess.run(init)
        starting_epoch = 0
        summary_writer = tf.summary.FileWriter('./tfsummary/'+tag, sess.graph)
        # if len(glob.glob(model_path+".meta"))>0:
        #     saver.restore(sess, model_path)
        #     print("Model restored.")
        #     starting_epoch = training_epochs
        if cfg.use_yaw_pitch_roll:
            total_batch = 8*int(datadb[0]['raw_yaw'].shape[1]/cfg.running_length/batch_size)
        elif cfg.use_xyz:
            total_batch = 8*int(datadb[0]['x'].shape[1]/cfg.running_length/fps/batch_size)
        elif cfg.use_phi_theta:
            total_batch = 8*int(datadb[0]['phi'].shape[1]/cfg.running_length/fps/batch_size)

        _current_state = np.zeros((num_layers, 2, batch_size, n_hidden))
        _concat_state_for_decoder = np.zeros((num_layers, 2, batch_size, n_hidden))
        for epoch in range(starting_epoch,starting_epoch+training_epochs,1):
            if epoch>10 and epoch%100==0:
                save_path = saver.save(sess, model_path)
                print("Model saved: %s" % save_path)
                lr_temp = cfg.LEARNING_RATE*(0.5**(epoch/100))
                print('epoch: ',epoch, ', change lr=lr*0.5, lr=', lr_temp)
                sess.run(tf.assign(lr, lr_temp))
            for step in range(total_batch):
                # TODO: zero before every new minibatch?
                # _current_state = np.zeros((num_layers, 2, batch_size, n_hidden))
    
                # batch_x, batch_y, batch_seqlen = trainset.next(batch_size)
                (batch_x, batch_y, batch_x_others),_,_,batch_x_others_further = data_io._get_next_minibatch(datadb,batch_size)
                # ## collapse last dimension, batch*len*(num_user*2)
                # batch_x = np.reshape(batch_x,[batch_size,batch_x.shape[1],-1])
                # batch_seqlen = np.array([cfg.running_length]*batch_size) #fixed length
              
                #get conv LSTM's batch
                batch_x_past,batch_x_future,batch_y2 = data_io._get_next_minibatch(datadb,batch_size,'convLSTM')
                # Run optimization op (backprop)
                _, _current_state = sess.run([train_op,current_state],
                                             feed_dict={x: batch_x, y: batch_y,
                                             x_others: batch_x_past,
                                             # last_x: batch_x[:,-1,:][:,np.newaxis,:],
                                             dropout: 0.1,
                                             # others_future_further:batch_x_others_further,
                                             # seqlen: batch_seqlen,
                                             # others_future: batch_x_others,
                                             init_state: _current_state})
                count = (step+1)*batch_size+epoch*total_batch*batch_size
                if step% display_step == 0 or step==0:
                    # # Calculate batch accuracy & loss
                    loss,summary,_current_state,ux_temp,uy_temp,uz_temp,varx_temp,vary_temp,varz_temp = sess.run(
                                    [cost,summary_op,current_state,ux,uy,uz,varx,vary,varz], 
                                    feed_dict={x: batch_x, y: batch_y,
                                               x_others: batch_x_past, 
                                               # last_x: batch_x[:,-1,:][:,np.newaxis,:],
                                               dropout: 0.1,
#                                                 others_future_further:batch_x_others_further,
                                                # seqlen: batch_seqlen,
                                                # others_future: batch_x_others,
                                                init_state: _current_state})
             
#                     loss,summary,_clipped_gradients,_gv,_current_state,_states_series,ux_temp,uy_temp,uz_temp,varx_temp,vary_temp,varz_temp = sess.run(
#                                     [cost,summary_op,clipped_gradients[2:],gvs[2:],current_state,states_series,ux,uy,uz,varx,vary,varz], 
#                                     feed_dict={x: batch_x, y: batch_y,
#                                                 x_others: batch_x_past,
#                                                dropout: 0.1,
#                                                 #others_future_further:batch_x_others_further,
#                                                 # seqlen: batch_seqlen,
#                                                 # others_future: batch_x_others,
#                                                 init_state: _current_state})                    
                    
                    
#                     loss,summary,_inspect1,_inspect2 = sess.run(
#                                     [cost,summary_op,inspect1,inspect2], 
#                                     feed_dict={x: batch_x, y: batch_y,
#                                                 dropout: 0.1,
#                                                 others_future_further:batch_x_others_further,
#                                                 # seqlen: batch_seqlen,
#                                                 # others_future: batch_x_others,
#                                                 init_state: _current_state})                    
                    


                    summary_writer.add_summary(summary, float(count))
                    print("Step " + str(count) + ", Minibatch Loss= " +                           "{:.6f}".format(loss))

        print("Optimization Finished!")
        save_path = saver.save(sess, model_path)
        print("Model saved: %s" % save_path)  


# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# ## testing

# In[ ]:

# test
is_test=True
with tf.Session() as sess:
    # Run the initializer
    sess.run(init) 
    saver.restore(sess, model_path)
    print("Model restored. ",model_path)
    data_io_test = DataLayer(datadb, random=False, is_test=True)
    batch_seqlen = np.array([cfg.running_length]*batch_size) #fixed length
    test_out = []
    gt_out = []

    for ii in range(10):
        # every test time feed in zero state?
        _current_state = np.zeros((num_layers, 2, batch_size, n_hidden))
        (batch_x, batch_y, batch_x_others), batch_y_further,db_index,batch_x_others_further = data_io_test._get_next_minibatch(datadb,batch_size)
        gt_out.append(batch_y_further)

        #get conv LSTM's batch
        batch_x_past,batch_x_future,batch_y2 = data_io_test._get_next_minibatch(datadb,batch_size,'convLSTM')
        batch_x_others_past_fut = np.concatenate((batch_x_past,batch_x_future),axis=1)        
        for predict_step in range(cfg.predict_step):
            # feed prediction back as input!
            if cfg.use_xyz:
                loss,ux_temp,uy_temp,uz_temp,varx_temp,vary_temp,varz_temp,_current_state = sess.run(
                                        [cost,ux,uy,uz,varx,vary,varz,current_state],
                                        feed_dict={x: batch_x, y: batch_y,
                                        x_others: batch_x_past, 
                                        # last_x: batch_x[:,-1,:][:,np.newaxis,:],
                                        dropout: 0.0,
                                        # seqlen: batch_seqlen,
                                        # others_future: batch_x_others,
#                                         others_future_further:batch_x_others_further,
                                        init_state: _current_state})
                test_out.append([ux_temp,uy_temp,uz_temp,varx_temp,vary_temp,varz_temp])
                temp_newdata = np.stack((generate_fake_batch(ux_temp,varx_temp),
                                generate_fake_batch(uy_temp,vary_temp),
                                generate_fake_batch(uz_temp,varz_temp)),axis=-1)[:,np.newaxis,:,:].reshape((batch_size,1,-1))

                print ('loss ',loss)
            batch_x = np.concatenate((batch_x[:,1:,:],temp_newdata),axis=1)
            batch_x_past = batch_x_others_past_fut[:,predict_step+1:predict_step+1+cfg.predict_step,:,:,:]
            
            
            ###exeperiments
#             batch_x = np.repeat(temp_newdata, 5, axis=1)
#             batch_x = np.concatenate((np.zeros((32,4,90)),temp_newdata),axis=1)
#             batch_x = np.zeros((32,5,90))
        
    pickle.dump(test_out,open('LSTM_test_out'+tag+'.p','wb'))
    pickle.dump(gt_out,open('LSTM_gt_out'+tag+'.p','wb'))
    print("Test finished!")





# In[ ]:




# In[ ]:




# In[ ]:




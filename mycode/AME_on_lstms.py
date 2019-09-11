"""
Attentive Mixture of Experts
Each other user has one LSTM. Compute the similarity between <h_tar,h_i>

"""
from __future__ import print_function
from keras.models import Model
from keras.layers import Input, LSTM, Dense, Reshape, Add
from keras.layers import Lambda,Concatenate,Flatten
from keras.layers import Permute,multiply,Softmax
from keras.layers import TimeDistributed
from keras.callbacks import ModelCheckpoint,ReduceLROnPlateau,EarlyStopping
from keras import backend as K
from keras.models import load_model
from keras import activations,optimizers
import sys,glob,io,random
if './360video/' not in sys.path:
    sys.path.insert(0, './360video/')
from mycode.dataLayer import DataLayer
import mycode.cost as costfunc
from mycode.config import cfg
from mycode.dataIO import clip_xyz
import mycode.utility as util
from random import shuffle  
import matplotlib.pyplot as plt
import _pickle as pickle
import numpy as np
import pdb
from keras.layers.wrappers import Bidirectional
from mycode.saliency_CNN import *
from tqdm import tqdm
from collections import OrderedDict
import h5py

batch_size = cfg.batch_size
epochs = 200
latent_dim = 32

fps = cfg.fps
num_encoder_tokens = 3*fps
num_decoder_tokens = 6
max_encoder_seq_length = cfg.running_length
max_decoder_seq_length = cfg.predict_step
num_user = 34 #48 
use_generator = False




mlp_mixing = False
ame = True
oth_from_past = True #others' LSTM starts from the very beginning
shared_LSTM = True
shared_WiVi = True

## utility layers
flatten_layer = Flatten()
expand_dim_layer = Lambda(lambda x: K.expand_dims(x,1))
expand_dim_layer_1 = Lambda(lambda x: K.expand_dims(x,-1))
Concatenatelayer = Concatenate(axis=1)
Concatenatelayer_1 = Concatenate(axis=-1)
get_dim1_layer = Lambda(lambda x: x[:,0,:])

reduce_sum_layer = Lambda(lambda x: K.sum(x, axis=1))#collapsing user dim
reduce_sum_layer_1 = Lambda(lambda x: K.sum(x, axis=-1))


def generate_fake_batch(x):
    """generate new data for 1 second using predicted mean and variance"""
    # batch_size = 64
    # fps =30
    mu = x[0]
    var = x[1]
    temp = K.random_normal(shape = (batch_size,fps,1), mean=mu,stddev=var)
    return temp

generate_fake_batch_layer = Lambda(lambda x: generate_fake_batch(x))



### ====================Graph def====================
#2 layer encoder
if cfg.input_mean_var:
    encoder_inputs = Input(shape=(None, 6))
    if cfg.stateful_across_batch:
        encoder_inputs = Input(batch_shape=(batch_size, None, 6))
else:
    encoder_inputs = Input(shape=(None, num_encoder_tokens))

encoder1 = LSTM(latent_dim, stateful=cfg.stateful_across_batch, return_state=True, return_sequences=True)
encoder1_outputs, state_h_1, state_c_1 = encoder1(encoder_inputs)

encoder2 = LSTM(latent_dim, stateful=cfg.stateful_across_batch, return_state=True, return_sequences=True)
encoder2_outputs, state_h_2, state_c_2 = encoder2(encoder1_outputs)

encoder1_states = [state_h_1, state_c_1]
encoder2_states = [state_h_2, state_c_2]

##2 layer decoder
decoder1_states_inputs = encoder1_states
decoder2_states_inputs = encoder2_states

if cfg.sample_and_refeed:
    decoder_inputs = Input(shape=(1, num_encoder_tokens))    
else:
    decoder_inputs = Input(shape=(1, num_decoder_tokens))    

decoder_lstm1 = LSTM(latent_dim, stateful=cfg.stateful_across_batch, return_sequences=True, return_state=True)
decoder_lstm2 = LSTM(latent_dim, stateful=cfg.stateful_across_batch, return_sequences=True, return_state=True)
if cfg.predict_mean_var:
    decoder_dense = Dense(num_decoder_tokens,activation='tanh')
else:
    decoder_dense = Dense(num_encoder_tokens,activation='tanh')


if oth_from_past:
    others_inputs = Input(shape=(max_encoder_seq_length+max_decoder_seq_length,(num_user-1),3*fps))    
else:
    # others_inputs = Input(shape=(max_decoder_seq_length,(num_user-1),6))
    others_inputs = Input(shape=(max_decoder_seq_length,(num_user-1),3*fps))

if mlp_mixing:
    mixing = Dense(num_decoder_tokens,activation='tanh')
if ame:
    if not shared_LSTM and not shared_WiVi: #all separate
        W_i_pool = []
        V_i_pool = []
        lstm_pool = []
        for user_ind in range(num_user):
            W_i_pool.append(Dense(256,activation='relu'))
            V_i_pool.append(Dense(256,activation='relu'))
            lstm_pool.append(LSTM(latent_dim*2, stateful=cfg.stateful_across_batch, return_state=True, return_sequences=True)) #latent_dim
    if shared_LSTM:
        lstm = LSTM(latent_dim, stateful=cfg.stateful_across_batch, return_state=True, return_sequences=True)
    if shared_WiVi:
        W_i=Dense(256,activation='relu')
        V_i=Dense(256,activation='relu')
get_dim2_layer = Lambda(lambda x: x[:,:,0,:])
expand_dim2_layer = Lambda(lambda x: K.expand_dims(x,2))
oth_states_list=[]
for user_ind in range(num_user-1):
    if shared_LSTM:
        oth_states, state_h_oth, state_c_oth = lstm(get_dim2_layer(util.slice_layer(2,user_ind,user_ind+1)(others_inputs)))
    else:
        oth_states, state_h_oth, state_c_oth = lstm_pool[user_ind](get_dim2_layer(util.slice_layer(2,user_ind,user_ind+1)(others_inputs)))
    oth_states_list.append(expand_dim2_layer(oth_states))
oth_states_list = Lambda(lambda x: K.concatenate(x, axis=2))(oth_states_list)#(batch,10,33,64) or (batch,20,33,64) 


# ************************************************************
def ui_usi_similarity(u_i_list,u_si_list):
    logit = reduce_sum_layer_1(multiply([u_si_list,u_i_list]))#(batch,34)
    alpha_list = Softmax()(logit)
    return alpha_list

def merge_experts_contribution(alpha_list,concat_outputs):
    # weighted sum on locations
    contributions = []
    for ii in range(num_user):
        alpha = util.slice_layer(1,ii,ii+1)(alpha_list)
        contribution = util.slice_layer(1,ii,ii+1)(concat_outputs)
        contributions.append(multiply([alpha,contribution]))

    contributions = Lambda(lambda x: K.concatenate(x, axis=1))(contributions)
    outputs = reduce_sum_layer(contributions)
    outputs = expand_dim_layer(outputs)
    return outputs

def merge_experts_contribution2(alpha_list,concat_hidden):
    # weighted sum on hidden states
    contributions = []
    for ii in range(num_user):
        alpha = util.slice_layer(1,ii,ii+1)(alpha_list)
        contribution = util.slice_layer(1,ii,ii+1)(concat_hidden)
        contributions.append(multiply([alpha,contribution]))

    contributions = Lambda(lambda x: K.concatenate(x, axis=1))(contributions)
    outputs = reduce_sum_layer(contributions)
    final_dense = Dense(num_decoder_tokens,activation='tanh')
    outputs = final_dense(outputs)
    outputs = expand_dim_layer(outputs)
    return outputs

all_outputs = []
inputs = decoder_inputs
for time_ind in range(max_decoder_seq_length):
    decoder1_outputs, state_decoder1_h, state_decoder1_c = decoder_lstm1(inputs,
                                         initial_state=decoder1_states_inputs)
    decoder1_states_inputs = [state_decoder1_h, state_decoder1_c]
    decoder2_outputs, state_decoder2_h, state_decoder2_c = decoder_lstm2(decoder1_outputs,initial_state=decoder2_states_inputs)
    decoder2_states_inputs = [state_decoder2_h, state_decoder2_c]

    if oth_from_past:
        #let the hidden states evolve from past but only use the future part
        gt_mean_var_oth = util.slice_layer(1,time_ind+max_encoder_seq_length,time_ind+max_encoder_seq_length+1)(others_inputs) 
        # oth_states_slice = util.slice_layer(1,time_ind+max_encoder_seq_length,time_ind+max_encoder_seq_length+1)(oth_states_list)  #use h_t
        oth_states_slice = util.slice_layer(1,time_ind+max_encoder_seq_length-1,time_ind+max_encoder_seq_length)(oth_states_list)  #use h_t-1
    else:
        gt_mean_var_oth = util.slice_layer(1,time_ind,time_ind+1)(others_inputs) 
        oth_states_slice = util.slice_layer(1,time_ind,time_ind+1)(oth_states_list) 

    if cfg.sample_and_refeed:
        concat_outputs = Concatenatelayer([get_dim1_layer(gt_mean_var_oth),inputs])#concat 90
    else:
        decoder_pred = decoder_dense(decoder2_outputs)
        concat_outputs = Concatenatelayer([get_dim1_layer(gt_mean_var_oth),decoder_pred])#concat 6

    if mlp_mixing:
        concat_outputs = Flatten()(concat_outputs)
        outputs = mixing(concat_outputs)
        outputs = expand_dim_layer(outputs)
    if ame:
        u_i_list = []
        u_si_list = []
        for user_ind in range(num_user-1):
            #for all other users
            # h_i = decoder_pred
            h_i = decoder2_outputs #also use hidden states for h_tar
            if not shared_WiVi:                
                W_i = W_i_pool[user_ind]
                V_i = V_i_pool[user_ind]
            h_all = util.slice_layer(1,user_ind,user_ind+1)(get_dim1_layer(oth_states_slice)) #use hidden_i as the h_i
            u_i_list.append(W_i(h_all))
            # u_i_list.append(h_all)#no embedding
            u_si_list.append(V_i(h_i))
            # u_si_list.append(h_i)
        #for target user
        # h_i = decoder_pred
        h_i = decoder2_outputs#also use hidden states for h_tar
        h_all = h_i
        if not shared_WiVi:                
            W_i = W_i_pool[-1]
            V_i = V_i_pool[-1]
        u_i_list.append(W_i(h_all))
        # u_i_list.append(h_all)
        u_si_list.append(V_i(h_i))
        # u_si_list.append(h_i)

        assert len(u_i_list)==num_user
        assert len(u_si_list)==num_user
        u_i_list = Lambda(lambda x: K.concatenate(x, axis=1))(u_i_list)#(batch,34,256)
        u_si_list = Lambda(lambda x: K.concatenate(x, axis=1))(u_si_list)#(batch,34,256)
        alpha_list = ui_usi_similarity(u_i_list,u_si_list)#(batch,34)
        outputs = merge_experts_contribution(alpha_list,concat_outputs)
        # outputs = merge_experts_contribution2(alpha_list,u_i_list)


    if cfg.predict_mean_var and cfg.sample_and_refeed:
        outputs = decoder_dense(outputs)
        #for training            
        ### generated from gaussian
        ux_temp = util.slice_layer(2,0,1)(outputs)
        uy_temp = util.slice_layer(2,1,2)(outputs)
        uz_temp = util.slice_layer(2,2,3)(outputs)
        varx_temp = util.slice_layer(2,3,4)(outputs)
        vary_temp = util.slice_layer(2,4,5)(outputs)
        varz_temp = util.slice_layer(2,5,6)(outputs)

        temp_newdata = expand_dim_layer(expand_dim_layer(Concatenatelayer_1([generate_fake_batch_layer([ux_temp,varx_temp]),
                            generate_fake_batch_layer([uy_temp,vary_temp]),
                            generate_fake_batch_layer([uz_temp,varz_temp])])))
        inputs = Reshape((1,3*fps))(temp_newdata)
    else:
        inputs = outputs

    all_outputs.append(outputs)


## Concatenate all predictions
decoder_outputs = Lambda(lambda x: K.concatenate(x, axis=1))(all_outputs)

RMSprop = optimizers.RMSprop(lr=1e-4, rho=0.9, epsilon=None, decay=0.0)
model = Model([encoder_inputs, others_inputs, decoder_inputs], decoder_outputs)
model.compile(optimizer=RMSprop, loss='mean_squared_error',metrics=['accuracy'])
#use two losses
# model = Model([encoder_inputs, others_inputs, decoder_inputs], [decoder_outputs,decoder_outputs])
# model.compile(optimizer='Nadam', loss=[costfunc.likelihood_loss,"mean_squared_error"],loss_weights=[1,1])







#### ========================================data============================================================
def _reshape_others_data(_video_db_oth):
    ## to match Input shape: others_inputs
    _video_db_oth = _video_db_oth.transpose((1,2,0,3))
    _video_db_oth = _video_db_oth.reshape((_video_db_oth.shape[0],_video_db_oth.shape[1],_video_db_oth.shape[2],
                    fps,3))
    return _video_db_oth

#### load cached data format4 
dataformat = 'format4' #shanghaitech
option='stride10_cut_head/'
_video_db_tar = util.load_h5('./cache/'+dataformat+'/train/'+option+'_video_db_tar.h5','_video_db_tar')
_video_db_future_tar = util.load_h5('./cache/'+dataformat+'/train/'+option+'_video_db_future_tar.h5','_video_db_future_tar')
_video_db_future_input_tar = util.load_h5('./cache/'+dataformat+'/train/'+option+'_video_db_future_input_tar.h5','_video_db_future_input_tar')
_video_db_oth = util.load_h5('./cache/'+dataformat+'/train/'+option+'_video_db_oth.h5','_video_db_oth')
_video_db_future_oth = util.load_h5('./cache/'+dataformat+'/train/'+option+'_video_db_future_oth.h5','_video_db_future_oth')
# _video_db_future_input_oth = util.load_h5('./cache/'+dataformat+'/train/'+option+'_video_db_future_input_oth.h5','_video_db_future_input_oth')


_video_db_oth = _reshape_others_data(_video_db_oth)
_video_db_future_oth = _reshape_others_data(_video_db_future_oth)
_video_db_tar = _video_db_tar.reshape((_video_db_tar.shape[0],_video_db_tar.shape[1],fps,3))
_video_db_future_tar = _video_db_future_tar.reshape((_video_db_tar.shape[0],_video_db_tar.shape[1],fps,3))
_video_db_future_input_tar = _video_db_future_input_tar.reshape((_video_db_tar.shape[0],_video_db_tar.shape[1],fps,3))


print('other data shape: ',_video_db_oth.shape)
print('other data shape: ',_video_db_future_oth.shape)
print('target user data shape: ',_video_db_tar.shape)
print('target user data shape: ',_video_db_future_tar.shape)


#### prepare training data
if cfg.input_mean_var:
    ### target user
    encoder_input_data = util.get_gt_target_xyz(_video_db_tar)
    decoder_target_data = util.get_gt_target_xyz(_video_db_future_tar)
    # encoder_input_data = util.get_gt_target_xyz(_video_db_oth_all)
    # decoder_target_data = util.get_gt_target_xyz(_video_db_future_oth_all)
    # ### other users
    others_input_data = util.get_gt_target_xyz_oth(_video_db_future_oth)
    if not cfg.teacher_forcing:
        decoder_input_data = encoder_input_data[:,-1,:][:,np.newaxis,:]
    else:
        decoder_input_data = util.get_gt_target_xyz(_video_db_future_input_tar)

else:
    ### target user
    _video_db_tar = _video_db_tar.reshape((_video_db_tar.shape[0],_video_db_tar.shape[1],-1))
    encoder_input_data = _video_db_tar
    # decoder_target_data = _video_db_future_tar.reshape((_video_db_future_tar.shape[0],_video_db_future_tar.shape[1],-1)) #predict raw
    decoder_target_data = util.get_gt_target_xyz(_video_db_future_tar)#predict mean/var
    decoder_input_data = encoder_input_data[:,-1,:][:,np.newaxis,:]
    # decoder_input_data = util.get_gt_target_xyz(encoder_input_data[:,-1,:][:,np.newaxis,:])
    ### other users
    if oth_from_past:
        _video_db_past_future_oth = np.concatenate((_video_db_oth,_video_db_future_oth),axis=1)
        others_input_data = _video_db_past_future_oth.reshape((_video_db_past_future_oth.shape[0],_video_db_past_future_oth.shape[1],num_user-1,-1))        
    else:
        others_input_data = _video_db_future_oth.reshape((_video_db_future_oth.shape[0],_video_db_future_oth.shape[1],num_user-1,-1))
    # others_input_data = util.get_gt_target_xyz_oth(_video_db_future_oth)



if cfg.sample_and_refeed or cfg.stateful_across_batch:
    # if using the generate fake batch layer, the dataset size has to
    # be dividable by the batch size
    sample_ind = util.rand_sample_ind(encoder_input_data.shape[0],0,batch_size,validation_ratio=0.1)
    if not cfg.shuffle_data:
        sample_ind = sorted(sample_ind)
    encoder_input_data = util.rand_sample(encoder_input_data,sample_ind)
    decoder_input_data = util.rand_sample(decoder_input_data,sample_ind)
    decoder_target_data = util.rand_sample(decoder_target_data,sample_ind)
    others_input_data = util.rand_sample(others_input_data,sample_ind)
    # sanity check
    ind = np.random.randint(encoder_input_data.shape[0])
    assert encoder_input_data[ind,-1,:].sum()==decoder_input_data[ind,0,:].sum()


### ====================Training====================
# tag='AMEonLSTMs_pair_similarity_sep26'
# tag='AMEonLSTMs_tar_also_hidden_pair_similarity_sep26' ##also use hidden states for h_tar, but h_i dim 64, h_t 32
# tag='AMEonLSTMs_tar_also_hidden_pair_similarity_noembedding_sep27'#others use 32 dim LSTM
# tag='AMEonLSTMs_tar_hidden_pair_sim_NLL_sep27'
# tag='AMEonLSTMs_tar_hidden_pair_sim_NLL_MSE_sep28'
# tag='AMEonLSTMs_oth_sharedLSTM_sep30'
# tag='AMEonLSTMs_oth_sharedLSTM_allraw_sep30'#encoder, decoder and others' LSTM all using raw trj
# tag='AMEonLSTMs_oth_sharedLSTM_allraw_weightedhidden_sep30'
# tag='AMEonLSTMs_oth_starts_past_oct23'
# tag='AMEonLSTMs_oth_starts_past_ht-1_oct23' 
tag='AMEonLSTMs_oth_starts_past_ht-1_sharedLSTM_sharedwi_nov16'  #all 90 input  both hidden states sim



model_checkpoint = ModelCheckpoint(tag+'{epoch:02d}-{val_loss:.4f}.h5', monitor='val_loss', save_best_only=False)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                 patience=2, min_lr=1e-6)
stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='auto')


# model.fit([encoder_input_data, others_input_data, decoder_input_data], [decoder_target_data,decoder_target_data_muvar],#two losses
model.fit([encoder_input_data, others_input_data, decoder_input_data], decoder_target_data,
      batch_size=batch_size,
      epochs=epochs,
      validation_split=0.1,
      shuffle=False, 
      initial_epoch=0,
      callbacks=[model_checkpoint, reduce_lr, stopping])



### ====================Testing===================
# model.load_weights('AMEonLSTMs_oth_sharedLSTM_allraw_weightedhidden_sep3034-0.0935.h5')
# model.load_weights('AMEonLSTMs_oth_starts_past_ht-1_oct2310-0.0948.h5')
# model.load_weights('AMEonLSTMs_pair_similarity_sep2606-0.0900.h5')
# dataformat = 'format4' #shanghaitech
# option='stride10_cut_head/'
# select_k_neighbours=False
# thu_tag=''
dataformat = 'format5_tsinghua_by_sec_interp' #tsinghua
option=''
thu_tag='_thu_'
select_k_neighbours=True
_video_db_tar = util.load_h5('./cache/'+dataformat+'/test/'+option+'_video_db_tar.h5','_video_db_tar')
_video_db_future_tar = util.load_h5('./cache/'+dataformat+'/test/'+option+'_video_db_future_tar.h5','_video_db_future_tar')
_video_db_future_input_tar = util.load_h5('./cache/'+dataformat+'/test/'+option+'_video_db_future_input_tar.h5','_video_db_future_input_tar')
_video_db_oth = util.load_h5('./cache/'+dataformat+'/test/'+option+'_video_db_oth.h5','_video_db_oth')
_video_db_future_oth = util.load_h5('./cache/'+dataformat+'/test/'+option+'_video_db_future_oth.h5','_video_db_future_oth')


if cfg.input_mean_var:
    _video_db_tar = _video_db_tar.reshape((_video_db_tar.shape[0],_video_db_tar.shape[1],fps,3))
_video_db_future_tar = _video_db_future_tar.reshape((_video_db_tar.shape[0],_video_db_tar.shape[1],fps,3))
_video_db_future_input_tar = _video_db_future_input_tar.reshape((_video_db_tar.shape[0],_video_db_tar.shape[1],fps,3))
_video_db_future_oth = _reshape_others_data(_video_db_future_oth)

if oth_from_past:
    _video_db_oth = _reshape_others_data(_video_db_oth)
    _video_db_past_future_oth = np.concatenate((_video_db_oth,_video_db_future_oth),axis=1)

if select_k_neighbours:
    # _video_db_future_oth = util.get_random_k_other_users(_video_db_future_oth)
    _video_db_past_future_oth = util.get_random_k_other_users(_video_db_past_future_oth)



def decode_sequence_fov(input_seq,others_fut_input_seq):
    last_location = input_seq[:,-1,:][:,np.newaxis,:]
    # last_location = util.get_gt_target_xyz(input_seq[0,-1,:][np.newaxis,np.newaxis,:])
    decoded_sentence = model.predict([input_seq,others_fut_input_seq,last_location])
    return decoded_sentence


if cfg.sample_and_refeed:
    test_batch_size = batch_size
else:
    test_batch_size = 1
gt_sentence_list = []
decoded_sentence_list = []
for seq_index in range(0,_video_db_tar.shape[0],test_batch_size):
    # input_seq = util.get_gt_target_xyz(_video_db_tar[seq_index: seq_index + test_batch_size,:,:])
    input_seq = _video_db_tar[seq_index: seq_index + test_batch_size,:]

    if oth_from_past:
        others_fut_input_seq = _video_db_past_future_oth[seq_index: seq_index + test_batch_size,:].reshape(-1,20,num_user-1,fps*3)        
    else:
        # others_fut_input_seq = _video_db_future_oth[seq_index: seq_index + test_batch_size,:].reshape(-1,10,num_user-1,fps*3)
        others_fut_input_seq = util.get_gt_target_xyz_oth(_video_db_future_oth[seq_index: seq_index + test_batch_size,:])

    decoded_sentence = decode_sequence_fov(input_seq,others_fut_input_seq)
    decoded_sentence_list+=[decoded_sentence]
    gt_sentence = _video_db_future_tar[seq_index: seq_index + test_batch_size,:,:]
    gt_sentence_list+=[gt_sentence]
    # print('-')
    # decoder_target = util.get_gt_target_xyz(gt_sentence)
    # print('Decoded sentence - decoder_target:', np.squeeze(np.array(decoded_sentence))[:,:3]-np.squeeze(decoder_target)[:,:3])

pickle.dump(decoded_sentence_list,open('decoded_sentence'+thu_tag+tag+'.p','wb'))
pickle.dump(gt_sentence_list,open('gt_sentence_list'+thu_tag+tag+'.p','wb'))
print('Testing finished!')







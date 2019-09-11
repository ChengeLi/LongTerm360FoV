"""
Attentive Mixture of Experts
Please see the plot 'ame in detail' for details
"""
from __future__ import print_function
from keras.models import Model
from keras.layers import Input, LSTM, Dense, Reshape, Add
from keras.layers import Lambda,Concatenate,Flatten
from keras.layers import Permute,multiply,Softmax
from keras.layers import TimeDistributed,Subtract
from keras.callbacks import ModelCheckpoint,ReduceLROnPlateau,EarlyStopping
from keras import backend as K
from keras.models import load_model
from keras import activations
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
import h5py
from tqdm import tqdm
from collections import OrderedDict

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
oth_from_past = False #others' LSTM starts from the very beginning



## utility layers
flatten_layer = Flatten()
expand_dim_layer = Lambda(lambda x: K.expand_dims(x,1))
expand_dim_layer_1 = Lambda(lambda x: K.expand_dims(x,-1))
Concatenatelayer = Concatenate(axis=1)
Concatenatelayer_1 = Concatenate(axis=-1)
get_dim1_layer = Lambda(lambda x: x[:,0,:])

reduce_sum_layer = Lambda(lambda x: K.sum(x, axis=1))#collapsing user dim
reduce_sum_layer_1 = Lambda(lambda x: K.sum(x, axis=-1))
sigmoid = activations.get('sigmoid')



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

decoder_inputs = Input(shape=(1, num_decoder_tokens))    
# decoder_inputs = Input(shape=(1, num_encoder_tokens))    

decoder_lstm1 = LSTM(latent_dim, stateful=cfg.stateful_across_batch, return_sequences=True, return_state=True)
decoder_lstm2 = LSTM(latent_dim, stateful=cfg.stateful_across_batch, return_sequences=True, return_state=True)
if cfg.predict_mean_var:
    decoder_dense = Dense(num_decoder_tokens,activation='tanh')
else:
    decoder_dense = Dense(num_encoder_tokens,activation='tanh')


if oth_from_past:
    others_inputs = Input(shape=(max_encoder_seq_length+max_decoder_seq_length,(num_user-1),6))    
else:
    others_inputs = Input(shape=(max_decoder_seq_length,(num_user-1),6))

if mlp_mixing:
    mixing = Dense(num_decoder_tokens,activation='tanh')
if cfg.use_embedding_AME:
    W_i_pool = []
    V_i_pool = []
    for user_ind in range(num_user):
        W_i_pool.append(Dense(256,activation='relu'))
        V_i_pool.append(Dense(256,activation='relu'))
    # W_i_pool = TimeDistributed(Dense(256,activation='relu'))
    # V_i_pool = TimeDistributed(Dense(256,activation='relu'))

oth_past_dense = Dense(1,activation='relu')#condense the past 10 seconds others' location into 1 sec



# ************************************************************
def score_func(u_i_list,u_si_list):
    # dot product
    sim =  multiply([u_si_list,u_i_list])
    # general
    # sim_dense = Dense(num_user,activation='relu',use_bias=False)
    # sim = sim_dense()

    ### exp(-(distance))
    # square_layer = Lambda(lambda x: K.square(x))
    # distance = square_layer(Subtract()([u_i_list,u_si_list])) #shape=(?, 34, 6)
    # exp_layer = Lambda(lambda x: K.exp(-x))
    # sim = exp_layer(distance)
    return sim


def ui_usi_similarity(u_i_list,u_si_list):
    # get attention alpha_i
    # sim_list = multiply(usi,u_i_list)
    # denominator = reduce_sum_layer(K.exp(sim_list))
    # for ii in range(num_user):
    #     usi = u_si_list[ii]
    #     ui = u_i_list[ii]
    #     numerator = K.exp(multiply(usi,ui))
    #     alpha.append(numerator/denominator)

    # logit = [expand_dim_layer(reduce_sum_layer_1(dot_prod)) for dot_prod in (list(multiply([usi,ui]) for usi,ui in zip(u_si_list,u_i_list)))]
    # logit = Lambda(lambda x: K.concatenate(x, axis=1))(logit)
    logit = reduce_sum_layer_1(score_func(u_i_list,u_si_list))#(batch,34)
    alpha_list = Softmax()(logit)
    return alpha_list


def merge_experts_contribution(alpha_list,concat_outputs):
    # weighted sum
    contributions = []
    for ii in range(num_user):
        alpha = util.slice_layer(1,ii,ii+1)(alpha_list)
        contribution = util.slice_layer(1,ii,ii+1)(concat_outputs)
        contributions.append(multiply([alpha,contribution]))

    contributions = Lambda(lambda x: K.concatenate(x, axis=1))(contributions)
    outputs = reduce_sum_layer(contributions)
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
    decoder_pred = decoder_dense(decoder2_outputs)

    if oth_from_past:
        #use the last 10 locations as the h_i
        gt_mean_var_oth = util.slice_layer(1,time_ind+max_encoder_seq_length-10,time_ind+max_encoder_seq_length)(others_inputs) #(,10,33,6)
        gt_mean_var_oth = Permute((2,3,1))(gt_mean_var_oth) #shape=(?, 33, 6, 10)
        gt_mean_var_oth = oth_past_dense(gt_mean_var_oth) #shape=(?, 33, 6, 1)
        gt_mean_var_oth = Permute((3,1,2))(gt_mean_var_oth) #permute back:shape=(?, 1, 33, 6)
    else:
        gt_mean_var_oth = util.slice_layer(1,time_ind,time_ind+1)(others_inputs) 

    concat_outputs = Concatenatelayer([get_dim1_layer(gt_mean_var_oth),decoder_pred])
    if mlp_mixing:
        concat_outputs = Flatten()(concat_outputs)
        outputs = mixing(concat_outputs)
        outputs = expand_dim_layer(outputs)
    if ame:
        # concat_outputs_flat = Flatten()(concat_outputs)
        # h_all = expand_dim_layer(concat_outputs_flat) #(batch,1,34*6)
        u_i_list = []
        u_si_list = []
        for user_ind in range(num_user-1):
            #for all other users
            h_i = decoder_pred # h_i = util.slice_layer(1,user_ind,user_ind+1)(get_dim1_layer(gt_mean_var_oth))#(batch,1,6)
            h_all = util.slice_layer(1,user_ind,user_ind+1)(get_dim1_layer(gt_mean_var_oth)) #use h_i instead of h_all
            if cfg.use_embedding_AME:
                W_i = W_i_pool[user_ind]
                u_i_list.append(W_i(h_all))
                V_i = V_i_pool[user_ind]
                u_si_list.append(V_i(h_i))
            else:
                u_i_list.append(h_all)
                u_si_list.append(h_i)

        #for target user
        h_i = decoder_pred
        h_all = h_i
        if cfg.use_embedding_AME:        
            W_i = W_i_pool[-1]
            u_i_list.append(W_i(h_all))
            V_i = V_i_pool[-1]
            u_si_list.append(V_i(h_i))
        else:
            u_i_list.append(h_all)
            u_si_list.append(h_i)            

        assert len(u_i_list)==num_user
        assert len(u_si_list)==num_user
        u_i_list = Lambda(lambda x: K.concatenate(x, axis=1))(u_i_list)#(batch,34,256)
        u_si_list = Lambda(lambda x: K.concatenate(x, axis=1))(u_si_list)#(batch,34,256)
        alpha_list = ui_usi_similarity(u_i_list,u_si_list)
        outputs = merge_experts_contribution(alpha_list,concat_outputs)

    all_outputs.append(outputs)
    inputs = outputs

## Concatenate all predictions
decoder_outputs = Lambda(lambda x: K.concatenate(x, axis=1))(all_outputs)

model = Model([encoder_inputs, others_inputs, decoder_inputs], decoder_outputs)
model.compile(optimizer='Adam', loss='mean_squared_error',metrics=['accuracy'])



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
select_k_neighbours=False
# dataformat = 'format5_tsinghua_by_sec_interp' #tsinghua
# option=''
# select_k_neighbours=True
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
    # decoder_target_data = _video_db_future_tar #predict raw
    decoder_target_data = util.get_gt_target_xyz(_video_db_future_tar)#predict mean/var
    # decoder_input_data = _video_db_tar
    decoder_input_data = util.get_gt_target_xyz(encoder_input_data[:,-1,:][:,np.newaxis,:])
    ### other users
    if oth_from_past:
        _video_db_past_future_oth = np.concatenate((_video_db_oth,_video_db_future_oth),axis=1)
        others_input_data = util.get_gt_target_xyz_oth(_video_db_past_future_oth)
    else:
        # others_input_data = _video_db_future_oth
        others_input_data = util.get_gt_target_xyz_oth(_video_db_future_oth)




### ====================Training====================
# tag='AME_tar_similarity_sep26'
# tag='AME_pair_similarity_sep26'#sim between hi and ht
# tag = 'AME_pair_sim_pastoth_oct24'  #h_all=h_oth_i(last 10 seconds)
# tag='AME_loc_simi_noembed__nov1'
# tag='AME_loc_simi_noembed_exp_distance_nov1' #use the exp(-distance) as the similarity measure
tag='AME_loc_simi_timeshift_nov1' #with separate embeddings, dot product

model_checkpoint = ModelCheckpoint(tag+'{epoch:02d}-{val_loss:.4f}.h5', monitor='val_loss', save_best_only=False)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                 patience=10, min_lr=1e-6)
stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='auto')


if cfg.time_shift:
    encoder_input_data = encoder_input_data[:,:-1,:]#delete the last sec
model.fit([encoder_input_data, others_input_data, decoder_input_data], decoder_target_data,
      batch_size=batch_size,
      epochs=epochs,
      validation_split=0.1,
      shuffle=True, initial_epoch=0,
      callbacks=[model_checkpoint, reduce_lr, stopping])



### ====================Testing===================
# model.load_weights('AME_pair_similarity_sep2608-0.0900.h5')
#or use h5
_video_db_tar = util.load_h5('./cache/'+dataformat+'/test/'+option+'_video_db_tar.h5','_video_db_tar')
_video_db_future_tar = util.load_h5('./cache/'+dataformat+'/test/'+option+'_video_db_future_tar.h5','_video_db_future_tar')
_video_db_future_input_tar = util.load_h5('./cache/'+dataformat+'/test/'+option+'_video_db_future_input_tar.h5','_video_db_future_input_tar')
_video_db_oth = util.load_h5('./cache/'+dataformat+'/test/'+option+'_video_db_oth.h5','_video_db_oth')
_video_db_future_oth = util.load_h5('./cache/'+dataformat+'/test/'+option+'_video_db_future_oth.h5','_video_db_future_oth')


_video_db_future_oth = _reshape_others_data(_video_db_future_oth)
if select_k_neighbours:
    _video_db_future_oth = util.get_random_k_other_users(_video_db_future_oth)
if cfg.input_mean_var:
    _video_db_tar = _video_db_tar.reshape((_video_db_tar.shape[0],_video_db_tar.shape[1],fps,3))
_video_db_future_tar = _video_db_future_tar.reshape((_video_db_tar.shape[0],_video_db_tar.shape[1],fps,3))
_video_db_future_input_tar = _video_db_future_input_tar.reshape((_video_db_tar.shape[0],_video_db_tar.shape[1],fps,3))

if oth_from_past:
    _video_db_oth = _reshape_others_data(_video_db_oth)
    _video_db_past_future_oth = np.concatenate((_video_db_oth,_video_db_future_oth),axis=1)



def decode_sequence_fov(input_seq,others_fut_input_seq):
    # Encode the input as state vectors.
    if cfg.input_mean_var:
        last_location = input_seq[0,-1,:][np.newaxis,np.newaxis,:]
    else:
        last_location = util.get_gt_target_xyz(input_seq[0,-1,:][np.newaxis,np.newaxis,:])
    if cfg.time_shift:
        input_seq = input_seq[:,:-1,:]
    decoded_sentence = model.predict([input_seq,others_fut_input_seq,last_location])
    return decoded_sentence



gt_sentence_list = []
decoded_sentence_list = []
for seq_index in range(0,_video_db_tar.shape[0]):
    if cfg.input_mean_var:
        input_seq = util.get_gt_target_xyz(_video_db_tar[seq_index: seq_index + 1,:,:])
    else:
        input_seq = _video_db_tar[seq_index: seq_index + 1,:]

    if oth_from_past:
        others_fut_input_seq = util.get_gt_target_xyz_oth(_video_db_past_future_oth[seq_index: seq_index + 1,:])
    else:
        others_fut_input_seq = util.get_gt_target_xyz_oth(_video_db_future_oth[seq_index: seq_index + 1,:])

    decoded_sentence = decode_sequence_fov(input_seq,others_fut_input_seq)
    decoded_sentence_list+=[decoded_sentence]
    gt_sentence = _video_db_future_tar[seq_index: seq_index + 1,:,:]
    gt_sentence_list+=[gt_sentence]
    # print('-')
    # decoder_target = util.get_gt_target_xyz(gt_sentence)
    # print('Decoded sentence - decoder_target:', np.squeeze(np.array(decoded_sentence))[:,:3]-np.squeeze(decoder_target)[:,:3])

# thu_tag='_thu_'
thu_tag=''
pickle.dump(decoded_sentence_list,open('decoded_sentence'+thu_tag+tag+'.p','wb'))
pickle.dump(gt_sentence_list,open('gt_sentence_list'+thu_tag+tag+'.p','wb'))
print('Testing finished!')






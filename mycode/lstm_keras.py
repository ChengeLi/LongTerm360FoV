"""
single LSTM (not seqence modeling)
Keras
"""
from __future__ import print_function
from keras.models import Model
from keras.layers import Input, LSTM, Dense
from keras.layers import Lambda,Concatenate,Flatten,Reshape
from keras.callbacks import ModelCheckpoint,ReduceLROnPlateau,EarlyStopping
from keras import backend as K
from keras.models import load_model
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


batch_size = 32  # Batch size for training.
epochs = 200  # Number of epochs to train for.
latent_dim = 64  # Latent dimensionality of the encoding space.

fps = 30
num_encoder_tokens = 3*fps
num_decoder_tokens = 6
max_encoder_seq_length = cfg.running_length
max_decoder_seq_length = cfg.predict_step

expand_dim_layer = Lambda(lambda x: K.expand_dims(x,1))
Concatenatelayer1 = Concatenate(axis=-1)

def generate_fake_batch(x):
    """generate new data for 1 second using predicted mean and variance"""
    mu = x[0]
    var = x[1]
    temp = K.random_normal(shape = (batch_size,fps), mean=mu,stddev=var)
    return temp

generate_fake_batch_layer = Lambda(lambda x: generate_fake_batch(x))



# ************************************
#                                   **
#     1st part: 10-step input       **
#   1-step loss during training     **
#                                   **
# ************************************
### ====================Graph def====================
# single layer LSTM
if not cfg.input_mean_var:
    inputs = Input(shape=(None, num_encoder_tokens))
else:
    inputs = Input(shape=(None, num_decoder_tokens))    
lstm = LSTM(latent_dim, return_state=True)
# encoder_outputs, state_h, state_c = lstm(inputs)
# states = [state_h, state_c]


output_dense = Dense(num_decoder_tokens,activation='tanh')


all_outputs = []
for time_ind in range(max_decoder_seq_length):
    this_inputs = util.slice_layer(1,time_ind,time_ind+1)(inputs)
    if time_ind==0:
        decoder_states, state_h, state_c = lstm(this_inputs)#no initial states
    else:
        decoder_states, state_h, state_c = lstm(this_inputs,
                                         initial_state=states)
    outputs = output_dense(decoder_states)
    all_outputs.append(expand_dim_layer(outputs))
    # this_inputs = outputs
    states = [state_h, state_c]

all_outputs = Lambda(lambda x: K.concatenate(x, axis=1))(all_outputs)

model = Model(inputs, all_outputs)
model.compile(optimizer='Adam', loss='mean_squared_error',metrics=['accuracy'])

#### ========================================data============================================================
video_data_train = pickle.load(open('./360video/data/shanghai_dataset_xyz_train.p','rb'),encoding='latin1')    
video_data_train = clip_xyz(video_data_train)
datadb = video_data_train.copy()
# assert cfg.data_chunk_stride=1
_video_db,_video_db_future,_video_db_future_input = util.get_data(datadb,pick_user=False,num_user=34)

if cfg.input_mean_var:
    input_data = util.get_gt_target_xyz(_video_db_future_input)
else:
    input_data = _video_db_future_input
target_data = util.get_gt_target_xyz(_video_db_future)

### ====================Training====================
tag = 'single_LSTM_keras_sep24'
model_checkpoint = ModelCheckpoint(tag+'{epoch:02d}-{val_loss:.4f}.h5', monitor='val_loss', save_best_only=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                 patience=3, min_lr=1e-6)
stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=0, mode='auto')
model.fit(input_data, target_data,
          batch_size=batch_size,
          epochs=epochs,
          validation_split=0.1,
          shuffle=False,
          initial_epoch=0,
          callbacks=[model_checkpoint, reduce_lr, stopping])



### ====================Testing====================
# =================1-step input during testing======================

# define sampling model. note the "this_inputs = outputs"
if not cfg.input_mean_var:
    inputs = Input(shape=(1, num_encoder_tokens))
else:
    inputs = Input(shape=(1, num_decoder_tokens))    
lstm = LSTM(latent_dim, return_state=True)
output_dense = Dense(num_decoder_tokens,activation='tanh')


all_outputs = []
this_inputs = inputs
for time_ind in range(max_decoder_seq_length):
    if time_ind==0:
        decoder_states, state_h, state_c = lstm(this_inputs)#no initial states
    else:
        decoder_states, state_h, state_c = lstm(this_inputs,
                                         initial_state=states)
    outputs = output_dense(decoder_states)
    if cfg.predict_mean_var and cfg.sample_and_refeed:    
        ux_temp = util.slice_layer(1,0,1)(outputs)
        uy_temp = util.slice_layer(1,1,2)(outputs)
        uz_temp = util.slice_layer(1,2,3)(outputs)
        varx_temp = util.slice_layer(1,3,4)(outputs)
        vary_temp = util.slice_layer(1,4,5)(outputs)
        varz_temp = util.slice_layer(1,5,6)(outputs)

        temp_newdata = expand_dim_layer(Concatenatelayer1([generate_fake_batch_layer([ux_temp,varx_temp]),
                            generate_fake_batch_layer([uy_temp,vary_temp]),
                            generate_fake_batch_layer([uz_temp,varz_temp])]))
        this_inputs = temp_newdata

    all_outputs.append(expand_dim_layer(outputs))
    states = [state_h, state_c]

all_outputs = Lambda(lambda x: K.concatenate(x, axis=1))(all_outputs)

model = Model(inputs, all_outputs)


model.load_weights('single_LSTM_keras_sep2434-0.0120.h5')


# data
video_data_test = pickle.load(open('./360video/data/shanghai_dataset_xyz_test.p','rb'),encoding='latin1')
video_data_test = clip_xyz(video_data_test)
datadb = video_data_test.copy()
_,_video_db_future,_video_db_future_input = util.get_data(datadb,pick_user=False)
if cfg.input_mean_var:
    _video_db = util.get_gt_target_xyz(_video_db_future_input)
else:
    _video_db = _video_db_future_input #use this as input

def decode_sequence_fov(input_seq):
    last_location = input_seq[:,0,:][:,np.newaxis,:] #1-step input during testing
    if cfg.input_mean_var:
        last_mu_var = util.get_gt_target_xyz(last_location)
    else:
        last_mu_var = last_location
    decoded_sentence = model.predict(last_mu_var)
    return decoded_sentence



gt_sentence_list = []
decoded_sentence_list = []
for seq_index in range(0,_video_db.shape[0],batch_size):
    input_seq = _video_db[seq_index: seq_index +  batch_size,:,:]

    if input_seq.shape[0]<batch_size:
        break
    decoded_sentence = decode_sequence_fov(input_seq)
    decoded_sentence_list+=[decoded_sentence]
    gt_sentence = _video_db_future[seq_index: seq_index + batch_size,:,:]
    gt_sentence_list+=[gt_sentence]
    decoder_target = util.get_gt_target_xyz(gt_sentence)
    # print('-')
    # print('Decoded sentence - decoder_target:', np.squeeze(np.array(decoded_sentence))[:,:3]-np.squeeze(decoder_target)[:,:3])

pickle.dump(decoded_sentence_list,open('decoded_sentence'+tag+'.p','wb'))
pickle.dump(gt_sentence_list,open('gt_sentence_list'+tag+'.p','wb'))
print('Testing finished!')




# ************************************
#                                   **
#     2nd part: 1-step input        **
#   10-step loss during training    **
#                                   **
# ************************************
if not cfg.input_mean_var:
    inputs = Input(shape=(1, num_encoder_tokens))
else:
    inputs = Input(shape=(1, num_decoder_tokens))    
lstm = LSTM(latent_dim, return_state=True)
output_dense = Dense(num_decoder_tokens,activation='tanh')


all_outputs = []
this_inputs = inputs
for time_ind in range(max_decoder_seq_length):
    if time_ind==0:
        decoder_states, state_h, state_c = lstm(this_inputs)#no initial states
    else:
        decoder_states, state_h, state_c = lstm(this_inputs,
                                         initial_state=states)
    outputs = output_dense(decoder_states)
    if cfg.predict_mean_var and cfg.sample_and_refeed:    
        ux_temp = util.slice_layer(1,0,1)(outputs)
        uy_temp = util.slice_layer(1,1,2)(outputs)
        uz_temp = util.slice_layer(1,2,3)(outputs)
        varx_temp = util.slice_layer(1,3,4)(outputs)
        vary_temp = util.slice_layer(1,4,5)(outputs)
        varz_temp = util.slice_layer(1,5,6)(outputs)

        temp_newdata = expand_dim_layer(Concatenatelayer1([generate_fake_batch_layer([ux_temp,varx_temp]),
                            generate_fake_batch_layer([uy_temp,vary_temp]),
                            generate_fake_batch_layer([uz_temp,varz_temp])]))
        this_inputs = temp_newdata

    all_outputs.append(expand_dim_layer(outputs))
    states = [state_h, state_c]

all_outputs = Lambda(lambda x: K.concatenate(x, axis=1))(all_outputs)
model = Model(inputs, all_outputs)
model.compile(optimizer='Adam', loss='mean_squared_error',metrics=['accuracy'])


#### ========================================data============================================================
video_data_train = pickle.load(open('./360video/data/shanghai_dataset_xyz_train.p','rb'),encoding='latin1') #76414,10,90
video_data_train = clip_xyz(video_data_train)
datadb = video_data_train.copy()
# assert cfg.data_chunk_stride=1
_video_db,_video_db_future,_video_db_future_input = util.get_data(datadb,pick_user=False,num_user=34)

if cfg.input_mean_var:
    input_data = util.get_gt_target_xyz(_video_db_future_input)
else:
    input_data = _video_db_future_input
input_data = input_data[:,0,:][:,np.newaxis,:] #1-step input
target_data = util.get_gt_target_xyz(_video_db_future)


# if using the generate fake batch layer, the dataset size has to
# be dividable by the batch size
validation_ratio=0.1
if cfg.sample_and_refeed or cfg.stateful_across_batch:
    sample_ind = util.rand_sample_ind(input_data.shape[0],0,batch_size,validation_ratio=validation_ratio)
    if not cfg.shuffle_data:
        sample_ind = sorted(sample_ind)
    input_data = util.rand_sample(input_data,sample_ind)
    target_data = util.rand_sample(target_data,sample_ind)

### ====================Training====================
tag = 'single_LSTM_keras_10steploss_sep25'
model_checkpoint = ModelCheckpoint(tag+'{epoch:02d}-{val_loss:.4f}.h5', monitor='val_loss', save_best_only=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                 patience=3, min_lr=1e-6)
stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=0, mode='auto')
model.fit(input_data, target_data,
          batch_size=batch_size,
          epochs=epochs,
          validation_split=validation_ratio,
          shuffle=False,
          initial_epoch=0,
          callbacks=[model_checkpoint, reduce_lr, stopping])

### ====================Testing====================
# model.load_weights('single_LSTM_keras_10steploss_sep2540-0.1204.h5')
# data
# video_data_test = pickle.load(open('./360video/data/shanghai_dataset_xyz_test.p','rb'),encoding='latin1') #20528,10,90
# video_data_test = clip_xyz(video_data_test)
# datadb = video_data_test.copy()
# _,_video_db_future,_video_db_future_input = util.get_data(datadb,pick_user=False)

dataformat = 'format5_tsinghua_by_sec_interp' #tsinghua
option=''
# _video_db_tar = util.load_h5('./cache/'+dataformat+'/test/'+option+'_video_db_tar.h5','_video_db_tar')
_video_db_future = util.load_h5('./cache/'+dataformat+'/test/'+option+'_video_db_future_tar.h5','_video_db_future_tar') #(6768, 10, 90)
_video_db_future_input = util.load_h5('./cache/'+dataformat+'/test/'+option+'_video_db_future_input_tar.h5','_video_db_future_input_tar')
thu_tag='_thu_'

if cfg.input_mean_var:
    _video_db = util.get_gt_target_xyz(_video_db_future_input)
else:
    _video_db = _video_db_future_input #use this as input

def decode_sequence_fov(input_seq):
    last_location = input_seq[:,0,:][:,np.newaxis,:] #1-step input during testing
    if cfg.input_mean_var:
        last_mu_var = util.get_gt_target_xyz(last_location)
    else:
        last_mu_var = last_location
    decoded_sentence = model.predict(last_mu_var)
    return decoded_sentence


gt_sentence_list = []
decoded_sentence_list = []
for seq_index in range(0,_video_db.shape[0],batch_size):
    input_seq = _video_db[seq_index: seq_index +  batch_size,:,:]
    if input_seq.shape[0]<batch_size:
        break
    decoded_sentence = decode_sequence_fov(input_seq)
    decoded_sentence_list+=[decoded_sentence]
    gt_sentence = _video_db_future[seq_index: seq_index + batch_size,:,:]
    gt_sentence_list+=[gt_sentence]
    decoder_target = util.get_gt_target_xyz(gt_sentence)
    # print('-')
    # print('Decoded sentence - decoder_target:', np.squeeze(np.array(decoded_sentence))[:,:3]-np.squeeze(decoder_target)[:,:3])


pickle.dump(decoded_sentence_list,open('decoded_sentence'+thu_tag+tag+'.p','wb'))
pickle.dump(gt_sentence_list,open('gt_sentence_list'+thu_tag+tag+'.p','wb'))
print('Testing finished!')














# ************************************
#                                   **
#     3rd part: 10-step input       **
#   10-step loss during training    **
#                                   **
# ************************************

if not cfg.input_mean_var:
    inputs = Input(shape=(None, num_encoder_tokens))
else:
    inputs = Input(shape=(None, num_decoder_tokens))    
lstm = LSTM(latent_dim, return_state=True)
encoder_outputs, state_h, state_c = lstm(inputs)
states = [state_h, state_c]

output_dense = Dense(num_decoder_tokens,activation='tanh')

all_outputs = []
for time_ind in range(max_decoder_seq_length):
    if time_ind==0:
        decoder_states = encoder_outputs
    else:
        decoder_states, state_h, state_c = lstm(this_inputs)
    outputs = output_dense(decoder_states)
    all_outputs.append(expand_dim_layer(outputs))
    if cfg.predict_mean_var and cfg.sample_and_refeed:    
        ux_temp = util.slice_layer(1,0,1)(outputs)
        uy_temp = util.slice_layer(1,1,2)(outputs)
        uz_temp = util.slice_layer(1,2,3)(outputs)
        varx_temp = util.slice_layer(1,3,4)(outputs)
        vary_temp = util.slice_layer(1,4,5)(outputs)
        varz_temp = util.slice_layer(1,5,6)(outputs)

        temp_newdata = expand_dim_layer(Concatenatelayer1([generate_fake_batch_layer([ux_temp,varx_temp]),
                            generate_fake_batch_layer([uy_temp,vary_temp]),
                            generate_fake_batch_layer([uz_temp,varz_temp])]))
        this_inputs = temp_newdata

all_outputs = Lambda(lambda x: K.concatenate(x, axis=1))(all_outputs)

model = Model(inputs, all_outputs)
model.compile(optimizer='Adam', loss='mean_squared_error',metrics=['accuracy'])

#### ========================================data============================================================
video_data_train = pickle.load(open('./360video/data/shanghai_dataset_xyz_train.p','rb'),encoding='latin1')    
video_data_train = clip_xyz(video_data_train)
datadb = video_data_train.copy()
# assert cfg.data_chunk_stride=1
_video_db,_video_db_future,_video_db_future_input = util.get_data(datadb,pick_user=False,num_user=34)

if cfg.input_mean_var:
    input_data = util.get_gt_target_xyz(_video_db)
else:
    input_data = _video_db
target_data = util.get_gt_target_xyz(_video_db_future)


# if using the generate fake batch layer, the dataset size has to
# be dividable by the batch size
validation_ratio=0.1
if cfg.sample_and_refeed or cfg.stateful_across_batch:
    sample_ind = util.rand_sample_ind(input_data.shape[0],0,batch_size,validation_ratio=validation_ratio)
    if not cfg.shuffle_data:
        sample_ind = sorted(sample_ind)
    input_data = util.rand_sample(input_data,sample_ind)
    target_data = util.rand_sample(target_data,sample_ind)

### ====================Training====================
tag = 'single_LSTM_keras_10input_10steploss_model3_sep25'
model_checkpoint = ModelCheckpoint(tag+'{epoch:02d}-{val_loss:.4f}.h5', monitor='val_loss', save_best_only=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                 patience=3, min_lr=1e-6)
stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=0, mode='auto')
model.fit(input_data, target_data,
          batch_size=batch_size,
          epochs=epochs,
          validation_split=validation_ratio,
          shuffle=False,
          initial_epoch=0,
          callbacks=[model_checkpoint, reduce_lr, stopping])



### ====================Testing====================
# data
video_data_test = pickle.load(open('./360video/data/shanghai_dataset_xyz_test.p','rb'),encoding='latin1')
video_data_test = clip_xyz(video_data_test)
datadb = video_data_test.copy()
_video_db,_video_db_future,_video_db_future_input = util.get_data(datadb,pick_user=False)
if cfg.input_mean_var:
    _video_db = util.get_gt_target_xyz(_video_db)

def decode_sequence_fov(input_seq):
    last_location = input_seq #10-step input during testing (last 10 sec)
    if cfg.input_mean_var:
        last_mu_var = util.get_gt_target_xyz(last_location)
    else:
        last_mu_var = last_location
    decoded_sentence = model.predict(last_mu_var)
    return decoded_sentence


gt_sentence_list = []
decoded_sentence_list = []
for seq_index in range(0,_video_db.shape[0],batch_size):
    input_seq = _video_db[seq_index: seq_index +  batch_size,:,:]
    if input_seq.shape[0]<batch_size:
        break
    decoded_sentence = decode_sequence_fov(input_seq)
    decoded_sentence_list+=[decoded_sentence]
    gt_sentence = _video_db_future[seq_index: seq_index + batch_size,:,:]
    gt_sentence_list+=[gt_sentence]
    decoder_target = util.get_gt_target_xyz(gt_sentence)
    # print('-')
    # print('Decoded sentence - decoder_target:', np.squeeze(np.array(decoded_sentence))[:,:3]-np.squeeze(decoder_target)[:,:3])

pickle.dump(decoded_sentence_list,open('decoded_sentence'+tag+'.p','wb'))
pickle.dump(gt_sentence_list,open('gt_sentence_list'+tag+'.p','wb'))
print('Testing finished!')









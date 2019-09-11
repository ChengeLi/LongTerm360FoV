"""
Using gt mean and var from others' data. 
2 layer fc-lstm seq2seq without teacher forcing.
MLP to mix these mean and var. 
"""
from __future__ import print_function
from keras.models import Model
from keras.layers import Input, LSTM, Dense, Reshape, Add
from keras.layers import Lambda,Concatenate,Flatten,ConvLSTM2D
from keras.layers import Permute,Conv2D,MaxPooling2D,multiply
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
from collections import OrderedDict
from mycode.data_generator_including_saliency import *
from mycode.heatmap_generator import gauss_2d,discretization

batch_size = cfg.batch_size
epochs = 200
latent_dim = 32

fps = cfg.fps
num_encoder_tokens = 3*fps
num_decoder_tokens = 6
max_encoder_seq_length = cfg.running_length
max_decoder_seq_length = cfg.predict_step
num_user = 34 #48 
use_generator = True
select_k_neighbours=False
mixing_saliency=False

mlp_mixing = True
conv_mixing = False



## utility layers
flatten_layer = Flatten()
expand_dim_layer = Lambda(lambda x: K.expand_dims(x,1))
Concatenatelayer = Concatenate(axis=1)
get_dim1_layer = Lambda(lambda x: x[:,0,:])

sigmoid = activations.get('sigmoid')
# GLU_layer = Lambda(lambda x: multiply([x[0],sigmoid(x[1])]))
GLU_layer = Lambda(lambda x: multiply([x[1],sigmoid(x[0])]))


### ====================Graph def====================
#2 layer encoder
if cfg.input_mean_var:
    encoder_inputs = Input(shape=(None, 6))
    if cfg.stateful_across_batch:
        encoder_inputs = Input(batch_shape=(batch_size, None, 6))
else:
    encoder_inputs = Input(shape=(None, num_encoder_tokens))
if cfg.use_saliency:
    img_h,img_w = (256/4, 512/4)
    saliency_inputs = Input(shape=(max_decoder_seq_length,img_h,img_w,2))

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
decoder_lstm1 = LSTM(latent_dim, stateful=cfg.stateful_across_batch, return_sequences=True, return_state=True)
decoder_lstm2 = LSTM(latent_dim, stateful=cfg.stateful_across_batch, return_sequences=True, return_state=True)
decoder_dense = Dense(num_decoder_tokens,activation='tanh')
# decoder_dense = Dense(2,activation='tanh')#predict theta, phi



if mlp_mixing:
    mixing = Dense(3*num_decoder_tokens,activation='tanh')
    mixing1 = Dense(2*num_decoder_tokens,activation='tanh')
    mixing2 = Dense(num_decoder_tokens,activation=None)
    gating = Dense(num_user*6,activation=None) #gating

elif conv_mixing:
    mixing = Conv2D(filters=8, kernel_size=(1,3), strides=(1, 1),padding='same',
                     activation='relu', dilation_rate=(1, 1),
                     input_shape=(1,6,num_user))
    mixing1 = Conv2D(filters=8, kernel_size=(1,3), strides=(1, 1),padding='same',
                     activation='relu', dilation_rate=(1, 1),
                     input_shape=(1,6,8))
    mixing2 = Conv2D(filters=1, kernel_size=(1,3), strides=(1, 1),padding='same',
                     activation='relu', dilation_rate=(1, 1),
                     input_shape=(1,6,8))

if cfg.use_saliency:
    saliency_mixing = Dense(num_decoder_tokens,activation=None)


all_outputs = []
inputs = decoder_inputs
for time_ind in range(max_decoder_seq_length):
    # Run the decoder on one timestep
    # 2-layer fclstm, without teacher forcing
    decoder1_outputs, state_decoder1_h, state_decoder1_c = decoder_lstm1(decoder_inputs,
                                         initial_state=decoder1_states_inputs)
    decoder1_states_inputs = [state_decoder1_h, state_decoder1_c]
    decoder2_outputs, state_decoder2_h, state_decoder2_c = decoder_lstm2(decoder1_outputs,initial_state=decoder2_states_inputs)
    decoder2_states_inputs = [state_decoder2_h, state_decoder2_c]
    decoder_pred = decoder_dense(decoder2_outputs)

    #saliency CNN feature
    if cfg.use_saliency:
        _saliency = get_CNN_fea(saliency_inputs,time_ind,final_dim=num_decoder_tokens)

    if mlp_mixing:
        concat_outputs = Flatten()(decoder_pred)
        outputs = mixing(concat_outputs)
        outputs = mixing1(outputs)
        gating_weights = gating(outputs)
        outputs = multiply([gating_weights,concat_outputs])
        outputs = mixing2(outputs)
        if cfg.use_saliency:
            if mixing_saliency:
                outputs = Concatenatelayer([outputs,_saliency])#add saliency features
                outputs = saliency_mixing(outputs)
            else: #saliency residual
                outputs = Add()([outputs,_saliency])
        outputs = expand_dim_layer(outputs)
    elif conv_mixing:
        # use conv layer to mix
        concat_outputs = Permute((2, 1))(concat_outputs)
        outputs = mixing(expand_dim_layer(concat_outputs))
        outputs = mixing1(outputs)
        outputs = mixing2(outputs)
        outputs = Permute((2, 1))(get_dim1_layer(outputs))

    all_outputs.append(outputs)
    inputs = outputs
    # inputs = decoder_pred
    # states = [state_h, state_c]


## Concatenate all predictions
decoder_outputs = Lambda(lambda x: K.concatenate(x, axis=1))(all_outputs)

model = Model([encoder_inputs, decoder_inputs, saliency_inputs], decoder_outputs)
model.compile(optimizer='Adam', loss='mean_squared_error',metrics=['accuracy'])




def data_sanity_check(_video_db_tar,_video_db_future_tar,_video_db_future_input_tar):
    sample_ind = np.random.randint(0,_video_db_future_input_tar.shape[0])
    assert (_video_db_tar[sample_ind,:][-1,:]-_video_db_future_input_tar[sample_ind,:][0,:]).sum()==0
    print(np.abs(_video_db_tar[sample_ind,:][-1,:]-_video_db_future_tar[sample_ind,:][0,:]))
    
def _reshape_others_data(_video_db_oth):
    ## to match Input shape: others_fut_inputs
    _video_db_oth = _video_db_oth.transpose((1,2,0,3))
    _video_db_oth = _video_db_oth.reshape((_video_db_oth.shape[0],_video_db_oth.shape[1],_video_db_oth.shape[2],
                    fps,3))
    return _video_db_oth

def get_segment_index(datadb):
    """segment time is used to get the visual/saliency information"""
    #match in time!!!!
    if cfg.use_saliency:
        segment_index_tar = util.get_time_for_visual(datadb)
        segment_index_tar_future = OrderedDict()
        for key in segment_index_tar.keys():
            segment_index_tar_future[key] = np.array(segment_index_tar[key])+max_encoder_seq_length
    return segment_index_tar,segment_index_tar_future


#### ========================================data============================================================
if use_generator:
    video_data_train = pickle.load(open('./360video/data/shanghai_dataset_xyz_train.p','rb'))    
    datadb_train = clip_xyz(video_data_train)
    video_data_test = pickle.load(open('./360video/data/shanghai_dataset_xyz_test.p','rb'))    
    datadb_test = clip_xyz(video_data_test)

    #saliency
    if cfg.use_saliency:
        segment_index_tar,segment_index_tar_future = get_segment_index(datadb_train)
        mygenerator = generator_train2(datadb_train,segment_index_tar,segment_index_tar_future,phase='train')

        segment_index_tar_test,segment_index_tar_future_test = get_segment_index(datadb_test)
        mygenerator_val = generator_train2(datadb_test,segment_index_tar_test,segment_index_tar_future_test,phase='val')
    else:       
        # no saliency
        mygenerator = generator_train2(datadb_train,phase='train')
        mygenerator_val = generator_train2(datadb_test,phase='val')


### ====================Training====================
tag = 'fc_mlpmixing_saliency_residual_sep11'

model_checkpoint = ModelCheckpoint(tag+'{epoch:02d}-{val_loss:.4f}.h5', monitor='val_loss', save_best_only=False)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                 patience=10, min_lr=1e-6)
stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=3, verbose=0, mode='auto')

if use_generator:
    # num_samples = 5404/cfg.batch_size
    # num_samples = 24000
    num_samples = 10000
    model.fit_generator(mygenerator,steps_per_epoch=num_samples, epochs=epochs,
                    validation_data=mygenerator_val, validation_steps=100,
                    callbacks=[model_checkpoint, reduce_lr, stopping],
                    use_multiprocessing=False,
                    initial_epoch=0)
else:
    model.fit([encoder_input_data, others_fut_input_data, decoder_input_data], decoder_target_data,
    # model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
          batch_size=batch_size,
          epochs=epochs,
          validation_split=0.1,
          shuffle=True, initial_epoch=0,
          callbacks=[model_checkpoint, reduce_lr, stopping])





### ====================Testing===================
model.load_weights('fc_mlpmixing_saliency_residual_sep1102-0.0828.h5')
if cfg.use_saliency:
    segment_index_tar_test,segment_index_tar_future_test = get_segment_index(datadb_test)
    mygenerator_test = generator_train2(datadb_test,segment_index_tar_test,segment_index_tar_future_test,phase='test')
else:
    mygenerator_test = generator_train2(datadb_test,phase='test')

test_out = []
gt_out = []
while len(test_out)<1261:
    x,gt_temp = mygenerator_test.next()
    test_out_temp = model.predict_on_batch(x)
    test_out.append(test_out_temp)
    gt_out.append(gt_temp)

pickle.dump(test_out,open('decoded_sentence'+tag+'.p','wb'))
pickle.dump(gt_out,open('gt_sentence_list'+tag+'.p','wb'))
print('Testing finished!')















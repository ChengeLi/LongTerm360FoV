"""
target user only seq2seq. 
With attention from KULC
Python 3.6
"""
from __future__ import print_function
from keras.models import Model
from keras.layers import Input, LSTM, Dense, Embedding
from keras.layers import Lambda,Concatenate,Flatten,ConvLSTM2D
from keras.callbacks import ModelCheckpoint,ReduceLROnPlateau,EarlyStopping
from keras import backend as K
from keras.models import load_model
import sys,glob,io,random
if './360video/' not in sys.path:
    sys.path.insert(0, './360video/')
import mycode.cost as costfunc
from mycode.config import cfg
from mycode.dataIO import clip_xyz
import mycode.utility as util
from random import shuffle
import matplotlib.pyplot as plt
import _pickle as pickle
import numpy as np
import pdb
from kulc.attention import AttentionRNNWrapper #,ExternalAttentionRNNWrapper
from mycode.data_generator_including_saliency import *

Concatenatelayer_1 = Concatenate(axis=-1)
batch_size = 32  # Batch size for training.
epochs = 50  # Number of epochs to train for.
latent_dim = 64  # Latent dimensionality of the encoding space.

fps = 30
num_encoder_tokens = 3*fps
num_decoder_tokens = 6
max_encoder_seq_length = cfg.running_length
max_decoder_seq_length = cfg.predict_step
num_user = 34
use_attention_encoder=True
use_attention_decoder=False
target_user_only = True
use_one_layer = True
use_embedding = False
if use_embedding:
    vocabulary_size = 2000  #divide (-1,1) into 2000
    embedding_size = 512


get_dim_layer = Lambda(lambda x: x[:,:,0])



### ====================Graph def====================
if not cfg.input_mean_var:
    encoder_inputs = Input(shape=(None, num_encoder_tokens))
    # encoder_inputs = Input(batch_shape=(batch_size, None, num_encoder_tokens))
else:
    encoder_inputs = Input(shape=(None, num_decoder_tokens))    

if use_embedding:
    # CAUTION!! this embedding takes integer as input!!
    # also the input should be (batch*T), i.e. only 1 dimension integer for each T!!!
    dense_before_embed = Dense(1)
    encoder_inputs1 = get_dim_layer(dense_before_embed(encoder_inputs))
    encoder_inputs_embedded = Embedding(vocabulary_size, embedding_size, input_length=max_encoder_seq_length)(encoder_inputs1)

if use_one_layer:
    encoder = LSTM(latent_dim,return_sequences=True,return_state=True)
    if use_attention_encoder:
        attented_encoder = AttentionRNNWrapper(encoder)
        encoder_outputs, state_h, state_c  = attented_encoder(encoder_inputs)
    else:
        encoder_outputs, state_h, state_c = encoder(encoder_inputs)
    states = [state_h, state_c]

else:
    encoder1 = LSTM(latent_dim, stateful=cfg.stateful_across_batch, return_state=True, return_sequences=True)
    encoder1_outputs, state_h_1, state_c_1 = encoder1(encoder_inputs)
    encoder2 = LSTM(latent_dim, stateful=cfg.stateful_across_batch, return_state=True, return_sequences=True)
    if use_attention_encoder:
        attented_encoder = AttentionRNNWrapper(encoder2) #only attention on encoder2??
        encoder2_outputs, state_h_2, state_c_2 = attented_encoder(encoder1_outputs)
    else:
        encoder2_outputs, state_h_2, state_c_2 = encoder2(encoder1_outputs)

    encoder1_states = [state_h_1, state_c_1]
    encoder2_states = [state_h_2, state_c_2]

    ##2 layer decoder
    decoder1_states_inputs = encoder1_states
    decoder2_states_inputs = encoder2_states

# decoder:
decoder_inputs = Input(shape=(1, num_decoder_tokens))
if cfg.include_time_ind:
    time_ind_input = Input(shape=(max_decoder_seq_length,1)) #concat as input, not for decay

if use_one_layer:
    decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
    if use_attention_decoder:
        decoder_lstm = AttentionRNNWrapper(decoder_lstm)
else:
    decoder_lstm1 = LSTM(latent_dim, stateful=cfg.stateful_across_batch, return_sequences=True, return_state=True)
    decoder_lstm2 = LSTM(latent_dim, stateful=cfg.stateful_across_batch, return_sequences=True, return_state=True)
    if use_attention_decoder:
        decoder_lstm1 = AttentionRNNWrapper(decoder_lstm1)
        decoder_lstm2 = AttentionRNNWrapper(decoder_lstm2)


decoder_dense = Dense(num_decoder_tokens,activation='tanh')



all_outputs = []
inputs = decoder_inputs
for time_ind in range(max_decoder_seq_length):
    # if cfg.include_time_ind: #as input
    #     this_time_ind_input = util.slice_layer(1,time_ind,time_ind+1)(time_ind_input)
    #     inputs = Concatenatelayer_1([inputs,this_time_ind_input])

    if use_one_layer:
        decoder_states, state_h, state_c = decoder_lstm(inputs,initial_state=states)
        states = [state_h, state_c]
        if cfg.include_time_ind: #as embedding input
            this_time_ind_input = util.slice_layer(1,time_ind,time_ind+1)(time_ind_input)
            decoder_states = Concatenatelayer_1([decoder_states,this_time_ind_input])

        outputs = decoder_dense(decoder_states)
    else:
        decoder1_outputs, state_decoder1_h, state_decoder1_c = decoder_lstm1(inputs,
                                             initial_state=decoder1_states_inputs)
        decoder1_states_inputs = [state_decoder1_h, state_decoder1_c]
        decoder2_outputs, state_decoder2_h, state_decoder2_c = decoder_lstm2(decoder1_outputs,
                                            initial_state=decoder2_states_inputs)
        decoder2_states_inputs = [state_decoder2_h, state_decoder2_c]
        outputs = decoder_dense(decoder2_outputs)

    all_outputs.append(outputs)
    inputs = outputs


decoder_outputs = Lambda(lambda x: K.concatenate(x, axis=1))(all_outputs)

if cfg.include_time_ind:
    model = Model([encoder_inputs, decoder_inputs, time_ind_input], decoder_outputs)
else:
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.compile(optimizer='Adam', loss='mean_squared_error',metrics=['accuracy'])



#### ====================data====================
# video_data_train = pickle.load(open('./360video/data/shanghai_dataset_xyz_train.p','rb'),encoding='latin1')    
# datadb_train = clip_xyz(video_data_train)
# video_data_test = pickle.load(open('./360video/data/shanghai_dataset_xyz_test.p','rb'),encoding='latin1')    
# datadb_test = clip_xyz(video_data_test)
      
# mygenerator = generator_train2(datadb_train,phase='train')
# mygenerator_val = generator_train2(datadb_test,phase='val')

video_data_train = pickle.load(open('./360video/data/shanghai_dataset_xyz_train.p','rb'),encoding='latin1')    
video_data_train = clip_xyz(video_data_train)
_video_db,_video_db_future,_video_db_future_input = util.get_data(video_data_train,pick_user=False)
if cfg.input_mean_var:
    encoder_input_data = util.get_gt_target_xyz(_video_db)
else:
    encoder_input_data = _video_db
    # if use_embedding:
    #     # first convert encoder_inputs into integers
    #     encoder_input_data+=1
    #     step = 2./vocabulary_size
    #     encoder_input_data = np.int32(encoder_input_data//step)
decoder_target_data = util.get_gt_target_xyz(_video_db_future)
decoder_input_data = util.get_gt_target_xyz(_video_db_future_input)[:,0,:][:,np.newaxis,:]
if cfg.include_time_ind:
    time_ind_data = np.repeat(np.arange(max_decoder_seq_length).reshape(-1,1),decoder_input_data.shape[0],axis=-1).transpose((1,0))[:,:,np.newaxis]

### ====================Training====================
# tag='taronly_seq2seq_2layer_full_attention_embedding_nonint' #only attention on the 2nd layer of encoder, but both layers in decoder? odd...
# tag='tar_only_one_layer_fullattention' #only attention on the 2nd layer of encoder, but both layers in decoder? odd...
# tag='tar_only_one_layer_decoder_attention_dec3'
# tag='tar_only_one_layer_encoder_attention_dec3'
# tag='tar_only_one_layer_encoder_attention_timeindinput_dec19' #concat 1-d time index as input
tag='tar_only_one_layer_encoder_attention_timeindembed_dec20' #concat 1-d time index as input



model_checkpoint = ModelCheckpoint(tag+'{epoch:02d}-{val_loss:.4f}.h5', monitor='val_loss', save_best_only=False)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                 patience=10, min_lr=1e-6)
stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=0, mode='auto')


# num_samples = 10000
# model.fit_generator(mygenerator,steps_per_epoch=num_samples, epochs=epochs,
#                 validation_data=mygenerator_val, validation_steps=100,
#                 callbacks=[model_checkpoint, reduce_lr, stopping],
#                 use_multiprocessing=False, shuffle=True,
#                 initial_epoch=0)


# history = model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
history = model.fit([encoder_input_data, decoder_input_data, time_ind_data], decoder_target_data,
          batch_size=batch_size,
          epochs=epochs,
          validation_split=0.1,
          shuffle=True,
          initial_epoch=0,
          callbacks=[model_checkpoint, reduce_lr, stopping])

# save loss plot
# plt.plot(history.history["loss"])
# plt.plot(history.history["val_loss"])
# plt.title("social model loss")
# plt.ylabel("loss")
# plt.xlabel("epoch")
# plt.legend(["train", "test"], loc="upper right")
# plt.savefig(os.path.join(out_dir, "test={}_loss.png".format(
# config.test_dataset_kind)))
pickle.dump(history.history, open(tag+'_training_loss.p','wb'))





### ====================Testing====================
# model.load_weights('taronly_seq2seq_attention03-0.1087.h5')
# model.load_weights('taronly_seq2seq_2layer_attention23-0.1247.h5')
# model.load_weights('taronly_seq2seq_2layer_full_attention08-0.1107.h5')
# model.load_weights('taronly_seq2seq_2layer_full_attention_embedding08-0.1273.h5')

video_data_test = pickle.load(open('./360video/data/shanghai_dataset_xyz_test.p','rb'),encoding='latin1')
video_data_test = clip_xyz(video_data_test)
_video_db,_video_db_future,_video_db_future_input = util.get_data(video_data_test,pick_user=False)

if cfg.input_mean_var:
    _video_db = util.get_gt_target_xyz(_video_db)

def decode_sequence_fov(input_seq):
    # Encode the input as state vectors.
    last_location = input_seq[0,-1,:][np.newaxis,np.newaxis,:]
    if not cfg.input_mean_var:
        last_mu_var = util.get_gt_target_xyz(last_location)
    else:
        last_mu_var = last_location
    if cfg.include_time_ind:
        time_ind_data = np.repeat(np.arange(max_decoder_seq_length).reshape(-1,1),input_seq.shape[0],axis=-1).transpose((1,0))[:,:,np.newaxis]
        decoded_sentence = model.predict([input_seq,last_mu_var,time_ind_data])
    else:
        decoded_sentence = model.predict([input_seq,last_mu_var])
    return decoded_sentence


gt_sentence_list = []
decoded_sentence_list = []
for seq_index in range(_video_db.shape[0]):
    input_seq = _video_db[seq_index: seq_index + 1,:,:]
    decoded_sentence = decode_sequence_fov(input_seq)
    decoded_sentence_list+=[decoded_sentence]
    gt_sentence = _video_db_future[seq_index: seq_index + 1,:,:]
    gt_sentence_list+=[gt_sentence]
    decoder_target = util.get_gt_target_xyz(gt_sentence)
    # print('-')
    # print('Decoded sentence - decoder_target:', np.squeeze(np.array(decoded_sentence))[:,:3]-np.squeeze(decoder_target)[:,:3])

pickle.dump(decoded_sentence_list,open('decoded_sentence'+tag+'.p','wb'))
pickle.dump(gt_sentence_list,open('gt_sentence_list'+tag+'.p','wb'))
print('Testing finished!')


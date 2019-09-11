'''
- An encoder LSTM turns input sequences to 2 state vectors
    (we keep the last LSTM state and discard the outputs).
- A decoder LSTM is trained to turn the target sequences into
    the same sequence but offset by one timestep in the future,
    a training process called "teacher forcing" in this context.
    Is uses as initial state the state vectors from the encoder.
    Effectively, the decoder learns to generate `targets[t+1...]`
    given `targets[...t]`, conditioned on the input sequence.
- In inference mode, when we want to decode unknown input sequences, we:
    - Encode the input sequence into state vectors
    - Start with a target sequence of size 1
        (just the start-of-sequence character)
    - Feed the state vectors and 1-char target sequence
        to the decoder to produce predictions for the next character
    - Sample the next character using these predictions
        (we simply use argmax).
    - Append the sampled character to the target sequence
    - Repeat until we generate the end-of-sequence character or we
        hit the character limit.
'''
from __future__ import print_function

from keras.models import Model
from keras.layers import Input, LSTM, Dense
from keras.callbacks import ModelCheckpoint,ReduceLROnPlateau,EarlyStopping
import sys,glob,io,random
if '/scratch/wz1219/FoV' not in sys.path:
    sys.path.insert(0, '/scratch/wz1219/FoV')
from mycode.dataLayer import DataLayer
import mycode.cost as costfunc
# from mycode.provide_hidden_state import multilayer_perceptron_hidden_state_series,multilayer_perceptron_hidden_state,dynamicRNN_hidden_state
from mycode.config import cfg
from mycode.dataIO import clip_xyz
import _pickle as pickle
import numpy as np
import matplotlib.pyplot as plt
import pdb

batch_size = 64  # Batch size for training.
epochs = 200  # Number of epochs to train for.
latent_dim = 64  # Latent dimensionality of the encoding space.

fps = 30
num_encoder_tokens = 3*fps
num_decoder_tokens = 6
max_encoder_seq_length = cfg.running_length
max_decoder_seq_length = cfg.predict_step
num_user = 48

## data preparation
if cfg.use_xyz:
    #all_video_data = pickle.load(open('/scratch/wz1219/FoV/data/exp_1_code.p','rb'))
    all_video_data = pickle.load(open('/scratch/wz1219/FoV/data/exp_1_xyz.p','rb'))
    data_dim = 3
#all_video_data = clip_xyz(all_video_data)
datadb = all_video_data.copy()


def reshape2second_stacks(per_video_db,collapse_user=False):
    """reshape from N* *90 into M*max_encoder_seq_length*90"""
    # split into chunks of max_encoder_seq_length seconds
    per_video_db = np.array(np.split(per_video_db,per_video_db.shape[1]/max_encoder_seq_length,axis=1))
    # decoder target
    per_video_db_future = per_video_db[1:,:,:,:]
    last_location = per_video_db[:-1,:,-1,:][:,:,np.newaxis,:]
    # decoder input
    per_video_db_future_input = np.concatenate((last_location,per_video_db_future[:,:,:-1,:]),axis=2)

    if collapse_user:
        #mix user and time
        #reshape, collapse user dimension
        per_video_db = per_video_db[:-1,:,:,:]
        per_video_db = np.reshape(per_video_db,(-1,max_encoder_seq_length,num_encoder_tokens))
        per_video_db_future = np.reshape(per_video_db_future,(-1,max_encoder_seq_length,num_encoder_tokens))
        per_video_db_future_input = np.reshape(per_video_db_future_input,(-1,max_encoder_seq_length,num_encoder_tokens))
    else:
        #keep the user dimension in dim 0
        #userful for other users' data
        per_video_db = per_video_db[:-1,:,:,:]
        per_video_db = per_video_db.transpose((1,0,2,3))
        per_video_db_future = per_video_db_future.transpose((1,0,2,3))
        per_video_db_future_input = per_video_db_future_input.transpose((1,0,2,3))

    return per_video_db,per_video_db_future,per_video_db_future_input


def get_data(datadb,pick_user=False):
    if not pick_user:
        ### concat all 9 videos and all users
        # don't distinguish users or videos during training
        _video_db = np.zeros((1,max_encoder_seq_length,num_encoder_tokens))
        _video_db_future = np.zeros((1,max_encoder_seq_length,num_encoder_tokens))
        _video_db_future_input = np.zeros((1,max_encoder_seq_length,num_encoder_tokens))
        for _video_ind in datadb.keys():
            per_video_db = np.stack((datadb[_video_ind]['x'],datadb[_video_ind]['y'],datadb[_video_ind]['z']),axis=-1)
            #per_video_db = np.stack((datadb[_video_ind]['one_hot_code_row'],datadb[_video_ind]['one_hot_code_col']),axis = -1)
            per_video_db = per_video_db[:,per_video_db.shape[1]-per_video_db.shape[1]/fps/max_encoder_seq_length*fps*max_encoder_seq_length:,:] #cut head to make dividable by seconds
            #one_hot_matrix = np.zeros((per_video_db.shape[0],per_video_db.shape[1],72))
            #for i in range(per_video_db.shape[0]):
            #    for j in range(per_video_db.shape[1]):
            #        temp = np.zeros((6,12))
            #        temp[int(per_video_db[i,j,0]),int(per_video_db[i,j,1])] = 1
            #        one_hot_matrix[i,j,:] = temp.flatten()
            #        print(int(per_video_db[i,j,0]),int(per_video_db[i,j,1]))
            #per_video_db = one_hot_matrix
            per_video_db = np.reshape(per_video_db,(per_video_db.shape[0],per_video_db.shape[1]/fps,num_encoder_tokens))

            per_video_db, per_video_db_future, per_video_db_future_input = reshape2second_stacks(per_video_db,collapse_user=True)
            _video_db = np.concatenate((_video_db, per_video_db),axis=0)
            _video_db_future = np.concatenate((_video_db_future, per_video_db_future),axis=0)
            _video_db_future_input = np.concatenate((_video_db_future_input, per_video_db_future_input),axis=0)
        return _video_db[1:,:,:], _video_db_future[1:,:,:], _video_db_future_input[1:,:,:]
    else:
        _video_db_tar = np.zeros((1,max_encoder_seq_length,num_encoder_tokens))
        _video_db_future_tar = np.zeros((1,max_encoder_seq_length,num_encoder_tokens))
        _video_db_future_input_tar = np.zeros((1,max_encoder_seq_length,num_encoder_tokens))

        _video_db_oth = np.zeros((num_user-1,1,max_encoder_seq_length,num_encoder_tokens))
        _video_db_future_oth = np.zeros((num_user-1,1,max_encoder_seq_length,num_encoder_tokens))
        _video_db_future_input_oth = np.zeros((num_user-1,1,max_encoder_seq_length,num_encoder_tokens))

        for _video_ind in datadb.keys():
            # for each video, pick out a random target user, split target user and other users
            _target_user = np.random.randint(0,num_user)
            per_video_db = np.stack((datadb[_video_ind]['x'],datadb[_video_ind]['y'],datadb[_video_ind]['z']),axis=-1)
            per_video_db = per_video_db[:,per_video_db.shape[1]-per_video_db.shape[1]/fps/max_encoder_seq_length*fps*max_encoder_seq_length:,:] #cut head to make dividable by seconds
            per_video_db = np.reshape(per_video_db,(per_video_db.shape[0],per_video_db.shape[1]/fps,num_encoder_tokens))
            
            per_video_db_tar = per_video_db[_target_user,:][np.newaxis,:,:]
            per_video_db_oth = np.delete(per_video_db,_target_user,axis=0)


            per_video_db_tar, per_video_db_future_tar, per_video_db_future_input_tar = reshape2second_stacks(per_video_db_tar,collapse_user=True)
            per_video_db_oth, per_video_db_future_oth, per_video_db_future_input_oth = reshape2second_stacks(per_video_db_oth,collapse_user=False)

            _video_db_tar = np.concatenate((_video_db_tar, per_video_db_tar),axis=0)
            _video_db_future_tar = np.concatenate((_video_db_future_tar, per_video_db_future_tar),axis=0)
            _video_db_future_input_tar = np.concatenate((_video_db_future_input_tar, per_video_db_future_input_tar),axis=0)

            _video_db_oth = np.concatenate((_video_db_oth, per_video_db_oth),axis=1)
            _video_db_future_oth = np.concatenate((_video_db_future_oth, per_video_db_future_oth),axis=1)
            _video_db_future_input_oth = np.concatenate((_video_db_future_input_oth, per_video_db_future_input_oth),axis=1)

        return _video_db_tar[1:,:,:], _video_db_future_tar[1:,:,:], _video_db_future_input_tar[1:,:,:], \
                _video_db_oth[:,1:,:,:], _video_db_future_oth[:,1:,:,:], _video_db_future_input_oth[:,1:,:,:]


## only use one video for now
# _video_ind = 0
# per_video_db = np.stack((datadb[_video_ind]['x'],datadb[_video_ind]['y'],datadb[_video_ind]['z']),axis=-1)
# per_video_db = per_video_db[:,per_video_db.shape[1]-per_video_db.shape[1]/fps*fps:,:] #cut head to make dividable by seconds
# per_video_db = np.reshape(per_video_db,(per_video_db.shape[0],per_video_db.shape[1]/fps,90))

def get_gt_target_xyz(y):
    """get gt mean var"""
    target_x = y[:,:,0::3]
    target_y = y[:,:,1::3]
    target_z = y[:,:,2::3]
    gt_mean_x = np.mean(target_x, axis=2)[:,:,np.newaxis]
    gt_var_x = np.var(target_x, axis=2)[:,:,np.newaxis]
    gt_mean_y = np.mean(target_y, axis=2)[:,:,np.newaxis]
    gt_var_y = np.var(target_y, axis=2)[:,:,np.newaxis]
    gt_mean_z = np.mean(target_z, axis=2)[:,:,np.newaxis]
    gt_var_z = np.var(target_z, axis=2)[:,:,np.newaxis]
    return np.concatenate((gt_mean_x,gt_mean_y,gt_mean_z,gt_var_x,gt_var_y,gt_var_z),axis=2)

def get_gt_target_row_col(y):
    """get gt mean var"""
    target_row = y[:,:,0::2]
    target_col = y[:,:,1::2]
    gt_mean_row = np.mean(target_row, axis=2)[:,:,np.newaxis]
    gt_var_row = np.var(target_row, axis=2)[:,:,np.newaxis]
    gt_mean_col = np.mean(target_col, axis=2)[:,:,np.newaxis]
    gt_var_col = np.var(target_col, axis=2)[:,:,np.newaxis]
    return np.concatenate((gt_mean_row,gt_mean_col,gt_var_row,gt_var_col),axis=2)

def get_gt_target_72(y):
     """get gt mean var"""
     for i in range(72):
         target_code = y[:,:,i::72]
         gt_mean = np.mean(target_code,axis=2)[:,:,np.newaxis]
         #gt_var = np.var(target_code,axis=2)[:,:,np.newaxis]
         if i == 0:
             gt = gt_mean
         else:
             gt = np.concatenate((gt,gt_mean),axis=2)
     return gt
         
         

#assign data
## only use one video for now
# encoder_input_data = per_video_db[:,100:110,:]
# temp = per_video_db[:,110:120,:].copy()
# temp1 = np.zeros_like(temp)
# temp1[:,1:,:] = temp[:,:-1,:].copy()
# temp1[:,0,:] = encoder_input_data[:,-1,:]



_video_db,_video_db_future,_video_db_future_input = get_data(datadb,pick_user=False)
num_samples = _video_db.shape[0] # Number of samples to train on.
num_testing_sample = 1000
print(_video_db.shape)


#encoder_input_data = _video_db[:-num_testing_sample,:,:]######################
encoder_input_data = get_gt_target_xyz(_video_db)[:-num_testing_sample,:,:]
decoder_target_data = get_gt_target_xyz(_video_db_future)[:-num_testing_sample,:,:]
decoder_input_data = get_gt_target_xyz(_video_db_future_input)[:-num_testing_sample,:,:]
#decoder_target_data = get_gt_target_row_col(_video_db_future)[:-num_testing_sample,:,:]
#decoder_input_data = get_gt_target_row_col(_video_db_future_input)[:-num_testing_sample,:,:]
#decoder_target_data = get_gt_target_72(_video_db_future)[:-num_testing_sample,:,:]
#decoder_input_data = get_gt_target_72(_video_db_future_input)[:-num_testing_sample,:,:]

### ====================Graph def====================
# Define an input sequence and process it.
encoder_inputs = Input(shape=(None, 6))
encoder = LSTM(latent_dim, return_state=True)
encoder_outputs, state_h, state_c = encoder(encoder_inputs)
# We discard `encoder_outputs` and only keep the states.
encoder_states = [state_h, state_c]

# Set up the decoder, using `encoder_states` as initial state.
decoder_inputs = Input(shape=(None, num_decoder_tokens))
# We set up our decoder to return full output sequences,
# and to return internal states as well. We don't use the
# return states in the training model, but we will use them in inference.
decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_inputs,
                                     initial_state=encoder_states)
decoder_dense = Dense(num_decoder_tokens,activation='tanh')
decoder_outputs = decoder_dense(decoder_outputs)



### ====================Training====================
model_checkpoint = ModelCheckpoint('fov_s2s_noteacherforcing_epoch{epoch:02d}-{val_loss:.4f}.h5', monitor='val_loss', save_best_only=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                 patience=3, min_lr=1e-6)
stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='auto')
# Define the model that will turn
# `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

# Run training
model.compile(optimizer='Adam', loss='mean_squared_error',metrics=['accuracy'])
model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
          batch_size=batch_size,
          epochs=epochs,
          validation_split=0.2,
          shuffle=True,
          callbacks=[model_checkpoint, reduce_lr, stopping])
# Save model
model.save('fov_s2s_tanh.h5')

print("aaaaaaaaaaaaaaaa")
### ====================Testing====================
# Next: inference mode (sampling).
# Here's the drill:
# 1) encode input and retrieve initial decoder state
# 2) run one step of decoder with this initial state
# and a "start of sequence" token as target.
# Output will be the next target token
# 3) Repeat with the current target token and current states

# Define sampling models
encoder_model = Model(encoder_inputs, encoder_states)

decoder_state_input_h = Input(shape=(latent_dim,))
decoder_state_input_c = Input(shape=(latent_dim,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
decoder_outputs, state_h, state_c = decoder_lstm(
    decoder_inputs, initial_state=decoder_states_inputs)
decoder_states = [state_h, state_c]
decoder_outputs = decoder_dense(decoder_outputs)
decoder_model = Model(
    [decoder_inputs] + decoder_states_inputs,
    [decoder_outputs] + decoder_states)


# from keras.models import load_model
# temp = load_model('fov_s2s.h5')

def decode_sequence_fov(input_seq):
    # Encode the input as state vectors.
    states_value = encoder_model.predict(get_gt_target_xyz(input_seq))

    last_location = input_seq[0,-1,:][np.newaxis,np.newaxis,:]
    last_mu_var = get_gt_target_xyz(last_location) ########xyz
    target_seq = last_mu_var
    # target_seq = np.zeros((1, 1, num_decoder_tokens))

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    decoded_sentence = []
    for ii in range(cfg.predict_step):
        output_tokens, h, c = decoder_model.predict(
            [target_seq] + states_value)

        decoded_sentence+=[output_tokens]

        # # Update the target sequence (of length 1).
        # target_seq = np.zeros((1, 1, num_decoder_tokens))
        target_seq = output_tokens

        # Update states
        states_value = [h, c]

    return decoded_sentence


gt_sentence_list = []
decoded_sentence_list = []
for seq_index in range(num_samples-num_testing_sample,num_samples-num_testing_sample+100):
    # Take one sequence (part of the training set)
    # for trying out decoding.
    input_seq = _video_db[seq_index: seq_index + 1,:,:]
    #input_seq = get_gt_xyz
    decoded_sentence = decode_sequence_fov(input_seq)
    decoded_sentence_list+=[decoded_sentence]
    gt_sentence = _video_db_future[seq_index: seq_index + 1,:,:]
    gt_sentence_list+=[gt_sentence]
    decoder_target = get_gt_target_xyz(gt_sentence)##############xyz
    # print('-')
    # print('Decoded sentence - decoder_target:', np.squeeze(np.array(decoded_sentence))[:,:3]-np.squeeze(decoder_target)[:,:3])

pickle.dump(decoded_sentence_list,open('decoded_sentence.p','wb'))
pickle.dump(gt_sentence_list,open('gt_sentence_list.p','wb'))
print('Testing finished!')







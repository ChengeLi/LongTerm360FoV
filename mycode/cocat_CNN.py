"""
seq2seq without teacher forcing, 
- use CNN to extract features from other users
- no seperate loss for CNN 
- encoder-decoder for target user
- concat CNN feature with encoder/decoder hidden states and then predict
"""

from __future__ import print_function
from keras.models import Model
from keras.layers import Input, LSTM, Dense
from keras.layers import Lambda,Concatenate,Flatten,ConvLSTM2D
from keras.layers import Reshape,Permute
from keras.layers import Conv2D,MaxPooling2D,BatchNormalization
from keras.callbacks import ModelCheckpoint,ReduceLROnPlateau,EarlyStopping
from keras import backend as K
from keras.models import load_model
from keras import optimizers
import sys,glob,io,random
if './360video/' not in sys.path:
    sys.path.insert(0, './360video/')
from mycode.dataLayer import DataLayer
import mycode.cost as costfunc
from mycode.config import cfg
from mycode.dataIO import clip_xyz
from mycode.utility import reshape2second_stacks,get_data
from mycode.utility import get_shuffle_index,shuffle_data,get_gt_target_xyz,get_gt_target_xyz_oth
from mycode.utility import slice_layer,rand_sample_ind,rand_sample
from random import shuffle
import matplotlib.pyplot as plt
import _pickle as pickle
import numpy as np
import pdb

experiment = 1
batch_size = 32  # Batch size for training.
epochs = 200  # Number of epochs to train for.
latent_dim = 64  # Latent dimensionality of the encoding space.

fps = 30
num_encoder_tokens = 3*fps
num_decoder_tokens = 6
max_encoder_seq_length = cfg.running_length
max_decoder_seq_length = cfg.predict_step
num_user = 48



# --------------------------
use_fclstm_tar = False
kernel_size = cfg.conv_kernel_size


## utility layers
flatten_layer = Flatten()
expand_dim_layer = Lambda(lambda x: K.expand_dims(x,1))
Concatenatelayer = Concatenate(axis=1)
Concatenatelayer1 = Concatenate(axis=-1)
Concatenatelayer_dim3 = Concatenate(axis=3)
get_dim1_layer = Lambda(lambda x: x[:,0,:])
reshape_layer = Reshape((1,fps,-1))
transpose_layer = Permute((1, 3, 2, 4))#note that the first dimension cannot be permuted.


def generate_fake_batch(x):
    """generate new data for 1 second using predicted mean and variance"""
    mu = x[0]
    var = x[1]
    temp = K.random_normal(shape = (batch_size,fps,1), mean=mu,stddev=K.sqrt(var))
    return temp

generate_fake_batch_layer = Lambda(lambda x: generate_fake_batch(x))




### ====================Graph def====================
###======CNN on others' past encoder======
whole_span = max_encoder_seq_length+max_decoder_seq_length
encoder_inputs_oth = Input(shape=(whole_span,1,fps,(num_user-1)*3))

latent_dim = 4
cnnlayer1 = Conv2D(filters=latent_dim, kernel_size=(1, kernel_size), strides=(1, 1),padding='same',
                 activation='relu', dilation_rate=(1, 1),
                 input_shape=(1,fps,(num_user-1)*3))

cnnlayer2 = Conv2D(filters=latent_dim*2, kernel_size=(1, kernel_size), strides=(1, 1),padding='same',
                 activation='relu', dilation_rate=(1, 1),
                 input_shape=(1,fps,latent_dim))

cnnlayer3 = Conv2D(filters=latent_dim*4, kernel_size=(1, kernel_size), strides=(1, 1),padding='same',
                 activation='relu', dilation_rate=(1, 1),
                 input_shape=(1,fps,latent_dim*2))

bnlayer = BatchNormalization(axis=-1,center=True, scale=True)




# userpooling = MaxPooling2D(pool_size=(num_user-1, 1), strides=None, padding='valid', data_format=None)


def _get_cnn_fea(cnn_input):
    cnn_oth_output1 = cnnlayer1(cnn_input)
    cnn_oth_output2 = cnnlayer2(cnn_oth_output1)
    cnn_oth_output3 = cnnlayer3(cnn_oth_output2)
    # cnn_oth_output3 = userpooling(cnn_oth_output3)
    return cnn_oth_output3



if use_fclstm_tar:
    ###======target user encoder======
    encoder_inputs = Input(shape=(None, num_encoder_tokens))
    encoder = LSTM(latent_dim, return_sequences=True, return_state=True)
    encoder_outputs, state_h, state_c = encoder(encoder_inputs)
    states = [state_h, state_c]

    ###======target user decoder======
    # Set up the decoder for target branch
    # only process one timestep at a time.
    # decoder_inputs = Input(shape=(1, num_decoder_tokens))
    decoder_inputs = Input(shape=(1, num_encoder_tokens))
    decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
else:
    ###======convLSTM target user encoder======
    latent_dim_target = 16
    encoder_inputs = Input(shape=(max_encoder_seq_length,1,fps,3))

    convlstm_encoder = ConvLSTM2D(filters=latent_dim_target, kernel_size=(1, kernel_size),
                       input_shape=(1,fps,3),dropout=cfg.dropout_rate, recurrent_dropout=0.0,
                       stateful=False,
                       padding='same', return_sequences=True, return_state=True)
    pst_outputs_sqns0, pst_state_h, pst_state_c = convlstm_encoder(encoder_inputs)
    states0 = [pst_state_h, pst_state_c]
    convlstm_encoder1 = ConvLSTM2D(filters=latent_dim_target/2, kernel_size=(1, kernel_size),
                       input_shape=(1,fps,latent_dim_target),dropout=cfg.dropout_rate, recurrent_dropout=0.0,
                       stateful=False,
                       padding='same', return_sequences=True, return_state=True)
    pst_outputs_sqns1, pst_state_h, pst_state_c = convlstm_encoder1(pst_outputs_sqns0)
    states1 = [pst_state_h, pst_state_c]

    convlstm_encoder2 = ConvLSTM2D(filters=latent_dim_target/4, kernel_size=(1, kernel_size),
                       input_shape=(1,fps,latent_dim_target/2),dropout=cfg.dropout_rate, recurrent_dropout=0.0,
                        stateful=False,
                        padding='same', return_sequences=True, return_state=True)
    pst_outputs_sqns2, pst_state_h, pst_state_c = convlstm_encoder2(pst_outputs_sqns1)
    states2 = [pst_state_h, pst_state_c]
    pst_outputs_sqns = Concatenatelayer1([pst_outputs_sqns0,pst_outputs_sqns1,pst_outputs_sqns2])

    ###======convLSTM on target future decoder======
    if cfg.stateful_across_batch:
        decoder_inputs = Input(batch_shape=(batch_size,1,1,fps,3))
    else:
        decoder_inputs = Input(shape=(1,1,fps,3))

    convlstm_decoder = ConvLSTM2D(filters=latent_dim_target, kernel_size=(1, kernel_size),
                       input_shape=(1,fps,3),dropout=cfg.dropout_rate, recurrent_dropout=0.0,
                       stateful=False,
                       padding='same', return_sequences=True, return_state=True)

    convlstm_decoder1 = ConvLSTM2D(filters=latent_dim_target/2, kernel_size=(1, kernel_size),
                       input_shape=(1,fps,latent_dim_target),dropout=cfg.dropout_rate, recurrent_dropout=0.0,
                       stateful=False,
                       padding='same', return_sequences=True, return_state=True)

    convlstm_decoder2 = ConvLSTM2D(filters=latent_dim_target/4, kernel_size=(1, kernel_size),
                       input_shape=(1,fps,latent_dim_target/2),dropout=cfg.dropout_rate, recurrent_dropout=0.0,
                       stateful=False,
                       padding='same', return_sequences=True, return_state=True)

encoder_dense = Dense(3,activation='tanh')
if cfg.predict_mean_var:
    decoder_dense = Dense(num_decoder_tokens,activation=None)
else:
    decoder_dense = Dense(3,activation=None)




## concat states
all_outputs = []
inputs = decoder_inputs
all_outputs_target_past = []

for time_ind in range(max_encoder_seq_length):
    #predict for target user's past (reconstruction)
    encoder_outputs_slice = slice_layer(1,time_ind,time_ind+1)(pst_outputs_sqns)
    # get cnn feature
    cnn_input = get_dim1_layer(slice_layer(1,time_ind,time_ind+1)(encoder_inputs_oth))
    cnn_oth_output3 = _get_cnn_fea(cnn_input)
    concat_cnn_state = Concatenatelayer_dim3([cnn_oth_output3,get_dim1_layer(encoder_outputs_slice)]) 
    outputs = encoder_dense(concat_cnn_state)
    all_outputs_target_past.append(outputs)

for time_ind in range(max_decoder_seq_length):
    # decoder
    fut_outputs_sqns0, fut_state_h, fut_state_c = convlstm_decoder([inputs]+states0)
    states0 = [fut_state_h, fut_state_c]
    fut_outputs_sqns1, fut_state_h, fut_state_c = convlstm_decoder1([fut_outputs_sqns0]+states1)
    states1 = [fut_state_h, fut_state_c]
    fut_outputs_sqns2, fut_state_h, fut_state_c = convlstm_decoder2([fut_outputs_sqns1]+states2)
    states2 = [fut_state_h, fut_state_c]
    decoder_states = Concatenatelayer1([fut_outputs_sqns0,fut_outputs_sqns1,fut_outputs_sqns2])

    # get cnn feature
    cnn_input = get_dim1_layer(slice_layer(1,time_ind+max_encoder_seq_length,time_ind+max_encoder_seq_length+1)(encoder_inputs_oth))
    cnn_oth_output3 = _get_cnn_fea(cnn_input)
    concat_cnn_state = Concatenatelayer_dim3([cnn_oth_output3,get_dim1_layer(decoder_states)]) 
    if cfg.predict_mean_var:
        concat_cnn_state_flat = flatten_layer(concat_cnn_state)
        outputs = decoder_dense(concat_cnn_state_flat)
    else:
        outputs = decoder_dense(concat_cnn_state)

    outputs = expand_dim_layer(outputs)
    if cfg.sample_and_refeed:
        ### generated from gaussian
        ux_temp = slice_layer(2,0,1)(outputs)
        uy_temp = slice_layer(2,1,2)(outputs)
        uz_temp = slice_layer(2,2,3)(outputs)
        varx_temp = slice_layer(2,3,4)(outputs)
        vary_temp = slice_layer(2,4,5)(outputs)
        varz_temp = slice_layer(2,5,6)(outputs)

        temp_newdata = expand_dim_layer(expand_dim_layer(Concatenatelayer1([generate_fake_batch_layer([ux_temp,varx_temp]),
                            generate_fake_batch_layer([uy_temp,vary_temp]),
                            generate_fake_batch_layer([uz_temp,varz_temp])])))
        inputs = temp_newdata  
    else:
        inputs = expand_dim_layer(outputs)
    all_outputs.append(outputs)


#### both encoder and decoder losses
# decoder_outputs = Lambda(lambda x: K.concatenate(x, axis=1))(all_outputs)
# encoder_reconstruct_tar = Lambda(lambda x: K.concatenate(x, axis=1))(all_outputs_target_past)

# # Define and compile model as previously
# model = Model([encoder_inputs, encoder_inputs_oth, decoder_inputs],
#                [decoder_outputs,encoder_reconstruct_tar])
# Adam = optimizers.Adam(lr=0.001,clipnorm=1)
# model.compile(optimizer='Adam', loss=['mean_squared_error','mean_squared_error'],
#               loss_weights=[1,0.1])

#### only decoder loss
decoder_outputs = Lambda(lambda x: K.concatenate(x, axis=1))(all_outputs)

# Define and compile model as previously
model = Model([encoder_inputs, encoder_inputs_oth, decoder_inputs],
               decoder_outputs)
Adam = optimizers.Adam(lr=0.001,clipnorm=1)
model.compile(optimizer='Adam', loss='mean_squared_error')




#### ====================data====================
## get target user data and other user's data
# _video_db_tar, _video_db_future_tar, _video_db_future_input_tar, \
# _video_db_oth,_video_db_future_oth,_video_db_future_input_oth = get_data(datadb,pick_user=True)

## load cached data
_video_db_tar = pickle.load(open('./cache/format2/_video_db_tar_exp'+str(experiment)+'.p','rb'))
_video_db_future_tar = pickle.load(open('./cache/format2/_video_db_future_tar_exp'+str(experiment)+'.p','rb'))
_video_db_future_input_tar = pickle.load(open('./cache/format2/_video_db_future_input_tar_exp'+str(experiment)+'.p','rb'))
_video_db_oth = pickle.load(open('./cache/format2/_video_db_oth_exp'+str(experiment)+'.p','rb'))
_video_db_future_oth = pickle.load(open('./cache/format2/_video_db_future_oth_exp'+str(experiment)+'.p','rb'))
_video_db_future_input_oth = pickle.load(open('./cache/format2/_video_db_future_input_oth_exp'+str(experiment)+'.p','rb'))


def data_sanity_check(_video_db_tar,_video_db_future_tar,_video_db_future_input_tar):
    sample_ind = np.random.randint(0,_video_db_future_input_tar.shape[0])
    assert (_video_db_tar[sample_ind,:][-1,:]-_video_db_future_input_tar[sample_ind,:][0,:]).sum()==0
    print(np.abs(_video_db_tar[sample_ind,:][-1,:]-_video_db_future_tar[sample_ind,:][0,:]))
    
def _reshape_others_data(_video_db_oth):
    ## to match Input shape: encoder_inputs_oth
    _video_db_oth = _video_db_oth.transpose((1,2,0,3))
    _video_db_oth = _video_db_oth.reshape((_video_db_oth.shape[0],_video_db_oth.shape[1],_video_db_oth.shape[2],
                    fps,3))
    return _video_db_oth
def _reshape_others_data2(_video_db_oth):
    """collapse user index dimension, merging into xyz as channel dimension"""
    #from (N, 10, 47, 30, 3) to (N, 10, 30, 47*3)
    _video_db_oth = _video_db_oth.transpose((0,1,3,2,4))
    _video_db_oth = _video_db_oth.reshape((_video_db_oth.shape[0],_video_db_oth.shape[1],_video_db_oth.shape[2],-1))
    return _video_db_oth



def get_whole_span(_video_db_oth):
    # NOTE: must for unshuffled data!
    assert cfg.shuffle_data==False
    # get adjacent time periods: past+future as whole span
    # from  (N, 10, 47, 30, 3) to (N/2, 20, 47, 30, 3)
    length = _video_db_oth.shape[0]
    # _video_db_oth_span = np.zeros((length,2*_video_db_oth.shape[1],_video_db_oth.shape[2],_video_db_oth.shape[3],_video_db_oth.shape[4]))
    _video_db_oth_span = np.zeros((length,2*_video_db_oth.shape[1],_video_db_oth.shape[2],_video_db_oth.shape[3]))
    for ii in range(length-1):
        temp = np.concatenate((_video_db_oth[ii,:],_video_db_oth[ii+1,:]))
        _video_db_oth_span[ii] = temp
    #the last row is all zero!
    return _video_db_oth_span


_video_db_oth = _reshape_others_data(_video_db_oth)
_video_db_future_oth = _reshape_others_data(_video_db_future_oth)
# _video_db_future_input_oth = _reshape_others_data(_video_db_future_input_oth)
_video_db_oth = _reshape_others_data2(_video_db_oth)
_video_db_future_oth = _reshape_others_data2(_video_db_future_oth)

_video_db_tar = _video_db_tar.reshape((_video_db_tar.shape[0],_video_db_tar.shape[1],fps,3))
_video_db_future_tar = _video_db_future_tar.reshape((_video_db_tar.shape[0],_video_db_tar.shape[1],fps,3))

total_num_samples = _video_db_tar.shape[0]
num_testing_sample = int(0.15*total_num_samples)#use last 1000 as test



if cfg.shuffle_data:
    #### shuffle the whole dataset
    # index_shuf = get_shuffle_index(total_num_samples)
    index_shuf = pickle.load(open('index_shuf'+'_exp'+str(experiment)+'.p','rb'))
    print('Shuffle data before training and testing.')
    _video_db_tar = shuffle_data(index_shuf,_video_db_tar)
    _video_db_future_tar = shuffle_data(index_shuf,_video_db_future_tar)
    _video_db_future_input_tar = shuffle_data(index_shuf,_video_db_future_input_tar)

    _video_db_oth = shuffle_data(index_shuf,_video_db_oth)
    _video_db_future_oth = shuffle_data(index_shuf,_video_db_future_oth)
    # _video_db_future_input_oth = shuffle_data(index_shuf,_video_db_future_input_oth)


#### prepare training data
# data_sanity_check(_video_db_tar,_video_db_future_tar,_video_db_future_input_tar)

### target user
encoder_input_data = _video_db_tar[:-num_testing_sample,:,:][:,:,np.newaxis,:,:]
# decoder_target_data = _video_db_future_tar[:-num_testing_sample,:,:]
decoder_input_data = _video_db_tar[:-num_testing_sample,-1,:][:,np.newaxis,np.newaxis,:]
decoder_target_data = get_gt_target_xyz(_video_db_future_tar)[:-num_testing_sample,:,:]
# decoder_input_data = get_gt_target_xyz(_video_db_tar)[:-num_testing_sample,-1,:][:,np.newaxis,:]
# decoder_input_data1 = get_gt_target_xyz(_video_db_future_input_tar)[:-num_testing_sample,0,:][:,np.newaxis,:]

### other users
_video_db_oth_span = get_whole_span(_video_db_oth)
others_pst_input_data = _video_db_oth_span[:-num_testing_sample][:,:,np.newaxis,:,:]

def _get_next_timestetp_data(input_data):
    """delete the first time stamp data and append the first time stamp in next chunk"""
    assert cfg.shuffle_data==False
    input_data_next = np.zeros_like(input_data)
    temp = input_data[:-1,1:,:]
    temp2 = input_data[1:,0,:][:,np.newaxis,:]
    input_data_next[:-1,:] = np.concatenate((temp,temp2),axis=1)
    return input_data_next

def _get_next_timestetp_data_span(input_data):
    #for spanned data
    assert cfg.shuffle_data==False
    input_data_next = np.zeros_like(input_data)
    time_span = input_data.shape[1]
    temp = input_data[:-1,1:,:]
    temp2 = input_data[1:,time_span/2+0,:][:,np.newaxis,:]
    input_data_next[:-1,:] = np.concatenate((temp,temp2),axis=1)
    return input_data_next

encoder_input_data_next = _get_next_timestetp_data(encoder_input_data[:,:,0,:,:])
# data_sanity_check(encoder_input_data,decoder_target_data,decoder_input_data)


## ensure dividable by batch size
sample_ind = rand_sample_ind(total_num_samples,num_testing_sample,batch_size)
encoder_input_data = rand_sample(encoder_input_data,sample_ind)
decoder_input_data = rand_sample(decoder_input_data,sample_ind)
decoder_target_data = rand_sample(decoder_target_data,sample_ind)
others_pst_input_data = rand_sample(others_pst_input_data,sample_ind)



### ====================Training====================
# model = load_model('convLSTM_endec_11_256tanh_epoch12-1.2859.h5')
# model = load_model('convLSTM_wholespan_targetrecons_trj_decodernotanh_epoch10-0.1658.h5')
# tag = 'convLSTM_wholespan_targetrecons_trj_decodernotanh_epoch'
# tag = 'convLSTM_wholespan_targetrecons_trj_decodernotanh_epoch'
# tag = 'concat_cnn_kernel_1_5_convlstmlatent16_epoch'
tag = 'concat_cnn1_5_16_meanvar_epoch'

model_checkpoint = ModelCheckpoint(tag+'{epoch:02d}-{val_loss:.4f}.h5', monitor='val_loss', save_best_only=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                 patience=3, min_lr=1e-6)
stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=0, mode='auto')
# model.fit([encoder_input_data, others_pst_input_data, decoder_input_data],
#             [decoder_target_data,encoder_input_data_next],
model.fit([encoder_input_data, others_pst_input_data, decoder_input_data],
          decoder_target_data,
          batch_size=batch_size,
          epochs=epochs,
          validation_split=0.2,
          # shuffle=cfg.shuffle_data, 
          shuffle=True, 
          initial_epoch=0,
          callbacks=[model_checkpoint, reduce_lr, stopping])




### ====================Testing====================
# model = load_model('concat_cnn_kernel_1_5_epoch18-0.2037.h5')
model = load_model('backup_concat_cnn_kernel_47_5_epoch12-0.2047.h5')
# model = load_model('concat_cnn_kernel_1_5_convlstmlatent16_epoch21-0.1499.h5')
# model = load_model('concat_cnn1_5_16_meanvar_epoch20-0.0661.h5')
def decode_sequence_fov(input_seq,others_pst_input_seq):            
    if input_seq.shape[0]>1:
        last_location = input_seq[:,-1,:][:,np.newaxis,:]
    elif input_seq.shape[0]==1: 
        last_location = input_seq[0,-1,:][np.newaxis,np.newaxis,:]
    # [decoded_sentence,encoder_reconstruct_sentence_tar] = model.predict([input_seq,others_pst_input_seq,last_location])
    # return decoded_sentence,encoder_reconstruct_sentence_tar
    decoded_sentence = model.predict([input_seq,others_pst_input_seq,last_location])
    return decoded_sentence



gt_sentence_list = []
decoded_sentence_list = []
gt_sentence_oth_list = []
encoder_reconstruct_tar_list = []
gt_sentence_recons_tar_list = []

_video_db_oth_span_next = _get_next_timestetp_data_span(_video_db_oth_span[total_num_samples-num_testing_sample:,:])
_video_db_tar_next = _get_next_timestetp_data(_video_db_tar[total_num_samples-num_testing_sample:,:])

# for seq_index in range(total_num_samples-num_testing_sample,total_num_samples-1):
# for seq_index in range(total_num_samples-num_testing_sample,total_num_samples-num_testing_sample+100):
for seq_index in range(total_num_samples-num_testing_sample,total_num_samples-num_testing_sample+320,batch_size):
    input_seq = _video_db_tar[seq_index: seq_index + batch_size,:,:][:,:,np.newaxis,:,:]
    input_seq_next = _video_db_tar_next[seq_index-(total_num_samples-num_testing_sample): seq_index-(total_num_samples-num_testing_sample) + batch_size,:,:][:,:,np.newaxis,:,:]

    others_pst_input_seq = _video_db_oth_span[seq_index: seq_index + batch_size,:][:,:,np.newaxis,:,:]
    # decoded_sentence,encoder_reconstruct_sentence_tar = decode_sequence_fov(input_seq,others_pst_input_seq)
    decoded_sentence = decode_sequence_fov(input_seq,others_pst_input_seq)
    
    decoded_sentence_list+=[decoded_sentence]
    # encoder_reconstruct_tar_list+=[encoder_reconstruct_sentence_tar]

    gt_sentence = _video_db_future_tar[seq_index: seq_index + batch_size,:,:]
    last_location = input_seq[0,-1,:][np.newaxis,:]
    gt_sentence_list+=[gt_sentence]
    # gt_sentence_list+=[np.concatenate((last_location,gt_sentence),axis=1)]

    # gt_sentence_recons_tar = input_seq #reconstruction
    gt_sentence_recons_tar = input_seq_next #prediction
    last_location = input_seq[0,0,:][np.newaxis,np.newaxis,:]
    gt_sentence_recons_tar_list+=[gt_sentence_recons_tar]
    # gt_sentence_recons_tar_list+=[np.concatenate((last_location,gt_sentence_recons_tar),axis=1)]

    # print('-')
    # decoder_target = get_gt_target_xyz(gt_sentence)
    # print('Decoded sentence - decoder_target:', np.squeeze(np.array(decoded_sentence))[:,:3]-np.squeeze(decoder_target)[:,:3])

pickle.dump(decoded_sentence_list,open('decoded_sentence.p','wb'))
pickle.dump(gt_sentence_list,open('gt_sentence_list.p','wb'))


pickle.dump(encoder_reconstruct_tar_list,open('decoded_sentence.p','wb'))
pickle.dump(gt_sentence_recons_tar_list,open('gt_sentence_list.p','wb'))

print('Testing finished!')











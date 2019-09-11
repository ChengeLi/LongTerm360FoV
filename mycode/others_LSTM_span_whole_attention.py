"""
seq2seq without teacher forcing, 
- others' future convLSTM spans the whole time axis: both past and future
only one convLSTM (no encoder-decoder for others)
- others' future convLSTM has its own loss function for self-reconstruction
- encoder-decoder for target user
- concat states with decoder LSTM and then predict
"""
from __future__ import print_function
from keras.models import Model
from keras.layers import Input, LSTM, Dense
from keras.layers import Lambda,Concatenate,Flatten,ConvLSTM2D
from keras.layers import Reshape,Permute
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
import mycode.utility as util
from random import shuffle
import matplotlib.pyplot as plt
import _pickle as pickle
import numpy as np
import pdb
from mycode.attention_decoder import AttentionDecoder
from keras.layers.wrappers import Bidirectional

# experiment = 1
batch_size = 32  # Batch size for training.
epochs = 200  # Number of epochs to train for.
latent_dim = 64  # Latent dimensionality of the encoding space.

fps = 30
num_encoder_tokens = 3*fps
num_decoder_tokens = 6
max_encoder_seq_length = cfg.running_length
max_decoder_seq_length = cfg.predict_step
num_user = 34 #48


# --------------------------
use_fclstm_tar = True
use_attention_decoder = True
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
###======convLSTM on others' past encoder======
## span
whole_span = max_encoder_seq_length+max_decoder_seq_length
if cfg.input_mean_var:
    encoder_inputs_oth = Input(shape=(whole_span,1,(num_user-1),6))
else:
    encoder_inputs_oth = Input(shape=(whole_span,1,fps,(num_user-1)*3))
    # encoder_inputs_oth = Input(shape=(max_encoder_seq_length,num_user-1,fps,3))

latent_dim = 32
other_lstm_encoder = ConvLSTM2D(filters=latent_dim, kernel_size=(1, kernel_size),
                   input_shape=(1,fps,(num_user-1)*3),dropout=cfg.dropout_rate, recurrent_dropout=0.0,
                   padding='same', return_sequences=True, return_state=True)
outputs_sqns_oth0, others_state_h, others_state_c = other_lstm_encoder(encoder_inputs_oth)

other_lstm_encoder1 = ConvLSTM2D(filters=latent_dim/2, kernel_size=(1, kernel_size),
                   input_shape=(1, fps, latent_dim),dropout=cfg.dropout_rate, recurrent_dropout=0.0,
                   padding='same', return_sequences=True, return_state=True)
outputs_sqns_oth1, others_state_h, others_state_c = other_lstm_encoder1(outputs_sqns_oth0)
other_lstm_encoder2 = ConvLSTM2D(filters=latent_dim/4, kernel_size=(1, kernel_size),
                   input_shape=(1, fps, latent_dim/2),dropout=cfg.dropout_rate, recurrent_dropout=0.0,
                   padding='same', return_sequences=True, return_state=True)
outputs_sqns_oth2, others_state_h, others_state_c = other_lstm_encoder2(outputs_sqns_oth1)
# states_oth = [others_state_h, others_state_c]
outputs_sqns_oth = Concatenatelayer1([outputs_sqns_oth0,outputs_sqns_oth1,outputs_sqns_oth2])

if cfg.predict_mean_var:
    pred_conv_lstm_dense = Dense(6,activation=None)
else:
    pred_conv_lstm_dense = Dense((num_user-1)*3,activation=None)
flatten_conv_lstm_state_dense = Dense(256)

if use_fclstm_tar:
    latent_dim_target_fclstm = 64
    ###======target user encoder======
    if cfg.input_mean_var:
        encoder_inputs = Input(shape=(None, 6))
    else:
        if cfg.stateful_across_batch:
            encoder_inputs = Input(batch_shape=(batch_size,None, num_encoder_tokens))
        else:
            encoder_inputs = Input(shape=(None, num_encoder_tokens))

    if use_attention_decoder:
        encoder = Bidirectional(LSTM(latent_dim_target_fclstm, return_sequences=True), merge_mode='concat')
        encoder_outputs = encoder(encoder_inputs)
    else:
        encoder = LSTM(latent_dim_target_fclstm,stateful=cfg.stateful_across_batch,return_sequences=True, return_state=True)
        encoder_outputs, state_h, state_c = encoder(encoder_inputs)
        encoder_states = [state_h, state_c]

    ###======target user decoder======
    # Set up the decoder for target branch
    if cfg.input_mean_var:
        if not cfg.teacher_forcing:
            decoder_inputs = Input(shape=(1, 6))
        else:
            decoder_inputs = Input(shape=(None, 6))
    else:        
        decoder_inputs = Input(shape=(1, num_encoder_tokens))
        if cfg.stateful_across_batch:
            decoder_inputs = Input(batch_shape=(batch_size, 1, num_encoder_tokens))
        else:
            decoder_inputs = Input(shape=(1, num_encoder_tokens))

    if use_attention_decoder:
        decoder_lstm =AttentionDecoder(latent_dim_target_fclstm,num_decoder_tokens,stateful=cfg.stateful_across_batch,return_sequences=True)
    else:
        decoder_lstm = LSTM(latent_dim_target_fclstm,stateful=cfg.stateful_across_batch,return_sequences=True, return_state=True)

else:
    ###======convLSTM target user encoder======
    latent_dim_target = 8
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

    concat_input_dim = (latent_dim+latent_dim/2+latent_dim/4)+3
    convlstm_decoder = ConvLSTM2D(filters=latent_dim_target, kernel_size=(1, kernel_size),
                       input_shape=(1,fps,concat_input_dim),dropout=cfg.dropout_rate, recurrent_dropout=0.0,
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




if cfg.predict_mean_var:
    encoder_dense = Dense(6,activation='tanh')
    decoder_dense = Dense(6,activation=None)
else:
    encoder_dense = Dense(3,activation='tanh')
    decoder_dense = Dense(3,activation=None)



## concat states
all_outputs = []
all_outputs_oth= []
if not cfg.teacher_forcing:
    inputs = decoder_inputs
else:
    inputs = util.slice_layer(1,0,1)(decoder_inputs)
all_outputs_target_past = []


for time_ind in range(max_encoder_seq_length):
    #predict for others' past (reconstruction)
    outputs_sqns_oth_slice = util.slice_layer(1,time_ind,time_ind+1)(outputs_sqns_oth)
    outputs_oth = pred_conv_lstm_dense(outputs_sqns_oth_slice)
    all_outputs_oth.append(outputs_oth)

    #predict for target user's past (reconstruction)
    if use_fclstm_tar:
        encoder_outputs_slice = util.slice_layer(1,time_ind,time_ind+1)(encoder_outputs)
    else:
        encoder_outputs_slice = util.slice_layer(1,time_ind,time_ind+1)(pst_outputs_sqns)
    outputs = encoder_dense(encoder_outputs_slice)
    all_outputs_target_past.append(outputs)

if use_fclstm_tar: 
    states = encoder_states

for time_ind in range(max_decoder_seq_length):
    outputs_sqns_oth_slice = util.slice_layer(1,time_ind+max_encoder_seq_length,time_ind+max_encoder_seq_length+1)(outputs_sqns_oth)

    if use_fclstm_tar:
        if use_attention_decoder:
            ##how to initialize states for the decoder??
            decoder_states = decoder_lstm(inputs)
        else:
            decoder_states, state_h, state_c = decoder_lstm(inputs,initial_state=states)
        states = [state_h, state_c]
        convlstm_state = flatten_layer(outputs_sqns_oth_slice)
        convlstm_state = flatten_conv_lstm_state_dense(convlstm_state)
        concat_state = Concatenatelayer([get_dim1_layer(decoder_states),convlstm_state])
        outputs = decoder_dense(concat_state)
        outputs = expand_dim_layer(outputs)
    else:
        concat_inputs = Concatenatelayer1([inputs,outputs_sqns_oth_slice])
        fut_outputs_sqns0, fut_state_h, fut_state_c = convlstm_decoder([concat_inputs]+states0)
        states0 = [fut_state_h, fut_state_c]
        fut_outputs_sqns1, fut_state_h, fut_state_c = convlstm_decoder1([fut_outputs_sqns0]+states1)
        states1 = [fut_state_h, fut_state_c]
        fut_outputs_sqns2, fut_state_h, fut_state_c = convlstm_decoder2([fut_outputs_sqns1]+states2)
        states2 = [fut_state_h, fut_state_c]
        decoder_states = Concatenatelayer1([fut_outputs_sqns0,fut_outputs_sqns1,fut_outputs_sqns2])
    
        if cfg.predict_mean_var:
            decoder_states_flat = flatten_layer(decoder_states)
            outputs = decoder_dense(decoder_states_flat)
        else:
            outputs = decoder_dense(decoder_states)
            # outputs = expand_dim_layer(outputs)
    if not cfg.teacher_forcing:
        if cfg.predict_mean_var and cfg.sample_and_refeed:
            outputs = expand_dim_layer(outputs)
            ### generated from gaussian
            ux_temp = util.slice_layer(2,0,1)(outputs)
            uy_temp = util.slice_layer(2,1,2)(outputs)
            uz_temp = util.slice_layer(2,2,3)(outputs)
            varx_temp = util.slice_layer(2,3,4)(outputs)
            vary_temp = util.slice_layer(2,4,5)(outputs)
            varz_temp = util.slice_layer(2,5,6)(outputs)

            temp_newdata = expand_dim_layer(expand_dim_layer(Concatenatelayer1([generate_fake_batch_layer([ux_temp,varx_temp]),
                                generate_fake_batch_layer([uy_temp,vary_temp]),
                                generate_fake_batch_layer([uz_temp,varz_temp])])))
            inputs = temp_newdata  
        else:
            inputs = outputs
    else:
        if time_ind<max_decoder_seq_length-1:
            inputs = util.slice_layer(1,time_ind+1,time_ind+2)(decoder_inputs)


    all_outputs.append(outputs)

    ### predict others' future (reconstruction)
    outputs_oth = pred_conv_lstm_dense(outputs_sqns_oth_slice)
    all_outputs_oth.append(outputs_oth)



# Concatenate all predictions
decoder_outputs = Lambda(lambda x: K.concatenate(x, axis=1))(all_outputs)
decoder_outputs_oth = Lambda(lambda x: K.concatenate(x, axis=1))(all_outputs_oth)
encoder_reconstruct_tar = Lambda(lambda x: K.concatenate(x, axis=1))(all_outputs_target_past)

# Define and compile model as previously
model = Model([encoder_inputs, encoder_inputs_oth, decoder_inputs],
               [decoder_outputs,decoder_outputs_oth,encoder_reconstruct_tar])
Adam = optimizers.Adam(lr=0.001,clipnorm=1)
model.compile(optimizer='Adam', loss=['mean_squared_error','mean_squared_error','mean_squared_error'],
              loss_weights=[1,1,1])



#### ====================data====================
## get target user data and other user's data
# _video_db_tar, _video_db_future_tar, _video_db_future_input_tar, \
# _video_db_oth,_video_db_future_oth,_video_db_future_input_oth = get_data(datadb,pick_user=True)

# ## load cached data
# _video_db_tar = pickle.load(open('./cache/format2/_video_db_tar_exp'+str(experiment)+'.p','rb'))
# _video_db_future_tar = pickle.load(open('./cache/format2/_video_db_future_tar_exp'+str(experiment)+'.p','rb'))
# _video_db_future_input_tar = pickle.load(open('./cache/format2/_video_db_future_input_tar_exp'+str(experiment)+'.p','rb'))
# _video_db_oth = pickle.load(open('./cache/format2/_video_db_oth_exp'+str(experiment)+'.p','rb'))
# _video_db_future_oth = pickle.load(open('./cache/format2/_video_db_future_oth_exp'+str(experiment)+'.p','rb'))
# _video_db_future_input_oth = pickle.load(open('./cache/format2/_video_db_future_input_oth_exp'+str(experiment)+'.p','rb'))

#### load cached data format4 
dataformat = 'format4'
_video_db_tar = pickle.load(open('./cache/'+dataformat+'/_video_db_tar.p','rb'))
_video_db_future_tar = pickle.load(open('./cache/'+dataformat+'/_video_db_future_tar.p','rb'))
_video_db_future_input_tar = pickle.load(open('./cache/'+dataformat+'/_video_db_future_input_tar.p','rb'))
_video_db_oth = pickle.load(open('./cache/'+dataformat+'/_video_db_oth.p','rb'))
_video_db_future_oth = pickle.load(open('./cache/'+dataformat+'/_video_db_future_oth.p','rb'))
_video_db_future_input_oth = pickle.load(open('./cache/'+dataformat+'/_video_db_future_input_oth.p','rb'))


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
    assert _video_db_oth.shape[2]==num_user-1
    if len(_video_db_oth.shape)==4:
        _video_db_oth_span = np.zeros((length,2*_video_db_oth.shape[1],_video_db_oth.shape[2],_video_db_oth.shape[3]))
    elif len(_video_db_oth.shape)==5:
        _video_db_oth_span = np.zeros((length,2*_video_db_oth.shape[1],_video_db_oth.shape[2],_video_db_oth.shape[3],_video_db_oth.shape[4]))

    for ii in range(length-1):
        temp = np.concatenate((_video_db_oth[ii,:],_video_db_oth[ii+1,:]))
        _video_db_oth_span[ii] = temp
    #the last row is all zero!
    return _video_db_oth_span

def delete_across_video_span(_video_db_oth_span):
    """TODO: delete chunks that spans across two videos"""
    assert _video_db_oth_span.shape[1]==11040*2 #only for data loaded from old cached 
    video_chunk_length_list_accu =  [672, 1536, 2880, 3600, 4464, 7440, 9168, 9744, 11040]#stride=10
    for ind in video_chunk_length_list_accu:
        pdb.set_trace()
        _video_db_oth_span[ind,:] = 0
    return _video_db_oth_span




def _split_out_test_set(_video_db_tar,_video_db_future_tar,_video_db_future_input_tar,
                        _video_db_oth,_video_db_future_oth):

    """in order to test on a specific video instead of the last few,
    this func will split out the testing chunks and shift the test set into the end
    """
    assert _video_db_tar.shape[0]==11040 #only for data loaded from old cached 
    # def _get_testing_index():
    #     # for data that are ordered firstly using video index,
    #     # i.e.: one video (contains all users) followeed by another video
    #     # video_length_list = [4762,5834,8798,5005,5966,18997,11279,4121,8471]
    #     video_chunk_length_list = [672,864,1344,720,864,2976,1728,576,1296]#stride=10
    #     video_chunk_length_list_accu = []
    #     prev = 0
    #     for ii in range(len(video_chunk_length_list)):
    #         prev += video_chunk_length_list[ii]
    #         video_chunk_length_list_accu.append(prev)
        
    #     testing_index = (video_chunk_length_list_accu[cfg.test_video_ind-1],video_chunk_length_list_accu[cfg.test_video_ind])
    #     num_testing_sample = testing_index[1]-testing_index[0]
    #     assert num_testing_sample==video_chunk_length_list[cfg.test_video_ind]
    #     return testing_index,num_testing_sample

    # def shift_test_into_the_end(_video_db_tar,testing_index):
    #     # for data that are ordered firstly using video index,
    #     _video_db_tar_train = np.concatenate([_video_db_tar[:testing_index[0],:,:],_video_db_tar[testing_index[1]:,:,:]],axis=0)
    #     _video_db_tar_test = _video_db_tar[testing_index[0]:testing_index[1],:,:]
    #     _video_db_tar_new = np.concatenate([_video_db_tar_train,_video_db_tar_test],axis=0)
    #     return _video_db_tar_new


    def _get_testing_index_tar():
        #for data that's ordered firstly by target_user index, eg. _video_db_tar  
        # i.e.: one user rolls over all videos first, another user rolls etc
        video_chunk_length_list = np.array([672,864,1344,720,864,2976,1728,576,1296])/num_user #stride=10
        video_chunk_length_list_accu = []
        prev = 0
        for ii in range(len(video_chunk_length_list)):
            prev += video_chunk_length_list[ii]
            video_chunk_length_list_accu.append(prev)
        
        testing_index_each_user = (video_chunk_length_list_accu[cfg.test_video_ind-1],video_chunk_length_list_accu[cfg.test_video_ind])
        num_testing_sample = num_user*(testing_index_each_user[1]-testing_index_each_user[0])
        return video_chunk_length_list_accu,testing_index_each_user,num_testing_sample

    def shift_test_into_the_end_tar(_example_db,testing_index):
        #for data that's ordered firstly by target_user index, eg. _video_db_tar, _video_db_oth
        assert _example_db.shape[0]==11040
        newtrain = np.zeros_like(_example_db)[0,:][np.newaxis,:]
        newtest = np.zeros_like(_example_db)[0,:][np.newaxis,:]
        gap = int(video_chunk_length_list_accu[-1])
        for _target_user in range(num_user):
            start = int(_target_user*gap+testing_index[0])
            end = int(_target_user*gap+testing_index[1])
            train = np.concatenate([_example_db[_target_user*gap:start,:],_example_db[end:(_target_user+1)*gap,:]],axis=0)
            test = _example_db[start:end,:]
            newtrain = np.concatenate([newtrain,train])
            newtest = np.concatenate([newtest,test])

        _example_db_new = np.concatenate([newtrain[1:,:],newtest[1:,:]],axis=0)
        return _example_db_new

    video_chunk_length_list_accu,testing_index,num_testing_sample = _get_testing_index_tar()
    #target user
    _video_db_tar_new = shift_test_into_the_end_tar(_video_db_tar,testing_index)
    _video_db_future_tar_new = shift_test_into_the_end_tar(_video_db_future_tar,testing_index)
    _video_db_future_input_tar_new = shift_test_into_the_end_tar(_video_db_future_input_tar,testing_index)

    #others
    _video_db_oth_new = shift_test_into_the_end_tar(_video_db_oth,testing_index)
    _video_db_future_oth_new = shift_test_into_the_end_tar(_video_db_future_oth,testing_index)

    return _video_db_tar_new,_video_db_future_tar_new,_video_db_future_input_tar_new,\
            _video_db_oth_new,_video_db_future_oth_new,\
            num_testing_sample


print('other data shape: ',_video_db_oth.shape)
print('other data shape: ',_video_db_future_oth.shape)
print('target user data shape: ',_video_db_tar.shape)
print('target user data shape: ',_video_db_future_tar.shape)
print('Preprocessing....')
_video_db_oth = _reshape_others_data(_video_db_oth)
_video_db_future_oth = _reshape_others_data(_video_db_future_oth)
# _video_db_future_input_oth = _reshape_others_data(_video_db_future_input_oth)
# _video_db_oth = _reshape_others_data2(_video_db_oth)
# _video_db_future_oth = _reshape_others_data2(_video_db_future_oth)

_video_db_tar = _video_db_tar.reshape((_video_db_tar.shape[0],_video_db_tar.shape[1],fps,3))
_video_db_future_tar = _video_db_future_tar.reshape((_video_db_tar.shape[0],_video_db_tar.shape[1],fps,3))
_video_db_future_input_tar = _video_db_future_input_tar.reshape((_video_db_tar.shape[0],_video_db_tar.shape[1],fps,3))

total_num_samples = _video_db_tar.shape[0]
# num_testing_sample = int(0.15*total_num_samples)#use last few as test
_video_db_tar,_video_db_future_tar,_video_db_future_input_tar,\
                            _video_db_oth,_video_db_future_oth,\
                            num_testing_sample = _split_out_test_set(
                                                _video_db_tar,_video_db_future_tar,_video_db_future_input_tar,
                                                _video_db_oth,_video_db_future_oth)


print('other data shape: ',_video_db_oth.shape)
print('other data shape: ',_video_db_future_oth.shape)
print('target user data shape: ',_video_db_tar.shape)
print('target user data shape: ',_video_db_future_tar.shape)


if cfg.shuffle_data:
    #### shuffle the whole dataset
    # index_shuf = util.get_shuffle_index(total_num_samples)
    index_shuf = pickle.load(open('index_shuf'+'_exp'+str(experiment)+'.p','rb'))
    print('Shuffle data before training and testing.')
    _video_db_tar = util.shuffle_data(index_shuf,_video_db_tar)
    _video_db_future_tar = util.shuffle_data(index_shuf,_video_db_future_tar)
    _video_db_future_input_tar = util.shuffle_data(index_shuf,_video_db_future_input_tar)

    _video_db_oth = util.shuffle_data(index_shuf,_video_db_oth)
    _video_db_future_oth = util.shuffle_data(index_shuf,_video_db_future_oth)
    # _video_db_future_input_oth = util.shuffle_data(index_shuf,_video_db_future_input_oth)


#### prepare training data
# data_sanity_check(_video_db_tar,_video_db_future_tar,_video_db_future_input_tar)
_video_db_oth_span = get_whole_span(_video_db_oth)
#change shape from (11040, 20, 47, 30, 3) to (11040, 20, 1, 30, 141)
_video_db_oth_span = _video_db_oth_span.transpose(0,1,3,2,4)
_video_db_oth_span = _video_db_oth_span.reshape(_video_db_oth_span.shape[0],_video_db_oth_span.shape[1],_video_db_oth_span.shape[2],-1)
_video_db_oth_span = _video_db_oth_span[:,:,np.newaxis,:,:]
if cfg.input_mean_var:
    ### target user
    encoder_input_data = util.get_gt_target_xyz(_video_db_tar[:-num_testing_sample,:,:])
    decoder_target_data = util.get_gt_target_xyz(_video_db_future_tar[:-num_testing_sample,:,:])
    ### other users
    others_pst_input_data = util.get_gt_target_xyz_oth(_video_db_oth_span)[:-num_testing_sample][:,:,np.newaxis,:]    
    if not cfg.teacher_forcing:
        decoder_input_data = encoder_input_data[:,-1,:][:,np.newaxis,:]
    else:
        decoder_input_data = util.get_gt_target_xyz(_video_db_future_input_tar[:-num_testing_sample,:,:])

else:
    ### target user
    encoder_input_data = _video_db_tar[:-num_testing_sample,:,:][:,:,np.newaxis,:,:]
    decoder_target_data = _video_db_future_tar[:-num_testing_sample,:,:][:,:,np.newaxis,:,:]
    decoder_input_data = _video_db_tar[:-num_testing_sample,-1,:][:,np.newaxis,np.newaxis,:]
    ### other users
    others_pst_input_data = _video_db_oth_span[:-num_testing_sample]




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

encoder_input_data_next = _get_next_timestetp_data(encoder_input_data)
decoder_target_data_oth = _get_next_timestetp_data_span(others_pst_input_data)

data_sanity_check(encoder_input_data,decoder_target_data,decoder_input_data)

## ensure dividable by batch size
sample_ind = util.rand_sample_ind(total_num_samples,num_testing_sample,batch_size)
encoder_input_data = util.rand_sample(encoder_input_data,sample_ind)
decoder_input_data = util.rand_sample(decoder_input_data,sample_ind)
decoder_target_data = util.rand_sample(decoder_target_data,sample_ind)
others_pst_input_data = util.rand_sample(others_pst_input_data,sample_ind)
encoder_input_data_next = util.rand_sample(encoder_input_data_next,sample_ind)
decoder_target_data_oth = util.rand_sample(decoder_target_data_oth,sample_ind)


### ====================Training====================
# model = load_model('convLSTM_endec_11_256tanh_epoch12-1.2859.h5')
# model = load_model('convLSTM_wholespan_targetrecons_trj_decodernotanh_epoch10-0.1658.h5')
# tag = 'convLSTM_wholespan_targetrecons_trj_decodernotanh_epoch'
# tag = '3_3layerconvLSTM_wholespan_latent32_8_pred_err_concat_input_epoch'
# tag = '3_3layerconvLSTM_wholespan_concat_input_meanvar_epoch'
# tag = 'convLSTM_wholespan_fclstm_meanvarinput_TFor_epoch'
# tag = '2recons1predloss_raw_epoch'
tag = 'fc_seq2seq_attention+convlstm_others_aug18'

model_checkpoint = ModelCheckpoint(tag+'{epoch:02d}-{val_loss:.4f}.h5', monitor='val_loss', save_best_only=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                 patience=3, min_lr=1e-6)
stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=0, mode='auto')
model.fit([encoder_input_data, others_pst_input_data, decoder_input_data],
            [decoder_target_data,decoder_target_data_oth,encoder_input_data_next],
          batch_size=batch_size,
          epochs=epochs,
          validation_split=0.2,
          shuffle=cfg.shuffle_data, 
          initial_epoch=0,
          callbacks=[model_checkpoint, reduce_lr, stopping])




### ====================Testing====================
_video_db_tar = pickle.load(open('./cache/'+dataformat+'/test/_video_db_tar.p','rb'))
_video_db_future_tar = pickle.load(open('./cache/'+dataformat+'/test/_video_db_future_tar.p','rb'))
_video_db_future_input_tar = pickle.load(open('./cache/'+dataformat+'/test/_video_db_future_input_tar.p','rb'))
_video_db_oth = pickle.load(open('./cache/'+dataformat+'/test/_video_db_oth.p','rb'))
_video_db_future_oth = pickle.load(open('./cache/'+dataformat+'/test/_video_db_future_oth.p','rb'))
_video_db_future_input_oth = pickle.load(open('./cache/'+dataformat+'/test/_video_db_future_input_oth.p','rb'))

_video_db_tar = _video_db_tar.reshape((_video_db_tar.shape[0],_video_db_tar.shape[1],fps,3))
_video_db_future_tar = _video_db_future_tar.reshape((_video_db_tar.shape[0],_video_db_tar.shape[1],fps,3))
_video_db_future_input_tar = _video_db_future_input_tar.reshape((_video_db_tar.shape[0],_video_db_tar.shape[1],fps,3))
_video_db_oth = _reshape_others_data(_video_db_oth)
_video_db_future_oth = _reshape_others_data(_video_db_future_oth)
print('other data shape: ',_video_db_oth.shape)
print('other data shape: ',_video_db_future_oth.shape)
print('target user data shape: ',_video_db_tar.shape)
print('target user data shape: ',_video_db_future_tar.shape)


# model = load_model('convLSTM_endec_105_epoch55-0.1842.h5')
# model = load_model('convLSTM_endec_101_epoch13-0.1056.h5')
# model = load_model('backup_convLSTM_endec_11_epoch99-0.1952.h5')
# model = load_model('backup_convLSTM_endec_105_halftanh_epoch50-1.0356.h5')
# model = load_model('convLSTM_endec_101_halftanh_epoch108-1.0330.h5')
# model = load_model('convLSTM_endec_11_256tanh_epoch71-1.2156.h5')
# model = load_model('convLSTM_wholespan_11_epoch43-0.9892.h5')
# model.load_weights('3_3layerconvLSTM_wholespan_concat_input_meanvar_epoch42-0.2414.h5')

# model= load_model('convLSTM_wholespan_fclstm_meanvarinput_TFor_epoch25-0.0522.h5')
# model= load_model('2recons1predloss_raw_epoch23-0.3633.h5')

if cfg.teacher_forcing:
    # Define sampling models
    encoder_model = Model(encoder_inputs, [encoder_outputs]+encoder_states)

    others_model = Model(encoder_inputs_oth,outputs_sqns_oth)


    decoder_state_input_h = Input(shape=(latent_dim_target_fclstm,))
    decoder_state_input_c = Input(shape=(latent_dim_target_fclstm,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    decoder_outputs, state_h, state_c = decoder_lstm(
        decoder_inputs, initial_state=decoder_states_inputs)
    decoder_states = [state_h, state_c]


    outputs_sqns_oth_slice_input = Input(shape=(1,1,num_user-1,56))
    convlstm_state = flatten_layer(outputs_sqns_oth_slice_input)
    convlstm_state = flatten_conv_lstm_state_dense(convlstm_state)
    concat_state = Concatenatelayer([get_dim1_layer(decoder_outputs),convlstm_state])
    outputs = decoder_dense(concat_state)
    outputs = expand_dim_layer(outputs)

    decoder_model = Model([decoder_inputs, decoder_state_input_h, decoder_state_input_c, outputs_sqns_oth_slice_input],
        [outputs] + decoder_states)


def decode_sequence_fov_TF(input_seq,others_pst_input_seq):
    if input_seq.shape[0]>1:
        last_location = input_seq[:,-1,:][:,np.newaxis,:]
    elif input_seq.shape[0]==1: 
        last_location = input_seq[0,-1,:][np.newaxis,np.newaxis,:]

    # Encode the input as state vectors.
    encoder_output_val,states_value_h,states_value_c = encoder_model.predict(input_seq)

    # Encode the input as state vectors (others)
    outputs_sqns_oth_val = others_model.predict(others_pst_input_seq)

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    target_seq = last_location
    decoded_sentence = []
    for ii in range(cfg.predict_step):
        ## get state from others branch
        outputs_sqns_oth_slice_input_val = outputs_sqns_oth_val[:,ii+max_encoder_seq_length,:][:,:,np.newaxis,:]
        output_tokens, h, c = decoder_model.predict(
            [target_seq, states_value_h, states_value_c, outputs_sqns_oth_slice_input_val])

        decoded_sentence+=[output_tokens]
        # Update the target sequence (of length 1).
        target_seq = output_tokens
        # Update states
        states_value_h, states_value_c = [h, c]

    return decoded_sentence



def decode_sequence_fov(input_seq,others_pst_input_seq):            
    # Encode the input as state vectors.
    if input_seq.shape[0]>1:
        last_location = input_seq[:,-1,:][:,np.newaxis,:]
    elif input_seq.shape[0]==1: 
        last_location = input_seq[0,-1,:][np.newaxis,np.newaxis,:]
    # last_mu_var = util.get_gt_target_xyz(last_location)
    [decoded_sentence,decoded_sentence_oth,encoder_reconstruct_sentence_tar] = model.predict([input_seq,others_pst_input_seq,last_location])

    return decoded_sentence,decoded_sentence_oth,encoder_reconstruct_sentence_tar

gt_sentence_list = []
decoded_sentence_list = []
gt_sentence_oth_list = []
decoded_sentence_oth_list = []
encoder_reconstruct_tar_list = []
gt_sentence_recons_tar_list = []


_video_db_tar_next = _get_next_timestetp_data(_video_db_tar[total_num_samples-num_testing_sample:,:])
_video_db_oth_span_next = _get_next_timestetp_data_span(_video_db_oth_span[total_num_samples-num_testing_sample:,:])

# test_batch_size = 1
test_batch_size = batch_size

# for seq_index in range(total_num_samples-num_testing_sample,total_num_samples-1):
# for seq_index in range(total_num_samples-num_testing_sample,total_num_samples-num_testing_sample+100):
for seq_index in range(total_num_samples-num_testing_sample,total_num_samples-num_testing_sample+320,test_batch_size):
    if cfg.input_mean_var:
        input_seq = util.get_gt_target_xyz(_video_db_tar[seq_index: seq_index + test_batch_size,:,:])
        others_pst_input_seq = util.get_gt_target_xyz_oth(_video_db_oth_span[seq_index: seq_index + test_batch_size,:])[:,:,np.newaxis,:]
    else:
        input_seq = _video_db_tar[seq_index: seq_index + test_batch_size,:,:][:,:,np.newaxis,:,:]
        others_pst_input_seq = _video_db_oth_span[seq_index: seq_index + test_batch_size,:]#[:,:,np.newaxis,:]
    
    input_seq_next = _video_db_tar_next[seq_index-(total_num_samples-num_testing_sample): seq_index-(total_num_samples-num_testing_sample) + test_batch_size,:,:][:,:,np.newaxis,:,:]
    others_pst_input_seq_next = _video_db_oth_span_next[seq_index-(total_num_samples-num_testing_sample): seq_index-(total_num_samples-num_testing_sample)+ test_batch_size,:][:,:,np.newaxis,:]

    if cfg.teacher_forcing:
        decoded_sentence = decode_sequence_fov_TF(input_seq,others_pst_input_seq)
    else:
        decoded_sentence,decoded_sentence_oth,encoder_reconstruct_sentence_tar = decode_sequence_fov(input_seq,others_pst_input_seq)

    decoded_sentence_list+=[decoded_sentence]
    encoder_reconstruct_tar_list+=[encoder_reconstruct_sentence_tar]
    decoded_sentence_oth_list+=[decoded_sentence_oth]

    gt_sentence = _video_db_future_tar[seq_index: seq_index + test_batch_size,:,:]
    gt_sentence_list+=[gt_sentence]
    
    # gt_sentence_recons_tar = input_seq #reconstruction
    gt_sentence_recons_tar = input_seq_next #prediction
    gt_sentence_recons_tar_list+=[gt_sentence_recons_tar]
    # gt_sentence_oth = others_pst_input_seq #reconstruction
    gt_sentence_oth = others_pst_input_seq_next #prediction
    gt_sentence_oth_list+=[gt_sentence_oth]


    # print('-')
    # decoder_target = util.get_gt_target_xyz(gt_sentence)
    # print('Decoded sentence - decoder_target:', np.squeeze(np.array(decoded_sentence))[:,:3]-np.squeeze(decoder_target)[:,:3])

pickle.dump(decoded_sentence_list,open('decoded_sentence.p','wb'))
pickle.dump(gt_sentence_list,open('gt_sentence_list.p','wb'))

# pickle.dump(encoder_reconstruct_tar_list,open('decoded_sentence.p','wb'))
# pickle.dump(gt_sentence_recons_tar_list,open('gt_sentence_list.p','wb'))

# pickle.dump(decoded_sentence_oth_list,open('decoded_sentence.p','wb'))
# pickle.dump(gt_sentence_oth_list,open('gt_sentence_list.p','wb'))
# print('Testing finished!')











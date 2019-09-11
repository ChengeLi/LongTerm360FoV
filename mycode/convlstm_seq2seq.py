"""
seq2seq: both encoder and decoder used convLSTM
"""
from __future__ import print_function
from keras.models import Model
from keras.layers import Input, LSTM, Dense, Add, Softmax
from keras.layers import Lambda,Concatenate,Flatten,ConvLSTM2D
from keras.layers import Permute,Conv2D
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
from mycode.utility import reshape2second_stacks,get_data,_create_one_hot
from mycode.utility import get_shuffle_index,shuffle_data,get_gt_target_xyz,get_gt_target_xyz_oth
from mycode.utility import slice_layer,rand_sample_ind,rand_sample
from random import shuffle
import matplotlib.pyplot as plt
import _pickle as pickle
import numpy as np
import pdb
from keras.layers import Conv1D
from keras import optimizers
from mycode.utility import generate_fake_batch_numpy

# experiment = 1 
batch_size = 32
epochs = 200
latent_dim = 16

fps = 30
num_encoder_tokens = 3*fps
num_decoder_tokens = 6
max_encoder_seq_length = cfg.running_length
max_decoder_seq_length = cfg.predict_step

if cfg.use_one_hot:
    channel_num = 648#one-hot
else:
    channel_num = 3#xyz


from keras.layers import BatchNormalization
# bnlayer = BatchNormalization(axis=-1,center=True, scale=True)

def generate_fake_batch(x):
    """generate new data for 1 second using predicted mean and variance"""
    # batch_size = 64
    # fps =30
    mu = x[0]
    var = x[1]
    temp = K.random_normal(shape = (batch_size,fps,1), mean=mu,stddev=var)
    return temp

generate_fake_batch_layer = Lambda(lambda x: generate_fake_batch(x))


## utility layers
expand_dim_layer = Lambda(lambda x: K.expand_dims(x,1))
get_dim_layer = Lambda(lambda x: x[:,0,0,:,:])
get_dim_layer1 = Lambda(lambda x: x[:,0,:,:,:])
flatten_layer = Flatten()
# get_dim1_layer = Lambda(lambda x: x[:,0,:])
Concatenatelayer = Concatenate(axis=2)
Concatenatelayer1 = Concatenate(axis=-1)


### ====================Graph def====================
if cfg.use_one_hot:
    input_shape1 = (1,36,18,fps)
    input_shape2 = (1,36,18,latent_dim*2)
    input_shape3 = (1,36,18,latent_dim)
else:
    if cfg.input_mean_var:
        input_shape1 = (1,1,1,num_decoder_tokens)
        input_shape2 = (1,1,1,latent_dim*2)
        input_shape3 = (1,1,1,latent_dim)
    else:
        input_shape1 = (1,1,fps,channel_num)
        input_shape2 = (1,1,fps,latent_dim*2)
        input_shape3 = (1,1,fps,latent_dim)
###======convLSTM on target past encoder======
kernel_size = cfg.conv_kernel_size
if cfg.stateful_across_batch:
    if cfg.use_one_hot:     ### spatial one-hot matrix
        encoder_inputs = Input(batch_shape=(batch_size, max_encoder_seq_length,36,18,fps))    
    else:
        encoder_inputs = Input(batch_shape=(batch_size, max_encoder_seq_length, 1,fps,channel_num))
else:
    if cfg.input_mean_var:
        encoder_inputs = Input(shape=(max_encoder_seq_length,1,1,num_decoder_tokens))
    else:
        encoder_inputs = Input(shape=(max_encoder_seq_length,1,fps,channel_num))

convlstm_encoder = ConvLSTM2D(filters=latent_dim*2, kernel_size=(kernel_size, kernel_size),
                   input_shape=input_shape1,
                   dilation_rate=cfg.dilation_rate,
                   dropout=cfg.dropout_rate, recurrent_dropout=0.0,
                   stateful=cfg.stateful_across_batch,
                   padding='same', return_sequences=True, return_state=True)
pst_outputs_sqns, pst_state_h0, pst_state_c0 = convlstm_encoder(encoder_inputs)
states0 = [pst_state_h0, pst_state_c0]
convlstm_encoder1 = ConvLSTM2D(filters=latent_dim, kernel_size=(kernel_size, kernel_size),
                   input_shape=input_shape2,
                   dilation_rate=cfg.dilation_rate,
                   dropout=cfg.dropout_rate, recurrent_dropout=0.0,
                   stateful=cfg.stateful_across_batch,
                   padding='same', return_sequences=True, return_state=True)
# pst_outputs_sqns, pst_state_h1, pst_state_c1 = convlstm_encoder1(Concatenatelayer1([encoder_inputs,pst_outputs_sqns]))
pst_outputs_sqns, pst_state_h1, pst_state_c1 = convlstm_encoder1(pst_outputs_sqns)
states1 = [pst_state_h1, pst_state_c1]

convlstm_encoder2 = ConvLSTM2D(filters=latent_dim/2, kernel_size=(kernel_size, kernel_size),
                   input_shape=input_shape3,
                   dilation_rate=cfg.dilation_rate,
                   dropout=cfg.dropout_rate, recurrent_dropout=0.0,
                    stateful=cfg.stateful_across_batch,
                    padding='same', return_sequences=True, return_state=True)
# pst_outputs_sqns, pst_state_h2, pst_state_c2 = convlstm_encoder2(Concatenatelayer1([encoder_inputs,pst_outputs_sqns]))
pst_outputs_sqns, pst_state_h2, pst_state_c2 = convlstm_encoder2(pst_outputs_sqns)
states2 = [pst_state_h2, pst_state_c2]



###======convLSTM on target future decoder======
if cfg.stateful_across_batch:
    if not cfg.input_mean_var:
        if cfg.use_one_hot:     ### spatial one-hot matrix
            decoder_inputs = Input(batch_shape=(batch_size,1,36,18,fps))
        else:
            decoder_inputs = Input(batch_shape=(batch_size,1,1,fps,channel_num))            
    else:
        decoder_inputs = Input(batch_shape=(batch_size,1,num_decoder_tokens))            
else:
    if not cfg.input_mean_var:
        decoder_inputs = Input(shape=(1,1,fps,channel_num))
    else:
        decoder_inputs = Input(shape=(1,1,1,num_decoder_tokens))    


convlstm_decoder = ConvLSTM2D(filters=latent_dim*2, kernel_size=(kernel_size, kernel_size),
                   input_shape=input_shape1,
                   dilation_rate=cfg.dilation_rate,
                   dropout=cfg.dropout_rate, recurrent_dropout=0.0,
                   stateful=cfg.stateful_across_batch,
                   padding='same', return_sequences=True, return_state=True)

convlstm_decoder1 = ConvLSTM2D(filters=latent_dim, kernel_size=(kernel_size, kernel_size),
                   input_shape=input_shape2,
                   dilation_rate=cfg.dilation_rate,
                   dropout=cfg.dropout_rate, recurrent_dropout=0.0,
                   stateful=cfg.stateful_across_batch,
                   padding='same', return_sequences=True, return_state=True)

convlstm_decoder2 = ConvLSTM2D(filters=latent_dim/2, kernel_size=(kernel_size, kernel_size),
                   input_shape=input_shape3,
                   dilation_rate=cfg.dilation_rate,
                   dropout=cfg.dropout_rate, recurrent_dropout=0.0,
                   stateful=cfg.stateful_across_batch,
                   padding='same', return_sequences=True, return_state=True)




if cfg.predict_mean_var:
    pred_conv_lstm_dense = Dense(6,activation=None)
    # pred_conv_lstm_dense_mu = Dense(3,activation='tanh')
    # pred_conv_lstm_dense_var = Dense(3,activation='relu')
##----------- 2D conv
if cfg.use_one_hot:
    pred_conv_lstm_conv = Conv2D(filters=512, kernel_size=(kernel_size,kernel_size), padding='same',
        activation='relu', use_bias=True, kernel_initializer='glorot_uniform')
    pred_conv_lstm_conv1 = Conv2D(filters=1024, kernel_size=(kernel_size,kernel_size), padding='same',
        activation='relu', use_bias=True, kernel_initializer='glorot_uniform')
    pred_conv_lstm_conv2 = Conv2D(filters=fps, kernel_size=(kernel_size,kernel_size), padding='same',
        activation='relu', use_bias=True, kernel_initializer='glorot_uniform')
else:
    ###----------- only temporal 1D conv
    pred_conv_lstm_conv = Conv1D(filters=512, kernel_size=7, padding='same',
        activation='relu', use_bias=True, kernel_initializer='glorot_uniform')
    pred_conv_lstm_conv1 = Conv1D(filters=1024, kernel_size=7, padding='same',
        activation='relu', use_bias=True, kernel_initializer='glorot_uniform')
    pred_conv_lstm_conv2 = Conv1D(filters=3, kernel_size=7, padding='same',
        activation='softmax', use_bias=True, kernel_initializer='glorot_uniform')
    # pred_conv_lstm_conv3 = Conv1D(filters=64, kernel_size=3, padding='same',
    #     activation='relu', use_bias=True, kernel_initializer='glorot_uniform')
    # pred_conv_lstm_conv4 = Conv1D(filters=32, kernel_size=3, padding='same',
    #     activation='relu', use_bias=True, kernel_initializer='glorot_uniform')
    # pred_conv_lstm_conv5 = Conv1D(filters=16, kernel_size=3, padding='same',
    #     activation='relu', use_bias=True, kernel_initializer='glorot_uniform')
    # pred_conv_lstm_conv6 = Conv1D(filters=channel_num, kernel_size=3, padding='same',
    #     activation='tanh', use_bias=True, kernel_initializer='glorot_uniform')
    # squeeze_for_residual = Dense(3,activation=None)


bnlayer0 = BatchNormalization(axis=-1,center=True, scale=True)
bnlayer1 = BatchNormalization(axis=-1,center=True, scale=True)
bnlayer2 = BatchNormalization(axis=-1,center=True, scale=True)
# bnlayer3 = BatchNormalization(axis=-1,center=True, scale=True)
# bnlayer4 = BatchNormalization(axis=-1,center=True, scale=True)
# bnlayer5 = BatchNormalization(axis=-1,center=True, scale=True)


all_outputs= []
inputs = decoder_inputs
for time_ind in range(max_decoder_seq_length):
    # multi-layer decoder
    fut_outputs_sqns0, fut_state_h, fut_state_c = convlstm_decoder([inputs]+states0)
    states0 = [fut_state_h, fut_state_c]
    fut_outputs_sqns1, fut_state_h, fut_state_c = convlstm_decoder1([fut_outputs_sqns0]+states1)
    states1 = [fut_state_h, fut_state_c]
    fut_outputs_sqns2, fut_state_h, fut_state_c = convlstm_decoder2([fut_outputs_sqns1]+states2)
    states2 = [fut_state_h, fut_state_c]

    fut_outputs_sqns = Concatenatelayer1([fut_outputs_sqns0,fut_outputs_sqns1,fut_outputs_sqns2])

    # fut_outputs_sqns = bnlayer(fut_outputs_sqns)
    ### predict others' future
    if cfg.predict_mean_var:
        fut_outputs_sqns = flatten_layer(get_dim_layer(fut_outputs_sqns))
        outputs = pred_conv_lstm_dense(fut_outputs_sqns)
        outputs = expand_dim_layer(outputs)
    else:
        ## use conv layer to predict
        if cfg.use_one_hot:
            outputs = pred_conv_lstm_conv(get_dim_layer1(fut_outputs_sqns))
            # outputs = bnlayer0(outputs)
            outputs = pred_conv_lstm_conv1(outputs)
            # outputs = bnlayer1(outputs)
            outputs = pred_conv_lstm_conv2(outputs)
            #channel-direction softmax
            outputs = Softmax(axis=-1)(outputs)
            outputs = expand_dim_layer(outputs)

        else:
            outputs = pred_conv_lstm_conv(get_dim_layer(fut_outputs_sqns))
            # outputs = bnlayer0(outputs)
            outputs = pred_conv_lstm_conv1(outputs)
            # outputs = bnlayer1(outputs)
            outputs = pred_conv_lstm_conv2(outputs)
            # outputs = bnlayer2(outputs)
            # outputs = pred_conv_lstm_conv3(outputs)
            # outputs = bnlayer3(outputs)
            # outputs = pred_conv_lstm_conv4(outputs)
            # outputs = bnlayer4(outputs)
            # outputs = pred_conv_lstm_conv5(outputs)
            # outputs = bnlayer5(outputs)
            # outputs = pred_conv_lstm_conv6(outputs)
            # residual
            # outputs = Add()([outputs,squeeze_for_residual(get_dim_layer(fut_outputs_sqns))])
            # outputs = Add()([outputs,get_dim_layer(inputs)])
            outputs = expand_dim_layer(outputs)
            outputs = expand_dim_layer(outputs)
    if cfg.predict_mean_var and cfg.sample_and_refeed:
        #for training            
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
        if cfg.input_mean_var:
            inputs = expand_dim_layer(expand_dim_layer(outputs))
        else:
            inputs = outputs

    all_outputs.append(outputs)

# Concatenate all predictions
decoder_outputs = Lambda(lambda x: K.concatenate(x, axis=1))(all_outputs)

model = Model([encoder_inputs, decoder_inputs],decoder_outputs)
# RMSprop = optimizers.RMSprop(lr=0.01,clipnorm=3)
# sgd = optimizers.sgd(lr=0.0001,clipnorm=1)
model.compile(optimizer='RMSprop', loss=costfunc._mse)
# model.compile(optimizer='RMSprop', loss='mean_squared_error')
# model.compile(optimizer='RMSprop', loss=costfunc.likelihood_loss)
# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# weights=np.ones(30)
# model.compile(loss=costfunc.weighted_categorical_crossentropy(weights), optimizer='adam', metrics=['accuracy'])



#### ====================data====================
def _reshape_others_data(_video_db):
    _video_db = _video_db.reshape((_video_db.shape[0],_video_db.shape[1],fps,3))
    return _video_db

if not cfg.use_one_hot:
    ##### data format 1
    # _video_db,_video_db_future,_video_db_future_input = get_data(datadb,pick_user=False)
    ##### data format 2
    # _video_db_tar = pickle.load(open('./cache/format2/_video_db_tar_exp'+str(experiment)+'.p','rb'))
    # _video_db_future_tar = pickle.load(open('./cache/format2/_video_db_future_tar_exp'+str(experiment)+'.p','rb'))
    # _video_db_future_input_tar = pickle.load(open('./cache/format2/_video_db_future_input_tar_exp'+str(experiment)+'.p','rb'))
    # _video_db = _video_db_tar
    # _video_db_future = _video_db_future_tar
    # _video_db_future_input = _video_db_future_input_tar
    ##### data format 3or4--3, ie Tsinghua train/test split(on users)
    # video_data_train = pickle.load(open('./360video/data/tsinghua_train_video_data.p','rb'))
    ##### data format 3or4--4, ie Shanghaitech train/test split(on videos)
    # video_data_train = pickle.load(open('./360video/data/shanghai_dataset_xyz_train.p','rb'))    
    #### data format 5
    video_data_train = pickle.load(open('./360video/temp/tsinghua_after_bysec_interpolation/tsinghua_train_video_data_over_video.p','rb'))    

    video_data_train = clip_xyz(video_data_train)
    datadb = video_data_train.copy()
    _video_db,_video_db_future,_video_db_future_input = get_data(datadb,pick_user=False)


    print('data loading finished...')
    #reshape to match input dimension
    _video_db = _reshape_others_data(_video_db)[:,:,np.newaxis,:,:]
    _video_db_future_input = _reshape_others_data(_video_db_future_input)
    _video_db_future = _reshape_others_data(_video_db_future)

    total_num_samples = _video_db.shape[0]
    # num_testing_sample = int(0.15*total_num_samples)#use last few as test
    num_testing_sample =1 #already pure train, don't have to save for test

    if cfg.shuffle_data:
        #shuffle the whole dataset
        index_shuf = pickle.load(open('index_shuf'+'_exp'+str(experiment)+'.p','rb'))
        _video_db = shuffle_data(index_shuf,_video_db)
        _video_db_future = shuffle_data(index_shuf,_video_db_future)
        _video_db_future_input = shuffle_data(index_shuf,_video_db_future_input)

    #prepare training data
    if cfg.input_mean_var:
        encoder_input_data = get_gt_target_xyz(_video_db[:-num_testing_sample,:,:].squeeze())[:,:,np.newaxis,np.newaxis,:]
        decoder_input_data = get_gt_target_xyz(_video_db_future_input[:-num_testing_sample,:])[:,0,:][:,np.newaxis,np.newaxis,np.newaxis,:]    
    else:
        encoder_input_data = _video_db[:-num_testing_sample,:,:]
        decoder_input_data = _video_db_future_input[:-num_testing_sample,0,:][:,np.newaxis,np.newaxis,:]
    
    if cfg.predict_mean_var:
        decoder_target_data = get_gt_target_xyz(_video_db_future)[:-num_testing_sample,:,:]
    else:
        decoder_target_data = _video_db_future[:-num_testing_sample,:,:][:,:,np.newaxis,:,:]

else:
    data_format=2
    name = '_video_db_tar'
    theta_index = pickle.load(open('./cache/format'+str(data_format)+'/'+name+'_theta_index_exp'+str(experiment)+'.p','rb'))    
    phi_index = pickle.load(open('./cache/format'+str(data_format)+'/'+name+'_phi_index_exp'+str(experiment)+'.p','rb'))    
    one_hot = _create_one_hot(theta_index,phi_index,vector=False)

    name = '_video_db_future_tar'
    theta_index_future = pickle.load(open('./cache/format'+str(data_format)+'/'+name+'_theta_index_exp'+str(experiment)+'.p','rb'))    
    phi_index_future = pickle.load(open('./cache/format'+str(data_format)+'/'+name+'_phi_index_exp'+str(experiment)+'.p','rb'))    
    one_hot_future = _create_one_hot(theta_index_future,phi_index_future,vector=False)

    name = '_video_db_future_input_tar'
    theta_index_future_input = pickle.load(open('./cache/format'+str(data_format)+'/'+name+'_theta_index_exp'+str(experiment)+'.p','rb'))    
    phi_index_future_input = pickle.load(open('./cache/format'+str(data_format)+'/'+name+'_phi_index_exp'+str(experiment)+'.p','rb'))    
    one_hot_future_input = _create_one_hot(theta_index_future_input,phi_index_future_input,vector=False)

    total_num_samples = one_hot.shape[0]
    num_testing_sample = int(0.15*total_num_samples)#use last few as test

    #prepare training data
    encoder_input_data = one_hot[:-num_testing_sample,:,:].transpose(0,1,3,4,2)
    decoder_input_data = one_hot_future_input[:-num_testing_sample,0,:][:,np.newaxis,:,:].transpose(0,1,3,4,2)
    decoder_target_data = one_hot_future[:-num_testing_sample,:,:].transpose(0,1,3,4,2)



if cfg.sample_and_refeed or cfg.stateful_across_batch:
    # if using the generate fake batch layer, the dataset size has to
    # be dividable by the batch size
    sample_ind = rand_sample_ind(total_num_samples,num_testing_sample,batch_size)
    if not cfg.shuffle_data:
        sample_ind = sorted(sample_ind)
    encoder_input_data = rand_sample(encoder_input_data,sample_ind)
    decoder_input_data = rand_sample(decoder_input_data,sample_ind)
    decoder_target_data = rand_sample(decoder_target_data,sample_ind)
    # sanity check
    ind = np.random.randint(encoder_input_data.shape[0])
    assert encoder_input_data[ind,-1,:].sum()==decoder_input_data[ind,0,:].sum()



### ====================Training====================
# tag = 'weightedce_onehot_mat_3layertanh_stateful_noshuffle_raw_512-1024july10'
# tag = 'convLSTMtar_seqseq_THU_traintest_split_NLL_Aug7'
# tag = 'convLSTMtar_seqseq_shanghai_traintest_split_Aug9'
# tag = 'convLSTMtar_seqseq_shanghai_traintest_split_predmeanvar_Aug9'
# tag = 'convLSTMtar_seqseq_shanghai_traintest_split_meanvarmeanvar_Aug10'
# tag = 'convLSTMtar_seqseq_dilation_predmeanvar_Aug21'##NOT finished!!!
tag = 'convLSTMtar_seqseq_THU_predmeanvar_Sep5'

model_checkpoint = ModelCheckpoint(tag+'_epoch{epoch:02d}-{val_loss:.4f}.h5', monitor='val_loss', save_best_only=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                 patience=3, min_lr=1e-6)
stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=0, mode='auto')
model.fit([encoder_input_data, decoder_input_data],decoder_target_data,
          batch_size=batch_size, 
          epochs=epochs,
          validation_split=0.2,
          shuffle=cfg.shuffle_data, initial_epoch=0,
          callbacks=[model_checkpoint, reduce_lr, stopping])


### ====================Testing====================
### data format 3
# video_data_test = pickle.load(open('./360video/data/tsinghua_test_video_data.p','rb'))
# ### data format 4
# video_data_test = pickle.load(open('./360video/data/shanghai_dataset_xyz_test.p','rb'))
### data format 5
video_data_test = pickle.load(open('./360video/temp/tsinghua_after_bysec_interpolation/tsinghua_test_video_data_over_video.p','rb'))

video_data_test = clip_xyz(video_data_test)
datadb = video_data_test.copy()
_video_db,_video_db_future,_video_db_future_input = get_data(datadb,pick_user=False)
_video_db = _reshape_others_data(_video_db)[:,:,np.newaxis,:,:]
_video_db_future_input = _reshape_others_data(_video_db_future_input)
_video_db_future = _reshape_others_data(_video_db_future)


# model = load_model('2layer_convLSTM_notanh_seq2seq_fakebatch_epoch14-0.0782.h5')#cannot deploy, has errors!
# model = load_model('2layer_convLSTM_notanh_seq2seq_fakebatch_june11_epoch07-0.0787.h5')#cannot deploy, has errors!
# model.load_weights('backup_convLSTM_seq2seq_on_videodb_tar_BN_july2_epoch13-0.0908.h5')
# model.load_weights('backup_convLSTM_seq2seq_on_videodb_tar_july2_epoch29-0.0754.h5')
# model = load_model('3layertanh_stateful_shuffle_raw_july7_epoch16-0.1654.h5')
# model = load_model('onehot_3layertanh_stateful_noshuffle_raw_july9_epoch14-4.7314.h5')

# backup_convLSTM_seq2seq_on_videodb_tar_BN_july2_epoch12-0.0953
# backup_convLSTM_seq2seq_on_videodb_tar_BN_july2_epoch13-0.0908
# model = load_model('backup_convLSTM_seq2seq_on_videodb_tar_july2_epoch29-0.0754.h5')
# model = load_model('convLSTMtar_seqseq_THU_traintest_split_Aug7_epoch14-0.5000.h5')
# model.load_weights('convLSTMtar_seqseq_shanghai_traintest_split_meanvarmeanvar_Aug10_epoch33-0.1125.h5')
model.load_weights('convLSTMtar_seqseq_shanghai_traintest_split_predmeanvar_Aug9_epoch13-0.1090.h5')

if cfg.predict_mean_var and cfg.sample_and_refeed:
    create_sampling_model = True  
else:
    create_sampling_model = False    
if create_sampling_model:
    # Define sampling models
    encoder_outputs = [pst_state_h0, pst_state_c0,pst_state_h1, pst_state_c1,pst_state_h2, pst_state_c2]
    encoder_model = Model(encoder_inputs, encoder_outputs)

    states0_h = Input(shape=(1,fps,latent_dim*2))
    states0_c = Input(shape=(1,fps,latent_dim*2))
    states1_h = Input(shape=(1,fps,latent_dim))
    states1_c = Input(shape=(1,fps,latent_dim))
    states2_h = Input(shape=(1,fps,latent_dim/2))
    states2_c = Input(shape=(1,fps,latent_dim/2))

    fut_outputs_sqns00, fut_state_h0, fut_state_c0 = convlstm_decoder([decoder_inputs]+[states0_h, states0_c])
    fut_outputs_sqns11, fut_state_h1, fut_state_c1 = convlstm_decoder1([fut_outputs_sqns00]+[states1_h, states1_c])
    fut_outputs_sqns22, fut_state_h2, fut_state_c2 = convlstm_decoder2([fut_outputs_sqns11]+[states2_h, states2_c])

    fut_outputs_sqns012 = Concatenatelayer1([fut_outputs_sqns00,fut_outputs_sqns11,fut_outputs_sqns22])
    # fut_outputs_sqns012 = bnlayer(fut_outputs_sqns012)
    if cfg.predict_mean_var:
        fut_outputs_sqns012 = flatten_layer(get_dim_layer(fut_outputs_sqns012))
        outputs_sampling = pred_conv_lstm_dense(fut_outputs_sqns012)
    outputs_sampling = expand_dim_layer(outputs_sampling)

    decoder_states_outputs = [fut_state_h0,fut_state_c0,fut_state_h1,fut_state_c1,fut_state_h2, fut_state_c2]
    decoder_model = Model([encoder_inputs,decoder_inputs,
                        states0_h,states0_c,states1_h,states1_c,states2_h,states2_c],
                        [outputs_sampling]+decoder_states_outputs)


def decode_sequence_fov_sampling(input_seq):
    if input_seq.shape[0]>1:
        last_location = input_seq[:,-1,:][:,np.newaxis,:]
    elif input_seq.shape[0]==1: 
        last_location = input_seq[0,-1,:][np.newaxis,np.newaxis,:]

    h0, c0, h1, c1, h2, c2 = encoder_model.predict(input_seq)
    target_seq = last_location

    decoded_sentence = []
    for ii in range(max_decoder_seq_length):
        output_tokens, h0, c0, h1, c1, h2, c2 = decoder_model.predict([input_seq,target_seq, h0, c0, h1, c1, h2, c2])
        decoded_sentence+=[output_tokens]
        ux_temp,varx_temp = output_tokens[:,0,0],output_tokens[:,0,3]
        uy_temp,vary_temp = output_tokens[:,0,1],output_tokens[:,0,4]
        uz_temp,varz_temp = output_tokens[:,0,2],output_tokens[:,0,5]
        temp_newdata = np.stack((generate_fake_batch_numpy(ux_temp,varx_temp,batch_size=batch_size),
                        generate_fake_batch_numpy(uy_temp,vary_temp,batch_size=batch_size),
                        generate_fake_batch_numpy(uz_temp,varz_temp,batch_size=batch_size)),axis=-1)[:,np.newaxis,np.newaxis,:]
        target_seq = temp_newdata

    decoded_sentence = np.array(decoded_sentence)
    decoded_sentence = decoded_sentence.transpose(1,0,2,3)
    return decoded_sentence


def decode_sequence_fov(input_seq):
    # Encode the input as state vectors.
    if input_seq.shape[0]>1:
        last_location = input_seq[:,-1,:][:,np.newaxis,:]
    elif input_seq.shape[0]==1: 
        last_location = input_seq[0,-1,:][np.newaxis,np.newaxis,:]
    decoded_sentence = model.predict([input_seq,last_location])
    return decoded_sentence


gt_sentence_list = []
decoded_sentence_list = []
for seq_index in range(0,_video_db.shape[0],batch_size):
# for seq_index in range(total_num_samples-num_testing_sample,total_num_samples,batch_size):
# for seq_index in range(total_num_samples-num_testing_sample,total_num_samples-num_testing_sample+100):
    # Take one sequence (part of the training set)
    # for trying out decoding.
    if cfg.use_one_hot:
        input_seq = one_hot[seq_index: seq_index + batch_size,:,:].transpose(0,1,3,4,2)
    else:
        if cfg.input_mean_var:
            input_seq = get_gt_target_xyz(_video_db[seq_index: seq_index + batch_size,:,:].squeeze())[:,:,np.newaxis,np.newaxis,:]
        else:
            input_seq = _video_db[seq_index: seq_index + batch_size,:,:]

    if input_seq.shape[0]<batch_size:
        break
    if create_sampling_model:
        decoded_sentence = decode_sequence_fov_sampling(input_seq)
    else:
        decoded_sentence = decode_sequence_fov(input_seq)        
    
    if cfg.use_one_hot:
        ### for 1d one-hot vec:
        # max_ind = np.argmax(decoded_sentence,axis=-1)
        ### for 2d one-hot mat:
        max_ind = np.argmax(decoded_sentence.reshape(batch_size,cfg.predict_step,-1,fps),axis=-2)
        decoded_sentence = max_ind
        ## also use one-hot gt
        gt_sentence = one_hot_future[seq_index: seq_index + batch_size,:].transpose(0,1,3,4,2)       
        ### for 1d one-hot vec:
        # gt_max_ind = np.argmax(gt_sentence,axis=-1)
        ### for 2d one-hot mat:
        # gt_max_ind = np.argmax(gt_sentence.reshape(batch_size,cfg.predict_step,-1,fps),axis=-2)
        # gt_sentence = gt_max_ind
        ## use real gt
        gt_sentence = _video_db_future[seq_index: seq_index + batch_size,:]
    else:
        gt_sentence = _video_db_future[seq_index: seq_index + batch_size,:]

    decoded_sentence_list+=[decoded_sentence]
    gt_sentence_list+=[gt_sentence]

pickle.dump(decoded_sentence_list,open('decoded_sentence'+tag+'.p','wb'))
pickle.dump(gt_sentence_list,open('gt_sentence_list'+tag+'.p','wb'))
print('Testing finished!')




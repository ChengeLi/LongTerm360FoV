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
from tqdm import tqdm
from collections import OrderedDict
from mycode.data_generator_including_saliency import *

# experiment = 1 
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
mixing_saliency=False

target_user_only = False
model_others = False
if model_others:
    others_mlp = False
    others_lstm = True #also given gt mean,var
if not model_others:
    mlp_mixing = True
    conv_mixing = False

## utility layers
flatten_layer = Flatten()
expand_dim_layer = Lambda(lambda x: K.expand_dims(x,1))
expand_dim_layer_1 = Lambda(lambda x: K.expand_dims(x,-1))
Concatenatelayer = Concatenate(axis=1)
Concatenatelayer_1 = Concatenate(axis=-1)
get_dim1_layer = Lambda(lambda x: x[:,0,:])

reduce_sum_layer = Lambda(lambda x: K.sum(x, axis=1))#collapsing user dim
sigmoid = activations.get('sigmoid')
# GLU_layer = Lambda(lambda x: multiply([x[0],sigmoid(x[1])]))
GLU_layer = Lambda(lambda x: multiply([x[1],sigmoid(x[0])]))



### ====================Graph def====================
### 1-layer fc-lstm
# if cfg.input_mean_var:
#     encoder_inputs = Input(shape=(None, 6))
#     if cfg.stateful_across_batch:
#         encoder_inputs = Input(batch_shape=(batch_size, None, 6))
#         decoder_inputs = Input(batch_shape=(batch_size, 1, num_decoder_tokens))
# else:
#     encoder_inputs = Input(shape=(None, num_encoder_tokens))

# # 1 layer encoder
# encoder = LSTM(latent_dim, return_state=True)
# encoder_outputs, state_h, state_c = encoder(encoder_inputs)
# states = [state_h, state_c]

# # 1 layer decoder
# # Set up the decoder, which will only process one timestep at a time.
# decoder_inputs = Input(shape=(1, num_decoder_tokens))
# decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
# # decoder_lstm = LSTM(latent_dim, return_sequences=False, return_state=True, stateful=True)
# decoder_dense = Dense(num_decoder_tokens,activation='tanh')


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
if cfg.teacher_forcing:
    decoder_inputs = Input(shape=(max_decoder_seq_length, num_decoder_tokens))
else:
    decoder_inputs = Input(shape=(1, num_decoder_tokens))    
    # decoder_inputs = Input(shape=(1, num_encoder_tokens))    
decoder_lstm1 = LSTM(latent_dim, stateful=cfg.stateful_across_batch, return_sequences=True, return_state=True)
decoder_lstm2 = LSTM(latent_dim, stateful=cfg.stateful_across_batch, return_sequences=True, return_state=True)
if cfg.predict_mean_var:
    decoder_dense = Dense(num_decoder_tokens,activation='tanh')
else:
    decoder_dense = Dense(num_encoder_tokens,activation='tanh')


if cfg.teacher_forcing:
    decoder1_outputs, state_decoder1_h, state_decoder1_c = decoder_lstm1(decoder_inputs,initial_state=decoder1_states_inputs)
    decoder2_outputs, state_decoder2_h, state_decoder2_c = decoder_lstm2(decoder1_outputs,initial_state=decoder2_states_inputs)
    if target_user_only:
        decoder_outputs = decoder_dense(decoder2_outputs)
    else:
        if not model_others:
            decoder_preds = decoder_dense(decoder2_outputs)


if not target_user_only:
    others_fut_inputs = Input(shape=(max_decoder_seq_length,(num_user-1),6))
    if model_others:
        if others_mlp:
            #user mlp to model others
            others_dense1 = Dense(256,activation='relu')
            others_dense2 = Dense(latent_dim,activation='relu')
        elif others_lstm:
            others_lstm1 = Bidirectional(LSTM(latent_dim, stateful=cfg.stateful_across_batch, return_sequences=True, return_state=True), merge_mode='concat')
            others_lstm2 = Bidirectional(LSTM(latent_dim, stateful=cfg.stateful_across_batch, return_sequences=True, return_state=True), merge_mode='concat')
            # others_lstm1 = LSTM(latent_dim, stateful=cfg.stateful_across_batch, return_sequences=True, return_state=True)
            # others_lstm2 = LSTM(latent_dim, stateful=cfg.stateful_across_batch, return_sequences=True, return_state=True)

            others_fut_inputs0 = Reshape((max_decoder_seq_length,(num_user-1)*num_decoder_tokens))(others_fut_inputs)
            others_fut_inputs1 = others_lstm1(others_fut_inputs0)
            others_fut_inputs2 = others_lstm2(others_fut_inputs1)

            #lstm+mlp mixing
            # others_dense = Dense((num_user-1)*num_decoder_tokens,activation='relu')
            # others_fut_output = others_dense(others_fut_inputs2[0])
            # others_fut_output = Reshape((max_decoder_seq_length,num_user-1,num_decoder_tokens))(others_fut_output)
            # mixing = Dense(num_decoder_tokens,activation=None)
    else:
        if mlp_mixing:
            ### 1 layer mixing
            mixing = Dense(num_decoder_tokens,activation='tanh')
            ### 3 layer mixing
            # mixing = Dense(3*num_decoder_tokens,activation='tanh')
            # mixing1 = Dense(2*num_decoder_tokens,activation='tanh')
            # mixing2 = Dense(num_decoder_tokens,activation=None)
            # gating = Dense(num_user*6,activation=None) #gating

            ### softmax gating
            # mixing = Dense(256,activation='tanh')
            # mixing1 = Dense(512,activation='tanh')
            # mixing2 = Dense(256,activation=None)
            # gating = Dense(num_user,activation='softmax') #softmax gating
            # modulating = Lambda(lambda x: Concatenatelayer_1([expand_dim_layer_1(multiply([x[0],x[1][:,:,0]])),
            #                                                 expand_dim_layer_1(multiply([x[0],x[1][:,:,1]])),
            #                                                 expand_dim_layer_1(multiply([x[0],x[1][:,:,2]])),
            #                                                 expand_dim_layer_1(multiply([x[0],x[1][:,:,3]])),
            #                                                 expand_dim_layer_1(multiply([x[0],x[1][:,:,4]])),
            #                                                 expand_dim_layer_1(multiply([x[0],x[1][:,:,5]]))])) 
            

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
    # 1-layer fclstm
    # decoder_states, state_h, state_c = decoder_lstm(inputs,initial_state=states)
    # decoder_pred = decoder_dense(decoder_states)
    
    if not cfg.teacher_forcing:
        # 2-layer fclstm, without teacher forcing
        decoder1_outputs, state_decoder1_h, state_decoder1_c = decoder_lstm1(inputs,
                                             initial_state=decoder1_states_inputs)
        decoder1_states_inputs = [state_decoder1_h, state_decoder1_c]
        decoder2_outputs, state_decoder2_h, state_decoder2_c = decoder_lstm2(decoder1_outputs,initial_state=decoder2_states_inputs)
        decoder2_states_inputs = [state_decoder2_h, state_decoder2_c]

    if target_user_only:
        outputs = decoder_dense(decoder2_outputs)
    else:
        if model_others:
            # model others' trend
            if others_mlp:
                others_fut_inputs_slice = util.slice_layer(1,time_ind,time_ind+1)(others_fut_inputs)
                others_fut_inputs1 = Flatten()(others_fut_inputs_slice)
                others_fut_inputs1 = others_dense1(others_fut_inputs1)
                others_fut_inputs1 = others_dense2(others_fut_inputs1)
                if cfg.teacher_forcing:
                    concat_state = Concatenatelayer([others_fut_inputs1,get_dim1_layer(util.slice_layer(1,time_ind,time_ind+1)(decoder2_outputs))])
                else:
                    concat_state = Concatenatelayer([others_fut_inputs1,get_dim1_layer(decoder2_outputs)])
                outputs = expand_dim_layer(decoder_dense(concat_state))
            elif others_lstm:
                #LSTM only
                others_fut_inputs2_slice = util.slice_layer(1,time_ind,time_ind+1)(others_fut_inputs2[0])
                concat_state = Concatenatelayer([get_dim1_layer(others_fut_inputs2_slice),get_dim1_layer(decoder2_outputs)])
                ###use Gated Linear Unit instead of concatenating
                # concat_state = GLU_layer([get_dim1_layer(others_fut_inputs2_slice),get_dim1_layer(decoder2_outputs)])
                outputs = expand_dim_layer(decoder_dense(concat_state))
                #LSTM +mlp mixing                    
                # decoder_pred = decoder_dense(decoder2_outputs)
                # concat_state = Concatenatelayer([get_dim1_layer(util.slice_layer(1,time_ind,time_ind+1)(others_fut_output)),decoder_pred])
                # outputs = expand_dim_layer(mixing(Flatten()(concat_state)))
        else:  
            if not cfg.teacher_forcing:
                decoder_pred = decoder_dense(decoder2_outputs)
            else:
                decoder_pred = util.slice_layer(1,time_ind,time_ind+1)(decoder_preds)

            #saliency CNN feature
            if cfg.use_saliency:
                _saliency = get_CNN_fea(saliency_inputs,time_ind,final_dim=num_decoder_tokens)

            # given others' gt mean and variance
            # directly concat others' mean and variance
            gt_mean_var_oth = util.slice_layer(1,time_ind,time_ind+1)(others_fut_inputs) 
            
            # get nearest K neighbours TODO!!!
            # neighbor_ind = util.find_k_neighbours_TF(decoder_pred, gt_mean_var_oth, k=5) 
            concat_outputs = Concatenatelayer([get_dim1_layer(gt_mean_var_oth),decoder_pred])
            if mlp_mixing:
                ##1 layer or 3 layers mixing
                concat_outputs = Flatten()(concat_outputs)
                outputs = mixing(concat_outputs)
                # outputs = mixing1(outputs)
                # outputs = mixing2(outputs)
                ### softmax gating
                # gating_weights = gating(outputs)
                # outputs = multiply([gating_weights,concat_outputs])
                ### multipy on user level
                # concat_outputs = Reshape([num_user,6])(concat_outputs)
                # outputs = modulating([gating_weights,concat_outputs])
                # outputs = reduce_sum_layer(outputs)


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


if target_user_only:
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
elif cfg.use_saliency:
    model = Model([encoder_inputs, others_fut_inputs, decoder_inputs, saliency_inputs], decoder_outputs)
else:
    model = Model([encoder_inputs, others_fut_inputs, decoder_inputs], decoder_outputs)
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
    video_data_train = pickle.load(open('./360video/data/shanghai_dataset_xyz_train.p','rb'),encoding='latin')    
    datadb_train = clip_xyz(video_data_train)
    video_data_test = pickle.load(open('./360video/data/shanghai_dataset_xyz_test.p','rb'),encoding='latin')    
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

else:
    #### create data for tsinghua dataset (train, test) split on users, format5_tsinghua_by_sec_interp
    # video_data_train = pickle.load(open('./360video/temp/tsinghua_after_bysec_interpolation/tsinghua_train_video_data_over_users.p','rb'),encoding='latin')    
    # video_data_train = pickle.load(open('./360video/temp/tsinghua_after_bysec_interpolation/tsinghua_train_video_data_over_video.p','rb'),encoding='latin')     
    # video_data_train = clip_xyz(video_data_train)
    # datadb = video_data_train.copy()
    # video_data_test = pickle.load(open('./360video/temp/tsinghua_after_bysec_interpolation/tsinghua_test_video_data_over_video.p','rb'))    
    # video_data_test = clip_xyz(video_data_test)
    # datadb = video_data_test.copy()
    # _video_db_tar, _video_db_future_tar, _video_db_future_input_tar, \
    # _video_db_oth,_video_db_future_oth,_video_db_future_input_oth = util.get_data(datadb,pick_user=True,num_user=datadb[0]['x'].shape[0])

    ##### create data format 3or4--4
    # video_data_train = pickle.load(open('./360video/data/shanghai_dataset_xyz_train.p','rb'))    
    # video_data_train = clip_xyz(video_data_train)
    # datadb = video_data_train.copy()
    # video_data_test = pickle.load(open('./360video/data/shanghai_dataset_xyz_test.p','rb'))    
    # video_data_test = clip_xyz(video_data_test)
    # datadb = video_data_test.copy()
    # _video_db_tar, _video_db_future_tar, _video_db_future_input_tar, \
    # _video_db_oth,_video_db_future_oth,_video_db_future_input_oth = util.get_data(datadb,pick_user=True,num_user=34)

    # #### load cached data
    # dataformat = 'formtat2'
    # _video_db_tar = pickle.load(open('./cache/'+dataformat+'/_video_db_tar_exp'+str(experiment)+'.p','rb'))
    # _video_db_future_tar = pickle.load(open('./cache/'+dataformat+'/_video_db_future_tar_exp'+str(experiment)+'.p','rb'))
    # _video_db_future_input_tar = pickle.load(open('./cache/'+dataformat+'/_video_db_future_input_tar_exp'+str(experiment)+'.p','rb'))
    # _video_db_oth = pickle.load(open('./cache/'+dataformat+'/_video_db_oth_exp'+str(experiment)+'.p','rb'))
    # _video_db_future_oth = pickle.load(open('./cache/'+dataformat+'/_video_db_future_oth_exp'+str(experiment)+'.p','rb'))
    # _video_db_future_input_oth = pickle.load(open('./cache/'+dataformat+'/_video_db_future_input_oth_exp'+str(experiment)+'.p','rb'))
    #### load cached data format4 
    dataformat = 'format4' #shanghaitech
    option='stride10_cut_head/'
    dataformat = 'format5_tsinghua_by_sec_interp' #tsinghua
    option=''
    select_k_neighbours=True
    _video_db_tar = util.load_h5('./cache/'+dataformat+'/train/'+option+'_video_db_tar.h5','_video_db_tar')
    _video_db_future_tar = util.load_h5('./cache/'+dataformat+'/train/'+option+'_video_db_future_tar.h5','_video_db_future_tar')
    _video_db_future_input_tar = util.load_h5('./cache/'+dataformat+'/train/'+option+'_video_db_future_input_tar.h5','_video_db_future_input_tar')
    _video_db_oth = util.load_h5('./cache/'+dataformat+'/train/'+option+'_video_db_oth.h5','_video_db_oth')
    _video_db_future_oth = util.load_h5('./cache/'+dataformat+'/train/'+option+'_video_db_future_oth.h5','_video_db_future_oth')
    _video_db_future_input_oth = util.load_h5('./cache/'+dataformat+'/train/'+option+'_video_db_future_input_oth.h5','_video_db_future_input_oth')


    _video_db_oth = _reshape_others_data(_video_db_oth)
    _video_db_future_oth = _reshape_others_data(_video_db_future_oth)
    _video_db_tar = _video_db_tar.reshape((_video_db_tar.shape[0],_video_db_tar.shape[1],fps,3))
    _video_db_future_tar = _video_db_future_tar.reshape((_video_db_tar.shape[0],_video_db_tar.shape[1],fps,3))
    _video_db_future_input_tar = _video_db_future_input_tar.reshape((_video_db_tar.shape[0],_video_db_tar.shape[1],fps,3))


    total_num_samples = _video_db_tar.shape[0]
    # num_testing_sample = int(0.15*total_num_samples)
    num_testing_sample = 1

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

        # _video_db_oth = util.shuffle_data(index_shuf,_video_db_oth)
        _video_db_future_oth = util.shuffle_data(index_shuf,_video_db_future_oth)
        # _video_db_future_input_oth = util.shuffle_data(index_shuf,_video_db_future_input_oth)


    #### prepare training data
    if cfg.input_mean_var:
        ### target user
        encoder_input_data = util.get_gt_target_xyz(_video_db_tar[:-num_testing_sample,:,:])
        decoder_target_data = util.get_gt_target_xyz(_video_db_future_tar[:-num_testing_sample,:,:])
        # encoder_input_data = util.get_gt_target_xyz(_video_db_oth_all[:-num_testing_sample,:,:])
        # decoder_target_data = util.get_gt_target_xyz(_video_db_future_oth_all[:-num_testing_sample,:,:])
        # ### other users
        others_fut_input_data = util.get_gt_target_xyz_oth(_video_db_future_oth[:-num_testing_sample])
        if not cfg.teacher_forcing:
            decoder_input_data = encoder_input_data[:,-1,:][:,np.newaxis,:]
        else:
            decoder_input_data = util.get_gt_target_xyz(_video_db_future_input_tar[:-num_testing_sample,:,:])
            # decoder_input_data = util.get_gt_target_xyz(_video_db_future_input_oth_all[:-num_testing_sample,:,:])

    else:
        ### target user
        _video_db_tar = _video_db_tar.reshape((_video_db_tar.shape[0],_video_db_tar.shape[1],-1))
        encoder_input_data = _video_db_tar[:-num_testing_sample,:,:]
        # decoder_target_data = _video_db_future_tar[:-num_testing_sample,:,:][:,:,np.newaxis,:,:]#predict raw
        decoder_target_data = util.get_gt_target_xyz(_video_db_future_tar[:-num_testing_sample,:,:])#predict mean/var
        # decoder_input_data = _video_db_tar[:-num_testing_sample,-1,:][:,np.newaxis,np.newaxis,:]
        decoder_input_data = util.get_gt_target_xyz(encoder_input_data[:,-1,:][:,np.newaxis,:])
        ### other users
        # others_fut_input_data = _video_db_future_oth[:-num_testing_sample][:,:,np.newaxis,:]
        others_fut_input_data = util.get_gt_target_xyz_oth(_video_db_future_oth[:-num_testing_sample])


### ====================Training====================
# tag = 'noTF_mlpmixing_'
tag='fctar_seqseq_mlpmixing_shanghai_traintest_split_predmeanvar_Aug9'
# tag = 'fctar_seqseq_mlpmixing_shanghai_traintest_split_meanvarmeanvar_Aug9_epoch'
# tag = 'fctar_seqseq_mlpmixing3layer_predmeanvar_Aug20_epoch'
# tag = 'fctar_seqseq_othersMLP_predmeanvar_Aug20'
# tag = 'fctar_seqseq_othersLSTM_predmeanvar_Aug20'
# tag = 'fctar_seqseq_othersLSTM+mlpmixing_predmeanvar_Aug21'
# tag = 'fctar_seqseq_others_nonBi-LSTM_predmeanvar_Aug24'
# tag = 'fctar_seqseq_mlpmixing_KNN_predmeanvar_aug29_' #didn't finish yet!!
# tag = 'fctar_seqseq_mlpmixing_THUformat5_sep4'
# tag = 'fctar_seqseq_othersBiLSTM_THUformat5_sep4'

# tag = 'fc_mlpmixing_shanghai_generator_stride1_bs16_sep9'  
# tag = 'fc_mlpmixing_shanghai_plussaliency_stride1_bs8_sep9'
# tag = 'fc_mlpmixing_shanghai_plussaliency_stride1_bs8_sep10'
# tag = 'fc_mlpmixing_shanghai_plussaliency_stride10_bs8_sep11'
# tag = 'fc_mlpmixing_saliency_residual_sep11'

# tag = 'othersLSTM_GLU2_sep10'
# tag='mlpmixing_gating_sep10'
# tag='mlpmixing_softmaxgating_sep14'

tag = 'fctar_seqseq_othersLSTM_predmeanvar_oct30'#bi-lstm


model_checkpoint = ModelCheckpoint(tag+'{epoch:02d}-{val_loss:.4f}.h5', monitor='val_loss', save_best_only=False)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                 patience=10, min_lr=1e-6)
stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=3, verbose=0, mode='auto')

if use_generator:
    # num_samples = 5404/cfg.batch_size
    # num_samples = 24000
    num_samples = 10000

    tqdm(model.fit_generator(mygenerator,steps_per_epoch=num_samples, epochs=epochs,
                    validation_data=mygenerator_val, validation_steps=100,
                    callbacks=[model_checkpoint, reduce_lr, stopping],
                    use_multiprocessing=False, shuffle=True,
                    initial_epoch=0))
else:
    model.fit([encoder_input_data, others_fut_input_data, decoder_input_data], decoder_target_data,
    # model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
          batch_size=batch_size,
          epochs=epochs,
          validation_split=0.1,
          shuffle=True, initial_epoch=0,
          callbacks=[model_checkpoint, reduce_lr, stopping])





### ====================Testing===================
# model = load_model('fc_mlpmixing_shanghai_generator_stride10_sep803-0.0822.h5')
# model.load_weights('fc_mlpmixing_shanghai_plussaliency_stride10_bs8_sep1101-0.0776.h5')
# model.load_weights('fc_mlpmixing_saliency_residual_sep1101-0.0786.h5')
# model.load_weights('fc_mlpmixing_saliency_residual_sep1102-0.0828.h5')
# model.load_weights('fc_mlpmixing_saliency_residual_sep1104-0.1102.h5')
# model = load_model('mlpmixing_gating_sep1009-0.0908.h5')

video_data_test = pickle.load(open('./360video/data/shanghai_dataset_xyz_test.p','rb'))    
datadb_test = clip_xyz(video_data_test)
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




# _video_db_tar = pickle.load(open('./cache/'+dataformat+'/test/'+option+'_video_db_tar.p','rb'))
# _video_db_future_tar = pickle.load(open('./cache/'+dataformat+'/test/'+option+'_video_db_future_tar.p','rb'))
# _video_db_future_input_tar = pickle.load(open('./cache/'+dataformat+'/test/'+option+'_video_db_future_input_tar.p','rb'))
# # _video_db_oth = pickle.load(open('./cache/'+dataformat+'/test/'+option+'_video_db_oth.p','rb'))
# _video_db_future_oth = pickle.load(open('./cache/'+dataformat+'/test/'+option+'_video_db_future_oth.p','rb'))
# # _video_db_future_input_oth = pickle.load(open('./cache/'+dataformat+'/test/'+option+'_video_db_future_input_oth.p','rb'))

#or use h5
_video_db_tar = util.load_h5('./cache/'+dataformat+'/test/'+option+'_video_db_tar.h5','_video_db_tar')
_video_db_future_tar = util.load_h5('./cache/'+dataformat+'/test/'+option+'_video_db_future_tar.h5','_video_db_future_tar')
_video_db_future_input_tar = util.load_h5('./cache/'+dataformat+'/test/'+option+'_video_db_future_input_tar.h5','_video_db_future_input_tar')
_video_db_future_oth = util.load_h5('./cache/'+dataformat+'/test/'+option+'_video_db_future_oth.h5','_video_db_future_oth')


_video_db_future_oth = _reshape_others_data(_video_db_future_oth)
if select_k_neighbours:
    _video_db_future_oth = util.get_random_k_other_users(_video_db_future_oth)

if cfg.input_mean_var:
    _video_db_tar = _video_db_tar.reshape((_video_db_tar.shape[0],_video_db_tar.shape[1],fps,3))
_video_db_future_tar = _video_db_future_tar.reshape((_video_db_tar.shape[0],_video_db_tar.shape[1],fps,3))
_video_db_future_input_tar = _video_db_future_input_tar.reshape((_video_db_tar.shape[0],_video_db_tar.shape[1],fps,3))


# model.load_weights('shuffle_tardb_seq2seq_51-0.0152.h5')
# model.load_weights('mlpmixing_others_mean_var_mixing_36-0.0156.h5')
# model.load_weights('given_others_mean_var_mixing_46-0.0150.h5')
model.load_weights('fctar_seqseq_mlpmixing_shanghai_traintest_split_predmeanvar_Aug9_epoch12-0.0903.h5')
# model = load_model('fctar_seqseq_othersLSTM+mlpmixing_predmeanvar_Aug2107-0.0917.h5')
model.load_weights('fctar_seqseq_othersLSTM_predmeanvar_Aug2006-0.0919.h5')
# model = load_model('fctar_seqseq_othersMLP_predmeanvar_Aug2007-0.0913.h5')
# model = load_model('fctar_seqseq_mlpmixing3layer_predmeanvar_Aug20_epoch16-0.0905.h5')
# model = load_model('fctar_seqseq_others_nonBi-LSTM_predmeanvar_Aug2404-0.0909.h5')

# model = load_model('fc_mlpmixing_shanghai_generator_sep803-0.0374.h5')

def decode_sequence_fov(input_seq,others_fut_input_seq):
    # Encode the input as state vectors.
    if cfg.input_mean_var:
        last_location = input_seq[0,-1,:][np.newaxis,np.newaxis,:]
    else:
        last_location = util.get_gt_target_xyz(input_seq[:,-1,:][:,np.newaxis,:])
    if target_user_only:
        decoded_sentence = model.predict([input_seq,last_location])
    else:
        decoded_sentence = model.predict([input_seq,others_fut_input_seq,last_location])
    return decoded_sentence

if cfg.teacher_forcing:
    # Define sampling models
    encoder_outputs = [state_h_1, state_c_1, state_h_2, state_c_2]
    encoder_model = Model(encoder_inputs, encoder_outputs)
 
    decoder1_state_input_h = Input(shape=(latent_dim,))
    decoder1_state_input_c = Input(shape=(latent_dim,))
    decoder2_state_input_h = Input(shape=(latent_dim,))
    decoder2_state_input_c = Input(shape=(latent_dim,))
    decoder1_states_inputs = [decoder1_state_input_h, decoder1_state_input_c]
    decoder2_states_inputs = [decoder2_state_input_h, decoder2_state_input_c]

    decoder_inputs = Input(shape=(1, num_decoder_tokens))
    decoder_outputs1, state_h1, state_c1 = decoder_lstm1(decoder_inputs, initial_state=decoder1_states_inputs)
    decoder_outputs2, state_h2, state_c2 = decoder_lstm2(decoder_outputs1,initial_state=decoder2_states_inputs)

    decoder_states_outputs = [state_h1, state_c1, state_h2, state_c2]

    if target_user_only:
        outputs = decoder_dense(decoder_outputs2)
        decoder_model = Model([decoder_inputs,
                        decoder1_state_input_h, decoder1_state_input_c,decoder2_state_input_h, decoder2_state_input_c],
                            [outputs]+decoder_states_outputs)

    else:
        others_fut_inputs = Input(shape=(max_decoder_seq_length,(num_user-1),6))
        if model_others:
            if others_mlp:        
                others_fut_inputs1 = Flatten()(others_fut_inputs)
                others_fut_inputs1 = others_dense1(others_fut_inputs1)
                others_fut_inputs1 = others_dense2(others_fut_inputs1)
                concat_state = Concatenatelayer([others_fut_inputs1,get_dim1_layer(util.slice_layer(1,0,1)(decoder_outputs2))])
                outputs = expand_dim_layer(decoder_dense(concat_state))
            else:
                raise NotImplementedError
        else:
            decoder_pred = decoder_dense(decoder_outputs2)
            #directly concat others' mean and variance
            gt_mean_var_oth = util.slice_layer(1,time_ind,time_ind+1)(others_fut_inputs)        
            concat_outputs = Concatenatelayer([get_dim1_layer(gt_mean_var_oth),decoder_pred])
            if mlp_mixing:
                concat_outputs = Flatten()(concat_outputs)
                outputs = expand_dim_layer(fconcat_outputs)
            elif conv_mixing:
                # use conv layer to mix
                concat_outputs = Permute((2, 1))(concat_outputs)
                outputs = mixing(expand_dim_layer(concat_outputs))
                outputs = mixing1(outputs)
                outputs = mixing2(outputs)
                outputs = Permute((2, 1))(get_dim1_layer(outputs))



        decoder_model = Model([decoder_inputs,
                            decoder1_state_input_h, decoder1_state_input_c,decoder2_state_input_h, decoder2_state_input_c,
                            others_fut_inputs],
                            [outputs, state_h1, state_c1, state_h2, state_c2])


def decode_sequence_fov_TF(input_seq,others_fut_input_seq):
    if input_seq.shape[0]>1:
        last_location = input_seq[:,-1,:][:,np.newaxis,:]
    elif input_seq.shape[0]==1: 
        last_location = input_seq[0,-1,:][np.newaxis,np.newaxis,:]

    h1, c1, h2, c2 = encoder_model.predict(input_seq)
    target_seq = last_location

    decoded_sentence = []
    for ii in range(max_decoder_seq_length):
        if target_user_only:
            output_tokens, h1, c1, h2, c2 = decoder_model.predict([target_seq,h1, c1, h2, c2])
        else:
            output_tokens, h1, c1, h2, c2 = decoder_model.predict([target_seq,h1, c1, h2, c2,others_fut_input_seq])
        decoded_sentence+=[output_tokens]
        target_seq = output_tokens
        states_value = [h1, c1, h2, c2]

    return decoded_sentence



gt_sentence_list = []
decoded_sentence_list = []
for seq_index in range(0,_video_db_tar.shape[0],batch_size):
# for seq_index in range(total_num_samples-num_testing_sample,total_num_samples):
# for seq_index in range(total_num_samples-num_testing_sample,total_num_samples-num_testing_sample+100):
    # input_seq = _video_db_tar[seq_index: seq_index + 1,:,:]
    # others_fut_input_seq = _video_db_future_oth[seq_index: seq_index + 1,:]
    if cfg.input_mean_var:
        input_seq = util.get_gt_target_xyz(_video_db_tar[seq_index: seq_index + batch_size,:,:])
    else:
        input_seq = _video_db_tar[seq_index: seq_index + batch_size,:]
    others_fut_input_seq = util.get_gt_target_xyz_oth(_video_db_future_oth[seq_index: seq_index + batch_size,:])

    if cfg.teacher_forcing:
        decoded_sentence = decode_sequence_fov_TF(input_seq,others_fut_input_seq)
    else:
        decoded_sentence = decode_sequence_fov(input_seq,others_fut_input_seq)
    decoded_sentence_list+=[decoded_sentence]
    gt_sentence = _video_db_future_tar[seq_index: seq_index + batch_size,:,:]
    gt_sentence_list+=[gt_sentence]
    # print('-')
    # decoder_target = util.get_gt_target_xyz(gt_sentence)
    # print('Decoded sentence - decoder_target:', np.squeeze(np.array(decoded_sentence))[:,:3]-np.squeeze(decoder_target)[:,:3])

# thu_tag='_thu_'
thu_tag=''
pickle.dump(decoded_sentence_list,open('decoded_sentence'+thu_tag+tag+'.p','wb'))
pickle.dump(gt_sentence_list,open('gt_sentence_list'+thu_tag+tag+'.p','wb'))
print('Testing finished!')














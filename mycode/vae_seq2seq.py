from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from keras.models import Model
from keras.layers import Input, LSTM, Dense, add, multiply, GRU, Conv1D
from keras.layers import Lambda,Concatenate,Flatten,ConvLSTM2D,BatchNormalization
from keras.datasets import mnist
from keras.callbacks import ModelCheckpoint,ReduceLROnPlateau,EarlyStopping
from keras import backend as K
from keras.models import load_model
from keras import optimizers
import keras.losses as losses
# from keras.losses import mse, binary_crossentropy
# from keras.utils import plot_model
import sys,glob,io,random,os
from random import shuffle
import matplotlib.pyplot as plt
import _pickle as pickle
import numpy as np
if './360video/' not in sys.path:
    sys.path.insert(0, './360video/')
from mycode.dataLayer import DataLayer
import mycode.cost as costfunc
from mycode.config import cfg
from mycode.dataIO import clip_xyz
import mycode.utility as util
import pdb


Concatenatelayer = Concatenate(axis=1)
starting_epoch = 62
is_train = True
use_partial_tf = False
reconstruction_loss_weight = 5

def sampling(args):
    """Reparameterization trick by sampling fr an isotropic unit Gaussian.
    # Arguments:
        args (tensor): mean and log of variance of Q(z|X)
    # Returns:
        z (tensor): sampled latent vector
    """

    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean=0 and std=1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon




def get_kl_loss(z_mean,z_log_var,starting_epoch):
    kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    kl_loss_weight = K.minimum(K.log(starting_epoch*1.+1.),10) #beta
    return kl_loss_weight*kl_loss



def get_vae_loss(y_true, y_pred):
    # reconstruction_loss = losses.mean_squared_error(y_true, y_pred)
    reconstruction_loss = K.mean(losses.mean_squared_error(y_true, y_pred))
    # reconstruction_loss = K.sum(losses.mean_squared_error(y_true, y_pred))
    # reconstruction_loss = losses.binary_crossentropy(y_true, y_pred)

    # reconstruction_loss *= original_dim #why scale up?
    reconstruction_loss *= reconstruction_loss_weight
    kl_loss = get_kl_loss(z_mean,z_log_var,starting_epoch)
    vae_loss = K.mean(reconstruction_loss + kl_loss)
    return vae_loss

if __name__ == '__main__':
    batch_size = 32 
    epochs = 50
    latent_dim = 64 #for LSTM

    fps = 30
    num_encoder_tokens = 3*fps
    num_decoder_tokens = 6
    max_encoder_seq_length = cfg.running_length
    max_decoder_seq_length = cfg.predict_step

    # network parameters for vae
    intermediate_dim = 512
    vae_latent_dim = 20

    # VAE model = encoder + decoder
    # build encoder model

    # input embedding module
    # encoder_inputs = Input(shape=(None, num_encoder_tokens), name='encoder_input')
    encoder_inputs = Input(shape=(None, num_decoder_tokens), name='encoder_input')

    ##----------- 1D conv
    conv0 = Conv1D(filters=128, kernel_size=cfg.conv_kernel_size, padding='same',
        activation='relu', use_bias=True, kernel_initializer='glorot_uniform')
    conv1 = Conv1D(filters=256, kernel_size=cfg.conv_kernel_size, padding='same',
        activation='relu', use_bias=True, kernel_initializer='glorot_uniform')
    conv2 = Conv1D(num_decoder_tokens, 1, kernel_initializer='he_normal', activation='linear', padding='same', strides=1)

    embed_inputs = conv0(encoder_inputs)
    embed_inputs = conv1(embed_inputs)
    embed_inputs = conv2(embed_inputs)


    # seq2seq
    # encoder = LSTM(latent_dim, return_state=True)
    # encoder_outputs, state_h, state_c = encoder(encoder_inputs)
    # states = [state_h, state_c]
    encoder = GRU(latent_dim, return_state=True)
    encoder_outputs, state_h = encoder(embed_inputs)

    # vae part
    x = Dense(intermediate_dim, activation='relu')(state_h) #only use h
    z_mean = Dense(vae_latent_dim, name='z_mean')(x)
    z_log_var = Dense(vae_latent_dim, name='z_log_var')(x)

    # use reparameterization trick to push the sampling out as input
    # note that "output_shape" isn't necessary with the TensorFlow backend
    z = Lambda(sampling, output_shape=(vae_latent_dim,), name='z')([z_mean, z_log_var])

    # instantiate encoder model
    encoder = Model(encoder_inputs, [z_mean, z_log_var, z], name='encoder')
    encoder.summary()
    # plot_model(encoder, to_file='vae_mlp_encoder.png', show_shapes=True)


    # build decoder model
    if is_train:
        decoder_inputs = Input(shape=(None, num_decoder_tokens), name='decoder_input')
    else:
        decoder_inputs = Input(shape=(1, num_decoder_tokens))
    # decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
    decoder_gru = GRU(latent_dim, return_sequences=True, return_state=True)
    decoder_dense = Dense(num_decoder_tokens,activation='tanh') 


    # latent_inputs = Input(shape=(vae_latent_dim,), name='z_sampling') #used as the initial states!
    # h0 = Dense(latent_dim, activation='tanh')(latent_inputs)
    h0 = Dense(latent_dim, activation='tanh')(encoder(encoder_inputs)[2])
    all_outputs = []
    if is_train:
        inputs = util.slice_layer(1,0,1)(decoder_inputs)
    else:
        inputs = decoder_inputs    
    states = h0
    for time_ind in range(max_decoder_seq_length):
        teacher_key = np.random.randint(1,max_decoder_seq_length/2)
        # also use the same embedding for the decoder
        inputs = conv0(inputs)
        inputs = conv1(inputs)
        inputs = conv2(inputs)
        # decoder_states, state_h, state_c = decoder_lstm(inputs,initial_state=states)        
        # states = [state_h, state_c]
        decoder_states, state_h = decoder_gru(inputs,initial_state=states)        
        states = state_h
        outputs = decoder_dense(decoder_states)
        all_outputs.append(outputs)
        if time_ind%teacher_key==0 and is_train and use_partial_tf:#provide gt signal
            inputs = util.slice_layer(1,time_ind,time_ind+1)(decoder_inputs)
        else:
            inputs = outputs


    decoder_outputs = Lambda(lambda x: K.concatenate(x, axis=1))(all_outputs)
    # # instantiate decoder model
    # decoder = Model([decoder_inputs,latent_inputs], decoder_outputs, name='decoder')
    # decoder.summary()
    # # plot_model(decoder, to_file='vae_mlp_decoder.png', show_shapes=True)


    # instantiate VAE model
    # vae_outputs = decoder([decoder_inputs, encoder(encoder_inputs)[2]])
    # vae = Model([encoder_inputs,decoder_inputs], vae_outputs, name='vae_seq2seq') 
    vae = Model([encoder_inputs,decoder_inputs], decoder_outputs, name='vae_seq2seq') 

    adam = optimizers.Adam(lr=0.005,decay=0.95)
    vae.compile(optimizer=adam,loss = get_vae_loss)
    vae.summary()



    #### ====================data====================
    video_data_train = pickle.load(open('./360video/data/shanghai_dataset_xyz_train.p','rb'),encoding='latin1')    
    datadb = video_data_train.copy()
    _video_db,_video_db_future,_video_db_future_input = util.get_data(datadb,pick_user=False)
    #### only do reconstruction!!
    # _video_db_future = _video_db[:,::-1,:]

    if cfg.input_mean_var:
        encoder_input_data = util.get_gt_target_xyz(_video_db)
    else:
        encoder_input_data = _video_db
    decoder_target_data = util.get_gt_target_xyz(_video_db_future)
    decoder_input_data = util.get_gt_target_xyz(_video_db_future_input)[:,0,:][:,np.newaxis,:]
    # decoder_input_data = util.get_gt_target_xyz(_video_db_future_input)
    # decoder_input_data = np.zeros_like(decoder_input_data) ##zero out the decoder input


    ### ====================Training====================
    # tag='vae_seq2seq_dec5'
    # tag='vae_seq2seq_fullTF_dec5'
    # tag='vae_seq2seq_vae20_dec5' #vae_latent_dim=20
    # tag='vae_seq2seq_vae100_dec5'
    # tag='vae_seq2seq_vae20_zerodecoderinput_dec5'
    # tag='vae_seq2seq_vae20_kl_logannealing_dec6'
    # tag='vae_seq2seq_vae20_kl_logannealing_reconsonly_dec6' #only do reconstruction #not trained
    # tag='vae_seq2seq_vae20_kl_logannealing_partialTF_dec6'#random teacher key
    # tag='vae_seq2seq_vae20_kl_logannealing_input_embedding_dec6'
    tag='vae_seq2seq_vae20_kl_logannealing_bothinput_embedding_dec6'


    model_checkpoint = ModelCheckpoint(tag+'{epoch:02d}-{val_loss:.4f}.h5', monitor='val_loss', save_best_only=False)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                     patience=3, min_lr=1e-6)
    stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=0, mode='auto')
    
    # vae.fit([encoder_input_data,decoder_input_data],decoder_target_data,
    #           batch_size=batch_size,
    #           epochs=epochs,
    #           validation_split=0.1,
    #           shuffle=cfg.shuffle_data,
    #           initial_epoch=starting_epoch,
    #           callbacks=[model_checkpoint, reduce_lr, stopping])



    # increase the batch size when perfomance saturated 

    # spynet to get optical flow 

    for starting_epoch in range(starting_epoch,epochs):
        print('starting_epoch:',starting_epoch)
        vae.compile(optimizer=adam,loss = get_vae_loss)
        vae.fit([encoder_input_data,decoder_input_data],decoder_target_data,
              batch_size=batch_size,
              epochs=starting_epoch+1,
              validation_split=0.1,
              shuffle=cfg.shuffle_data,
              initial_epoch=starting_epoch,
              callbacks=[model_checkpoint, reduce_lr, stopping])




    ### ====================Testing====================
    video_data_test = pickle.load(open('./360video/data/shanghai_dataset_xyz_test.p','rb'),encoding='latin1')
    thu_tag=''
    ### data format 5
    # video_data_test = pickle.load(open('./360video/temp/tsinghua_after_bysec_interpolation/tsinghua_test_video_data_over_video.p','rb'),encoding='latin1')
    # thu_tag='_thu_'

    video_data_test = clip_xyz(video_data_test)
    datadb = video_data_test.copy()
    _video_db,_video_db_future,_video_db_future_input = util.get_data(datadb,pick_user=False)
    if cfg.input_mean_var:
        _video_db = util.get_gt_target_xyz(_video_db)

    def decode_sequence_fov(input_seq):
        last_location = input_seq[0,-1,:][np.newaxis,np.newaxis,:]
        if cfg.input_mean_var:
            last_mu_var = last_location
        else:            
            last_mu_var = util.get_gt_target_xyz(last_location)
        # last_mu_var = np.zeros_like(last_mu_var) ##zero out the decoder input
        decoded_sentence = vae.predict([input_seq,last_mu_var])
        return decoded_sentence


    def get_z_from_encoder(input_seq):
        z_mean00, z_log_var00, z00= encoder.predict(input_seq)
        return z_mean00




    gt_sentence_list = []
    decoded_sentence_list = []
    input_list=[]
    z_mean_list=[]
    for seq_index in range(_video_db.shape[0]):
        input_seq = _video_db[seq_index: seq_index + 1,:,:]
        decoded_sentence = decode_sequence_fov(input_seq)               
        decoded_sentence_list+=[decoded_sentence]
        gt_sentence = _video_db_future[seq_index: seq_index + 1,:,:]
        input_list.append(input_seq)
        gt_sentence_list+=[gt_sentence]
        decoder_target = util.get_gt_target_xyz(gt_sentence)
        # print('-')
        # print('Decoded sentence - decoder_target:', np.squeeze(np.array(decoded_sentence))[:,:3]-np.squeeze(decoder_target)[:,:3])

        z_mean_list.append(get_z_from_encoder(input_seq))



    pickle.dump(decoded_sentence_list,open('decoded_sentence'+thu_tag+tag+'.p','wb'))
    pickle.dump(gt_sentence_list,open('gt_sentence_list'+thu_tag+tag+'.p','wb'))
    print('Testing finished!')

    pickle.dump(z_mean_list,open('z_mean_list'+tag+'.p','wb'))







    grid_x = np.linspace(-4, 4, 30)
    grid_y = np.linspace(-4, 4, 30)[::-1]

    decoded_from_z=[]
    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            z_sample = np.array([[xi, yi]])
            decoded_from_z.append(decoder.predict([last_mu_var,z_sample]))

    pickle.dump(decoded_from_z,open('sampled_result_'+tag+'.p','wb'))









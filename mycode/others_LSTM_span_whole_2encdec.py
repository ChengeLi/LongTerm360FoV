"""
seq2seq without teacher forcing, 
- 2 encoder-decoders
- past encoder and future decoder for others 
- others' future convLSTM has its own loss function for self-reconstruction
- concat states with decoder LSTM and then predict
"""
def generate_fake_batch(mu,var):
    """generate new data for 1 second using predicted mean and variance"""
    temp = []
    for ii in range(batch_size):
        temp.append(np.random.normal(mu[ii], var[ii], fps*1))
    return temp


def _refeed_out_as_input(outputarray):
    """feed back output as input in decoders"""
    # TODO: fix me
    ux_temp = outputarray[:,0,0]
    uy_temp = outputarray[:,0,1]
    uz_temp = outputarray[:,0,2]
    varx_temp = outputarray[:,0,3]
    vary_temp = outputarray[:,0,4]
    varz_temp = outputarray[:,0,5]
    temp_newdata = np.stack((generate_fake_batch(ux_temp,varx_temp),
                    generate_fake_batch(uy_temp,vary_temp),
                    generate_fake_batch(uz_temp,varz_temp)),axis=-1)[:,np.newaxis,:,:].reshape((batch_size,1,-1))
    return temp_newdata




### ====================Graph def====================


###======convLSTM on others' past encoder======
other_lstm_encoder = ConvLSTM2D(filters=latent_dim, kernel_size=(num_user-1, 3),
                   input_shape=(1, num_user-1, fps, 3),
                   padding='same', return_sequences=True, return_state=True)
## span
# whole_span = max_encoder_seq_length+max_decoder_seq_length
# encoder_inputs_oth = Input(shape=(whole_span,num_user-1,fps,3))
encoder_inputs_oth = Input(shape=(max_encoder_seq_length,num_user-1,fps,3))
pst_outputs_sqns, pst_others_state_h, pst_others_state_c = other_lstm_encoder(encoder_inputs_oth)
states_oth = [pst_others_state_h, pst_others_state_c]

###======convLSTM on others' future decoder======
# Set up the decoder for others' branch
# decoder_inputs_oth = Input(shape=(max_encoder_seq_length,num_user-1,fps,3))
decoder_inputs_oth = Input(shape=(1,num_user-1,fps,3))
other_lstm_decoder = ConvLSTM2D(filters=latent_dim, kernel_size=(num_user-1, 3),
                   input_shape=(1, num_user-1, fps, 3),
                   padding='same', return_sequences=True, return_state=True)
pred_conv_lstm_dense = Dense(3,activation='tanh')
flatten_conv_lstm_state_dense = Dense(256)




###======target user encoder======
encoder_inputs = Input(shape=(None, num_encoder_tokens))
encoder = LSTM(latent_dim, return_state=True)
encoder_outputs, state_h, state_c = encoder(encoder_inputs)
states = [state_h, state_c]

###======target user decoder======
# Set up the decoder for target branch
# only process one timestep at a time.
decoder_inputs = Input(shape=(1, num_decoder_tokens))
# decoder_inputs = Input(shape=(1, num_encoder_tokens))
decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_dense = Dense(num_decoder_tokens,activation='tanh')



## utility layers
flatten_layer = Flatten()
expand_dim_layer = Lambda(lambda x: K.expand_dims(x,1))
Concatenatelayer = Concatenate(axis=1)
get_dim1_layer = Lambda(lambda x: x[:,0,:])

## concat states
all_outputs = []
all_outputs_oth= []
inputs = decoder_inputs
others_inputs = decoder_inputs_oth
for time_ind in range(max_decoder_seq_length):
    # Run the decoder on one timestep
    decoder_states, state_h, state_c = decoder_lstm(inputs,initial_state=states)
    fut_outputs_sqns, fut_others_state_h, fut_others_state_c = \
            other_lstm_decoder([others_inputs]+states_oth)
    # fut_outputs = slice_layer(1,time_ind,time_ind+1)(fut_outputs_sqns)
    convlstm_state = flatten_layer(get_dim1_layer(fut_outputs_sqns))
    convlstm_state = flatten_conv_lstm_state_dense(convlstm_state)
    concat_state = Concatenatelayer([get_dim1_layer(decoder_states),convlstm_state])
    outputs = decoder_dense(concat_state)
    outputs = expand_dim_layer(outputs)
    all_outputs.append(outputs)

    inputs = outputs
    # inputs = _refeed_out_as_input(outputs)
    states = [state_h, state_c]

    ### predict others' future
    outputs_oth = pred_conv_lstm_dense(fut_outputs_sqns)
    all_outputs_oth.append(outputs_oth)
    others_inputs = outputs_oth
    states_oth = [fut_others_state_h, fut_others_state_c]



# Concatenate all predictions
decoder_outputs = Lambda(lambda x: K.concatenate(x, axis=1))(all_outputs)
decoder_outputs_oth = Lambda(lambda x: K.concatenate(x, axis=1))(all_outputs_oth)


# Define and compile model as previously
model = Model([encoder_inputs, encoder_inputs_oth, decoder_inputs, decoder_inputs_oth],
              [decoder_outputs, decoder_outputs_oth])
model.compile(optimizer='Adam', loss=['mean_squared_error','mean_squared_error'],
              loss_weights=[1,1])



#### ====================data====================
## get target user data and other user's data
# _video_db_tar, _video_db_future_tar, _video_db_future_input_tar, \
# _video_db_oth,_video_db_future_oth,_video_db_future_input_oth = get_data(datadb,pick_user=True)

## load cached data
_video_db_tar = pickle.load(open('./cache/_video_db_tar.p','rb'))
_video_db_future_tar = pickle.load(open('./cache/_video_db_future_tar.p','rb'))
_video_db_future_input_tar = pickle.load(open('./cache/_video_db_future_input_tar.p','rb'))
_video_db_oth = pickle.load(open('./cache/_video_db_oth.p','rb'))
_video_db_future_oth = pickle.load(open('./cache/_video_db_future_oth.p','rb'))
_video_db_future_input_oth = pickle.load(open('./cache/_video_db_future_input_oth.p','rb'))


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

def get_whole_span(_video_db_oth):
    # get adjacent time periods: past+future as whole span
    # from  (N, 10, 47, 30, 3) to (N/2, 20, 47, 30, 3)
    half_index = _video_db_oth.shape[0]/2
    _video_db_oth_span = np.zeros((half_index,2*_video_db_oth.shape[1],_video_db_oth.shape[2],_video_db_oth.shape[3],_video_db_oth.shape[4]))
    for ii in range(half_index):
        temp = np.concatenate((_video_db_oth[ii,:],_video_db_oth[ii+1,:]))
        _video_db_oth_span[ii] = temp
    return _video_db_oth_span


_video_db_oth = _reshape_others_data(_video_db_oth)
_video_db_future_oth = _reshape_others_data(_video_db_future_oth)
# _video_db_future_input_oth = _reshape_others_data(_video_db_future_input_oth)
total_num_samples = _video_db_tar.shape[0]
num_testing_sample = int(0.15*total_num_samples)#use last 1000 as test

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
data_sanity_check(_video_db_tar,_video_db_future_tar,_video_db_future_input_tar)

### target user
encoder_input_data = _video_db_tar[:-num_testing_sample,:,:]
decoder_target_data = get_gt_target_xyz(_video_db_future_tar)[:-num_testing_sample,:,:]
decoder_input_data = get_gt_target_xyz(_video_db_tar)[:-num_testing_sample,-1,:][:,np.newaxis,:]
# decoder_input_data1 = get_gt_target_xyz(_video_db_future_input_tar)[:-num_testing_sample,0,:][:,np.newaxis,:]

### other users
others_pst_input_data = _video_db_oth[:-num_testing_sample,:]
decoder_target_data_oth = _video_db_future_oth[:-num_testing_sample,:]
others_decoder_input_data = _video_db_oth[:-num_testing_sample,-1,:][:,np.newaxis,:]
# others_decoder_input_data1 = _video_db_future_input_oth[:-num_testing_sample,0,:][:,np.newaxis,:]




### ====================Training====================
# model = load_model('convLSTM_endec_11_256tanh_epoch12-1.2859.h5')
tag = 'convLSTM_endec_11_256tanh_epoch'
model_checkpoint = ModelCheckpoint(tag+'{epoch:02d}-{val_loss:.4f}.h5', monitor='val_loss', save_best_only=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                 patience=3, min_lr=1e-6)
stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=0, mode='auto')
model.fit([encoder_input_data, others_pst_input_data, decoder_input_data, others_decoder_input_data],
            [decoder_target_data,decoder_target_data_oth],
          batch_size=batch_size,
          epochs=epochs,
          validation_split=0.2,
          shuffle=cfg.shuffle_data, initial_epoch=0,
          callbacks=[model_checkpoint, reduce_lr, stopping])




### ====================Testing====================
# model = load_model('convLSTM_endec_105_epoch55-0.1842.h5')
# model = load_model('convLSTM_endec_101_epoch13-0.1056.h5')
# model = load_model('backup_convLSTM_endec_11_epoch99-0.1952.h5')
# model = load_model('backup_convLSTM_endec_105_halftanh_epoch50-1.0356.h5')
# model = load_model('convLSTM_endec_101_halftanh_epoch108-1.0330.h5')
model = load_model('convLSTM_endec_11_256tanh_epoch71-1.2156.h5')
def decode_sequence_fov(input_seq,others_pst_input_seq):            
    # Encode the input as state vectors.
    last_location = input_seq[0,-1,:][np.newaxis,np.newaxis,:]
    last_mu_var = get_gt_target_xyz(last_location)
    last_location_oth = others_pst_input_seq[:,-1,:][:,np.newaxis,:]

    [decoded_sentence,decoded_sentence_oth] = model.predict([input_seq,others_pst_input_seq,last_mu_var,last_location_oth])

    return decoded_sentence,decoded_sentence_oth

gt_sentence_list = []
decoded_sentence_list = []
gt_sentence_oth_list = []
decoded_sentence_oth_list = []

for seq_index in range(total_num_samples-num_testing_sample,total_num_samples):
    input_seq = _video_db_tar[seq_index: seq_index + 1,:,:]
    others_pst_input_seq = _video_db_oth[seq_index: seq_index + 1,:]
    decoded_sentence,decoded_sentence_oth = decode_sequence_fov(input_seq,others_pst_input_seq)
    
    decoded_sentence_list+=[decoded_sentence]
    decoded_sentence_oth_list+=[decoded_sentence_oth]

    gt_sentence = _video_db_future_tar[seq_index: seq_index + 1,:,:]
    gt_sentence_list+=[gt_sentence]

    gt_sentence_oth = _video_db_future_oth[seq_index: seq_index + 1,:]
    gt_sentence_oth_list+=[gt_sentence_oth]


    # print('-')
    # decoder_target = get_gt_target_xyz(gt_sentence)
    # print('Decoded sentence - decoder_target:', np.squeeze(np.array(decoded_sentence))[:,:3]-np.squeeze(decoder_target)[:,:3])

pickle.dump(decoded_sentence_list,open('decoded_sentence.p','wb'))
pickle.dump(gt_sentence_list,open('gt_sentence_list.p','wb'))
print('Testing finished!')












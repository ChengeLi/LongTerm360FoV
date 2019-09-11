"""
Every user is sharing the same tar-only LSTM.
Implement an attention layer to learn a weight vector on all other users. 
Mix using this weight.
"""
import sys
if './360video/' not in sys.path:
    sys.path.insert(0, './360video/')




def onelayer_tar_seq2seq():
    if not cfg.input_mean_var:
        encoder_inputs = Input(shape=(None, num_encoder_tokens))
    else:
        encoder_inputs = Input(shape=(None, num_decoder_tokens))    
    encoder = LSTM(latent_dim, return_state=True)
    encoder_outputs, state_h, state_c = encoder(encoder_inputs)
    states = [state_h, state_c]
    decoder_inputs = Input(shape=(1, num_decoder_tokens))
    decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
    decoder_dense = Dense(num_decoder_tokens,activation='tanh')

    all_outputs = []
    inputs = decoder_inputs
    for _ in range(max_decoder_seq_length):
        decoder_states, state_h, state_c = decoder_lstm(inputs,
                                                 initial_state=states)
        outputs = decoder_dense(decoder_states)
        all_outputs.append(outputs)
        inputs = outputs
        states = [state_h, state_c]

    decoder_outputs = Lambda(lambda x: K.concatenate(x, axis=1))(all_outputs)
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

    decoder_model = Model([encoder_inputs, decoder_inputs],[decoder_states, state_h, state_c])
    return model, decoder_model



### pre-compute all hidden states of the other users for the target user.
num_user = 34
# model_oth #the whole seq2seq model for every other user
# decoder_oth #the decoder only model for every other user
model_oth, decoder_oth = onelayer_tar_seq2seq()
model_oth.load_weights('fctar_seqseq_shanghai_traintest_split_predmeanvar_Aug917-0.1119.h5')

for ii in range(num_user):
    decoder_states, state_h, state_c = decoder_oth.predict([input_seq,last_mu_var])




    decoder_states






# =================data==================
dataformat = 'format4' #shanghaitech
option='stride10_cut_head/'
_video_db_tar = util.load_h5('./cache/'+dataformat+'/train/'+option+'_video_db_tar.h5','_video_db_tar')
_video_db_future_tar = util.load_h5('./cache/'+dataformat+'/train/'+option+'_video_db_future_tar.h5','_video_db_future_tar')
_video_db_future_input_tar = util.load_h5('./cache/'+dataformat+'/train/'+option+'_video_db_future_input_tar.h5','_video_db_future_input_tar')
_video_db_oth = util.load_h5('./cache/'+dataformat+'/train/'+option+'_video_db_oth.h5','_video_db_oth')
_video_db_future_oth = util.load_h5('./cache/'+dataformat+'/train/'+option+'_video_db_future_oth.h5','_video_db_future_oth')
_video_db_future_input_oth = util.load_h5('./cache/'+dataformat+'/train/'+option+'_video_db_future_input_oth.h5','_video_db_future_input_oth')












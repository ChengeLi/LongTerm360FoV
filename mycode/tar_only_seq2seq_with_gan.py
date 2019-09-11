"""
target user only seq2seq model with GAN loss
"""

import sys,glob,io,random
if './360video/' not in sys.path:
    sys.path.insert(0, './360video/')
from mycode.config import cfg
import mycode.utility as util
import keras.backend as K
from keras.models import Sequential
from keras.models import Model
from keras.layers import Input, LSTM, Dense, Flatten, Activation, Lambda, Concatenate, Reshape
from keras.layers import Conv1D,MaxPooling1D,BatchNormalization
from keras import optimizers
from keras.losses import binary_crossentropy,mean_squared_error
import numpy as np
import h5py
from tqdm import tqdm
import _pickle as pickle
import datetime
import pdb
batch_size = 32
epochs = 200
latent_dim = 32

fps = 30
num_encoder_tokens = 3*fps
num_decoder_tokens = 6
max_encoder_seq_length = cfg.running_length
max_decoder_seq_length = cfg.predict_step
cfg.predict_mean_var = False
use_cnnD = True


class FOV_GAN(object):
    def __init__(self):
        self.D = None   # discriminator
        self.G = None   # generator
        self.AM = None  # adversarial model
        self.DM = None  # discriminator model
        self.ebD = None   
        self.ebAM = None 
        self.ebDM = None 

        self.encoder_inputs = Input(shape=(None, num_encoder_tokens))
        # self.decoder_inputs = Input(shape=(1, num_decoder_tokens))    
        self.decoder_inputs = Input(shape=(1, num_encoder_tokens))    

    def tar_seq2seq_generator(self):
        if self.G:
            return self.G
        ##2 layer encoder
        encoder1 = LSTM(latent_dim, stateful=cfg.stateful_across_batch, return_state=True, return_sequences=True)
        encoder1_outputs, state_h_1, state_c_1 = encoder1(self.encoder_inputs)

        encoder2 = LSTM(latent_dim, stateful=cfg.stateful_across_batch, return_state=True, return_sequences=True)
        encoder2_outputs, state_h_2, state_c_2 = encoder2(encoder1_outputs)

        encoder1_states = [state_h_1, state_c_1]
        encoder2_states = [state_h_2, state_c_2]

        ##2 layer decoder
        decoder1_states_inputs = encoder1_states
        decoder2_states_inputs = encoder2_states
        decoder_lstm1 = LSTM(latent_dim, stateful=cfg.stateful_across_batch, return_sequences=True, return_state=True)
        decoder_lstm2 = LSTM(latent_dim, stateful=cfg.stateful_across_batch, return_sequences=True, return_state=True)
        if cfg.predict_mean_var:
            decoder_dense = Dense(num_decoder_tokens,activation='tanh')
        else:
            decoder_dense = Dense(num_encoder_tokens,activation='tanh')

        all_outputs = []
        inputs = self.decoder_inputs
        for _ in range(max_decoder_seq_length):
            decoder1_outputs, state_decoder1_h, state_decoder1_c = decoder_lstm1(inputs,
                                                 initial_state=decoder1_states_inputs)
            decoder1_states_inputs = [state_decoder1_h, state_decoder1_c]
            decoder2_outputs, state_decoder2_h, state_decoder2_c = decoder_lstm2(decoder1_outputs,initial_state=decoder2_states_inputs)
            decoder2_states_inputs = [state_decoder2_h, state_decoder2_c]

            outputs = decoder_dense(decoder2_outputs)
            all_outputs.append(outputs)
            inputs = outputs

        self.decoder_outputs = Lambda(lambda x: K.concatenate(x, axis=1))(all_outputs)

        self.G = Model([self.encoder_inputs, self.decoder_inputs], self.decoder_outputs)
        print('Generator summary:')
        self.G.summary()
        return self.G


    def trj_descriminator(self):
        if self.D:
            return self.D
        self.D = Sequential()
        self.D.add(LSTM(latent_dim, stateful=False, return_state=False, return_sequences=True))#return the whole hidden sequnce
        self.D.add(Flatten())
        self.D.add(Dense(1))
        self.D.add(Activation('sigmoid'))
        # print('discriminator summary:')
        # self.D.summary()
        return self.D


    def trj_descriminator_cnn(self):
        if self.D:
            return self.D
        dilation_rate=1
        self.D = Sequential()
        self.D.add(Conv1D(latent_dim, 5, strides=1, padding='same', dilation_rate=dilation_rate, activation='relu'))
        # self.D.add(Conv1D(latent_dim*2, 5, strides=1, padding='same', dilation_rate=dilation_rate, activation='relu'))
        self.D.add(BatchNormalization())
        self.D.add(MaxPooling1D(pool_size=2, strides=None, padding='same'))
        # self.D.add(Conv1D(latent_dim*2, 5, strides=1, padding='same', dilation_rate=dilation_rate, activation='relu'))
        self.D.add(Conv1D(latent_dim, 5, strides=1, padding='same', dilation_rate=dilation_rate, activation='relu'))
        self.D.add(BatchNormalization())
        self.D.add(MaxPooling1D(pool_size=2, strides=None, padding='same'))

        self.D.add(Flatten())
        self.D.add(Dense(1))
        self.D.add(Activation('sigmoid'))
        return self.D


    def discriminator_model(self):
        if self.DM:
            return self.DM
        optimizer = optimizers.RMSprop(lr=0.0002, decay=6e-8)
        self.DM = Sequential()
        if use_cnnD:
            self.DM.add(self.trj_descriminator_cnn())
        else:        
            self.DM.add(self.trj_descriminator())
        self.DM.compile(loss='binary_crossentropy', optimizer=optimizer,\
            metrics=['accuracy'])
        return self.DM

    def adversarial_model(self):
        if self.AM:
            return self.AM
        optimizer = optimizers.RMSprop(lr=0.0001, decay=3e-8)
        self.decoder_outputs = self.tar_seq2seq_generator()([self.encoder_inputs, self.decoder_inputs])
        pred_trj = Concatenate(axis=1)([self.encoder_inputs,self.decoder_outputs])
        pred_trj = Reshape((2*max_decoder_seq_length,num_encoder_tokens))(pred_trj)
        if use_cnnD:
            self.score = self.trj_descriminator_cnn()(pred_trj)
        else:
            self.score = self.trj_descriminator()(pred_trj)
        self.AM = Model([self.encoder_inputs, self.decoder_inputs],[self.decoder_outputs,self.score])
        self.AM.compile(loss=['mean_squared_error','binary_crossentropy'], 
                        loss_weights=[1,1],\
                        optimizer=optimizer,\
                        metrics=['mae','accuracy'])
        return self.AM



    def trj_descriminator_energy(self,ebD_inputs):
        """energy based GAN discriminator"""
        if self.ebD:
            return self.ebD

        # ebD_inputs = Input(shape=(None, num_encoder_tokens))
        ##2 layer encoder
        encoder1 = LSTM(latent_dim, stateful=cfg.stateful_across_batch, return_state=True, return_sequences=True)
        encoder2 = LSTM(num_encoder_tokens, stateful=cfg.stateful_across_batch, return_state=True, return_sequences=True)
        encoder1_outputs, state_h_1, state_c_1 = encoder1(ebD_inputs)
        encoder2_outputs, state_h_2, state_c_2 = encoder2(encoder1_outputs)

        ##2 layer decoder
        decoder1_states_inputs = [state_h_1, state_c_1]
        decoder2_states_inputs = [state_h_2, state_c_2]
        decoder_lstm1 = LSTM(latent_dim, stateful=cfg.stateful_across_batch, return_state=False,return_sequences=True)
        decoder_lstm2 = LSTM(num_encoder_tokens, stateful=cfg.stateful_across_batch, return_state=False,return_sequences=True)
        
        decoder1_outputs = decoder_lstm1(encoder2_outputs,
                                             initial_state=decoder1_states_inputs)
        decoder2_outputs = decoder_lstm2(decoder1_outputs,
                                            initial_state=decoder2_states_inputs)

        # self.ebD = Model(ebD_inputs,decoder2_outputs)
        # return self.ebD
        return decoder2_outputs

    def discriminator_energy_model(self):
        if self.ebDM:
            return self.ebDM
        optimizer = optimizers.RMSprop(lr=0.0002, decay=6e-8)
        ebD_inputs_true = Input(shape=(None, num_encoder_tokens))
        ebD_inputs_fake = Input(shape=(None, num_encoder_tokens))
        ebD_out_true = self.trj_descriminator_energy(ebD_inputs_true)
        ebD_out_fake = self.trj_descriminator_energy(ebD_inputs_fake)

        self.ebDM = Model([ebD_inputs_true,ebD_inputs_fake],[ebD_out_true,ebD_out_fake])
        self.ebDM.compile(loss=['mean_squared_error','mean_squared_error'],
            loss_weights=[1,-1],\
            optimizer=optimizer,\
            metrics=['accuracy'])
        return self.ebDM

    def adversarial_energy_model(self):
        if self.ebAM:
            return self.ebAM
        optimizer = optimizers.RMSprop(lr=0.0001, decay=3e-8)
        self.decoder_outputs = self.tar_seq2seq_generator()([self.encoder_inputs, self.decoder_inputs])
        self.reconstructed_pred = self.trj_descriminator_energy(self.decoder_outputs)
        self.ebAM = Model([self.encoder_inputs, self.decoder_inputs],[self.decoder_outputs,self.reconstructed_pred])
        self.ebAM.compile(loss=['mean_squared_error','mean_squared_error'], 
                        loss_weights=[1,1],\
                        optimizer=optimizer,\
                        metrics=['mae','accuracy'])
        return self.ebAM


def train_with_history():
    # Fit the model
    history = model.fit(X, Y, validation_split=0.33, epochs=150, batch_size=10, verbose=0)
    # list all data in history
    print(history.history.keys())
    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()






if __name__ == '__main__':
    FOV_GAN = FOV_GAN()
    use_energy_based = False
    if not use_energy_based:
        discriminator = FOV_GAN.discriminator_model()
        adversarial = FOV_GAN.adversarial_model()
    else:
        discriminator = FOV_GAN.discriminator_energy_model()
        adversarial = FOV_GAN.adversarial_energy_model()
    generator = FOV_GAN.tar_seq2seq_generator()
    # tag = 'G2D1_concatpastfut_CNN_d_oct16'
    # tag = 'G10D1_concatpastfut_CNN_d_oct17'
    tag = 'G10D1_concatpastfut_smallCNN_d_shuffle_oct17'


    # =====data====
    #### load cached data format4 
    dataformat = 'format4' #shanghaitech
    option='stride10_cut_head/'
    _video_db_tar = util.load_h5('./cache/'+dataformat+'/train/'+option+'_video_db_tar.h5','_video_db_tar')
    _video_db_future_tar = util.load_h5('./cache/'+dataformat+'/train/'+option+'_video_db_future_tar.h5','_video_db_future_tar')
    _video_db_future_input_tar = util.load_h5('./cache/'+dataformat+'/train/'+option+'_video_db_future_input_tar.h5','_video_db_future_input_tar')

    # training
    epochs=70
    batch_size=64
    g_steps = 10
    d_steps = 1
    record_loss=True
    if record_loss:
        d_losses=[]
        a_losses=[]
    for epoch in range(epochs):
        for iii in range(0,_video_db_tar.shape[0], batch_size):
            batch_ind = np.random.choice(_video_db_tar.shape[0]-batch_size, 1)[0]
            d_steps_left = d_steps
            g_steps_left = g_steps
            past_trj = _video_db_tar[batch_ind:batch_ind+batch_size]
            future_trj = _video_db_future_tar[batch_ind:batch_ind+batch_size]
            decoder_input_trj = _video_db_future_input_tar[batch_ind:batch_ind+batch_size,0,:][:,np.newaxis,:]
            real_batch_size = past_trj.shape[0]
            for batch in range(d_steps+g_steps):
                if d_steps_left > 0:
                    pred_trj = generator.predict([past_trj,decoder_input_trj])
                    true_trj =  np.concatenate((past_trj, future_trj),axis=1)#20sec
                    fake_trj =  np.concatenate((past_trj, pred_trj),axis=1)#20sec
                    if not use_energy_based:
                        x = np.concatenate((true_trj, fake_trj),axis=0)
                        y = np.ones([2*real_batch_size, 1])
                        y[real_batch_size:, :] = 0
                        d_loss = discriminator.train_on_batch(x, y)
                        log_mesg = "%d: D loss: %f, acc: %f\n" % (epoch, d_loss[0], d_loss[1])
                    else:
                        d_loss = discriminator.train_on_batch([true_trj,fake_trj],[true_trj,fake_trj])
                        log_mesg = "%d: D 2 mse loss: %f, acc: %f,%f\n" % (epoch, d_loss[0], d_loss[3],d_loss[4])
                    d_losses.append(d_loss)
                    d_steps_left -= 1
                elif g_steps_left > 0:
                    if not use_energy_based:
                        y = np.ones([real_batch_size, 1])
                    else:
                        y = future_trj
                    a_loss = adversarial.train_on_batch([past_trj,decoder_input_trj], [future_trj,y])
                    a_losses.append(a_loss)
                    g_steps_left -= 1
                    if d_steps==0:
                        log_mesg=''
                    log_mesg = log_mesg+"A loss: "+str(a_loss[:3])+"\n"
                    # if epoch%20==0:
                    #     log_mesg = log_mesg+"  mae, acc: "+str(a_loss[3:]) 
                print(log_mesg)

    def save_model(epoch,tag=datetime.datetime.now().strftime("%Y-%m-%d_%H:%M")):
        if record_loss:
            pickle.dump(a_losses,open('a_losses_'+tag+'.p','wb'))
            pickle.dump(d_losses,open('d_losses_'+tag+'.p','wb'))
        discriminator.save_weights(tag+'_discriminator_'+str(epoch)+'.h5')
        adversarial.save_weights(tag+'_adversarial_'+str(epoch)+'.h5')
        generator.save_weights(tag+'_generator_'+str(epoch)+'.h5')
    save_model(epoch,tag)




    # ====testing data====
    _video_db_tar = util.load_h5('./cache/'+dataformat+'/test/'+option+'_video_db_tar.h5','_video_db_tar')
    _video_db_future_tar = util.load_h5('./cache/'+dataformat+'/test/'+option+'_video_db_future_tar.h5','_video_db_future_tar')
    _video_db_future_input_tar = util.load_h5('./cache/'+dataformat+'/test/'+option+'_video_db_future_input_tar.h5','_video_db_future_input_tar')
    def test(batch_size=256):
        prediction=[]
        gt_out=[]
        for batch_ind in range(0,_video_db_tar.shape[0], batch_size):
            past_trj = _video_db_tar[batch_ind:batch_ind+batch_size]
            decoder_input_trj = _video_db_future_input_tar[batch_ind:batch_ind+batch_size,0,:][:,np.newaxis,:]

            pred_trj = generator.predict([past_trj,decoder_input_trj])
            prediction.append(pred_trj)
            gt_trj = _video_db_future_tar[batch_ind:batch_ind+batch_size]
            gt_out.append(gt_trj)
        return prediction,gt_out
    prediction,gt_out = test()
    
    pickle.dump(prediction,open('decoded_sentence'+tag+'.p','wb'))
    pickle.dump(gt_out,open('gt_sentence_list'+tag+'.p','wb'))
    print('Testing finished!')


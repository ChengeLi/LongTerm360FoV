"""
dataLayer generates pure future batch_y. e.g: 
batch_x from 0-9
batch_y from 10-19

This dataLayer2 generates one-step shifted future batch_y. e.g: 
batch_x from 0-9
batch_y from 1-10
"""

# TODO: only changed line 257:                                             
# db_index[ii]*self.data_chunk_stride+1:db_index[ii]*self.data_chunk_stride+1+self.predict_len]
#need to change for other format: cos, sin, CNN format etc. 

import numpy as np
import pdb
from mycode.config import cfg
class DataLayer(object):
    def __init__(self, datadb, random=False, is_test=False):
        """Set the datadb to be used by this layer during training."""
        self.is_test = is_test
        # Also set a random flag
        self._random = random
        if cfg.use_mixed_dataset:
            self._video_ind = np.random.choice([s for s in range(37)])
            self._get_train_test_user_id()
        else:
            self._video_ind = np.random.choice([s for s in datadb.keys() if s!=cfg.test_video_ind])
            if is_test:
                if cfg.test_video_ind:
                    self._video_ind = cfg.test_video_ind
                else:
                    self.test_video_ind=0
                    self._video_ind = datadb.keys()[self.test_video_ind]
        self.running_length = cfg.running_length
        self.predict_len = cfg.predict_len
        self.predict_step = cfg.predict_step
        self.data_chunk_stride = cfg.data_chunk_stride
        self.fps = 30
        if cfg.process_in_seconds:
            #process in seconds
            self.running_length = cfg.running_length*self.fps
            self.predict_len = cfg.predict_len*self.fps
            self.predict_step = cfg.predict_step*self.fps #during testing
            self.data_chunk_stride = cfg.data_chunk_stride*self.fps  
            
        self._init_2(datadb)


    def _init_2(self,datadb):
        # select a new video, shuffle 
        if cfg.use_cos_sin:
            self.per_video_db = {}
            self.per_video_db['cos'] = datadb[self._video_ind]['cos_yaw']
            self.per_video_db['sin'] = datadb[self._video_ind]['sin_yaw']
            self._num_db = int(self.per_video_db['cos'].shape[1]/self.running_length)
            self._num_user = int(self.per_video_db['cos'].shape[0])
        elif cfg.use_xyz:
            self.data_dim = 3
            self.per_video_db = np.stack((datadb[self._video_ind]['x'],datadb[self._video_ind]['y'],datadb[self._video_ind]['z']),axis=-1)
            self.per_video_db = np.array(self.per_video_db)
            if cfg.use_overlapping_chunks:
                self._num_db = int((self.per_video_db.shape[1]-2*self.running_length)/self.data_chunk_stride)+1                
            else:
                self._num_db = int(self.per_video_db.shape[1]/self.running_length)
            self._num_user = self.per_video_db.shape[0]        
        elif cfg.use_yaw_pitch_roll:
            self.data_dim = 2
            # self.per_video_db = datadb[self._video_ind]['raw_yaw']
            self.per_video_db = np.stack((datadb[self._video_ind]['raw_yaw'],datadb[self._video_ind]['raw_pitch']),axis=-1)
            self.per_video_db = np.array(self.per_video_db)
            self._num_db = int(self.per_video_db.shape[1]/self.running_length)
            self._num_user = self.per_video_db.shape[0]
        elif cfg.use_phi_theta:
            self.data_dim = 2
            if cfg.predict_eos:
                self.data_dim += 1
                self.per_video_db = np.stack((datadb[self._video_ind]['theta'],datadb[self._video_ind]['phi'],datadb[self._video_ind]['eos']),axis=-1)
            else:
                self.per_video_db = np.stack((datadb[self._video_ind]['theta'],datadb[self._video_ind]['phi']),axis=-1)
            self.per_video_db = np.array(self.per_video_db)
            # self._num_db = int(self.per_video_db.shape[1]/self.running_length)
            # self._num_db = int(self.per_video_db.shape[1]-cfg.running_length)/self.data_chunk_stride
            self._num_db = int(self.per_video_db.shape[1]-self.running_length-self.predict_len)/self.data_chunk_stride
            self._num_user = self.per_video_db.shape[0]

        self._shuffle_datadb_inds()


    def _get_train_test_user_id(self):
        if self.is_test:
            self.user_ind = np.load('./data/merged_dataset_user_idx/test_video'+str(self._video_ind)+'.npy')
        else:
            self.user_ind = np.load('./data/merged_dataset_user_idx/train_video'+str(self._video_ind)+'.npy')


    def _shuffle_datadb_inds(self):
        """Randomly permute the training datadb."""
        # If the random flag is set, 
        # then the database is shuffled according to system time
        # Useful for the validation set
        if self._random:
            st0 = np.random.get_state()
            millis = int(round(time.time() * 1000)) % 4294967295
            np.random.seed(millis)
      
        if cfg.shuffle_data:
            self._perm = np.random.permutation(np.arange(self._num_db))
            # Restore the random state
            if self._random:
                np.random.set_state(st0)
        else:
           self._perm = np.arange(self._num_db)
        self._cur = 0





    def reshape_data_in_seconds(self,batch_x,batch_y,others_future):
        # change the shape from frame_length*3 to (frame_length/30)*(30*3)
        batch_x = np.reshape(batch_x,(batch_x.shape[0],batch_x.shape[1],batch_x.shape[2]/self.fps,self.fps*self.data_dim))
        batch_y = np.reshape(batch_y,(batch_y.shape[0],batch_y.shape[1]/self.fps,self.fps*self.data_dim))
        others_future = np.reshape(others_future,(others_future.shape[0],others_future.shape[1],others_future.shape[2]/self.fps,self.fps*self.data_dim))
        
        if cfg.own_history_only:
            batch_x = batch_x[:,0,:,:]
        return batch_x,batch_y,others_future


    def another_video(self,datadb):
        if self.is_test:
            self._video_ind = datadb.keys()[self.test_video_ind] #sequential
            self.test_video_ind+=1
            if self.test_video_ind==len(datadb.keys()):
                print('====You have finished all testing dataset, restart from beginning again!!!====')
        else:
            self._video_ind = np.random.choice([s for s in datadb.keys() if s!=cfg.test_video_ind])
        if cfg.use_mixed_dataset:
            self._get_train_test_user_id()
        if self.is_test:
            print('change to video',self._video_ind) 
        self._init_2(datadb) 
    
    def _get_next_minibatch_inds(self,datadb,batch_size):
        """Return the index (starting from where) for the next minibatch."""
        if self._cur + batch_size >= self._num_db:
            # after going over one video, select another one
            if cfg.use_more_video:
                self.another_video(datadb)
                while len(self._perm)==0:#video is even less than 20 seconds
                    print('video '+self._video_ind+' is even less than 20 seconds')
                    pdb.set_trace()
                    self.another_video(datadb)
                db_index = self._perm[self._cur:self._cur + batch_size]

        while len(db_index)<batch_size:
            db_index = list(db_index)
            db_index.append(np.random.choice(db_index))
        # while len(db_index)<batch_size:
        #     #one video is too short for one batch
        ### TODO: also need to handle self.per_video_db from multiple videos
        #     self._video_ind = np.random.choice([s for s in datadb.keys() if s!=cfg.test_video_ind])
        #     self._init_2(datadb) 
        #     pdb.set_trace()
        #     print('concat next video',self._video_ind)
        #     db_index+=self._perm[self._cur:self._cur + batch_size-len(db_index)]

        self._cur += batch_size
        return db_index


    def get_batch_CNN_format(self,db_index,target_uer_ID):
        """get a minibatch in n_usr*(2*t*fps)*3 format"""
        this_x = self.per_video_db[:,db_index*self.data_chunk_stride:db_index*self.data_chunk_stride+2*self.running_length,:].copy()
        
        # put target row in the bottom
        target_row = this_x[target_uer_ID,:]
        this_x = np.delete(this_x,target_uer_ID, axis=0)
        this_x = np.vstack((this_x,target_row[np.newaxis,:]))
        this_y = target_row[self.running_length:,:]
        this_x[-1,self.running_length:]=0
        return this_x[np.newaxis,:],this_y[np.newaxis,:]


    def get_batch_convLSTM_format(self,db_index,target_uer_ID):
        """get a minibatch in timestamp*47*fps*3"""
        this_x = self.per_video_db[:,db_index*self.data_chunk_stride:db_index*self.data_chunk_stride+self.running_length,:].copy()
        this_x_future = self.per_video_db[:,db_index*self.data_chunk_stride+self.running_length:db_index*self.data_chunk_stride+self.running_length+self.predict_step,:].copy()
        
        # delte target row
        this_x = np.delete(this_x,target_uer_ID, axis=0)
        target_row = this_x_future[target_uer_ID,:]
        this_x_future = np.delete(this_x_future,target_uer_ID, axis=0)

        # reshape into timestamp*47*fps*3"""
        this_x_per_second_past = this_x.reshape(self._num_user-1,cfg.running_length,self.fps,self.data_dim)
        this_x_per_second_past = this_x_per_second_past.transpose((1,0,2,3))

        this_x_per_second_future = this_x_future.reshape(self._num_user-1,cfg.running_length,self.fps,self.data_dim)
        this_x_per_second_future = this_x_per_second_future.transpose((1,0,2,3))
        this_y = target_row
        return this_x_per_second_past[np.newaxis,:],this_x_per_second_future[np.newaxis,:],this_y[np.newaxis,:]


    def _get_target_user_id(self):
        if cfg.fix_target_user:
            _target_user = cfg.target_uer_ID
        else:
            if cfg.use_mixed_dataset:
                _target_user = np.random.choice(self.user_ind)
            else:
                _target_user = np.random.randint(0,self._num_user)
        return _target_user

    def _get_next_minibatch(self,datadb,batch_size,format=None):
        _target_user = np.zeros(batch_size).astype('int')
        db_index = self._get_next_minibatch_inds(datadb,batch_size)
        """fetch a batch from tsinghua dataset"""
        if format == 'CNN':
            batch_x = np.zeros((1,self._num_user,self.running_length*2,self.data_dim))
            batch_y = np.zeros((1,self.running_length,self.data_dim))
            for ii in range(batch_size):
                _target_user[ii] = self._get_target_user_id()
                this_x,this_y = self.get_batch_CNN_format(db_index[ii],_target_user[ii])
                batch_x = np.vstack((batch_x,this_x))
                batch_y = np.vstack((batch_y,this_y))
            batch_x = batch_x[1:,:,:,:]
            batch_y = batch_y[1:,:,:]
            return batch_x,batch_y

        elif format == 'convLSTM':
            batch_x_past = np.zeros((1,cfg.running_length,self._num_user-1,self.fps,self.data_dim))
            batch_x_future = np.zeros((1,cfg.running_length,self._num_user-1,self.fps,self.data_dim))
            batch_y = np.zeros((1,self.predict_step,self.data_dim))
            for ii in range(batch_size):
                _target_user[ii] = self._get_target_user_id()
                this_x_per_second_past,this_x_per_second_future,this_y = self.get_batch_convLSTM_format(db_index[ii],_target_user[ii])
                batch_x_past = np.vstack((batch_x_past,this_x_per_second_past))
                batch_x_future = np.vstack((batch_x_future,this_x_per_second_future))
                batch_y = np.vstack((batch_y,this_y))
            batch_x_past = batch_x_past[1:,:,:,:,:]
            batch_x_future = batch_x_future[1:,:,:,:,:]
            batch_y = batch_y[1:,:,:]
            return batch_x_past,batch_x_future,batch_y


        else:
            if cfg.use_cos_sin:
                _target_user = np.random.randint(0,self._num_user)  
                temp_cos = self.per_video_db['cos'][:,db_index*self.data_chunk_stride:db_index*self.data_chunk_stride+self.running_length].copy()
                temp_sin = self.per_video_db['sin'][:,db_index*self.data_chunk_stride:db_index*self.data_chunk_stride+self.running_length].copy()
                batch_x = np.vstack((temp_cos[np.newaxis,:],temp_sin[np.newaxis,:]))
                batch_x = batch_x.transpose((0,2,1)) #2*5000*48
                batch_x[:,self.running_length-self.predict_len:self.running_length,_target_user] = 0          
                temp_cos = self.per_video_db['cos'][_target_user,db_index*self.data_chunk_stride:db_index*self.data_chunk_stride+self.running_length]
                temp_sin = self.per_video_db['sin'][_target_user,db_index*self.data_chunk_stride:db_index*self.data_chunk_stride+self.running_length]
                batch_y = np.vstack((temp_cos[np.newaxis,:],temp_sin[np.newaxis,:]))

            else:
                # fetch batch_x
                if cfg.own_history_only:
                    batch_x = np.zeros((1,1,self.running_length,self.data_dim))
                else:
                    if cfg.include_own_history:
                        batch_x = np.zeros((1,self._num_user,self.running_length,self.data_dim))
                    else:
                        batch_x = np.zeros((1,self._num_user-1,self.running_length,self.data_dim))
                for ii in range(batch_size):
                    _target_user[ii] = self._get_target_user_id()
                    try:
                        this_x = self.per_video_db[:,db_index[ii]*self.data_chunk_stride:db_index[ii]*self.data_chunk_stride+self.running_length,:].copy()
                    except:
                        pdb.set_trace()
                    if cfg.stuff_zero:
                        this_x[_target_user[ii],self.running_length-self.predict_len:self.running_length,:] = 0
                    elif cfg.stuff_last:
                        this_x[_target_user[ii],self.running_length-self.predict_len:self.running_length,:] = this_x[_target_user[ii],self.running_length-self.predict_len,:]
                    
                    target_row = this_x[_target_user[ii],:]
                    if cfg.own_history_only:
                        batch_x = np.vstack((batch_x,target_row[np.newaxis,np.newaxis,:,:]))                        
                    else:
                        if cfg.include_own_history:
                            # put target row in the bottom
                            this_x = np.delete(this_x,_target_user[ii], axis=0)
                            this_x = np.vstack((this_x,target_row[np.newaxis,:]))
                        else:
                            this_x = np.delete(this_x,_target_user[ii], axis=0)
                        batch_x = np.vstack((batch_x,this_x[np.newaxis,:]))

                batch_x = batch_x[1:,:,:,:]

                # fetch batch_y
                if cfg.has_reconstruct_loss:
                    batch_y = np.zeros((1,cfg.self.running_length,self.data_dim))
                    others_future = np.zeros((1,self._num_user-1,cfg.self.running_length,self.data_dim))
                else:
                    batch_y = np.zeros((1,self.predict_len,self.data_dim))
                    batch_y_further = np.zeros((1,self.predict_step,self.data_dim))#predict_step = multiple*predict_len
                    others_future = np.zeros((1,self._num_user-1,self.predict_len,self.data_dim))
                    others_future_further = np.zeros((1,self._num_user-1,self.predict_step,self.data_dim))
                # rep_last_loc = np.zeros((1,self.predict_len))
                for ii in range(batch_size):
                    if cfg.has_reconstruct_loss:
                        batch_y = np.vstack((batch_y,self.per_video_db[_target_user[ii],
                                                        db_index[ii]*self.data_chunk_stride:db_index[ii]*self.data_chunk_stride+self.running_length]
                                                        .reshape((-1,self.running_length,self.data_dim))))
                    else:
                        batch_y = np.vstack((batch_y,self.per_video_db[_target_user[ii],
                                            db_index[ii]*self.data_chunk_stride+1:db_index[ii]*self.data_chunk_stride+1+self.predict_len]
                                            .reshape((-1,self.predict_len,self.data_dim))))
                        y_further = self.per_video_db[_target_user[ii],
                                   db_index[ii]*self.data_chunk_stride+1:db_index[ii]*self.data_chunk_stride+1+self.predict_step]
                        try:
                            batch_y_further = np.vstack((batch_y_further,y_further.reshape((-1,self.predict_step,self.data_dim))))
                        except: #don't have enough future data lasting for cfg.predict_step
                            if len(y_further)<self.predict_step:
                                # stack (1.123), bc the range should be [-1,1]
                                y_further = np.vstack((y_further,np.ones((self.predict_step-len(y_further),self.data_dim))*(1.123)))
                                batch_y_further = np.vstack((batch_y_further,y_further.reshape((-1,self.predict_step,self.data_dim))))


                    others_future_db = np.delete(self.per_video_db,_target_user[ii],axis=0)
                    if cfg.has_reconstruct_loss:
                        others_future_temp = others_future_db[:,db_index[ii]*self.data_chunk_stride:db_index[ii]*self.data_chunk_stride+self.running_length,:][np.newaxis,:]
                    else:
                        others_future_temp = others_future_db[:,db_index[ii]*self.data_chunk_stride+self.running_length:db_index[ii]*self.data_chunk_stride+self.running_length+self.predict_len][np.newaxis,:]
                        others_future_further_temp = others_future_db[:,db_index[ii]*self.data_chunk_stride+self.running_length:db_index[ii]*self.data_chunk_stride+self.running_length+self.predict_step][np.newaxis,:]
                    others_future = np.vstack((others_future,others_future_temp))
                    try:
                        others_future_further = np.vstack((others_future_further,others_future_further_temp))
                    except:
                        if others_future_further_temp.shape[2]<self.predict_step:
                            # stack (1.123), bc the range should be [-1,1]
                            others_future_further_temp = np.concatenate((others_future_further_temp,np.ones((1,self._num_user-1,self.predict_step-others_future_further_temp.shape[2],self.data_dim))*1.123),axis=2)
                            others_future_further = np.vstack((others_future_further,others_future_further_temp))

                    # last_val = np.array([self.per_video_db[_target_user[ii],db_index[ii]*self.data_chunk_stride+self.running_length-self.predict_len-1]]*self.predict_len).reshape((-1,self.predict_len))
                    # rep_last_loc = np.vstack((rep_last_loc,last_val))
                batch_y = batch_y[1:,:,:]
                batch_y_further = batch_y_further[1:,:,:]
                others_future = others_future[1:,:,:]
                others_future_further = others_future_further[1:,:,:]
                # rep_last_loc = rep_last_loc[1:,:]
            if cfg.process_in_seconds:
                batch_x,batch_y,others_future = self.reshape_data_in_seconds(batch_x,batch_y,others_future)
            else:
                batch_x = batch_x[:,0,:,:]
            return batch_x,batch_y,others_future,batch_y_further,db_index,others_future_further


    # def _insert_end_of_stroke(self):
    #     """create fake end of stroke signals
    #     eos will be set to 1 if:
    #     1) in the beginning and end of the whole trj
    #     2) whenever the theta has larger than pi changes 
    #        (~2pi change from pi to -pi or vice versa)
    #     """
    #     self.eos = np.zeros((self.per_video_db.shape[0],self.per_video_db.shape[1],1))
    #     self.eos[:,0,0]=1
    #     self.eos[:,-1,0]=1
    #     if cfg.use_residual_input and not cfg.normalize_residual:
    #         self.eos[:,1:,0][self.per_video_db[:,1:,0]>np.pi]=1
    #         self.eos[:,1:,0][self.per_video_db[:,1:,0]<-np.pi]=1
    #         ##change from diff to real location at the eos==1 
    #         pdb.set_trace()
    #         self.per_video_db[:,1:,:][self.eos[:,1:,0]==1] = original_per_video_db[:,1:,:][self.eos[:,1:,0]==1]
    #     elif not cfg.use_residual_input:
    #         diff = self.per_video_db[:,1:,0]-self.per_video_db[:,:-1,0]
    #         self.eos[:,1:,0][diff>np.pi]=1
    #         self.eos[:,1:,0][diff<-np.pi]=1
        

    #     self.per_video_db = np.concatenate([self.per_video_db,self.eos],axis=-1)







"""
keras data_generator
Datasets including others and saliency inputs are too large, cannot directly loading them all
Also, trajectory data is repeating in time, while saliency doesn't
"""
from collections import OrderedDict
import numpy as np
import sys
if '/scratch/wz1219/FoV' not in sys.path:
    sys.path.insert(0, '/scratch/wz1219/FoV')
from code.config import cfg
import code.utility as util
import pdb
import cPickle as p
fps = cfg.fps

def _prepare_data(per_video_db_tar, per_video_db_future_tar, per_video_db_future_input_tar,
                per_video_db_oth, per_video_db_future_oth, per_video_db_future_input_oth,
                phase='train'):
    
    def _reshape_others_data(_video_db_oth):
        ## to match Input shape: others_fut_inputs
        _video_db_oth = _video_db_oth.transpose((1,2,0,3,4,5))
        _video_db_oth = _video_db_oth.reshape((_video_db_oth.shape[0],_video_db_oth.shape[1],_video_db_oth.shape[2],
                        fps,18,36))
        return _video_db_oth

    #print('shape of per_video_db_oth:',per_video_db_oth.shape)
    #print('shape of per_video_db_tar:',per_video_db_tar.shape)
    per_video_db_oth = _reshape_others_data(per_video_db_oth)
    per_video_db_future_oth = _reshape_others_data(per_video_db_future_oth)
    per_video_db_tar = per_video_db_tar.reshape((per_video_db_tar.shape[0],per_video_db_tar.shape[1],fps,18,36))
    per_video_db_future_tar = per_video_db_future_tar.reshape((per_video_db_tar.shape[0],per_video_db_tar.shape[1],fps,18,36))
    per_video_db_future_input_tar = per_video_db_future_input_tar.reshape((per_video_db_tar.shape[0],per_video_db_tar.shape[1],fps,18,36))
    
    #print('other data shape: ',per_video_db_oth.shape)
    #print('other data shape: ',per_video_db_future_oth.shape)
    #print('target user data shape: ',per_video_db_tar.shape)
    #print('target user data shape: ',per_video_db_future_tar.shape)

    if cfg.input_mean_var:
        ### target user
        encoder_input_data = util.get_gt_target_xyz(per_video_db_tar)
        # ### other users
        others_fut_input_data = util.get_gt_target_xyz_oth(per_video_db_future_oth)
        if not cfg.teacher_forcing:
            decoder_input_data = encoder_input_data[:,-1,:][:,np.newaxis,:]
        else:
            decoder_input_data = util.get_gt_target_xyz(per_video_db_future_input_tar)
    else:
        ### target user
        #per_video_db_tar = per_video_db_tar.reshape((per_video_db_tar.shape[0],per_video_db_tar.shape[1],-1))
        encoder_input_data = per_video_db_tar
        print(encoder_input_data.shape)
        # decoder_input_data = _video_db_tar[:,np.newaxis,np.newaxis,:]
        decoder_input_data = encoder_input_data[:,-1,:][:,np.newaxis,:]
        ### other users
        others_fut_input_data = per_video_db_future_oth
    

    if cfg.predict_mean_var:
        decoder_target_data = util.get_gt_target_xyz(per_video_db_future_tar)#predict mean/var
    elif not cfg.predict_mean_var or  phase=='test':
        decoder_target_data = per_video_db_future_tar#[:,:,np.newaxis,:,:]#predict raw
        decoder_target_data = np.transpose(decoder_target_data,(0,1,4,3,2))
    encoder_input_data = np.transpose(encoder_input_data,(0,1,4,3,2))
    decoder_input_data = np.transpose(decoder_input_data,(0,1,4,3,2))
    #print('encoder_input_data shape: ',encoder_input_data.shape)
    #print('decoder_target_data shape: ',decoder_target_data.shape)
    #print('decoder_input_data shape: ',decoder_input_data.shape)
    # print('others_fut_input_data shape: ',others_fut_input_data.shape)
    return encoder_input_data,decoder_input_data,decoder_target_data#,others_fut_input_data



def get_data_per_vid_per_target(per_video_db,_video_ind,_target_user,num_user,stride=cfg.data_chunk_stride):
    per_video_db_tar = per_video_db[_target_user,:][np.newaxis,:,:]
    per_video_db_oth = np.delete(per_video_db,_target_user,axis=0)
    #print('shape of per_video_db_tar is: ',per_video_db_tar.shape)
    #print('shape of per_video_db_oth is: ',per_video_db_oth.shape)

    if per_video_db_oth.shape[0]<num_user-1:
        for concat_ind in range(per_video_db_oth.shape[0],num_user-1):
            duplicate_ind = np.random.randint(per_video_db_oth.shape[0])
            # print('duplicate_ind:',duplicate_ind)
            temp = per_video_db_oth[duplicate_ind,:][np.newaxis,:]
            per_video_db_oth = np.concatenate([per_video_db_oth,temp],axis=0)
    try:
        assert per_video_db_oth.shape[0]==num_user-1
    except:
        pdb.set_trace()
    # print('has %d other users'%(num_user-1))

    per_video_db_tar, per_video_db_future_tar, per_video_db_future_input_tar = util.reshape2second_stacks(per_video_db_tar,collapse_user=True,stride=stride)
    per_video_db_oth, per_video_db_future_oth, per_video_db_future_input_oth = util.reshape2second_stacks(per_video_db_oth,collapse_user=False,stride=stride)
    # print('_video_ind = ',_video_ind,'per_video_db.shape = ',per_video_db.shape)            
    # print('_video_ind = ',_video_ind,'per_video_db_tar.shape = ',per_video_db_tar.shape)            
    # print('_video_ind = ',_video_ind,'per_video_db_oth.shape = ',per_video_db_oth.shape)
    return per_video_db_tar, per_video_db_future_tar, per_video_db_future_input_tar,\
            per_video_db_oth, per_video_db_future_oth, per_video_db_future_input_oth,\


def generator_train2(datadb,segment_index_tar=None,segment_index_tar_future=None,phase='train'):   #datadb
    max_encoder_seq_length = cfg.running_length
    num_encoder_tokens = 3*fps
    batch_size = 1#cfg.batch_size
    stride=cfg.data_chunk_stride
    if  phase in ['val','test']:
        stride=10 #test stride=10
        datadb_keys_longerthan20 = ['146','153','158','190','079','128','058','136','032','019','031','118','170','160','028','209',
                                    '185','201','088','142','143','141','083','208','192','024','123','169','022','047','103','101',
                                    '043','161','091',]#videos less than 20 seconds: 155,001,124     # 154 005 205 121
    else:
        datadb_keys_longerthan20=['098','099','214','215','212','213','210','211','090','092','093','095','096','097','133','011',
        '130','014','015','016','134','018','139','138','122','125','127','129','029',
        '132','131','013','137','199','135','200','195','194','017','191','089','193','115','114','117','116',
        '111','110','113','112','033','030','119','056','034','035','057','173','172','052','053','085','049',
        '048','046','045','044','106','042','104','105','177','076','176','144','152','059','179','178',
        '054','055','175','174','050','051','171','198','071','145','182','183','180','181','186','041','184',
        '188','040','187','060','063','196','065','064','067','066','069','068','168','164','165','167',
        '163','189','038','039','151','150','074','075','072','073','157','156','159','078','062','100','037','147',
        '203','202','204','140','206',
        '120','126'] # videos less than 20 seconds: ['094',012,010,197,109,102,077,070,166,162,007]
        #  025 026 021 023 082 081 080 087 148 149 003 002 006 004 009 008 027 020

    ii=0
    while True:
        _video_ind = datadb_keys_longerthan20[ii%len(datadb_keys_longerthan20)]
        #print(ii%len(datadb_keys_longerthan20),'_video_ind',_video_ind)
        #per_video_db = p.load(open('/scratch/wz1219/FoV/data/shanghaitech1/shanghaitech_heatmap'+_video_ind+'.p','rb'))
        per_video_db = datadb[_video_ind]
        #per_video_db = np.stack((datadb[_video_ind]['x'],datadb[_video_ind]['y'],datadb[_video_ind]['z']),axis=-1)
        per_video_db = per_video_db['heatmap']
        per_video_db = util.cut_head_or_tail_less_than_1sec(per_video_db)
        #print('shape of per_video_db is: ',per_video_db.shape)
        per_video_db = np.reshape(per_video_db,(per_video_db.shape[0],per_video_db.shape[1]/fps,30,18,36))
        #print('shape of per_video_db is: ',per_video_db.shape)
        if per_video_db.shape[1]<max_encoder_seq_length*2:
            print('video %s only has %d seconds. skip...'%(_video_ind,per_video_db.shape[1]))
            pdb.set_trace()
            # continue
        
        for _target_user in range(per_video_db.shape[0]):
            # print('target user:',_target_user)
            print(_target_user)
            per_video_db_tar, per_video_db_future_tar, per_video_db_future_input_tar,\
            per_video_db_oth, per_video_db_future_oth, per_video_db_future_input_oth = get_data_per_vid_per_target(per_video_db,_video_ind,_target_user,num_user=34,stride=stride)
            '''
            if _target_user==0 and cfg.use_saliency:
                #only need to run 1 time
                ###visual saliency
                num_chunks = per_video_db_tar.shape[0]
                real_num_users = per_video_db.shape[0]
                assert len(segment_index_tar[_video_ind])/real_num_users == num_chunks
                # saliency for the whole video
                saliency_data = util.load_h5('./360video/temp/saliency_input_pad0_fulllen/input'+_video_ind+'.hdf5','input')
                # saliency for the prediction time
                future_saliency = saliency_data[segment_index_tar_future[_video_ind][:num_chunks],:].copy()
                del saliency_data #to save memory
                #downsample
                future_saliency = future_saliency[:,:,::4,::4,:]
                future_saliency = 1.*future_saliency/future_saliency.max() 
                #only use 2 channels
                future_saliency = np.concatenate((np.mean(future_saliency[:,:,:,:,:30],axis=-1)[:,:,:,:,np.newaxis],
                                                 np.mean(future_saliency[:,:,:,:,30:],axis=-1)[:,:,:,:,np.newaxis]),axis=-1)

            '''
            #print('before prepare_data, per_video_db_tar shape is:,',per_video_db_tar.shape)
            #print('before prepare_data, per_video_db_future_tar shape is:,',per_video_db_future_tar.shape)
            #print('before prepare_data, per_video_db_future_input_tar shape is:,',per_video_db_future_input_tar.shape)
            #print('before prepare_data, per_video_db_oth shape is:,',per_video_db_oth.shape)
            #print('before prepare_data, per_video_db_future_oth shape is:,',per_video_db_future_oth.shape)
            #print('before prepare_data, per_video_db_future_input_oth shape is:,', per_video_db_future_input_oth.shape)
            
            encoder_input_data,decoder_input_data,decoder_target_data = \
                            _prepare_data(per_video_db_tar, per_video_db_future_tar, per_video_db_future_input_tar,
                            per_video_db_oth, per_video_db_future_oth, per_video_db_future_input_oth,phase)   #other_fut_input_data

            local_counter=0



            #print('encoder_input_data shape:',encoder_input_data.shape)
            #print('others_fut_input_data shape:',others_fut_input_data.shape)
            #print('decoder_input_data shape:',decoder_input_data.shape)
            #print('decoder_target_data shape:',decoder_target_data.shape)





            while local_counter<encoder_input_data.shape[0]:
                # print(local_counter,local_counter+batch_size)
                # print(encoder_input_data[local_counter:local_counter+batch_size,:].shape)
                if cfg.use_saliency:
                    yield [encoder_input_data[local_counter:local_counter+batch_size,:],\
                          others_fut_input_data[local_counter:local_counter+batch_size,:],\
                          decoder_input_data[local_counter:local_counter+batch_size,:],\
                          future_saliency[local_counter:local_counter+batch_size,:]],\
                          decoder_target_data[local_counter:local_counter+batch_size,:]
                else:
                    #if encoder_input_data.shape[0] < batch_size:
                        
                    #    yield [np.repeat(encoder_input_data,32,axis = 0),\
                          #others_fut_input_data[local_counter:local_counter+batch_size,:],\
                    #      np.repeat(decoder_input_data,32,axis = 0),\
                    #      np.repeat(decoder_target_data,32,axis = 0)]



                    #else:
                    yield [encoder_input_data[local_counter:local_counter+batch_size,:],\
                              #others_fut_input_data[local_counter:local_counter+batch_size,:],\
                              decoder_input_data[local_counter:local_counter+batch_size,:]],\
                              decoder_target_data[local_counter:local_counter+batch_size,:]
                    print("sample yield, shape is: " ,encoder_input_data[local_counter:local_counter+batch_size,:].shape)                    
                local_counter+=batch_size

        ii+=1#next video

#print('flag')

#a = generator_train2()
#temp = a.next()











# train 162145 frames in total, ~5404 seconds
# temp=0
# for ii in ['098','099','214','215','212','213','210','211','090','092','093','095','096','097','133','011',
#         '130','014','015','016','134','018','139','138','025','122','026','021','125','023','127','129','029',
#         '132','131','013','137','199','135','200','195','194','017','191','089','193','115','114','117','116',
#         '111','110','113','112','033','030','119','056','034','035','057','173','172','052','053','085','049',
#         '048','046','045','044','106','042','104','105','177','076','176','144','152','059','179','178',
#         '054','055','175','174','050','051','171','198','071','145','182','183','180','181','186','041','184',
#         '188','040','187','060','063','196','065','064','067','066','069','068','168','164','165','167',
#         '163','189','038','039','151','150','074','075','072','073','157','156','159','078','062','100','037','147',
#         '203','202','204','140','206','082','081','080','087','148','149','003','002','006','004','009','008',
#         '120','027','126','020']:
#     temp+=datadb[ii]['x'].shape[1]


# for ii in ['146','153','154','158','190','079','128','058','136','032','019','031','118','170','160','028','209',
#                                     '185','201','088','142','143','141','083','208','192','024','123','169','005','022','047','103','101',
#                                     '043','161','205','091','121']:
#     temp+=datadb[ii]['x'].shape[1]



# for key in datadb_test.keys():
#     saliency_data = util.load_h5('./360video/temp/saliency_input_pad0_fulllen/input'+key+'.hdf5','input')
#     if len(segment_index_tar_future[key])!=0:
#         # if saliency_data.shape[0]-segment_index_tar_future[key].max()<0:
#         print(key, saliency_data.shape[0], datadb_test[key]['x'].shape[1]/30., segment_index_tar_future[key].max())
#         # pdb.set_trace()


# for key in datadb_train.keys():
#     print(datadb_train[key]['x'].shape)


# for key in shanghaitech.keys():
#     lat = shanghaitech[key]['latitude']
#     lon = shanghaitech[key]['longitude']
#     print(lat.shape,lon.shape)




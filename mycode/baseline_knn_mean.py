%matplotlib inline
import _pickle as p
import numpy as np
from bisect import bisect
import matplotlib.pyplot as plt
import pdb
from scipy.interpolate import spline
import random



evaluate_mode='frame'
dataset = p.load(open('C:/Users/weixi/Desktop/temp/shanghai_dataset_theta_phi.p','rb'), encoding='latin1')
test_video_idx=['146','153','154','158','190','079','128','058','136','032','019','031','118','170','160','028','209',
                                    '185','201','088','142','143','141','083','208','192','024','123','169','005','022','047','103','101',
                                    '043','161','205','091','121']




def k_nearest(k, center, sorted_data):
 
    distance_array = sorted_data - center
    distance = distance_array[:,0]**2+distance_array[:,1]**2
    others_k_idx = distance.argsort()[:k]
    
    return others_k_idx
    
def find_knn(current_loc, current_idx, others_seq,k=5):
    
    others_at_loc = np.mean(others_seq[:,current_idx-30:current_idx,:],axis=1)
    knn = k_nearest(k,current_loc,others_at_loc)
    prediction = 0
    for i in range(k):
        prediction += np.mean(others_seq[knn[i],current_idx:current_idx+30,:],axis=0)
        
    return prediction/k
    
def reshape2sec(seq):
    fps = 30
    sec = int(seq.shape[1]/fps)
    new_seq = np.zeros((sec,30))
    seq = np.reshape(seq,(seq.shape[1]))
    for i in range(sec):
        new_seq[i,:] = seq[i*30:(i+1)*30]
    return new_seq
    
def get_iou_or_hitrate(centroid, span, gt_centroid, gt_span):

    predicted_box = np.array([centroid[0]-span[0]/2,centroid[1]-span[1]/2,
                    centroid[0]+span[0]/2,centroid[1]+span[1]/2]).T

    gt_region = np.array([gt_centroid[0]-gt_span[0]/2,gt_centroid[1]-gt_span[1]/2,
                gt_centroid[0]+gt_span[0]/2,gt_centroid[1]+gt_span[1]/2]).T

    overlaps = []
    overlaps.append(bbox_overlaps_hit_rate(predicted_box,gt_region))

    return overlaps

def bbox_overlaps_hit_rate(boxes, query_boxes):
    """
    compute the hit rate: intersection over gt region size
    """
    overlaps = 0
    box_area = (
            (query_boxes[2] - query_boxes[0]) *
            (query_boxes[3] - query_boxes[ 1])
        )

    iw = (
                min(boxes[2], query_boxes[ 2]) -
                max(boxes[0], query_boxes[ 0])
        )
    if iw > 0:
            ih = (
                min(boxes[3], query_boxes[3]) -
                max(boxes[1], query_boxes[1])
            )
            if ih > 0:
                overlaps = iw * ih / box_area
    return overlaps


def boundary_cases(gt_centroid,centroid):
    condition1 = (gt_centroid[0]>2/3.0*np.pi)*(centroid[0]<-2/3.0*np.pi)
    centroid[0][condition1] = centroid[0][condition1]+2*np.pi

    condition2 = (gt_centroid[0]<-2/3.0*np.pi)*(centroid[0]>2/3.0*np.pi)
    gt_centroid[0][condition2] = gt_centroid[0][condition2]+2*np.pi

    
    return np.array(gt_centroid),np.array(centroid)








def _draw_result_curve(overlap_list,evaluate_mode,tag=''):
    overlap_list = np.array(overlap_list)
    predict_step = overlap_list.shape[1]
    mean_overlap = overlap_list
    if evaluate_mode=='second':
        plt.figure()
        plt.plot(range(1,predict_step+1),np.mean(overlap_list,axis=1))
        plt.scatter(range(1,predict_step+1),np.mean(overlap_list,axis=1))
        plt.xlabel('time in seconds')
    if evaluate_mode=='frame':
        mean_overlap_flat = mean_overlap.reshape(mean_overlap.shape[1])
        #plt.figure()
        plt.plot(mean_overlap_flat)
        mean_overlap = mean_overlap_flat
        plt.xlabel('time in frames')

    xaxis = np.linspace(0,1,len(mean_overlap))

    plt.grid('on')


def _eval_for_seq2seq(test_out,gt_out,draw=False,a=1):
    """gt shape: N*10*90
       test shape: N*10*6
    """
    predict_step = gt_out.shape[1]
    #print("predict_step:",predict_step)
    if draw:
        fig2,ax = plt.subplots(3,2)
    span = np.array([a*120,a*120])/180.0*np.pi #for now, don't predict span
    gt_span = np.array([120,120])/180.0*np.pi
    overlap_list = []
    gt_centroid_list = []
    centroid_list = []
        # gt_ux,gt_uy,gt_uz,gt_varx,gt_vary,gt_varz = np.split(target[ii,:,:],6,axis=1)
    #gt_ux = gt_out[0,:]
    #gt_uy = gt_out[1,:]
    #gt_uz = gt_out[2,:]
    gt_utheta = gt_out[0,:]
    gt_uphi = gt_out[1,:]
    #ux_temp = test_out[0,:]
    #uy_temp = test_out[1,:]
    #uz_temp = test_out[2,:]
    utheta = test_out[0,:]
    uphi = test_out[1,:]

    if evaluate_mode=='second':
        centroid = np.array([utheta.copy(), uphi.copy()])
    elif evaluate_mode=='frame':
        centroid = np.array([utheta.copy(), uphi.copy()])
    gt_centroid = np.array([gt_utheta.copy(), gt_uphi.copy()])

    gt_centroid,centroid = boundary_cases(gt_centroid,np.squeeze(centroid))

    if evaluate_mode=='second':
        overlaps = get_iou_or_hitrate(centroid, span, gt_centroid, gt_span)
    elif evaluate_mode=='frame':
        overlaps = []
        for time_ind in range(predict_step):
            overlaps.append(get_iou_or_hitrate(centroid[:,time_ind], span, gt_centroid[:,time_ind], gt_span))

    overlap_list.append(overlaps)
    gt_centroid_list.append(gt_centroid)
    centroid_list.append(centroid)

       
    return overlap_list,gt_centroid_list,centroid_list


##################################################################
######################     knn       #############################
##################################################################
k=5
overlap_list_video_knn= []
predict_step=300
for video_idx in test_video_idx:
    print(video_idx)
    for user_idx in range(dataset[video_idx]['latitude'].shape[0]):
        tar_user_idx = user_idx
        tar_user_lat = dataset[video_idx]['latitude'][tar_user_idx,:]
        tar_user_lon = dataset[video_idx]['longitude'][tar_user_idx,:]
        tar_user_x = dataset[video_idx]['x'][tar_user_idx,:]
        tar_user_y = dataset[video_idx]['y'][tar_user_idx,:]
        tar_user_z = dataset[video_idx]['z'][tar_user_idx,:]
    
        others_lat = np.delete(dataset[video_idx]['latitude'],tar_user_idx,axis=0)
        others_lon = np.delete(dataset[video_idx]['longitude'],tar_user_idx,axis=0)
        others_x = np.delete(dataset[video_idx]['z'],tar_user_idx,axis=0)
        others_y = np.delete(dataset[video_idx]['y'],tar_user_idx,axis=0)
        others_z = np.delete(dataset[video_idx]['z'],tar_user_idx,axis=0)
    
        prediction = np.zeros((10,2))#((tar_user_lat.shape[0]-1,2))
        others_lat_lon = np.zeros((others_lat.shape[0],others_lat.shape[1],2))
        others_lat_lon[:,:,0] = others_lat
        others_lat_lon[:,:,1] = others_lon
        #print(others_lat_lon.shape)
        for start_loc in range(30,tar_user_lat.shape[0]-300,300):
            current_loc = np.array([np.mean(tar_user_lat[start_loc-30:start_loc]),np.mean(tar_user_lon[start_loc-30:start_loc])])
            #print(start_loc)
            for i in range(10):
                prediction[i,:] = find_knn(current_loc, start_loc, others_lat_lon, k)
                current_loc = prediction[i,:]
        
            gt_out = np.zeros((2,10))
            test_out = np.zeros((2,10))    

            gt_out[0,:] = np.mean(tar_user_lat[start_loc:start_loc+300].reshape((10,30)),axis=1)
            gt_out[1,:] = np.mean(tar_user_lon[start_loc:start_loc+300].reshape((10,30)),axis=1)
 
       
            test_out[0,:] = prediction[:,0]
            test_out[1,:] = prediction[:,1]
    

            overlap_list_knn,gt_centroid_list_knn,centroid_list_knn = _eval_for_seq2seq(test_out,gt_out,draw=False)
            overlap_list_knn = np.array(overlap_list_knn)
    
            overlap_list_video_knn.append(overlap_list_knn) 


overlap_all_knn = np.array(overlap_list_video_knn)
b = np.mean(overlap_all_knn,axis=0).reshape((10,1))
plt.figure()
plt.plot(range(1,10+1),b)
plt.scatter(range(1,10+1),b)
plt.xlabel('time in seconds')     




######################################################################################
#################################    mean     ########################################
######################################################################################
overlap_list_all_mean = []
for video_idx in test_video_idx:
    for user_idx in range(dataset[video_idx]['latitude'].shape[0]):
        tar_user_idx = user_idx#random.randint(0,dataset[video_idx]['latitude'].shape[0]-1)    
        tar_user_lat = dataset[video_idx]['latitude'][tar_user_idx,:]
        tar_user_lon = dataset[video_idx]['longitude'][tar_user_idx,:]
        #print(tar_user_lat.shape)
    
        tar_user_x =dataset[video_idx]['x'][tar_user_idx,:]
        tar_user_y =dataset[video_idx]['y'][tar_user_idx,:]
        tar_user_z =dataset[video_idx]['z'][tar_user_idx,:]
        others_lat = np.delete(dataset[video_idx]['latitude'],tar_user_idx,axis=0)
        others_lon = np.delete(dataset[video_idx]['longitude'],tar_user_idx,axis=0)
        others_x = np.delete(dataset[video_idx]['z'],tar_user_idx,axis=0)
        others_y = np.delete(dataset[video_idx]['y'],tar_user_idx,axis=0)
        others_z = np.delete(dataset[video_idx]['z'],tar_user_idx,axis=0)
    
        prediction_lon = np.mean(others_lon,axis=0)
        prediction_lat = np.mean(others_lat,axis=0)
        prediction_x = np.mean(others_x,axis=0)
        prediction_y = np.mean(others_y,axis=0)
        prediction_z = np.mean(others_z,axis=0)
    
        gt_out = np.zeros((2,tar_user_lat.shape[0]))
        test_out = np.zeros((2,tar_user_lat.shape[0]))
        #gt_out[0,:] = tar_user_x
        #gt_out[1,:] = tar_user_y
        #gt_out[2,:] = tar_user_z
        gt_out[0,:] = tar_user_lon
        gt_out[1,:] = tar_user_lat
    
        #test_out[0,:] = prediction_x
        #test_out[1,:] = prediction_y
        #test_out[2,:] = prediction_z
        test_out[0,:] = prediction_lon
        test_out[1,:] = prediction_lat
    
        overlap_list_mean,gt_centroid_list_mean,centroid_list_mean = _eval_for_seq2seq(test_out,gt_out,draw=False)
    
        overlap_list_mean = np.array(overlap_list_mean)

        evaluate_mode='frame'
        tag = 'baseline_1_mean_over_all_others'
        overlap_list_all_mean.append(overlap_list_mean)
    
overlap_all_mean = []
for i in range(len(overlap_list_all_mean)):
    sec = len(overlap_list_all_mean[i])/30
    overlap_list_second = reshape2sec(overlap_list_all_mean[i])
    num_seq = overlap_list_second.shape[0]/10
    for j in range(int(num_seq)):
        overlap_all_mean.append(overlap_list_second[j*10:(j+1)*10,:])
a= np.array(overlap_all_mean)
b = np.mean(a,axis=0)
c = np.mean(b,axis=1)
print(c.shape)
plt.figure()
plt.plot(range(1,10+1),c)
plt.scatter(range(1,10+1),c)
plt.xlabel('time in seconds')     















"""
truncated linear baseline
"""




#history
#On HPC
# pickle.dump(_video_db_tar,open('./tar_own_history_THUtrain.p','wb'))



# ON local
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline
import _pickle as pickle
from sklearn import linear_model

# -------------------------------truncated linear--------------------------
def extrapolate(xi,yi,x,order=1):
    """xi,yi are knowns, x is the positions to be filled"""
    # spline order: 1 linear, 2 quadratic, 3 cubic ... 
    # do inter/extrapolation
    s = InterpolatedUnivariateSpline(xi, yi, k=order)
    y = s(x)
    # plt.plot(x, y)

    # # example showing the interpolation for linear, quadratic and cubic interpolation
    # plt.figure()
    # plt.plot(xi, yi)
    # for order in range(1, 4):
    #     s = InterpolatedUnivariateSpline(xi, yi, k=order)
    #     y = s(x)
    #     plt.plot(x, y)
    # plt.show()
    return y


### Shanghaitech
_video_db = pickle.load(open('./testresult/tar_own_history.p','rb'),encoding='latin1')
gt_out = pickle.load(open('./testresult/video_db_future.p','rb'),encoding='latin1')
### Tsinghua dataset
# ### _video_db = pickle.load(open('./testresult/tar_own_history_THUtrain.p','rb'),encoding='latin1')
# ### _video_db1 = pickle.load(open('./testresult/tar_own_history_THU.p','rb'),encoding='latin1')
# _video_db = pickle.load(open('./testresult/tar_own_history_THU2.p','rb'),encoding='latin1')
# gt_out = pickle.load(open('./testresult/_video_db_future_THU2.p','rb'))

x = np.linspace(0, 10, 300)
xi = np.linspace(-10, 0, 300)
_video_db = _video_db.reshape(-1,10,30,3)
_video_db = _video_db.reshape(-1,300,3)


# -------------------------------truncated linear--------------------------
# plt.figure()
extrapolated = np.zeros_like(_video_db)
for ind in range(_video_db.shape[0]):
    for jj in range(3): #xyz
        yi = _video_db[ind,:,jj]
        y = extrapolate(xi,yi,x,order=1)
        extrapolated[ind,:,jj] = y
        # plot(xi,yi)
        # plt.plot(x,y)
        # plt.draw()
        # pdb.set_trace()
        # plt.cla()

#clip the value to be (-1,1)
extrapolated[extrapolated<=-1]=-1 #Even worse!!why?
extrapolated[extrapolated>=1]=1
test_out = extrapolated.reshape(-1,10,30,3)
gt_out = gt_out.reshape(-1,10,30,3)



#### -------------------linear regression model-------------------
regr = linear_model.LinearRegression()
linear_reg = np.zeros_like(_video_db)
# plt.figure()
for ind in range(_video_db.shape[0]):
    for jj in range(3): #xyz
        yi = _video_db[ind,:,jj]
        y = extrapolate(xi,yi,x,order=1)
        # Train the model using the training sets
        regr.fit(xi[:,np.newaxis],yi)
        # Make predictions using the testing set
        y = regr.predict(x[:,np.newaxis])
        linear_reg[ind,:,jj] = y

        # plt.scatter(xi, yi,  color='black')
        # plt.plot(x, y, color='blue', linewidth=3)
        # plt.draw()
        # pdb.set_trace()
        # plt.cla()
        
#clip the value to be (-1,1)
linear_reg[linear_reg<=-1]=-1 #Even worse!!why?
linear_reg[linear_reg>=1]=1
test_out = linear_reg.reshape(-1,10,30,3)
gt_out = gt_out.reshape(-1,10,30,3)





#### -------------------persistence model-------------------
#last frame in the history
### Shanghaitech
_video_db = pickle.load(open('./testresult/tar_own_history.p','rb'),encoding='latin1')
gt_out = pickle.load(open('./testresult/video_db_future.p','rb'),encoding='latin1')
# ### Tsinghua dataset
# _video_db = pickle.load(open('./testresult/tar_own_history_THU2.p','rb'),encoding='latin1')
# gt_out = pickle.load(open('./testresult/_video_db_future_THU2.p','rb'))

gt_out = np.squeeze(np.array(gt_out))
_video_db = _video_db.reshape(-1,300,3)
persistence = np.zeros_like(_video_db)
for ind in range(persistence.shape[0]):
    pred_persistence = np.repeat(_video_db[ind,-1,:].reshape(1,3),300,axis=0)
    persistence[ind] = pred_persistence
test_out = persistence.reshape(-1,10,30,3)
gt_out = gt_out.reshape(-1,10,30,3)




# # #first frame in the future(~= last frame in the history)
# _video_db_future_input = gt_out
# _video_db_future_input = _video_db_future_input.reshape(-1,300,3)
# persistence = np.zeros_like(_video_db_future_input)
# for ind in range(persistence.shape[0]):
#     pred_persistence = np.repeat(_video_db_future_input[ind,0,:].reshape(1,3),300,axis=0)
#     persistence[ind] = pred_persistence
# test_out = persistence
# test_out = test_out.reshape(-1,10,30,3)
# gt_out = gt_out.reshape(-1,10,30,3)




# gt_out = gt_out.reshape(-1,10,30,3)
# var_pred = np.zeros_like(gt_out)
# for ind in range(gt_out.shape[0]):
#     # predictions = _VAR(train=gt_out[ind,:,:],test=None)
#     pred_persistence = np.repeat(gt_out[ind,-1,:,:].reshape(1,3),300,axis=0)
#     var_pred[ind] = pred_persistence
#     # gt = gt_out[ind,:,:]
# test_out = var_pred.reshape(-1,10,30,3)
# gt_out = gt_out.reshape(-1,10,30,3)






### persistence baselines for 2D representation:
# both shape: N,10,18,36
# for now use the first second gt as the persistence
persistence = np.zeros_like(gt_out)
for ii in range(persistence.shape[0]):
    persistence[ii,:,:,:] = gt_out[ii,0,:,:] 















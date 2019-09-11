"""
cost functions
"""
import tensorflow as tf
import numpy as np
from mycode.config import cfg
from keras import losses
from keras import backend as K
import pdb
import mycode.utility as util
from keras.layers import Lambda
def _modified_mse(out,y):
    """modified MSE loss func, calculate the min()"""
    diff = tf.abs(out-y)
    return tf.reduce_sum(tf.where(tf.greater(diff,tf.constant(np.pi)),
            tf.square(tf.constant(2*np.pi)-diff),
            tf.square(diff)))


def _mse(y_true, y_pred):
    # keras
    cost = losses.mean_squared_error(y_true, y_pred)
    if cfg.add_xyz_sum1:
        ux = y_pred[:,:,:,0][:,:,:,np.newaxis]
        uy = y_pred[:,:,:,1][:,:,:,np.newaxis]
        uz = y_pred[:,:,:,2][:,:,:,np.newaxis]
        reg = losses.mean_squared_error(1,ux**2+uy**2+uz**2)
        cost += 0.5*reg
    return cost

def _mean_var_cost_xyz(ux,uy,uz,varx,vary,varz,target,population_target=None,alpha=cfg.pop_alpha):
    assert cfg.use_xyz==True
    gt_mean_x,gt_mean_y,gt_mean_z,gt_var_x,gt_var_y,gt_var_z = target 

    cost_x = tf.losses.mean_squared_error(gt_mean_x,ux)+tf.losses.mean_squared_error(gt_var_x,varx)
    cost_y = tf.losses.mean_squared_error(gt_mean_y,uy)+tf.losses.mean_squared_error(gt_var_y,vary)
    cost_z = tf.losses.mean_squared_error(gt_mean_z,uz)+tf.losses.mean_squared_error(gt_var_z,varz)
    cost = cost_x+cost_y+cost_z

    # if cfg.add_xyz_sum1:
    #     reg = tf.losses.mean_squared_error(1,ux**2+uy**2+uz**2)
    #     cost += 0.05*reg
    if population_target:
        pop_mean_x,pop_mean_y,pop_mean_z,pop_var_x,pop_var_y,pop_var_z = population_target 
        cost_x_pop = tf.losses.mean_squared_error(pop_mean_x,ux)+tf.losses.mean_squared_error(pop_var_x,varx)
        cost_y_pop = tf.losses.mean_squared_error(pop_mean_y,uy)+tf.losses.mean_squared_error(pop_var_y,vary)
        cost_z_pop = tf.losses.mean_squared_error(pop_mean_z,uz)+tf.losses.mean_squared_error(pop_var_z,varz)
        cost = cost+alpha*(cost_x_pop+cost_y_pop+cost_z_pop)

    return tf.reduce_mean(cost)

def _mean_var_cost_phi_theta(u1,u2,var1,var2,target):
    assert  cfg.use_phi_theta==True
    gt_mean_phi,gt_mean_theta,gt_var_phi,gt_var_theta =target 
    cost_phi = tf.losses.mean_squared_error(gt_mean_phi,u1)+tf.losses.mean_squared_error(gt_var_phi,var1)
    cost_theta = tf.losses.mean_squared_error(gt_mean_theta,u2)+tf.losses.mean_squared_error(gt_var_theta,var2)
    cost = cost_phi+cost_theta
    return tf.reduce_mean(cost)


def Bhattacharyya_distance(u1,u2,var1,var2):
    """Bhattacharyya distance between two norm distributions"""
    inside_log = 1./4*(var1/(var2+1e-10)+var2/(var1+1e-10)+2)
    inside_log = tf.clip_by_value(inside_log,1e-10,100.)
    Db = 1./4*tf.log(inside_log) + 1./4*((u1-u2)**2/tf.clip_by_value((var1+var2),1e-10,8.0))
    return Db


def Wasserstein_distance(u1,u2,var1,var2):
    """Wasserstein distance between two norm distributions"""
    W = (u1-u2)**2+(var1+var2-2.*tf.sqrt(var1*var2))
    return W

def Kullback_Leibler_divergence_Gaussian(u1,u2,var1,var2):
    # KL-distance from N(u1,var1) to N(u2,var2)
    inside_log = tf.div(tf.sqrt(var2),(tf.sqrt(var1)+1e-10))
    inside_log = tf.clip_by_value(inside_log,1e-10,100)
    KLd = tf.add(tf.div((u1-u2)**2+var1-var2,2.*var2) , tf.log(inside_log))
    return KLd

def Kullback_Leibler_divergence(pk,qk):
    import scipy 
    return scipy.stats.entropy(pk,qk)


def _mean_var_cost_xyz_metric(ux,uy,uz,varx,vary,varz,target,metric_func):
    """use distance metrics other than euclidean distance
    such as Bhattacharyya or Wasserstein"""
    # eg. _mean_var_cost_xyz_metric(ux,uy,uz,varx,vary,varz,target,Bhattacharyya_distance)
    # or _mean_var_cost_xyz_metric(ux,uy,uz,varx,vary,varz,target,Wasserstein_distance)
    # or _mean_var_cost_xyz_metric(ux,uy,uz,varx,vary,varz,target,Kullback_Leibler_divergence_Gaussian)
    assert cfg.use_xyz==True
    gt_mean_x,gt_mean_y,gt_mean_z,gt_var_x,gt_var_y,gt_var_z =target 

    cost_x = metric_func(ux,gt_mean_x,varx,gt_var_x)
    cost_y = metric_func(uy,gt_mean_y,vary,gt_var_y)
    cost_z = metric_func(uz,gt_mean_z,varz,gt_var_z)

    cost = tf.reduce_mean(cost_x)+tf.reduce_mean(cost_y)+tf.reduce_mean(cost_z)
    return cost,cost_x,cost_y,cost_z


def duplicate(tensor):
    return tf.ones([32,30])*tensor




# TODO
def conditional_prob_loss(predict_sample,target,population_target):
    """derived from P(xt|Yt,x_{t-T})"""
    gt_mean_x,gt_mean_y,gt_mean_z,gt_var_x,gt_var_y,gt_var_z = target 
    pop_mean_x,pop_mean_y,pop_mean_z,pop_var_x,pop_var_y,pop_var_z = population_target 

    # cost_x = tf.losses.mean_squared_error(gt_mean_x,ux)/(varx+1e-7)+tf.log(varx)
    # cost_y = tf.losses.mean_squared_error(gt_mean_y,uy)/(vary+1e-7)+tf.log(vary)
    # cost_z = tf.losses.mean_squared_error(gt_mean_z,uz)/(varz+1e-7)+tf.log(varz)

    x = tf.slice(predict_sample,[0,0,0],[-1,-1,1])[:,:,0]
    y = tf.slice(predict_sample,[0,0,1],[-1,-1,1])[:,:,0]
    z = tf.slice(predict_sample,[0,0,2],[-1,-1,1])[:,:,0]

    cost = (tf.losses.mean_squared_error(x,duplicate(gt_mean_x))/tf.clip_by_value(2*gt_var_x,1e-10,5) \
    + tf.losses.mean_squared_error(y,duplicate(gt_mean_y))/tf.clip_by_value(2*gt_var_y,1e-10,5) \
    + tf.losses.mean_squared_error(z,duplicate(gt_mean_z))/tf.clip_by_value(2*gt_var_z,1e-10,5) \
    + tf.losses.mean_squared_error(x,duplicate(pop_mean_x))/tf.clip_by_value(2*pop_var_x,1e-10,5) \
    + tf.losses.mean_squared_error(y,duplicate(pop_mean_y))/tf.clip_by_value(2*pop_var_y,1e-10,5) \
    + tf.losses.mean_squared_error(z,duplicate(pop_mean_z))/tf.clip_by_value(2*pop_var_z,1e-10,5) \
    + tf.log( tf.clip_by_value(tf.sqrt(gt_var_x)*tf.sqrt(pop_var_x),1e-10,5)))

    return tf.reduce_mean(cost),duplicate(gt_mean_y),duplicate(pop_mean_z)





## NLL of gaussian distribution for keras
def likelihood_loss(y_true, y_pred):
    """
    if we assume the distribution follows the N(mean_pred,var_pred),
    then we can use the ground truth samples to compute the likelihood.
    Use the NLL as the cost.
    """
    #Note that var=sigma**2
    ux = util.slice_layer(2,0,1)(y_pred)
    uy = util.slice_layer(2,1,2)(y_pred)
    uz = util.slice_layer(2,2,3)(y_pred)
    varx = util.slice_layer(2,3,4)(y_pred)
    vary = util.slice_layer(2,4,5)(y_pred)
    varz = util.slice_layer(2,5,6)(y_pred)

    cliplayer = Lambda(lambda x: K.clip(K.abs(x), min_value=0.0001, max_value=2))
    cliplayer2 = Lambda(lambda x: K.clip(x, min_value=-2000, max_value=2000))

    varx = cliplayer(varx)
    vary = cliplayer(vary)
    varz = cliplayer(varz)

    ux = K.repeat_elements(ux,30,axis=-1)
    uy = K.repeat_elements(uy,30,axis=-1)
    uz = K.repeat_elements(uz,30,axis=-1)
    varx = K.repeat_elements(varx,30,axis=-1)
    vary = K.repeat_elements(vary,30,axis=-1)
    varz = K.repeat_elements(varz,30,axis=-1)

    x = y_true[:,:,0::3]
    y = y_true[:,:,1::3]
    z = y_true[:,:,2::3]
    lossx = K.log(varx+ K.epsilon())+((x-ux)**2)/(varx+ K.epsilon())
    lossy = K.log(vary+ K.epsilon())+((y-uy)**2)/(vary+ K.epsilon())
    lossz = K.log(varz+ K.epsilon())+((z-uz)**2)/(varz+ K.epsilon())

    # lossx = varx-1+ ((x-ux)**2)/(varx+ K.epsilon())
    # lossy = vary-1+ ((y-uy)**2)/(vary+ K.epsilon())
    # lossz = varz-1+ ((z-uz)**2)/(varz+ K.epsilon())

    lossx = cliplayer2(lossx)
    lossy = cliplayer2(lossy)
    lossz = cliplayer2(lossz)


    #constraint on x,y,z
    lambda_xyz=0
    lossxyz = lambda_xyz*(1-(ux**2+uy**2+uz**2))**2

    loss =  K.mean(K.sum(K.sum(lossx+lossy+lossz+lossxyz,axis=2),axis=1))
    return loss/cfg.running_length/cfg.fps


def likelihood_loss_tf(y_pred,y_true):
    ## NLL of gaussian distribution for TF
    ux = y_pred[0]
    uy = y_pred[1]
    uz = y_pred[2]
    varx = y_pred[3]
    vary = y_pred[4]
    varz = y_pred[5]

    if cfg.process_in_seconds:
        ux = expand(ux,-1,cfg.fps)
        uy = expand(uy,-1,cfg.fps)
        uz = expand(uz,-1,cfg.fps)

        varx = expand(varx,-1,cfg.fps)
        vary = expand(vary,-1,cfg.fps)
        varz = expand(varz,-1,cfg.fps)

    x = y_true[:,:,0::3]
    y = y_true[:,:,1::3]
    z = y_true[:,:,2::3]

    eps = 1e-20
    lossx = tf.log(varx+ eps)+((x-ux)**2)/(varx+ eps)
    lossy = tf.log(vary+ eps)+((y-uy)**2)/(vary+ eps)
    lossz = tf.log(varz+ eps)+((z-uz)**2)/(varz+ eps)

    lossx = tf.clip_by_value(lossx,-10,10)
    lossy = tf.clip_by_value(lossy,-10,10)
    lossz = tf.clip_by_value(lossz,-10,10)

    #constraint on x,y,z
    lambda_xyz=0
    lossxyz = lambda_xyz*tf.square(1-(tf.square(ux)+tf.square(uy)+tf.square(uz)))

    loss =  tf.reduce_mean(tf.reduce_sum(tf.reduce_sum(lossx+lossy+lossz+lossxyz,axis=2),axis=1))
    if cfg.process_in_seconds:
        return loss/cfg.running_length/cfg.fps
    else:
        return loss/cfg.running_length




def oneD_gaussian(x,mu,sigma):
    gaussian = tf.exp(-tf.square((x-mu)/sigma)/2)/tf.sqrt(2*np.pi*tf.square(sigma))
    return gaussian

def oneD_gaussian_loss(y_pred,y_true):
    ux = y_pred[0]
    uy = y_pred[1]
    uz = y_pred[2]
    varx = y_pred[3]
    vary = y_pred[4]
    varz = y_pred[5]

    ux = expand(ux,-1,cfg.fps)
    uy = expand(uy,-1,cfg.fps)
    uz = expand(uz,-1,cfg.fps)

    varx = expand(varx,-1,cfg.fps)
    vary = expand(vary,-1,cfg.fps)
    varz = expand(varz,-1,cfg.fps)

    x = y_true[:,:,0::3]
    y = y_true[:,:,1::3]
    z = y_true[:,:,2::3]

    gaussianx = oneD_gaussian(x,ux,tf.sqrt(varx))
    gaussiany = oneD_gaussian(y,uy,tf.sqrt(vary))
    gaussianz = oneD_gaussian(z,uz,tf.sqrt(varz))

    loss_gaussian = tf.reduce_mean(tf.reduce_sum(tf.reduce_sum(-tf.log(gaussianx+gaussiany+gaussianz + 1e-20),axis=2),axis=1))
    return loss_gaussian/cfg.running_length/cfg.fps



def likelihood_loss_phi_theta_tf(y_true, y_pred, use_reduce=True):
    """
    NLL loss for theta, phi, 2D gaussian distribution
    """
    #covariance matrix=[sigma_theta^2,rho*sigma_theta*sigma_phi;rho*sigma_theta*sigma_phi,sigma_phi^2]
    #prediction
    utheta = y_pred[0]
    uphi = y_pred[1]
    sigma_theta = y_pred[2]
    sigma_phi = y_pred[3]
    rho = y_pred[4]

    sigma_theta = tf.abs(sigma_theta)
    sigma_phi = tf.abs(sigma_phi)
    rho = tf.clip_by_value(rho,-1,1)

    #gt
    thetagt = y_true[:,0,0::2]
    phigt = y_true[:,0,1::2]

    #loss
    temp = (thetagt-utheta)**2/sigma_theta**2 + (phigt-uphi)**2/sigma_phi**2 - 2*rho*(thetagt-utheta)*(phigt-uphi)/sigma_theta/sigma_phi
    loss = tf.log(sigma_theta)+tf.log(sigma_phi)+0.5*tf.log(1-rho**2)+0.5/(1-rho**2)*(temp)
    if use_reduce:
        return  tf.reduce_mean(loss)
    else:
        return  loss



def mixture_likelihood_loss_phi_theta_tf(y_true, y_pred):
    """
    NLL loss for theta, phi, mixture of bivariate 2D gaussian distribution
    """   

    # input_end_storke is the indicator from the gt
    #ignore end_stroke for now
    # input_end_storke = y_true[-1]
    # y_true = y_true[:-1]

    mixture_pi,us,sigmas,rhos = y_pred
    n_mixture = mixture_pi.shape[1].value
    temp = 0
    for ii in range(n_mixture):
        utheta = tf.expand_dims(us[:,2*ii],1)
        uphi =  tf.expand_dims(us[:,2*ii+1],1)
        sigma_theta =  tf.expand_dims(sigmas[:,2*ii],1)
        sigma_phi =  tf.expand_dims(sigmas[:,2*ii+1],1)
        rho = tf.expand_dims(rhos[:,ii],1)
        weight = tf.expand_dims(mixture_pi[:,ii],1)
        temp+=weight*likelihood_loss_phi_theta_tf(y_true,[utheta,uphi,sigma_theta,sigma_phi,rho],use_reduce=False)

    # if input_end_storke==1:
    #     loss = -tf.log(temp)-tf.log(end_stroke)
    # else:
    #     loss = -tf.log(temp)-tf.log(1-end_stroke)
    return tf.reduce_mean(-tf.log(temp))/cfg.running_length


#modified based on https://github.com/snowkylin/rnn-handwriting-generation/blob/master/model.py
def bivariate_gaussian(x1, x2, mu1, mu2, sigma1, sigma2, rho):
    z = tf.square((x1 - mu1) / sigma1) + tf.square((x2 - mu2) / sigma2) \
        - 2 * rho * (x1 - mu1) * (x2 - mu2) / (sigma1 * sigma2)
    return tf.exp(-z / (2 * (1 - tf.square(rho)))) / \
           (2 * np.pi * sigma1 * sigma2 * tf.sqrt(1 - tf.square(rho)))



def _make_positive_semidefinite_tf(convariance_mat,n=3):
    e,v = tf.self_adjoint_eig(convariance_mat)
    min_eig = e[0]
    # e,v = tf.self_adjoint_eig(H)
    # e_pos = tf.maximum(0.0,e)+1e-6 #make sure positive definite 
    # e_sqrt = tf.diag(tf.sqrt(e_pos))
    # sq_H = tf.matmul(v,tf.matmul(e_sqrt,tf.transpose(v)))

    def f1(): 
        print('need to make the covariance matrix SPD, min_eig =',min_eig)
        return convariance_mat-10*min_eig * tf.eye(n)
    def f2(): return tf.reshape(tf.stack(convariance_mat),[n,n])
    convariance_mat = tf.cond(tf.less(min_eig,tf.constant(0.0)), f1, f2)       
    return convariance_mat



def multivariate_gaussian(x1, x2, x3, mu1, mu2, mu3, sigma1, sigma2, sigma3, rho12,rho13,rho23):
    ###3D
    n = 3
    # convariance_mat = np.array([[tf.square(sigma1),rho12*sigma1*sigma2,rho13*sigma1*sigma3],
    #                     [rho12*sigma1*sigma2,tf.square(sigma2),rho23*sigma2*sigma3],
    #                     [rho13*sigma1*sigma3,rho23*sigma2*sigma3,tf.square(sigma3)]])
    # return tf.pow(tf.matrix_determinant((2*np.pi)**n*convariance_mat),-1/2)*\
    # tf.exp(-1/2*tf.transpose([x1,x2,x3]-[mu1,mu2,mu3])*(tf.pow(convariance_mat,-1))*([x1,x2,x3]-[mu1,mu2,mu3]))
    likelihood = []
    for batch_ind in range(cfg.batch_size):
        for mixture_ind in range(x1.shape[1].value):
            for time_ind in range(x1.shape[2].value):
                sigma1_ = sigma1[batch_ind,mixture_ind,time_ind]
                sigma2_ = sigma2[batch_ind,mixture_ind,time_ind]
                sigma3_ = sigma3[batch_ind,mixture_ind,time_ind]
                mu1_ = mu1[batch_ind,mixture_ind,time_ind]
                mu2_ = mu2[batch_ind,mixture_ind,time_ind]
                mu3_ = mu3[batch_ind,mixture_ind,time_ind]
                rho12_ = rho12[batch_ind,mixture_ind,time_ind]
                rho13_ = rho13[batch_ind,mixture_ind,time_ind]
                rho23_ = rho23[batch_ind,mixture_ind,time_ind]
                x1_ = x1[batch_ind,mixture_ind,time_ind]
                x2_ = x2[batch_ind,mixture_ind,time_ind]
                x3_ = x3[batch_ind,mixture_ind,time_ind]
                convariance_mat = [[tf.square(sigma1_),rho12_*sigma1_*sigma2_,rho13_*sigma1_*sigma3_],
                                    [rho12_*sigma1_*sigma2_,tf.square(sigma2_),rho23_*sigma2_*sigma3_],
                                    [rho13_*sigma1_*sigma3_,rho23_*sigma2_*sigma3_,tf.square(sigma3_)]]
                convariance_mat = _make_positive_semidefinite_tf(convariance_mat,n)
                mvn = tf.contrib.distributions.MultivariateNormalFullCovariance(loc=[mu1_, mu2_, mu3_], covariance_matrix = convariance_mat)
                likelihood.append(mvn.prob([x1_,x2_,x3_]))

    return tf.reshape(tf.stack(likelihood),[cfg.batch_size,x1.shape[1].value,x1.shape[2].value])


def multivariate_gaussian2(x1, x2, x3, mu1, mu2, mu3, sigma1, sigma2, sigma3, rho12,rho13,rho23):
    ###3D
    n = 3

    def likelihood_mvn(x1_, x2_, x3_, mu1_, mu2_, mu3_, sigma1_, sigma2_, sigma3_, rho12_,rho13_,rho23_):
        convariance_mat = [[tf.square(sigma1_),rho12_*sigma1_*sigma2_,rho13_*sigma1_*sigma3_],
                            [rho12_*sigma1_*sigma2_,tf.square(sigma2_),rho23_*sigma2_*sigma3_],
                            [rho13_*sigma1_*sigma3_,rho23_*sigma2_*sigma3_,tf.square(sigma3_)]]
        convariance_mat = _make_positive_semidefinite_tf(convariance_mat,n)
        mvn = tf.contrib.distributions.MultivariateNormalFullCovariance(loc=[mu1_, mu2_, mu3_], covariance_matrix = convariance_mat)
        return mvn.prob([x1_,x2_,x3_])

    def flatten_me(t):
        return tf.reshape(t, [-1])

    elems_flat = tf.stack([flatten_me(x1),flatten_me(x2),flatten_me(x3),flatten_me(mu1),
                    flatten_me(mu2),flatten_me(mu3),flatten_me(sigma1),flatten_me(sigma2),
                    flatten_me(sigma3),flatten_me(rho12),flatten_me(rho13),flatten_me(rho23)])
    elems_flat = tf.transpose(tf.reshape(elems_flat,[12,-1]),(1,0)) #shape=1600*12 #1700=8*20*10
    likelihood = tf.map_fn(lambda x: likelihood_mvn(x[0],x[1],x[2],x[3],x[4],x[5],x[6],x[7],x[8],x[9],x[10],x[11]), elems_flat)
    return tf.reshape(tf.stack(likelihood),[cfg.batch_size,x1.shape[1].value,x1.shape[2].value])


def berrnoulli_loss(end_of_stroke,y_true):
    assert cfg.predict_eos
    data_dim=3
    if cfg.process_in_seconds:
        y_end_of_stroke = y_true[:,0,2::data_dim]
    else:
        y_end_of_stroke = y_true[:,:,2]

    prediction_length = y_true.shape[1].value
    end_of_stroke = expand(end_of_stroke,2,prediction_length)

    eps = 1e-20
    loss_bernoulli = tf.reduce_sum(
                        -tf.log((end_of_stroke + eps) * y_end_of_stroke
                        + (1 - end_of_stroke + eps) * (1 - y_end_of_stroke)))
    return loss_bernoulli


def expand(x, dim, N):
    return tf.concat([tf.expand_dims(x, dim) for _ in range(N)],dim)

def mixture_bivariate_gaussian_loss(y_true,y_pred):
    if cfg.predict_eos:
        end_of_stroke,mixture_pi,us,sigmas,rhos = y_pred
        data_dim=3
    else:
        mixture_pi,us,sigmas,rhos = y_pred
        data_dim=2
    n_mixture = mixture_pi.shape[1].value
    mu1 = us[:,0::2]
    mu2 = us[:,1::2]
    sigma1 = sigmas[:,0::2]
    sigma2 = sigmas[:,1::2]
    if cfg.process_in_seconds:
        # y_true.shape=N*1*60
        y1 = y_true[:,0,0::data_dim]#only allow one second for now
        y2 = y_true[:,0,1::data_dim]
        # duplicate mu,std for whole second  shape becomes: batch_size*n_mixture*fps
        mu1 = expand(mu1,2,cfg.fps)
        mu2 = expand(mu2,2,cfg.fps)
        sigma1 =  expand(sigma1,2,cfg.fps)
        sigma2 =  expand(sigma2,2,cfg.fps)
        rhos = expand(rhos,2,cfg.fps)
        mixture_pi = expand(mixture_pi,2,cfg.fps)
    else:
        y1 = y_true[:,:,0]
        y2 = y_true[:,:,1]
        prediction_length = y_true.shape[1].value
        mu1 = expand(mu1,2,prediction_length)
        mu2 = expand(mu2,2,prediction_length)
        sigma1 =  expand(sigma1,2,prediction_length)
        sigma2 =  expand(sigma2,2,prediction_length)
        rhos = expand(rhos,2,prediction_length)
        mixture_pi = expand(mixture_pi,2,prediction_length)


    temp = bivariate_gaussian(expand(y1, 1, n_mixture), 
                                                expand(y2, 1, n_mixture), 
                                                mu1, mu2, sigma1, sigma2, rhos)#shape=(batch_size, num_mixture, seq_len)
    gaussian = mixture_pi * temp
    loss_gaussian = tf.reduce_sum(-tf.log(tf.reduce_sum(gaussian, 1) + 1e-20)) #shape=empty after reduce_sum
    #process in second level: loss summing over batch*n_mixture*fps
    #process in frame level: loss summing over batch*n_mixture*fps

    if cfg.predict_eos:
        loss = loss_gaussian + cfg.berrnoulli_loss_weight*berrnoulli_loss(end_of_stroke,y_true)        
    else:
        loss = loss_gaussian

    if cfg.process_in_seconds:
        running_length = cfg.running_length*cfg.fps
    else:
        running_length = cfg.running_length
    return loss/(cfg.batch_size)/running_length



def mixture_3d_gaussian_loss(y_true,y_pred):
    ###multivariate gaussian NLL loss, for (x,y,z) 
    mixture_pi,us,sigmas,rhos = y_pred
    data_dim=3
    n_mixture = mixture_pi.shape[1].value
    mu1 = us[:,0::data_dim]
    mu2 = us[:,1::data_dim]
    mu3 = us[:,2::data_dim]
    sigma1 = sigmas[:,0::data_dim]
    sigma2 = sigmas[:,1::data_dim]
    sigma3 = sigmas[:,2::data_dim]
    rho12 = rhos[:,0::data_dim]
    rho13 = rhos[:,1::data_dim]
    rho23 = rhos[:,2::data_dim]
    
    if cfg.process_in_seconds:
        # y_true.shape=N*1*60
        y1 = y_true[:,0,0::data_dim]#only allow one second for now
        y2 = y_true[:,0,1::data_dim]
        y3 = y_true[:,0,2::data_dim]
        # duplicate mu,std for whole second  shape becomes: batch_size*n_mixture*fps
        mu1 = expand(mu1,2,cfg.fps)
        mu2 = expand(mu2,2,cfg.fps)
        mu3 = expand(mu3,2,cfg.fps)
        sigma1 =  expand(sigma1,2,cfg.fps)
        sigma2 =  expand(sigma2,2,cfg.fps)
        sigma3 =  expand(sigma3,2,cfg.fps)
        rho12 = expand(rho12,2,cfg.fps)
        rho13 = expand(rho13,2,cfg.fps)
        rho23 = expand(rho23,2,cfg.fps)
        mixture_pi = expand(mixture_pi,2,cfg.fps)
    else:
        y1 = y_true[:,:,0]
        y2 = y_true[:,:,1]
        y3 = y_true[:,:,2]
        prediction_length = y_true.shape[1].value
        mu1 = expand(mu1,2,prediction_length)
        mu2 = expand(mu2,2,prediction_length)
        mu3 = expand(mu3,2,prediction_length)
        sigma1 =  expand(sigma1,2,prediction_length)
        sigma2 =  expand(sigma2,2,prediction_length)
        sigma3 =  expand(sigma3,2,prediction_length)
        rho12 = expand(rho12,2,prediction_length)
        rho13 = expand(rho13,2,prediction_length)
        rho23 = expand(rho23,2,prediction_length)
        mixture_pi = expand(mixture_pi,2,prediction_length)

    gaussian =  multivariate_gaussian(expand(y1, 1, n_mixture), 
                                                expand(y2, 1, n_mixture), 
                                                expand(y3, 1, n_mixture), 
                                                mu1, mu2, mu3, 
                                                sigma1, sigma2, sigma3,
                                                rho12,rho13,rho23)
    loss_gaussian = tf.reduce_sum(-tf.log(tf.reduce_sum(gaussian, 1) + 1e-20))
    #process in second level: loss summing over batch*n_mixture*fps
    #process in frame level: loss summing over batch*n_mixture*fps

    loss = loss_gaussian

    if cfg.process_in_seconds:
        running_length = cfg.running_length*cfg.fps
    else:
        running_length = cfg.running_length
    return loss/(cfg.batch_size)/running_length









def weighted_categorical_crossentropy(weights):
    """
    @url: https://gist.github.com/wassname/ce364fddfc8a025bfab4348cf5de852d
    @author: wassname
    A weighted version of keras.objectives.categorical_crossentropy
    
    Variables:
        weights: numpy array of shape (C,) where C is the number of classes
    
    Usage:
        weights = np.array([0.5,2,10]) # Class one at 0.5, class 2 twice the normal weights, class 3 10x.
        loss = weighted_categorical_crossentropy(weights)
        model.compile(loss=loss,optimizer='adam')
    """
    
    weights = K.variable(weights)
        
    def loss(y_true, y_pred):
        # scale predictions so that the class probas of each sample sum to 1
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
        # clip to prevent NaN's and Inf's
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        # calc
        loss = weights*(y_true * K.log(y_pred))
        loss = -K.sum(loss, -1)
        return loss
    
    return loss





def total_variation_loss_keras(pred_raw_trj):
    # for keras
    # assert K.ndim(pred_raw_trj) == 4
    x = pred_raw_trj[:,:,0]
    y = pred_raw_trj[:,:,1]
    z = pred_raw_trj[:,:,2]

    deltax = K.square(x[:,:-1]-x[:,1:])
    deltay = K.square(y[:,:-1]-y[:,1:])
    deltaz = K.square(z[:,:-1]-z[:,1:])
    return K.sum(K.pow(deltax+deltay+deltaz, 1.25))

def total_variation_loss_tf(pred_raw_trj):
    # for tensorflow
    x = pred_raw_trj[:,:,0::3]
    y = pred_raw_trj[:,:,1::3]
    z = pred_raw_trj[:,:,2::3]

    deltax = tf.square(x[:,:-1]-x[:,1:])
    deltay = tf.square(y[:,:-1]-y[:,1:])
    deltaz = tf.square(z[:,:-1]-z[:,1:])
    return tf.reduce_sum(tf.pow(deltax+deltay+deltaz, 1.25))
    # return tf.reduce_mean(tf.pow(deltax+deltay+deltaz, 1.25)) should use reduce_mean?



def sum1reg_tf(pred_raw_trj):
    # for tensorflow
    if pred_raw_trj.shape[-1].value==90:
        x = pred_raw_trj[:,:,0::3]
        y = pred_raw_trj[:,:,1::3]
        z = pred_raw_trj[:,:,2::3]
    elif pred_raw_trj.shape[-1].value==3 or pred_raw_trj.shape[-1].value==6:
        x = pred_raw_trj[:,:,0]
        y = pred_raw_trj[:,:,1]
        z = pred_raw_trj[:,:,2]

    reg = tf.reduce_sum(tf.square(x**2+y**2+z**2-1))
    #tf.reduce_mean?
    return reg


def pred_raw_loss_tf(this_y,predict_sample,use_reg=False):
    lamdaTV = 0.1
    loss = tf.losses.mean_squared_error(this_y,predict_sample)+lamdaTV*total_variation_loss_tf(predict_sample)
    if use_reg:
        lambdareg = 0.1
        loss+=lambdareg*sum1reg_tf(predict_sample)
    return loss


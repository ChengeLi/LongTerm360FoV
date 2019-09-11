import numpy as np
from scipy.stats import norm
from sklearn.neighbors import KernelDensity
import _pickle as pickle

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn import mixture

temp = pickle.load(open('./data/exp_2_xyz.p','rb'))


gt_gmm_param = {}
for video_ind in temp.keys():
    print 'video_ind',video_ind
    datadb = temp[video_ind]
    gt_gmm_param[video_ind] = {}
    # X = datadb['z'][:,:100].flatten()[:,np.newaxis]
    # X_plot = np.linspace(-1, 1, 2*len(X))[:, np.newaxis]
    # bins = np.linspace(-1, 1, 300)

    for t in range(int(datadb['y'].shape[1]/100)+1):
        gt_gmm_param[video_ind][t] = {}
        X = datadb['y'][:,t*100:(t+1)*100].flatten()[:,np.newaxis]
        # fig, ax = plt.subplots()
        # # histogram 
        # ax.hist(X[:, 0], bins=bins, fc='#AAAAFF', normed=True)

        # for kernel in ['gaussian', 'tophat', 'epanechnikov']:
        #     kde = KernelDensity(kernel=kernel, bandwidth=0.05).fit(X)
        #     log_dens = kde.score_samples(X_plot)
        #     ax.plot(X_plot[:, 0], np.exp(log_dens), '-',
        #             label="kernel = '{0}'".format(kernel))
        # GMM
        # fit a Gaussian Mixture Model with two components
        clf = mixture.GaussianMixture(n_components=2, covariance_type='full')
        clf.fit(X)
        # log_dens = clf.score_samples(X_plot)
        # ax.plot(X_plot[:, 0], np.exp(log_dens), '-',label="GMM")

        # plt.legend(loc='best')
        # plt.show()
        # plt.savefig('gmm_z_'+str(t)+'.png')
        # plt.close()

        # get GMM params
        weight1,weight2 = clf.weights_
        mu1,mu2 = clf.means_[:,0]
        var1,var2 = clf.covariances_[:,0,0]

        # always keep the smaller-mean Gaussian as the first comp
        if weight1<=weight2:
            ind = [0,1]
        else:
            ind = [1,0]

        gt_gmm_param[video_ind][t]['weight1'] = clf.weights_[ind[0]]
        gt_gmm_param[video_ind][t]['weight2'] = clf.weights_[ind[1]]
        gt_gmm_param[video_ind][t]['mu1'] = clf.means_[:,0][ind[0]]
        gt_gmm_param[video_ind][t]['mu2'] = clf.means_[:,0][ind[1]]
        gt_gmm_param[video_ind][t]['var1'] = clf.covariances_[:,0,0][ind[0]]
        gt_gmm_param[video_ind][t]['var2'] = clf.covariances_[:,0,0][ind[1]]

        # plot GMM
        # plt.figure()
        # plt.plot(bins, weight1*1/(np.sqrt(var1) * np.sqrt(2 * np.pi))*np.exp( - (bins - mu1)**2 / (2 * var1) )
        #                 +weight2*1/(np.sqrt(var2) * np.sqrt(2 * np.pi))*np.exp( - (bins - mu2)**2 / (2 * var2)), 
        #                   linewidth=2, color='r')


pickle.dump(gt_gmm_param,open('./data/gt_gmm_param_exp2_y.p','wb'))




























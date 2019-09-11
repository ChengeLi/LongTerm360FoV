"""change distribution for new second based on population's choice"""

def product_of_2_gaussian(mu1,mu2,var1,var2):
    newmu = (mu1*var2+mu2*var1)/(var1+var2)
    newvar = (var1*var2)/(var1+var2)
    return newmu, newvar



def result_mu_var(gt_gmm_param,video_ind,t):
    w1 = gt_gmm_param[video_ind][t]['weight1']
    w2 = gt_gmm_param[video_ind][t]['weight2']
    mu1 = gt_gmm_param[video_ind][t]['mu1']
    mu2 = gt_gmm_param[video_ind][t]['mu2']
    var1 = gt_gmm_param[video_ind][t]['var1']
    var2 = gt_gmm_param[video_ind][t]['var2']
    # weight1*1/(np.sqrt(var1) * np.sqrt(2 * np.pi))*np.exp( - (bins - mu1)**2 / (2 * var1) )
    #                       +weight2*1/(np.sqrt(var2) * np.sqrt(2 * np.pi))*np.exp( - (bins - mu2)**2 / (2 * var2))


    # product of two gaussian (prior distribution and population distribution)
    # population distribution is a 2-component GMM
    # hence the resultant distribution is also a 2-component GMM
    newmu1, newvar1 = product_of_2_gaussian(mu1,mu_prior,var1,var_prior)
    newmu2, newvar2 = product_of_2_gaussian(mu2,mu_prior,var2,var_prior)
    return newmu1, newvar1, newmu2, newvar2 




















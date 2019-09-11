from __future__ import print_function
import tensorflow as tf
import sys,os,pdb
import numpy as np
if './360video/' not in sys.path:
    sys.path.insert(0, './360video/')
from mycode.dataLayer import DataLayer
import mycode.cost as costfunc
import _pickle as pickle
from mycode.config import cfg
from mycode.dataIO import clip_xyz
from mycode.utility import get_gt_target_xyz,generate_fake_batch,snapshot
from mycode.CNN import normalize2pi,normalize_cos_sin
from matplotlib import pyplot as plt

# Parameters
training_epochs = 200
batch_size = cfg.batch_size #need to be smaller than (one video length)/running_length
display_step = 10
fps = 30
data_dim = 3*fps


# Network Parameters
n_hidden_1 = 512  # 1st layer number of neurons
n_hidden_2 = 1024 # 2nd layer number of neurons
# n_hidden_3 = 8
# n_hidden_4 = 8
n_input = 48*cfg.running_length*data_dim
n_output = cfg.predict_len*6 #predict 1 step further
# tf Graph input
X = tf.placeholder("float", [None, n_input])
Y = tf.placeholder("float", [None, cfg.predict_len, data_dim])
#duplicate the last location
# LAST_LOC = tf.placeholder("float", [None, n_output]) 

is_test = False
# tag = '_200_50_modi_mse_2layer_16_16'
# tag = '_200_50_modi_mse_1layer_256'
tag = 'seconds_input48_'
# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1], stddev=0.001)),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2], stddev=0.001)),
    # 'h3': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3], stddev=0.001)),
    # 'h4': tf.Variable(tf.random_normal([n_hidden_3, n_hidden_4], stddev=0.001)),
    'out': tf.Variable(tf.random_normal([n_hidden_2, n_output], stddev=0.001))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1], stddev=0.001)),
    'b2': tf.Variable(tf.random_normal([n_hidden_2], stddev=0.001)),
    # 'b3': tf.Variable(tf.random_normal([n_hidden_3], stddev=0.001)),
    # 'b4': tf.Variable(tf.random_normal([n_hidden_4], stddev=0.001)),
    'out': tf.Variable(tf.random_normal([n_output], stddev=0.001))
}

 
# Create model
def multilayer_perceptron(x):
    # Hidden fully connected layer with 256 neurons
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    # Hidden fully connected layer with 256 neurons
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)
    # layer_3 = tf.add(tf.matmul(layer_2, weights['h3']), biases['b3'])
    # layer_3 = tf.nn.relu(layer_3)
    # layer_4 = tf.add(tf.matmul(layer_3, weights['h4']), biases['b4'])
    # layer_4 = tf.nn.relu(layer_4)
    # Output fully connected layer with a neuron for each class
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer




def _modified_mse(out,y):
    """modified MSE loss func, calculate the min()"""
    diff = tf.abs(out-y)
    return tf.reduce_sum(tf.where(tf.greater(diff,tf.constant(np.pi)),
            tf.square(tf.constant(2*np.pi)-diff),
            tf.square(diff)))

def flatten_batch(batch_x):
    temp = np.zeros((batch_size,n_input))
    for ii in range(batch_size):
        temp[ii,:] = batch_x[ii,:,:].flatten()
    return temp


# data IO
if cfg.use_xyz:
    all_video_data = pickle.load(open('./data/exp_1_xyz.p','rb'))
    # all_video_data = pickle.load(open('./data/exp_2_xyz.p','rb'))
elif cfg.use_cos_sin:
    all_video_data = pickle.load(open('./data/exp_2_raw_pair.p','rb'))
else:
    all_video_data = pickle.load(open('./data/exp_2_raw.p','rb'))


datadb = clip_xyz(all_video_data)
data_io = DataLayer(datadb, random=False, is_test=False)

# Construct model
mlp_output = multilayer_perceptron(X)
#residual structure to ensure smoothness
# out = mlp_output+LAST_LOC 
ux,uy,uz,varx,vary,varz = split_into_u_var(mlp_output)

# if cfg.use_cos_sin:
#     out = normalize_cos_sin(out)            
# else:
#     out = normalize2pi(out)

# Define loss and optimizer
# loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
#     logits=logits, labels=Y))

# if cfg.use_own_history:
#     predicted_tail =out[:,-cfg.predict_len:]
#     Y_tail = Y[:,-cfg.predict_len:]
#     # loss_op = tf.losses.mean_squared_error(predicted_tail,Y_tail)
#     loss_op = _modified_mse(predicted_tail,Y_tail)
# else:
#     loss_op = tf.losses.mean_squared_error(tf.squeeze(out),tf.squeeze(Y))

target = get_gt_target_xyz(Y)
loss_op = costfunc._mean_var_cost_xyz(ux,uy,uz,varx,vary,varz,target)
lr = tf.Variable(cfg.LEARNING_RATE, trainable=False)
optimizer = tf.train.AdamOptimizer(learning_rate=lr)
train_op = optimizer.minimize(loss_op)

# summary
all_losses_dict = {}
all_losses_dict['MSE_loss'] = loss_op
event_summaries = {}
event_summaries.update(all_losses_dict)
summaries = []
for key, var in event_summaries.items():
    summaries.append(tf.summary.scalar(key, var))

# summary_op = tf.summary.merge_all()
summary_op = tf.summary.merge(summaries)
saver = tf.train.Saver(max_to_keep=3)

# Initializing the variables
init = tf.global_variables_initializer()
if not is_test:
    with tf.Session() as sess:
        sess.run(init)
        summary_writer = tf.summary.FileWriter('./tfsummary/'+tag, sess.graph)
        # Training cycle
        total_batch = 8*int(datadb[0]['x'].shape[1]/cfg.running_length/fps/batch_size)
        for epoch in range(training_epochs):
            avg_cost = 0.
            # Loop over all batches
            for iter in range(total_batch):
                (batch_x, batch_y, batch_x_others),_,_,batch_x_others_further = data_io._get_next_minibatch(datadb,batch_size)
                batch_x = flatten_batch(batch_x)

                # Run optimization op (backprop) and cost op (to get loss value)
                _, c, summary = sess.run([train_op, loss_op, summary_op], feed_dict={X: batch_x,
                                                                Y: batch_y})
                                                                # LAST_LOC: rep_last_loc})
                # Compute average loss
                avg_cost += c / total_batch
                summary_writer.add_summary(summary, float(epoch*total_batch+iter))
            if epoch % display_step == 0:
                print("epoch:", '%04d' % (epoch+1), "cost={:.9f}".format(avg_cost))
            if epoch!=0 and epoch % 100 == 0:
                snapshot(sess, (epoch), saver,'MLP',tag)
                lr_temp = cfg.LEARNING_RATE*(0.5**(epoch/100))
                print('epoch: ',epoch, ', change lr=lr*0.5, lr=', lr_temp)
                sess.run(tf.assign(lr, lr_temp))

        snapshot(sess, (epoch), saver,'MLP',tag)
        print("Optimization Finished!")

elif is_test:
    with tf.Session() as sess:
        sess.run(init)
        filename = 'MLP_'+tag + '_epoch_{:d}'.format(100) + '.ckpt'
        filename = os.path.join(cfg.OUTPUT_DIR, filename)
        saver.restore(sess, filename)
        print("Model restored.")
        data_io_test = DataLayer(datadb, random=False, is_test=True)

        test_out = []
        gt_out = []
        for ii in range(10):
            (batch_x, batch_y, batch_x_others), batch_y_further,db_index,batch_x_others_further = data_io_test._get_next_minibatch(datadb,batch_size)
            gt_out.append(batch_y_further)
            for predict_step in range(cfg.predict_step):
                batch_x_flat = flatten_batch(batch_x)
                loss,ux_temp,uy_temp,uz_temp,varx_temp,vary_temp,varz_temp = sess.run(
                            [loss_op,ux,uy,uz,varx,vary,varz],
                                feed_dict={X: batch_x_flat,
                                Y: batch_y})
                                # LAST_LOC: rep_last_loc})
                test_out.append([ux_temp,uy_temp,uz_temp,varx_temp,vary_temp,varz_temp])
                temp_newdata = np.stack((generate_fake_batch(ux_temp,varx_temp),
                    generate_fake_batch(uy_temp,vary_temp),
                    generate_fake_batch(uz_temp,varz_temp)),axis=-1)[:,np.newaxis,:,:].reshape((batch_size,1,-1))

                if cfg.own_history_only:                    
                    batch_x = np.concatenate((batch_x[:,1:,:],temp_newdata),axis=1)
                else:
                    temp_newdata = np.concatenate((batch_x_others,temp_newdata[:,np.newaxis,:,:]),axis=1)
                    batch_x = np.concatenate((batch_x[:,:,1:,:],temp_newdata),axis=2)
    
    pickle.dump(test_out,open('MLP_test_out'+tag+'.p','wb'))
    pickle.dump(gt_out,open('MLP_gt_out'+tag+'.p','wb'))
    print("Finish testing.")







from __future__ import division, print_function, absolute_import
import tensorflow as tf
import sys,os,pdb
if './360video/' not in sys.path:
    sys.path.insert(0, './360video/')
from mycode.dataLayer import DataLayer
import numpy as np
import _pickle as pickle
from mycode.config import cfg
from mycode.dataIO import clip_xyz
import mycode.cost as costfunc
from mycode.cost import _modified_mse
from mycode.utility import get_gt_target_xyz,generate_fake_batch,snapshot,split_into_u_var

# Create some wrappers for simplicity
def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


def maxpool2d(x, k=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')


def normalize2pi(tensor):
    # constrain within [-pi,pi]
    tensor = tf.div(
        tf.subtract(tensor, tf.reduce_min(tensor)), 
        tf.subtract(tf.reduce_max(tensor), tf.reduce_min(tensor)))#0~1
    tensor = tf.multiply(tensor,tf.constant(2*np.pi))-tf.constant(np.pi)
    # tensor = tf.multiply(tensor,tf.constant(np.pi))-np.pi/2
    return tensor

def normalize_cos_sin(tensor):
    # TODO: how to constrain?
    # for now only -1~1
    tensor = tf.div(
        tf.subtract(tensor, tf.reduce_min(tensor)), 
        tf.subtract(tf.reduce_max(tensor), tf.reduce_min(tensor)))#0~1
    tensor = tf.subtract(tf.multiply(tensor,2), 1)
    return tensor


# Create model
def conv_net(x, weights, biases, dropout, num_user, running_length=cfg.running_length):
    # Reshape to match picture format [Height x Width x Channel]
    # Tensor input become 4-D: [Batch Size, Height, Width, Channel]
#     x = tf.reshape(x, shape=[-1, data_dim, cfg.running_length, num_user])
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    conv3 = conv2d(conv2, weights['wc3'], biases['bc3'])
    
    out = tf.nn.conv2d(conv3, weights['out'], strides=[1, 1, 1, 1], padding='SAME')
    out = tf.nn.bias_add(out, biases['out'])
    
    # fc layers
#     conv3_flat = tf.reshape(conv3, [-1, num_user*fps*cfg.running_length])
#     dense = tf.layers.dense(inputs=conv3_flat, units=1024, activation=tf.nn.relu)
#     dropout = tf.layers.dropout(
#                 inputs=dense, rate=dropout, training= (not is_test))
#     logits = tf.layers.dense(inputs=dropout, units=6)
    return conv1,conv2,conv3,out



def print_test(sess,batch_x,batch_y,test_out,gt_out):
    conv1out,conv2out,conv3out,outy,loss = sess.run(
                                    [conv1,conv2,conv3,out,loss_op], 
                                    feed_dict={X: batch_x[:,np.newaxis,:],
                                      Y: batch_y[:,np.newaxis,:],
                                      LAST_LOC: rep_last_loc[:,np.newaxis,:,np.newaxis],
                                      keep_prob: 1.0})

    print("Testing:", loss)
    print(conv1out.min(),conv1out.max())
    print(conv2out.min(),conv2out.max())
    print(conv3out.min(),conv3out.max())
    test_out.append(np.squeeze(outy[:,:,-cfg.predict_len:,:])+rep_last_loc)
    gt_out.append(batch_y)
    return test_out,gt_out



if __name__ == '__main__':
    # execfile('code/dataIO.py')
    # print('data preparation done!')
    
    tag = ''
    # Training Parameters
    is_test = False
    learning_rate = 0.001
    batch_size = 32
    display_step = 2
    training_epochs = 200
    fps = 30
    data_dim = 3
    num_user = 47
    if cfg.include_own_history:
        num_user = num_user+1
    # Network Parameters
    # num_input = num_user*cfg.running_length
    # num_output = 1*cfg.predict_len
    # if cfg.has_reconstruct_loss:
    #     num_output = 1*cfg.running_length
    dropout = 0.75 # Dropout, probability to keep units

    # tf Graph input
    X = tf.placeholder(tf.float32, [None,num_user,2*cfg.running_length*fps,data_dim])
    Y = tf.placeholder(tf.float32, [None,cfg.running_length*fps,data_dim])
    # Y = tf.placeholder(tf.float32, [None,cfg.predict_step,6]) #predict mu & var

    #duplicate the last location
    # LAST_LOC = tf.placeholder("float", [None,data_dim,cfg.predict_len,1]) 
    keep_prob = tf.placeholder(tf.float32) # dropout (keep probability)

    weights = {
        'wc1': tf.Variable(tf.random_normal([9, 9, data_dim, 256], stddev=0.1)),
        'wc2': tf.Variable(tf.random_normal([9, 9, 256, 512], stddev=0.1)),
        'wc3': tf.Variable(tf.random_normal([9, 9, 512, 512], stddev=0.1)),
    #     'out': tf.Variable(tf.random_normal([9, 9, 512, 1], stddev=0.1)),
        'out': tf.Variable(tf.random_normal([9, 9, 512, data_dim], stddev=0.1)),

        # fully connected
    #     'wd1': tf.Variable(tf.random_normal([48*300, 1024], stddev=0.01)),
    #     'outw': tf.Variable(tf.random_normal([1024, 6]))
    }

    biases = {
        'bc1': tf.Variable(tf.random_normal([256], stddev=0.1)),
        'bc2': tf.Variable(tf.random_normal([512], stddev=0.1)),
        'bc3': tf.Variable(tf.random_normal([512], stddev=0.1)),
        'out': tf.Variable(tf.random_normal([data_dim], stddev=0.1)),
    #     'bd1': tf.Variable(tf.random_normal([1024], stddev=0.01)),
    #     'outb': tf.Variable(tf.random_normal([6]))
    }   
 

    # Construct model
    conv1,conv2,conv3,out = conv_net(X, weights, biases, keep_prob, num_user, cfg.running_length)
    # if (not cfg.has_reconstruct_loss) and cfg.add_residual_link:
    #     finalout = out+LAST_LOC #residual link
    # else:
    #     finalout = out
    # if cfg.has_reconstruct_loss:
    #     loss_op1 = tf.losses.mean_squared_error(finalout[:,:,:,0],Y)
    #     loss_op = _modified_mse(finalout,Y)
    # else:
    #     predicted_tail = finalout[:,:,-cfg.predict_len:,0]
    #     loss_op1 = tf.losses.mean_squared_error(predicted_tail,Y)
    #     loss_op = _modified_mse(predicted_tail,Y)

    # ux,uy,uz,varx,vary,varz = split_into_u_var(out)
    # target = get_gt_target_xyz(Y)
    # loss_op = costfunc._mean_var_cost_xyz(ux,uy,uz,varx,vary,varz,target)

    predicted_tail = out[:,-1,-cfg.running_length*fps:,:]
    loss_op = tf.losses.mean_squared_error(predicted_tail,Y)

    lr = tf.Variable(cfg.LEARNING_RATE, trainable=False)
    optimizer = tf.train.AdamOptimizer(learning_rate=lr)
    train_op = optimizer.minimize(loss_op)



    # summary
    all_losses_dict = {}
    # all_losses_dict['MSE_loss'] = loss_op1
    all_losses_dict['modified_MSE_loss'] = loss_op
    event_summaries = {}
    event_summaries.update(all_losses_dict)
    summaries = []
    for key, var in event_summaries.items():
        summaries.append(tf.summary.scalar(key, var))
    summary_op = tf.summary.merge(summaries)
    saver = tf.train.Saver()


    # Evaluate model
    # correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
    # accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # Initialize the variables (i.e. assign their default value)
    init = tf.global_variables_initializer()

    ## data IO
    if cfg.use_xyz:
        all_video_data = pickle.load(open('./data/exp_1_xyz.p','rb'))
        # all_video_data = pickle.load(open('./data/exp_2_xyz.p','rb'))
    elif cfg.use_cos_sin:
        all_video_data = pickle.load(open('./data/exp_2_raw_pair.p','rb'))
    else:
        all_video_data = pickle.load(open('./data/exp_2_raw.p','rb'))

    datadb = clip_xyz(all_video_data)
    data_io = DataLayer(datadb, random=False, is_test=is_test)

    if not is_test:
        # Start training
        with tf.Session() as sess:
            # Run the initializer
            sess.run(init)
            summary_writer = tf.summary.FileWriter('./tfsummary/', sess.graph)
            total_batch = 8*int(datadb[0]['x'].shape[1]/cfg.running_length/fps/batch_size)
            for epoch in range(training_epochs):
                avg_cost = 0.
                for step in range(1, total_batch):
                    # print('step',step)
                    batch_x, batch_y = data_io._get_next_minibatch(datadb,batch_size,'CNN')
                    # Run optimization op (backprop)
                    _, c, summary = sess.run([train_op, loss_op, summary_op], feed_dict={X: batch_x,
                                                                    Y: batch_y,
                                                                    keep_prob: 0.85})
                    avg_cost += c / total_batch
                    summary_writer.add_summary(summary, float(epoch*total_batch+step))


                if epoch % display_step == 0:
                    print("epoch:", '%04d' % (epoch+1), "cost={:.9f}".format(avg_cost))
                if epoch!=0 and epoch % 100 == 0:
                    snapshot(sess, (epoch), saver,'CNN', tag)
                    lr_temp = cfg.LEARNING_RATE*(0.5**(epoch/100))
                    print('epoch: ',epoch, ', change lr=lr*0.5, lr=', lr_temp)
                    sess.run(tf.assign(lr, lr_temp))

            snapshot(sess, (epoch), saver,'CNN', tag)
            print("Optimization Finished!")

    elif is_test:
        # testing
        with tf.Session() as sess:
            # Run the initializer
            sess.run(init)
            # Restore variables
            filename = 'CNN_'+tag + '_epoch_{:d}'.format(100) + '.ckpt'
            filename = os.path.join(cfg.OUTPUT_DIR, filename)
            saver.restore(sess, filename)
            print("Model restored.")
            data_io_test = DataLayer(datadb, random=False, is_test=True)


            test_out = []
            gt_out = []
            for ii in range(10):
                batch_x, batch_y = data_io_test._get_next_minibatch(datadb,batch_size,'CNN')
                test_out,gt_out = print_test(sess,batch_x,batch_y,test_out,gt_out)
        



        pickle.dump(test_out,open('CNN_test_out'+tag+'.p','wb'))
        pickle.dump(gt_out,open('CNN_gt_out'+tag+'.p','wb'))

 











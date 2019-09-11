"""
self implemented attention layer. 
Compute similarity, context vector, attention weights.
"""


# # use attention
# self.W_att = tf.get_variable("W_att",shape=[self.h_dim,1]) # h x 1
# tmp3 = tf.matmul(tf.reshape(self.M,shape=[self.batch_size*self.XMAXLEN,self.h_dim]),self.W_att)
# # need 1 here so that later can do multiplication with h x L
# self.att = tf.nn.softmax(tf.reshape(tmp3,shape=[self.batch_size,1, self.XMAXLEN],name="att")) # nb x 1 x Xmax
# # print "att",self.att



    
    






    

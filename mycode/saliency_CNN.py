"""
Input: saliency image 60 channels
output: image features 
"""

from keras.layers import Conv2D,MaxPooling2D,Flatten,Dense
from keras.layers import Lambda
import sys
if './360video/' not in sys.path:
    sys.path.insert(0, './360video/')
import mycode.utility as util
get_dim1_layer = Lambda(lambda x: x[:,0,:])


pooling = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))
conv1 = Conv2D(filters=512, kernel_size=(3,3), strides=(1, 1),padding='same',
                 activation='relu', dilation_rate=(1, 1))
conv2 = Conv2D(filters=1024, kernel_size=(3,3), strides=(1, 1),padding='same',
                 activation='relu', dilation_rate=(1, 1))
conv3 = Conv2D(filters=1024, kernel_size=(3,3), strides=(1, 1),padding='same',
                 activation='relu', dilation_rate=(1, 1))
conv4 = Conv2D(filters=512, kernel_size=(3,3), strides=(1, 1),padding='same',
                 activation='relu', dilation_rate=(1, 1))
conv5 = Conv2D(64, (3, 3), activation='relu')


def get_CNN_fea(saliency_inputs,time_ind, final_dim=256):
    saliency_inputs_slice = util.slice_layer(1,time_ind,time_ind+1)(saliency_inputs)
    _saliency = conv1(get_dim1_layer(saliency_inputs_slice))
    _saliency = conv2(_saliency)
    _saliency = pooling(_saliency)
    _saliency = conv3(_saliency)
    _saliency = conv4(_saliency)
    _saliency = pooling(_saliency)
    _saliency = conv5(_saliency)
    _saliency = Flatten()(_saliency)
    _saliency = Dense(final_dim, activation='relu')(_saliency)
    return _saliency






















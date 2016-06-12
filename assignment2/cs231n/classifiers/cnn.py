import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *




#class Crpcra(object):
#  """
#  A variable number of layers convolutional network with the following architecture:
#  
#  [conv-relu-pool]xN - conv - relu - [affine]xM - [softmax or SVM]
#  
#  The network operates on minibatches of data that have shape (N, C, H, W)
#  consisting of N images, each with height H and width W and with C input
#  channels.
#  """
#  
#  def __init__(self, conv_layers, aff_layers, input_dim=(3, 32, 32), num_classes=10, weight_scale=1e-3, reg=0.0, use_batchnorm=False,dropout=0, dtype=np.float32):
#    """
#    Initialize a new network.
#    
#    Inputs:
#    - input_dim: Tuple (C, H, W) giving size of input data
#    - conv_layers: list of tuples with num_filters and filter_size for every conv layer
#    - aff_layers: list of number of units to use in every affine layer
#    - num_classes: Number of scores to produce from the final affine layer.
#    - weight_scale: Scalar giving standard deviation for random initialization
#      of weights.
#    - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=0 then
#      the network should not use dropout at all.
#    - use_batchnorm: Whether or not the network should use batch normalization.
#    - reg: Scalar giving L2 regularization strength
#    - dtype: numpy datatype to use for computation.
#    """
#    self.params = {}
#    self.reg = reg
#    self.dtype = dtype
#    self.use_batchnorm = use_batchnorm
#    self.use_dropout = dropout > 0
#    self.conv_num_layers = 1 + len(conv_layers)
#    self.aff_num_layers = 1+len(aff_layers)
#    
#    ############################################################################
#    # TODO: Initialize weights and biases for the three-layer convolutional    #
#    # network. Weights should be initialized from a Gaussian with standard     #
#    # deviation equal to weight_scale; biases should be initialized to zero.   #
#    # All weights and biases should be stored in the dictionary self.params.   #
#    # Store weights and biases for the convolutional layer using the keys 'W1' #
#    # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
#    # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
#    # of the output affine layer.                                              #
#    ############################################################################
#    
#    self.params['W1']=np.random.randn(conv_layers[0][0],input_dim[0],conv_layers[0][1],conv_layers[0][1])*weight_scale
#    self.params['b1']=np.zeros(conv_layers[0][0]) 
#    stride=1
#    pad=(conv_layers[0][1]-1)/2
#    out_conv_H=((input_dim[1]-conv_layers[0][1]+2*pad)/stride+1)/2
#    out_conv_W=((input_dim[2]-conv_layers[0][1]+2*pad)/stride+1)/2
#    for i in range(1,conv_num_layers-1):
#        self.params['W'+str(1+i)]=np.random.randn(conv_layers[i][0],conv_layers[i-1][0],conv_layers[i][1],conv_layrs[i][1])*weight_scale
#        self.params['b'+str(1+i)]=np.zeros(conv_layers[i][0]) 
#        out_conv_H=((out_conv_H-conv_layers[i][1]+2*pad)/stride+1)/2
#        out_conv_W=((out_conv_W-conv_layers[i][1]+2*pad)/stride+1)/2
#        
#    self.params['W'+str(1+conv_num_layers)]=np.random.randn(out_conv_H*out_conv_W*conv_layers[1][0],aff_layers[0])*weight_scale
#    self.params['b'+str(1+conv_num_layers)]=np.zeros(aff_layers[0])
#    
#    for i in range(aff_num_layers-1):
#        self.params['W'+str(conv_num_layers+1+i)]=np.random.randn(aff_layers[i],aff_layers[i+1])*weight_scale
#        self.params['b'+str(conv_num_layers+1+i)]=np.zeros(aff_layers[i+1])
#    
#    self.params['W'+str(conv_num_layers+1+aff_num_layers)]=np.random.randn(aff_layers[aff_num_layers],num_classes)*weight_scale
#    self.params['b'+str(conv_num_layers+1+aff_num_layers)]=np.zeros(num_classes)
#
##
##    #divide by 2 because of pooling layer. Again this is a default, so not sure whether it is valid
##    self.params['W2']=np.random.randn(out_conv_H/2*out_conv_W/2*num_filters, hidden_dim)*weight_scale
##    #self.params['W2']=    
##    self.params['b2']=np.zeros(hidden_dim)    
##    self.params['W3']=np.random.randn(hidden_dim,num_classes)*weight_scale
##    self.params['b3']=np.zeros(num_classes)
#    ############################################################################
#    #                             END OF YOUR CODE                             #
#    ############################################################################
#
#    for k, v in self.params.iteritems():
#      self.params[k] = v.astype(dtype)
#     
# 
#  def loss(self, X, y=None):
#    """
#    Evaluate loss and gradient for the three-layer convolutional network.
#    
#    Input / output: Same API as TwoLayerNet in fc_net.py.
#    """
#    W1, b1 = self.params['W1'], self.params['b1']
#    W2, b2 = self.params['W2'], self.params['b2']
#    W3, b3 = self.params['W3'], self.params['b3']
#    
#    # pass conv_param to the forward pass for the convolutional layer
#    filter_size = W1.shape[2]
#    conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}
#
#    # pass pool_param to the forward pass for the max-pooling layer
#    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}
#
#    scores = None
#    ############################################################################
#    # TODO: Implement the forward pass for the three-layer convolutional net,  #
#    # computing the class scores for X and storing them in the scores          #
#    # variable.                                                                #
#    ############################################################################
#    #conv - relu - 2x2 max pool - affine - relu - affine - softmax
#    #print "conv1"
#    h1, cache_h1=conv_relu_pool_forward(X, W1, b1, conv_param, pool_param)
#    #print "aff_relu1"
#    #print "p1=",p1.shape
#    #print "w2=", W2.shape
#    h2, cache_h2=affine_relu_forward(h1,W2,b2)
#    #print "aff_1"
#    scores, cache_h3=affine_forward(h2,W3,b3)
#    ############################################################################
#    #                             END OF YOUR CODE                             #
#    ############################################################################
#    
#    if y is None:
#      return scores
#    
#    loss, grads = 0, {}
#    ############################################################################
#    # TODO: Implement the backward pass for the three-layer convolutional net, #
#    # storing the loss and gradients in the loss and grads variables. Compute  #
#    # data loss using softmax, and make sure that grads[k] holds the gradients #
#    # for self.params[k]. Don't forget to add L2 regularization!               #
#    ############################################################################
#    loss, grad_s=softmax_loss(scores,y)
#    loss+=self.reg*0.5*(np.sum(W1**2)+np.sum(W2**2)+np.sum(W3**2))
#    dx3,dw3,db3=affine_backward(grad_s,cache_h3)
#    dx2,dw2,db2=affine_relu_backward(dx3,cache_h2)
#    
#    dx1, dw1, db1=conv_relu_pool_backward(dx2, cache_h1)    
#    
#    #dxp1=max_pool_backward_fast(dx2,cache_p1)
#    #dxr1=relu_backward(dxp1,cache_r1)
#    #dx1, dw1, db1=conv_backward_im2col(dxr1, cache_h1)
#    grads['W1']=dw1+self.reg*W1
#    grads['b1']=db1
#    grads['W2']=dw2+self.reg*W2
#    grads['b2']=db2
#    grads['W3']=dw3+self.reg*W3
#    grads['b3']=db3
#    ############################################################################
#    #                             END OF YOUR CODE                             #
#    ############################################################################
#    
#    return loss, grads


class ThreeLayerConvNet(object):
  """
  A three-layer convolutional network with the following architecture:
  
  conv - relu - 2x2 max pool - affine - relu - affine - softmax
  
  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """
  
  def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
               hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
               dtype=np.float32):
    """
    Initialize a new network.
    
    Inputs:
    - input_dim: Tuple (C, H, W) giving size of input data
    - num_filters: Number of filters to use in the convolutional layer
    - filter_size: Size of filters to use in the convolutional layer
    - hidden_dim: Number of units to use in the fully-connected hidden layer
    - num_classes: Number of scores to produce from the final affine layer.
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights.
    - reg: Scalar giving L2 regularization strength
    - dtype: numpy datatype to use for computation.
    """
    self.params = {}
    self.reg = reg
    self.dtype = dtype
    
    ############################################################################
    # TODO: Initialize weights and biases for the three-layer convolutional    #
    # network. Weights should be initialized from a Gaussian with standard     #
    # deviation equal to weight_scale; biases should be initialized to zero.   #
    # All weights and biases should be stored in the dictionary self.params.   #
    # Store weights and biases for the convolutional layer using the keys 'W1' #
    # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
    # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
    # of the output affine layer.                                              #
    ############################################################################
    self.params['W1']=np.random.randn(num_filters,input_dim[0],filter_size,filter_size)*weight_scale
    self.params['b1']=np.zeros(num_filters)
    #these are the defaults of size stride and pad so I used them, but not sure if this is valid
    stride=1
    pad=(filter_size-1)/2
    out_conv_H=(input_dim[1]-filter_size+2*pad)/stride+1
    out_conv_W=(input_dim[2]-filter_size+2*pad)/stride+1

    #divide by 2 because of pooling layer. Again this is a default, so not sure whether it is valid
    self.params['W2']=np.random.randn(out_conv_H/2*out_conv_W/2*num_filters, hidden_dim)*weight_scale
    #self.params['W2']=    
    self.params['b2']=np.zeros(hidden_dim)    
    self.params['W3']=np.random.randn(hidden_dim,num_classes)*weight_scale
    self.params['b3']=np.zeros(num_classes)
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)
     
 
  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the three-layer convolutional network.
    
    Input / output: Same API as TwoLayerNet in fc_net.py.
    """
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    W3, b3 = self.params['W3'], self.params['b3']
    
    # pass conv_param to the forward pass for the convolutional layer
    filter_size = W1.shape[2]
    conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}

    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the three-layer convolutional net,  #
    # computing the class scores for X and storing them in the scores          #
    # variable.                                                                #
    ############################################################################
    #conv - relu - 2x2 max pool - affine - relu - affine - softmax
    #print "conv1"
    h1, cache_h1=conv_relu_pool_forward(X, W1, b1, conv_param, pool_param)
    #print "aff_relu1"
    #print "p1=",p1.shape
    #print "w2=", W2.shape
    h2, cache_h2=affine_relu_forward(h1,W2,b2)
    #print "aff_1"
    scores, cache_h3=affine_forward(h2,W3,b3)
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    if y is None:
      return scores
    
    loss, grads = 0, {}
    ############################################################################
    # TODO: Implement the backward pass for the three-layer convolutional net, #
    # storing the loss and gradients in the loss and grads variables. Compute  #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    ############################################################################
    loss, grad_s=softmax_loss(scores,y)
    loss+=self.reg*0.5*(np.sum(W1**2)+np.sum(W2**2)+np.sum(W3**2))
    dx3,dw3,db3=affine_backward(grad_s,cache_h3)
    dx2,dw2,db2=affine_relu_backward(dx3,cache_h2)
    
    dx1, dw1, db1=conv_relu_pool_backward(dx2, cache_h1)    
    
    #dxp1=max_pool_backward_fast(dx2,cache_p1)
    #dxr1=relu_backward(dxp1,cache_r1)
    #dx1, dw1, db1=conv_backward_im2col(dxr1, cache_h1)
    grads['W1']=dw1+self.reg*W1
    grads['b1']=db1
    grads['W2']=dw2+self.reg*W2
    grads['b2']=db2
    grads['W3']=dw3+self.reg*W3
    grads['b3']=db3
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    return loss, grads
  

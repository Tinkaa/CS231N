import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *

class Crpcra(object):
  """
  A variable number of layers convolutional network with the following architecture:
  
  [conv-relu-pool]xN - conv - relu - [affine]xM - [softmax or SVM]
  
  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """
  
  def __init__(self, conv_layers, aff_layers, input_dim=(3, 32, 32), num_classes=10, weight_scale=1e-3, reg=0.0, use_batchnorm=False,dropout=0, dtype=np.float32, seed=None):
    """
    Initialize a new network.
    
    Inputs:
    - input_dim: Tuple (C, H, W) giving size of input data
    - conv_layers: list of tuples with num_filters and filter_size for every conv layer
    - aff_layers: list of number of units to use in every affine layer
    - num_classes: Number of scores to produce from the final affine layer.
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights.
    - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=0 then
      the network should not use dropout at all.
    - use_batchnorm: Whether or not the network should use batch normalization.
    - reg: Scalar giving L2 regularization strength
    - dtype: numpy datatype to use for computation.
    """

    self.params = {}
    self.conv_param={}
    self.reg = reg
    self.dtype = dtype
    self.use_batchnorm = use_batchnorm
    self.use_dropout = dropout > 0
    self.conv_num_layers = conv_num_layers = len(conv_layers)
    self.aff_num_layers = aff_num_layers = len(aff_layers)+1
    
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
    self.pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}
    aff_layers.append(num_classes)
    self.params['W1']=np.random.randn(conv_layers[0][0],input_dim[0],conv_layers[0][1],conv_layers[0][1])*weight_scale
    self.params['b1']=np.zeros(conv_layers[0][0]) 
    stride=1
    pad=(conv_layers[0][1]-1)/2
    out_conv_H=((input_dim[1]-conv_layers[0][1]+2*pad)/stride+1)/2
    out_conv_W=((input_dim[2]-conv_layers[0][1]+2*pad)/stride+1)/2
    self.conv_param['1']={'stride':1, 'pad':(conv_layers[0][1]-1)/2}    
    for i in range(1,conv_num_layers-1):
        self.params['W'+str(1+i)]=np.random.randn(conv_layers[i][0],conv_layers[i-1][0],conv_layers[i][1],conv_layers[i][1])*weight_scale
        self.params['b'+str(1+i)]=np.zeros(conv_layers[i][0]) 
        out_conv_H=((out_conv_H-conv_layers[i][1]+2*pad)/stride+1)/2
        out_conv_W=((out_conv_W-conv_layers[i][1]+2*pad)/stride+1)/2
        self.conv_param[str(1+i)]={'stride':1, 'pad':(conv_layers[i][1]-1)/2}        
        if use_batchnorm:        
            self.params['gamma'+str(1+i)]=np.ones(conv_layers[i][0])        
            self.params['beta'+str(1+i)]=np.zeros(conv_layers[i][0])        
     
    i=conv_num_layers-1
    self.params['W'+str(conv_num_layers)]=np.random.randn(conv_layers[i][0],conv_layers[i-1][0],conv_layers[i][1],conv_layers[i][1])*weight_scale    
    self.params['b'+str(1+i)]=np.zeros(conv_layers[i][0]) 
    self.conv_param[str(1+i)]={'stride':1, 'pad':(conv_layers[i][1]-1)/2}
    out_conv_H=((out_conv_H-conv_layers[i][1]+2*pad)/stride+1)
    out_conv_W=((out_conv_W-conv_layers[i][1]+2*pad)/stride+1)
    if use_batchnorm:        
        self.params['gamma'+str(1+i)]=np.ones(conv_layers[i][0])        
        self.params['beta'+str(1+i)]=np.zeros(conv_layers[i][0])  
    #self.params['W'+str(conv_num_layers)]=np.random.randn(out_conv_H*out_conv_W*conv_layers[conv_num_layers-1][0],aff_layers[0])*weight_scale   
    #aff_layers includes all the hidden dimensions and the num_classes
    

    self.params['W'+str(conv_num_layers+1)]=np.random.randn(out_conv_H*out_conv_W*conv_layers[conv_num_layers-1][0],aff_layers[0])*weight_scale 
    self.params['b'+str(conv_num_layers+1)]=np.zeros(aff_layers[0]) 
    if use_batchnorm:
        self.params['gamma'+str(conv_num_layers+1)]=np.ones(aff_layers[0])
        self.params['beta'+str(conv_num_layers+1)]=np.zeros(aff_layers[0]) 
            
    for i in range(1,aff_num_layers):
        self.params['W'+str(conv_num_layers+1+i)]=np.random.randn(aff_layers[i-1],aff_layers[i])*weight_scale
        self.params['b'+str(conv_num_layers+1+i)]=np.zeros(aff_layers[i])
        if use_batchnorm:
            self.params['gamma'+str(conv_num_layers+2+i)]=np.ones(aff_layers[i])
            self.params['beta'+str(conv_num_layers+2+i)]=np.zeros(aff_layers[i])        
        
    
    
    


        # When using dropout we need to pass a dropout_param dictionary to each
    # dropout layer so that the layer knows the dropout probability and the mode
    # (train / test). You can pass the same dropout_param to each dropout layer.
    self.dropout_param = {}
    if self.use_dropout:
      self.dropout_param = {'mode': 'train', 'p': dropout}
      if seed is not None:
        self.dropout_param['seed'] = seed
    
    # With batch normalization we need to keep track of running means and
    # variances, so we need to pass a special bn_param object to each batch
    # normalization layer. You should pass self.bn_params[0] to the forward pass
    # of the first batch normalization layer, self.bn_params[1] to the forward
    # pass of the second batch normalization layer, etc.
    self.bn_params = []
    if self.use_batchnorm:
      self.bn_params = [{'mode': 'train'} for i in xrange(self.num_layers - 1)]
      
    # Cast all parameters to the correct datatype  
    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)
     
 
  def loss(self, X, y=None):
     """
    Compute loss and gradient for a minibatch of data.

    Inputs:
    - X: Array of input data of shape (N, d_1, ..., d_k)
    - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

    Returns:
    If y is None, then run a test-time forward pass of the model and return:
    - scores: Array of shape (N, C) giving classification scores, where
      scores[i, c] is the classification score for X[i] and class c.

    If y is not None, then run a training-time forward and backward pass and
    return a tuple of:
    - loss: Scalar value giving the loss
    - grads: Dictionary with the same keys as self.params, mapping parameter
      names to gradients of the loss with respect to those parameters.
      """
     mode = 'test' if y is None else 'train'

    # Set train/test mode for batchnorm params and dropout param since they
    # behave differently during training and testing.
     if self.dropout_param is not None:
       self.dropout_param['mode'] = mode   
     if self.use_batchnorm:
       for bn_param in self.bn_params:
         bn_param[mode] = mode

     scores = None

    #Compute the forward pass for the crpcra network. The class scores for X are being stored in the scores variable.
     cache={}
     out=X
     for i in range(1,self.conv_num_layers):    
         out, cache[str(i)]=conv_relu_pool_forward(out, self.params['W'+str(i)], self.params['b'+str(i)], self.conv_param[str(i)], self.pool_param)

     out, cache[str(self.conv_num_layers)]=conv_relu_forward(out, self.params['W'+str(self.conv_num_layers)], self.params['b'+str(self.conv_num_layers)], self.conv_param[str(self.conv_num_layers)])
     
     for i in range(1,self.aff_num_layers): 
         if i==1:
             print "first out", out.shape
             print "first w", self.params['W'+str(self.conv_num_layers+i)].shape
         out,cache[str(self.conv_num_layers+i)]=affine_relu_forward(out,self.params['W'+str(self.conv_num_layers+i)],self.params['b'+str(self.conv_num_layers+i)])
         
     i=self.aff_num_layers    
     out,cache[str(self.conv_num_layers+i)]=affine_forward(out,self.params['W'+str(self.conv_num_layers+i)],self.params['b'+str(self.conv_num_layers+i)])
     scores=out

    
     if y is None:
       return scores
    
     loss, grads = 0, {}
   
   ##The backward pass for the CNN ##
     num_layers=self.conv_num_layers+self.aff_num_layers     
     loss, dx=softmax_loss(scores,y)
     reg_w=0
     for i in range(1,num_layers+1):
         reg_w+=np.sum(self.params['W'+str(i)]**2)
     loss+=self.reg*0.5*reg_w
     dx,dw,db=affine_backward(dx,cache[str(num_layers)])
     grads['W'+str(num_layers)]=dw+self.reg*self.params['W'+str(num_layers)]
     grads['b'+str(num_layers)]=db
     for i in range(num_layers-1,num_layers-self.aff_num_layers,-1):
         dx,dw,db=affine_relu_backward(dx,cache[str(i)])
         grads['W'+str(i)]=dw+self.reg*self.params['W'+str(i)]
         grads['b'+str(i)]=db

     dx,dw,db=conv_relu_backward(dx,cache[str(self.conv_num_layers)])
     grads['W'+str(self.conv_num_layers)]=dw+self.reg*self.params['W'+str(self.conv_num_layers)]
     grads['b'+str(self.conv_num_layers)]=db
    
     for i in range(self.conv_num_layers-1,0,-1):
         dx, dw, db=conv_relu_pool_backward(dx, cache[str(i)])
         grads['W'+str(i)]=dw+self.reg*self.params['W'+str(i)]
         grads['b'+str(i)]=db
    
    #dxp1=max_pool_backward_fast(dx2,cache_p1)
    #dxr1=relu_backward(dxp1,cache_r1)
    #dx1, dw1, db1=conv_backward_im2col(dxr1, cache_h1)

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
     return loss, grads
     
     
     
     
     
class Try2(object):
  """
  A three-layer convolutional network with the following architecture:
  
  conv - relu - 2x2 max pool - affine - relu - affine - softmax
  
  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """
  
  def __init__(self, conv_layers, aff_layers, input_dim=(3, 32, 32), num_classes=10, weight_scale=1e-3, reg=0.0,
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
    aff_layers.append(num_classes)
    self.num_layers=len(aff_layers)+len(conv_layers) #the +1 for last layer to num_classes
    self.conv_num=len(conv_layers)
    self.aff_num=len(aff_layers)
    self.conv_params={}
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
    self.params['W1']=np.random.randn(conv_layers[0][0],input_dim[0],conv_layers[0][1],conv_layers[0][1])*weight_scale
    self.params['b1']=np.zeros(conv_layers[0][0])
    
    #these are the defaults of size stride and pad so I used them, but not sure if this is valid
    stride=1
    pad=(conv_layers[0][1]-1)/2
    self.conv_params['1'] = {'stride': stride, 'pad': pad}
    #divide by 2 because of pooling layer. Again this is a default, so not sure whether it is valid
    out_conv_H=((input_dim[1]-conv_layers[0][1]+2*pad)/stride+1)/2
    out_conv_W=((input_dim[2]-conv_layers[0][1]+2*pad)/stride+1)/2
    
    for i in range(1,self.conv_num-1):
        print i
        self.params['W'+str(i+1)]=np.random.randn(conv_layers[i][0],conv_layers[i-1][0],conv_layers[i][1],conv_layers[i][1])*weight_scale
        self.params['b'+str(i+1)]=np.zeros(conv_layers[i][0])
        pad=(conv_layers[i][1]-1)/2
        out_conv_H=((out_conv_H-conv_layers[i][1]+2*pad)/stride+1)/2
        out_conv_W=((out_conv_W-conv_layers[i][1]+2*pad)/stride+1)/2
        self.conv_params[str(i+1)] = {'stride': stride, 'pad': pad}
    
    self.params['W'+str(self.conv_num)]=np.random.randn(conv_layers[self.conv_num-1][0],conv_layers[self.conv_num-2][0],conv_layers[self.conv_num-1][1],conv_layers[self.conv_num-1][1])*weight_scale
    self.params['b'+str(self.conv_num)]=np.zeros(conv_layers[self.conv_num-1][0])    
    pad=(conv_layers[self.conv_num-1][1]-1)/2    
    out_conv_H=(out_conv_H-conv_layers[self.conv_num-1][1]+2*pad)/stride+1
    out_conv_W=(out_conv_W-conv_layers[self.conv_num-1][1]+2*pad)/stride+1
    self.conv_params[str(self.conv_num)] = {'stride': stride, 'pad': pad}
    
    self.params['W'+str(self.conv_num+1)]=np.random.randn(out_conv_H*out_conv_W*conv_layers[self.conv_num-1][0], aff_layers[0])*weight_scale   
    self.params['b'+str(self.conv_num+1)]=np.zeros(aff_layers[0])  
    
    for i in range(1,self.aff_num):
        self.params['W'+str(self.conv_num+1+i)]=np.random.randn(aff_layers[i-1],aff_layers[i])*weight_scale
        self.params['b'+str(self.conv_num+1+i)]=np.zeros(aff_layers[i])
        
    self.pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}
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
    
    # pass conv_param to the forward pass for the convolutional layer

    

    # pass pool_param to the forward pass for the max-pooling layer
    

    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the three-layer convolutional net,  #
    # computing the class scores for X and storing them in the scores          #
    # variable.                                                                #
    ############################################################################
    #conv - relu - 2x2 max pool - affine - relu - affine - softmax
    #print "conv1"
    cache={}
    out=X
    for i in range(1,self.conv_num):
        out,cache[str(i)]=conv_relu_pool_forward(out, self.params['W'+str(i)], self.params['b'+str(i)], self.conv_params[str(i)], self.pool_param)
    
    out,cache[str(self.conv_num)]=conv_relu_forward(out,self.params['W'+str(self.conv_num)],self.params['b'+str(self.conv_num)],self.conv_params[str(self.conv_num)])    
    
    for i in range(1,self.aff_num):
        out, cache[str(self.conv_num+i)]=affine_relu_forward(out,self.params['W'+str(self.conv_num+i)],self.params['b'+str(self.conv_num+i)])
    
    scores, cache[str(self.num_layers)]=affine_forward(out,self.params['W'+str(self.num_layers)],self.params['b'+str(self.num_layers)])
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
    regw=0
    for i in range(1,self.num_layers):
        regw+=np.sum(self.params['W'+str(i)]**2)
    loss+=self.reg*0.5*regw
    dx,dw,db=affine_backward(grad_s,cache[str(self.num_layers)])
    grads['W'+str(self.num_layers)]=dw+self.reg*self.params['W'+str(self.num_layers)]
    grads['b'+str(self.num_layers)]=db    
    
    for i in range(self.num_layers-1,self.conv_num,-1):
        dx,dw,db=affine_relu_backward(dx,cache[str(i)])
        grads['W'+str(i)]=dw+self.reg*self.params['W'+str(i)]
        grads['b'+str(i)]=db
    dx, dw, db=conv_relu_backward(dx, cache[str(self.conv_num)])
    grads['W'+str(self.conv_num)]=dw+self.reg*self.params['W'+str(self.conv_num)]
    grads['b'+str(self.conv_num)]=db
    for i in range(self.conv_num-1,0,-1):
        dx, dw, db=conv_relu_pool_backward(dx, cache[str(i)]) 
        grads['W'+str(i)]=dw+self.reg*self.params['W'+str(i)]
        grads['b'+str(i)]=db

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    return loss, grads
  
class ConvNet(object):
    """
    This is a ConvNet to which an arbitrary combination of layers can be given as input. 
    The possible combinations are conv-relu (cr), conv-relu-pool(crp),affine-relu(ar). 
    The last layer is always an affine layer with a softmax classification and is not given as input
    """
    def __init__(self,structure,conv_layers,aff_layers,num_classes=10, weight_scale=1e-3, use_batchnorm=False, dropout=0, reg=0.0, input_dim=(3,32,32),dtype=np.float32):
        """
        Initialize a network
        
        Inputs:
        - structure: a list of strings indicating the layers, as indicated in the description above (cr, crp, ar)
        - conv_layers: a list of tuples with length the number of convolutional layers. The first index of the tuple is the number of filters, the second the filter size
        - aff_layers: a list with with length the number of affine layers, giving the size of the hidden dimensions
        - num_classes: Number of scores to produce from the final affine layer.
        - weight_scale: Scalar giving standard deviation for random initialization
          of weights.
        - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=0 then
          the network should not use dropout at all.
        - use_batchnorm: Whether or not the network should use batch normalization.
        - reg: Scalar giving L2 regularization strength
        - input_dim: Tuple (C, H, W) giving size of input data
        - dtype: numpy datatype to use for computation.
        """
        self.structure=structure
        self.weight_scale=weight_scale
        self.num_classes=num_classes
        self.dropout=dropout
        self.use_batchnorm=use_batchnorm
        self.reg=reg
        self.input_dim=input_dim
        self.num_layers=len(conv_layers)+len(aff_layers)+1
        
        self.params={}
        
        self.bn_params=[{'mode':'train'} for i in range(self.num_layers)]
        self.strides=[1 for i in range(len(conv_layers))]
        self.pads=[(conv_layers[s][1]-1)/2 for s in range(len(conv_layers))]
        
        C,H,W=input_dim
        
        conv=0
        pool=0
        ar=0
        for i in range(len(structure)):
            stri=str(i+1)
            lay=structure[i]
            if lay=='cr' or lay=='crp':
                self.__init_cr(stri,conv_layers[conv],C)
                C=conv_layers[conv][0]
                conv+=1
                if lay=='crp':
                    pool+=1
            elif lay=='ar':
                if ar==0:
                    self.__init_af(stri,H*W*conv_layers[-1][0]*0.25*pool,aff_layers[0])
                    ar+=1
                else:
                    self.__init_af(stri,aff_layers[ar-1],aff_layers[ar])
                    ar+=1
            else:
                raise ValueError('structure does not exist: %s' %structure[i])

        self.__init_finalaf(str(conv+ar+1),aff_layers[-1],num_classes)
    def loss(self,X,y=None):
        scores,cache=self.__forward_all(X)
        if y==None:
           return scores
        else:
            loss, grads=0,{}
            loss, dscores=softmax_loss(scores,y)
            reg_w=0
            for i in range(1,self.num_layers+1):
                reg_w+=np.sum(self.params['W'+str(i)]**2)
            loss+=self.reg*0.5*reg_w
            print loss
            grads=self.__backward_all(dscores,cache)
            return loss,grads
                
    def __init_cr(self,i,conv,C):
        
        """
        initialize a convolutional layer
        """
        self.params['W'+i]=np.random.randn(conv[0],C,conv[1],conv[1])*self.weight_scale
        self.params['b'+i]=np.zeros(conv[0])   
        if self.use_batchnorm:
            self.params['gamma'+i]=np.ones(conv[0])
            self.params['beta'+i]=np.zeros(conv[0])
            
    def __init_af(self,i,h1,h2):
        """
        Initialize an affine layer
        """
        self.params['W'+i]=np.random.randn(h1,h2)*self.weight_scale
        self.params['b'+i]=np.zeros(h2)
        if self.use_batchnorm:
            self.params['gamma'+i]=np.ones(h2)
            self.params['beta'+i]=np.zeros(h2)
            
    def __init_finalaf(self, i,h1,classes):
        """
        Initialize the final layer
        """
        self.params['W'+i]=np.random.randn(h1,classes)*self.weight_scale
        self.params['b'+i]=np.zeros(classes)
        
    
    def __forward_cr(self,i,inp,bn_param,conv_param):
        if self.use_batchnorm:
            out,cache=conv_bn_relu_forward(inp,self.params['W']+i,self.params['b'+i],self.params['gamma'+i],self.params['beta'+i],conv_param,bn_param)
        else:
            out,cache=conv_relu_forward(X,self.params['W'+i],self.params['b'+i],conv_param)
        return out,cache
        
    def __forward_crp(self,i,inp,bn_param,conv_parm,pool_param):
        if self.use_batchnorm:
            out,cache=conv_bn_relu_pool_forward(inp,self.params['W']+i,self.params['b'+i],self.params['gamma'+i],self.params['beta'+i],conv_param,bn_param,pool_param)
        else:
            out,cache=conv_relu_pool_forward(X,self.params['W'+i],self.params['b'+i],conv_param,pool_param)
        return out,cache
        
    def __forward_ar(self,i,inp):
        if self.use_batchnorm:
            out,cache=affine_bn_relu_forward(inp,self.params['W'+i],self.params['b'+i],bn_param,self.params['gamma'+i],self.params['beta'+i])
        else:
            out,cache=affine_relu_forward(inp,self.params['W'+i],self.params['b'+i])
        return out, cache
        
    def __forward_finalaf(self,i,inp):
        out,cache=affine_forward(inp,self.params['W'+i],self.params['b'+i])
        return out,cache
    
    def __forward_all(self,X):
         out=X
         cache={}
         pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}
         for i in range(len(self.structure)):
            stri=str(i+1)
            bn_param=self.bn_params[i]
            lay=self.structure[i]
            if lay=='cr':
                conv_param={'stride':self.strides[i],'pad':self.pads[i]}
                out,cache[stri]=self.__forward_cr(stri,out,bn_param,conv_param)
            if lay=='crp':
                conv_param={'stride':self.strides[i],'pad':self.pads[i]}
                out,cache[stri]=self.__forward_crp(stri,out,bn_param,conv_param,pool_param)
            if lay=='ar':
                out,cache[stri]=self.__forward_ar(stri,out)
         scores,cache[str(self.num_layers)]=self.__forward_finalaf(str(self.num_layers),out)
         return scores,cache
         
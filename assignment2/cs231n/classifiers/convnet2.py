import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *

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
        self.use_dropout=dropout > 0
        self.use_batchnorm=use_batchnorm
        self.reg=reg
        self.input_dim=input_dim
        self.num_layers=len(conv_layers)+len(aff_layers)+1
        
        self.dropout_param = {'mode': 'train', 'p': dropout}
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
                    self.__init_af(stri,H*W*conv_layers[-1][0]*0.25**pool,aff_layers[0])
                    ar+=1
                else:
                    self.__init_af(stri,aff_layers[ar-1],aff_layers[ar])
                    ar+=1
            else:
                raise ValueError('structure does not exist: %s' %structure[i])
        self.__init_finalaf(str(conv+ar+1),aff_layers[-1],num_classes)
        
    def loss(self,X,y=None):
        scores,cache=self.__forward_all(X)
        self.cache=cache
        if y is None:
           return scores
        else:
            loss, grads=0,{}
            loss, dscores=softmax_loss(scores,y)
            reg_w=0
            for i in range(1,self.num_layers+1):
                reg_w+=np.sum(self.params['W'+str(i)]**2)
            loss+=self.reg*0.5*reg_w
            #print loss
            self.__backward_all(dscores,cache)
            grads=self.grads
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
            out,cache=conv_bn_relu_forward(inp,self.params['W'+i],self.params['b'+i],self.params['gamma'+i],self.params['beta'+i],conv_param,bn_param)
        else:
            out,cache=conv_relu_forward(inp,self.params['W'+i],self.params['b'+i],conv_param)
        return out,cache
        
    def __forward_crp(self,i,inp,bn_param,conv_param,pool_param):
        if self.use_batchnorm:
            out,cache=conv_bn_relu_pool_forward(inp,self.params['W'+i],self.params['b'+i],self.params['gamma'+i],self.params['beta'+i],conv_param,bn_param,pool_param)
        else:
            out,cache=conv_relu_pool_forward(inp,self.params['W'+i],self.params['b'+i],conv_param,pool_param)
        return out,cache
        
    def __forward_ar(self,i,inp,bn_param):
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
                out,cache[stri]=self.__forward_ar(stri,out,bn_param)
                if self.use_dropout:
                    out, cache['drop'+stri]=dropout_forward(out,self.dropout_param)
         scores,cache[str(self.num_layers)]=self.__forward_finalaf(str(self.num_layers),out)
         return scores,cache
         
         
    def __backward_cr(self,stri,dx,cache):
        if self.use_batchnorm:
            dx,dw,db,dgamma,dbeta=conv_bn_relu_backward(dx,cache)
            self.grads['gamma'+stri]=dgamma
            self.grads['beta'+stri]=dbeta
        else:
            dx,dw,db=conv_relu_backward(dx,cache)
        self.grads['W'+stri]=dw+self.reg*self.params['W'+stri]
        self.grads['b'+stri]=db
        return dx
        
    def __backward_crp(self,stri,dx,cache):
        if self.use_batchnorm:
            dx,dw,db,dgamma,dbeta=conv_bn_relu_pool_backward(dx,cache)
            self.grads['gamma'+stri]=dgamma
            self.grads['beta'+stri]=dbeta
        else:
            dx,dw,db=conv_relu_pool_backward(dx,cache)
        self.grads['W'+stri]=dw+self.reg*self.params['W'+stri]
        self.grads['b'+stri]=db
        return dx
    
    def __backward_ar(self,stri,dx,cache):
        if self.use_batchnorm:
            dx,dw,db,dgamma,dbeta=affine_bn_relu_backward(dx,cache)
            self.grads['gamma'+stri]=dgamma
            self.grads['beta'+stri]=dbeta
        else:
            dx,dw,db=affine_relu_backward(dx,cache)
        self.grads['W'+stri]=dw+self.reg*self.params['W'+stri]
        self.grads['b'+stri]=db
        return dx
        
    def __backward_all(self,dscores,cache):
        dx=dscores
        self.grads={}
        dx,dw,db=affine_backward(dx,cache[str(self.num_layers)])
        self.grads['W'+str(self.num_layers)]=dw+self.reg*self.params['W'+str(self.num_layers)]
        self.grads['b'+str(self.num_layers)]=db
        for i in xrange(self.num_layers-2,-1,-1):
            stri=str(i+1)
            lay=self.structure[i]
            if lay=='cr':
                dx=self.__backward_cr(stri,dx,cache[stri])
            elif lay=='crp':
                dx=self.__backward_crp(stri,dx,cache[stri])
            elif lay=='ar':
                if self.use_dropout:
                    dx=dropout_backward(dx,cache['drop'+stri])
                dx=self.__backward_ar(stri,dx,cache[stri])
            
            
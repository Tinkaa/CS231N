import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  num_train=X.shape[0]
  num_classes=W.shape[1]
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  for i in xrange(num_train):
      scores=X[i].dot(W)
      scores-=np.max(scores)
      norm_prob=np.exp(scores)/np.sum(np.exp(scores))
      #print norm_prob[y[i]]
      loss+=-np.log(norm_prob[y[i]])
      #I don't know yet why this formula is the derivative of the softmax function. Have to understand this!
      for j in xrange(num_classes):
          dW[:,j]+=X[i]*(-1*(j==y[i])+np.exp(scores[j])/np.sum(np.exp(scores)))
  loss /= num_train
  dW /= num_train
  
  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(W * W)
  dW+=reg*W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  num_train=X.shape[0]

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################


  scores=X.dot(W)
  # the maximum element of a row is substracted from the score for numerical stability
  max_scores=np.max(scores,axis=1).reshape((num_train,1))
  scores-=max_scores
  # the formula for cross-entropy softmax loss is L_i=-log(e^scores[y[i]]/sum(e^scores[i]))
  norm_prob= np.exp(scores)/np.sum(np.exp(scores),axis=1).reshape((num_train,1))
  loss=np.sum(-np.log(norm_prob[np.arange(num_train),y]))
  loss/=num_train
  loss += 0.5*reg * np.sum(W*W)
  
  # the derivative of softmax is e^scores[j]/sum(e^scores[i])(-1 if j==y[i])
  #to update W one should multiply the derivative by X
  grad_scores=norm_prob
  cor_scores=np.zeros(grad_scores.shape)
  cor_scores[np.arange(num_train),y]=-1
  grad_scores+=cor_scores
  dW=X.T.dot(grad_scores)
  dW /= num_train
  dW+=reg*W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


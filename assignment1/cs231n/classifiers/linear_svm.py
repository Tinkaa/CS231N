import numpy as np
from random import shuffle

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

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
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in xrange(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    for j in xrange(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin
        dW[:,j]+=X[i]
        dW[:,y[i]]-=X[i]

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.

  loss /= num_train
  dW /= num_train

  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(W * W)

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################


  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero
  num_train=X.shape[0]
  num_classes=W.shape[1]
  scores=X.dot(W)
  cor_scores=scores[np.arange(num_train),y]
  cor_scores=cor_scores[:,np.newaxis]
  mar=scores-cor_scores+1
  margin=np.maximum(mar,0)
  margin[np.arange(num_train),y]=0

  loss=np.sum(margin)
  loss /= num_train

  # Add regularization
  loss += 0.5 * reg * np.sum(W * W)
#      
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  grad_scores=np.zeros(margin.shape)
  grad_scores[margin>0]=1
  num_scores=np.sum(grad_scores,axis=1)
  grad_scores[np.arange(num_train),y]=-num_scores
  dW=X.T.dot(grad_scores)
  dW /= num_train
  dW+=reg*W
#  binary_scores = np.array(margin != 0, dtype=np.float32)
#  print np.sum(binary_scores)
#  print np.sum(grad_scores)
#  # print("Our binary score's size : " + str(binary_scores.shape))
#  
#  # sum them column wise. This gets us the # of times we want to change for dWj
#  binary_score_col = np.sum(binary_scores, axis = 1)
#  # print("By summing up our binary score, we get : " + str(binary_score_col))
#  # print("Binary_score_col size: " + str(binary_score_col.shape))
#  # print("xrange(num_samples) : " + str(num_samples))
#  # print("y : " + str(y.shape))
#  binary_scores[np.arange(num_train), y] = -binary_score_col
#  print binary_scores
#  
#  dW = X.T.dot(binary_scores)/num_train + reg * W
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW

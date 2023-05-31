import numpy as np
from random import shuffle
from past.builtins import xrange

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

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_train = X.shape[0]
  num_classes = W.shape[1]
      
  scores = np.dot(X,W)
  for i in range(num_train):
        score_i = scores[i,:]
        score_i -= np.max(score_i) # numerical stability
        loss += -score_i[y[i]] + np.log(np.sum(np.exp(score_i)))
        
        
        score_i = np.exp(score_i)
        score_i /= np.sum(score_i)
        for j in range(len(score_i)):
            if(j == y[i]):
                dW[:, j] += (-1 + score_i[j])*X[i]
            else:
                dW[:, j] += score_i[j]*X[i]
  
  loss /= num_train
  dW /= num_train
  dW += 2*reg*W
  loss += reg*(np.sum(W*W))
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

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_train = X.shape[0]
  num_classes = W.shape[1]
    
  scores = np.dot(X,W)
  scores -= np.max(scores, axis = 1)[:, np.newaxis]
  exp_scores = np.exp(scores)
  sum_exp = np.sum(exp_scores, axis = 1)
  scores_ = exp_scores/sum_exp[:, np.newaxis]

  loss = -np.sum(np.log(scores_[range(num_train),y]))
  loss /= num_train
  
  Dscore = scores_
  Dscore[range(num_train), y] -= 1 # because of Softmax's cost function 
  dW = np.dot(X.T, Dscore)
  dW /= num_train 

  dW += 2*reg*W 
  loss += reg*np.sum(W*W)
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


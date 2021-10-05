from builtins import range
import random
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
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    num_train=X.shape[0]
    num_classes=W.shape[1]

    for i in range(num_train):
      scores=np.dot(X[i],W)
      scores-=np.max(scores)
      scores_exp=np.exp(scores)
      sum_score=np.sum(scores_exp)
      loss+=-scores[y[i]]+np.log(sum_score)
      for j in range(num_classes):
        if j==y[i]:
          dW[:,j]+=-X[i]
        dW[:,j]+=scores_exp[j]/sum_score*X[i]
    loss/=num_train
    loss+=0.5*reg*np.sum(W*W)
    dW/=num_train
    dW+=reg*W
    


    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

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
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    num_train=X.shape[0]
    scores=np.dot(X,W)#nxc
    shift_scores=scores-np.max(scores,1).reshape(-1,1)
    shift_scores_exp=np.exp(shift_scores)
    shift_scores_exp_sum=np.sum(shift_scores_exp,1)
    loss_mat=-shift_scores[np.arange(num_train),y]+np.log(shift_scores_exp_sum)
    loss=np.sum(loss_mat)
    loss/=num_train
    loss+=0.5*np.sum(W*W)

    dS=shift_scores_exp/shift_scores_exp_sum.reshape(-1,1)
    dS[np.arange(num_train),y]+=-1
    dW=np.dot(X.T,dS)
    dW=dW/num_train+reg*W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW

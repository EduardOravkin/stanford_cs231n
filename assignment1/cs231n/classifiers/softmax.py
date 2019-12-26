from builtins import range
import numpy as np
from random import shuffle
#from past.builtins import xrange

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

    num_train = X.shape[0]
    dim = X.shape[1]
    num_classes = W.shape[1]

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    loss = 0


    for i in range(num_train):

        Li = 0
        f = X[i].dot(W)  #f = np.matmul(X[i,:],W)  num_classes dimensional vector
        f = f - np.max(f)
        konst_i = np.sum(np.exp(f))

        loss += np.log(konst_i/np.exp(f[y[i]]))

        for j in range(num_classes):
            dW[:,j] +=  (np.exp(f[j])/konst_i) * X[i,:]

        dW[:,y[i]] += -X[i,:]


    loss = loss/num_train + reg*np.sum(W*W)
    dW = dW/num_train + 2*reg*W




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
    num_train = X.shape[0]
    dim = X.shape[1]
    num_classes = W.shape[1]

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    #NOTE THAT HERE I FORGOT TO SUBTRACT THE MAXIMUM VALUE (COMPARED TO NONVECTORIZED PART), WHEN I FORGOT TO DO IT IN
    #ASSIGNEMNT 2 THEN IT ACTUALLY CAUSED PROBLEMS. HERE IT IS FINE.

    F = np.matmul(X,W) #NxC scores matrix
    Konsts = np.log(np.sum(np.exp(F),axis=1)) #N dim vector

    loss = np.sum(Konsts-F[np.arange(num_train),y])/num_train # did I forget regularization?

    B = np.exp(F) # N x C
    A = ( (X.transpose())*1/(np.sum(B,axis=1)) ).transpose() # N x D matrix
    C = ( (np.ones((num_train,num_classes))*np.arange(num_classes)).transpose()==y ).astype(int) # CxN matrix


    dW = ( ( np.matmul(B.transpose(),A)   -  np.matmul(C,X) ).transpose() )/num_train + 2*reg*W # subtract the y_i components

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW

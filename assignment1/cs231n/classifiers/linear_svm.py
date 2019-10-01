from builtins import range
import numpy as np
from random import shuffle
#from past.builtins import xrange

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
    D=W.shape[0]
    C=W.shape[1]
    N=X.shape[0]
    #print(np.linalg.norm(dW))

    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    for i in range(num_train):
        positive_margins=0
        scores = X[i].dot(W)
        correct_class_score = scores[y[i]]
        for j in range(num_classes):
            if j == y[i]:
                continue
            margin = scores[j] - correct_class_score + 1 # note delta = 1
            if margin > 0:
                positive_margins += 1
                loss += margin
                dW[:,j] += X[i,:]

        dW[:,y[i]] =  dW[:,y[i]] - positive_margins*X[i,:]
        #print(np.linalg.norm(dW))
    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train

    # Add regularization to the loss.
    loss += reg * np.sum(W * W)

    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather that first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    #print(np.linalg.norm(dW))

    dW /= num_train
    dW += 2*reg*W


    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW



def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """
    loss = 0.0
    dW = np.zeros(W.shape) # initialize the gradient as zero
    D=W.shape[0]
    C=W.shape[1]
    N=X.shape[0]

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    #Wyi_cstack = (np.ones((C,D))*W[:,y[i]]).transpose()
    #B = (Bb.transpose()>0).astype(int)
    #A=1.0/float(N)*np.matmul(B,X)

    Y = np.matmul(X,W) # NxC
    #z = Y[np.column_stack((np.arange(N),y))]
    z = Y[np.arange(N),y]
    P = (np.ones((C,N))*z).transpose()
    B = (Y-P+1)*((Y-P+1>0).astype(int)) # NxC

    loss =1.0/float(N)*(np.sum(B) - N) + reg * np.sum(W * W)


    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the structured SVM     #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    #### baaaaaad, because it's slower than nonvectorized :(((

    for j in range(C):
        G=X*((np.ones((D,N))*((Y-P+1>0).astype(int)[:,j])).transpose()) # NxD
        dW[:,j] = np.sum(G,axis=0) #D dim vector

        f = np.matmul((Y-P+1>0).astype(int),np.ones(C)) # N dim vector
        g = (y==j).astype(int)  # N dim vector
        z = g*f # N dim vector

        H=X*((np.ones((D,N))*z).transpose()) # NxD    actually could have done (X.transpose()*z).transpose()
        dW[:,j] = dW[:,j] - np.sum(H,axis=0) #D dim vector

    dW /= N
    dW += 2*reg*W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW

from builtins import range
from builtins import object
import numpy as np

from cs231n.layers import *
from cs231n.layer_utils import *


class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network with ReLU nonlinearity and
    softmax loss that uses a modular layer design. We assume an input dimension
    of D, a hidden dimension of H, and perform classification over C classes.

    The architecure should be affine - relu - affine - softmax.

    Note that this class does not implement gradient descent; instead, it
    will interact with a separate Solver object that is responsible for running
    optimization.

    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to numpy arrays.
    """

    def __init__(self, input_dim=3*32*32, hidden_dim=100, num_classes=10,
                 weight_scale=1e-3, reg=0.0):
        """
        Initialize a new network.

        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: An integer giving the size of the hidden layer
        - num_classes: An integer giving the number of classes to classify
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - reg: Scalar giving L2 regularization strength.
        """
        self.params = {}
        self.reg = reg

        ############################################################################
        # TODO: Initialize the weights and biases of the two-layer net. Weights    #
        # should be initialized from a Gaussian centered at 0.0 with               #
        # standard deviation equal to weight_scale, and biases should be           #
        # initialized to zero. All weights and biases should be stored in the      #
        # dictionary self.params, with first layer weights                         #
        # and biases using the keys 'W1' and 'b1' and second layer                 #
        # weights and biases using the keys 'W2' and 'b2'.                         #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        self.params['W1'] = np.random.randn(input_dim,hidden_dim)*weight_scale
        self.params['b1'] = np.zeros(hidden_dim)
        self.params['W2'] = np.random.randn(hidden_dim,num_classes)*weight_scale
        self.params['b2'] = np.zeros(num_classes)


        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################


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
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the two-layer net, computing the    #
        # class scores for X and storing them in the scores variable.              #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        num_train, num_classes = X.shape[0], self.params['b2'].shape[0]

        out_a1 , cache_a1 = affine_forward(X,self.params['W1'],self.params['b1']) # NxH

        out_r1 , cache_r1 = relu_forward(out_a1) #NxH

        scores , cache_a2 = affine_forward(out_r1,self.params['W2'],self.params['b2']) #NxC


        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If y is None then we are in test mode so just return scores
        if y is None:
            return scores

        scores = (scores.transpose() - scores.max(axis=1)).transpose() # Had to renormalize because of practical reasons - see notes for softmax

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the two-layer net. Store the loss  #
        # in the loss variable and gradients in the grads dictionary. Compute data #
        # loss using softmax, and make sure that grads[k] holds the gradients for  #
        # self.params[k]. Don't forget to add L2 regularization!                   #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        #computing softmax loss

        Konsts = np.sum(np.exp(scores),axis=1) #N dim vector

        #print(np.min(Konsts))
        #print(np.max(Konsts))

        loss = np.sum(np.log(Konsts)-scores[np.arange(num_train),y])/num_train + 0.5*self.reg*(np.sum(self.params['W1']*self.params['W1'])+np.sum(self.params['W2']*self.params['W2']))

        #computing gradient

        dF = ( (np.exp(scores).transpose()/Konsts).transpose() )/num_train # NxC

        #print(scores.shape, dF.shape, num_classes, num_train, y.shape)

        dF = dF - ( (np.ones((num_classes,num_train))*y).transpose() == ( np.ones((num_train,num_classes))*np.arange(num_classes) ) ).astype(int)/num_train # NxC

        dout_r1 , grads['W2'], grads['b2'] = affine_backward(dF,cache_a2) #NxH, C, HxC

        dout_a1 = relu_backward(dout_r1,cache_r1) # NxH

        dx, grads['W1'], grads['b1'], = affine_backward(dout_a1, cache_a1) #NxD, H, DxH

        grads['W1'] = grads['W1'] + self.reg * self.params['W1'] # DxH and add regularization gradient
        grads['b1'] = grads['b1'] # H
        grads['W2'] = grads['W2'] + self.reg * self.params['W2'] # HxC and add regularization gradient
        grads['b2'] = grads['b2'] # C

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads


class FullyConnectedNet(object):
    """
    A fully-connected neural network with an arbitrary number of hidden layers,
    ReLU nonlinearities, and a softmax loss function. This will also implement
    dropout and batch/layer normalization as options. For a network with L layers,
    the architecture will be

    {affine - [batch/layer norm] - relu - [dropout]} x (L - 1) - affine - softmax

    where batch/layer normalization and dropout are optional, and the {...} block is
    repeated L - 1 times.

    Similar to the TwoLayerNet above, learnable parameters are stored in the
    self.params dictionary and will be learned using the Solver class.
    """

    def __init__(self, hidden_dims, input_dim=3*32*32, num_classes=10,
                 dropout=1, normalization=None, reg=0.0,
                 weight_scale=1e-2, dtype=np.float32, seed=None):
        """
        Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=1 then
          the network should not use dropout at all.
        - normalization: What type of normalization the network should use. Valid values
          are "batchnorm", "layernorm", or None for no normalization (the default).
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using
          this datatype. float32 is faster but less accurate, so you should use
          float64 for numeric gradient checking.
        - seed: If not None, then pass this random seed to the dropout layers. This
          will make the dropout layers deteriminstic so we can gradient check the
          model.
        """
        self.normalization = normalization
        self.use_dropout = dropout != 1
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}

        ############################################################################
        # TODO: Initialize the parameters of the network, storing all values in    #
        # the self.params dictionary. Store weights and biases for the first layer #
        # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
        # initialized from a normal distribution centered at 0 with standard       #
        # deviation equal to weight_scale. Biases should be initialized to zero.   #
        #                                                                          #
        # When using batch normalization, store scale and shift parameters for the #
        # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
        # beta2, etc. Scale parameters should be initialized to ones and shift     #
        # parameters should be initialized to zeros.                               #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        self.params['W1'] = np.random.randn(input_dim,hidden_dims[0])*weight_scale
        self.params['b1'] = np.zeros(hidden_dims[0])

        for i in np.arange(self.num_layers - 2) + 1:
            self.params['W'+str(i+1)] = np.random.randn(hidden_dims[i-1],hidden_dims[i])*weight_scale
            self.params['b'+str(i+1)] = np.zeros(hidden_dims[i])

        self.params['W'+str(self.num_layers)] = np.random.randn(hidden_dims[self.num_layers - 2],num_classes)*weight_scale
        self.params['b'+str(self.num_layers)] = np.zeros(num_classes)

        if self.normalization == 'batchnorm':

            for i in np.arange(self.num_layers - 1) + 1:
                #self.gamma[str(i)] = np.ones(hidden_dims[i-1])
                #self.beta[str(i)] = np.zeros(hidden_dims[i-1])
                self.params['gamma'+str(i)] = np.ones(hidden_dims[i-1])
                self.params['beta'+str(i)] = np.zeros(hidden_dims[i-1])

            #self.gamma[str(self.num_layers - 1)] = np.ones(hidden_dims[self.num_layers - 2])
            #self.beta[str(self.num_layers - 1)] = np.zeros(hidden_dims[self.num_layers - 2])

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

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
        if self.normalization=='batchnorm':
            self.bn_params = [{'mode': 'train'} for i in range(self.num_layers - 1)]
        if self.normalization=='layernorm':
            self.bn_params = [{} for i in range(self.num_layers - 1)]

        # Cast all parameters to the correct datatype
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)


    def loss(self, X, y=None):
        """
        Compute loss and gradient for the fully-connected net.

        Input / output: Same as TwoLayerNet above.
        """
        X = X.astype(self.dtype)
        mode = 'test' if y is None else 'train'

        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        if self.use_dropout:
            self.dropout_param['mode'] = mode
        if self.normalization=='batchnorm':
            for bn_param in self.bn_params:
                bn_param['mode'] = mode

        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the fully-connected net, computing  #
        # the class scores for X and storing them in the scores variable.          #
        #                                                                          #
        # When using dropout, you'll need to pass self.dropout_param to each       #
        # dropout forward pass.                                                    #
        #                                                                          #
        # When using batch normalization, you'll need to pass self.bn_params[0] to #
        # the forward pass for the first batch normalization layer, pass           #
        # self.bn_params[1] to the forward pass for the second batch normalization #
        # layer, etc.                                                              #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        num_train = X.shape[0]
        num_classes = self.params['b' + str(self.num_layers)].shape[0]

        cache = {}

        if self.normalization == 'batchnorm':

            if self.use_dropout:

                out, cache[str(1)] = affine_batch_relu_dropout_forward(X, self.params['W'+str(1)], self.params['b'+str(1)], self.params['gamma'+str(1)], self.params['beta'+str(1)], self.bn_params[0], self.dropout_param)

                for i in np.arange(self.num_layers - 2) + 1:

                    out, cache[str(i+1)] = affine_batch_relu_dropout_forward(out, self.params['W'+str(i+1)], self.params['b'+str(i+1)], self.params['gamma'+str(i+1)], self.params['beta'+str(i+1)], self.bn_params[i], self.dropout_param)

            else:

                out, cache[str(1)] = affine_batch_relu_forward(X, self.params['W'+str(1)], self.params['b'+str(1)], self.params['gamma'+str(1)], self.params['beta'+str(1)], self.bn_params[0])

                for i in np.arange(self.num_layers - 2) + 1:

                    out, cache[str(i+1)] = affine_batch_relu_forward(out, self.params['W'+str(i+1)], self.params['b'+str(i+1)], self.params['gamma'+str(i+1)], self.params['beta'+str(i+1)], self.bn_params[i])


        else:

            if self.use_dropout:

                out_a, cache['a_'+ str(1)] = affine_forward(X,self.params['W'+str(1)],self.params['b'+str(1)]) #could be written more nicely

                out_r, cache['r_'+ str(1)] = relu_forward(out_a)

                out_d, cache['d_' + str(1)] = dropout_forward(out_r, self.dropout_param)


                for i in np.arange(self.num_layers - 2) + 1:
                    out_a, cache['a_' + str(i+1)] = affine_forward(out_d,self.params['W'+str(i+1)],self.params['b'+str(i+1)]) #could be written more nicely

                    out_r, cache['r_' + str(i+1)] = relu_forward(out_a)

                    out_d, cache['d_' + str(i+1)] = dropout_forward(out_r, self.dropout_param)

                out = out_d

            else:

                out_a, cache['a_'+ str(1)] = affine_forward(X,self.params['W'+str(1)],self.params['b'+str(1)]) #could be written more nicely

                out_r, cache['r_'+ str(1)] = relu_forward(out_a)


                for i in np.arange(self.num_layers - 2) + 1:
                    out_a, cache['a_' + str(i+1)] = affine_forward(out_r,self.params['W'+str(i+1)],self.params['b'+str(i+1)]) #could be written more nicely

                    out_r, cache['r_' + str(i+1)] = relu_forward(out_a)

                out = out_r


        scores, cache[str(self.num_layers)] = affine_forward(out, self.params['W' + str(self.num_layers)],self.params['b' + str(self.num_layers)]) #NxC

        cache['a_' + str(self.num_layers)] = cache[str(self.num_layers)] # just in casee because of old version

        scores = (scores.transpose() - scores.max(axis=1)).transpose()


        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If test mode return early
        if mode == 'test':
            return scores

        loss, grads = 0.0, {}
        ############################################################################
        # TODO: Implement the backward pass for the fully-connected net. Store the #
        # loss in the loss variable and gradients in the grads dictionary. Compute #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        # When using batch/layer normalization, you don't need to regularize the scale   #
        # and shift parameters.                                                    #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        Konsts = np.sum(np.exp(scores),axis=1) #N dim vector

        loss = np.sum(np.log(Konsts)-scores[np.arange(num_train),y])/num_train

        regul_term = 0

        for i in range(self.num_layers):

            regul_term +=  np.sum(self.params['W' + str(i+1)]*self.params['W' + str(i+1)])

        loss = loss + regul_term * 0.5 * self.reg

        #computing gradient

        dF = ( (np.exp(scores).transpose()/Konsts).transpose() )/num_train # NxC

        #print(scores.shape, dF.shape, num_classes, num_train, y.shape)

        dF = dF - ( (np.ones((num_classes,num_train))*y).transpose() == ( np.ones((num_train,num_classes))*np.arange(num_classes) ) ).astype(int)/num_train # NxC


        #backward pass

        if self.normalization == 'batchnorm':

            if self.use_dropout:

                dout , grads['W' + str(self.num_layers)], grads['b' + str(self.num_layers)] = affine_backward(dF,cache[str(self.num_layers)])

                grads['W' + str(self.num_layers)] += self.reg * self.params['W' + str(self.num_layers)]


                for i in np.arange(self.num_layers - 1) + 1:

                    dout, grads['W' + str(self.num_layers - i)], grads['b' + str(self.num_layers - i)], grads['gamma' + str(self.num_layers - i)] , grads['beta' + str(self.num_layers - i)] = affine_batch_relu_dropout_backward(dout, cache[str(self.num_layers - i)])

                    grads['W' + str(self.num_layers - i)] += self.reg * self.params['W' + str(self.num_layers - i)]


                dx = dout

            else:

                dout , grads['W' + str(self.num_layers)], grads['b' + str(self.num_layers)] = affine_backward(dF,cache[str(self.num_layers)])

                grads['W' + str(self.num_layers)] += self.reg * self.params['W' + str(self.num_layers)]


                for i in np.arange(self.num_layers - 1) + 1:

                    dout, grads['W' + str(self.num_layers - i)], grads['b' + str(self.num_layers - i)], grads['gamma' + str(self.num_layers - i)] , grads['beta' + str(self.num_layers - i)] = affine_batch_relu_backward(dout, cache[str(self.num_layers - i)])

                    grads['W' + str(self.num_layers - i)] += self.reg * self.params['W' + str(self.num_layers - i)]


                dx = dout


        else:


            if self.use_dropout:

                dout_d , grads['W' + str(self.num_layers)], grads['b' + str(self.num_layers)] = affine_backward(dF,cache['a_' + str(self.num_layers)])

                grads['W' + str(self.num_layers)] += self.reg * self.params['W' + str(self.num_layers)]


                for i in np.arange(self.num_layers - 2) + 1:

                    dout_r = dropout_backward(dout_d, cache['d_' + str(self.num_layers - i)])

                    dout_a = relu_backward(dout_r, cache['r_' + str(self.num_layers - i)])

                    dout_d , grads['W' + str(self.num_layers - i)], grads['b' + str(self.num_layers - i)] = affine_backward(dout_a, cache['a_' + str(self.num_layers - i)])

                    grads['W' + str(self.num_layers - i)] += self.reg * self.params['W' + str(self.num_layers - i)]


                dout_r = dropout_backward(dout_d, cache['d_' + str(1)])

                dout_a = relu_backward(dout_r, cache['r_' + str(1)])

                dx , grads['W' + str(1)], grads['b' + str(1)] = affine_backward(dout_a, cache['a_' + str(1)])

                grads['W' + str(1)] += self.reg * self.params['W' + str(1)]


            else:

                dout_r , grads['W' + str(self.num_layers)], grads['b' + str(self.num_layers)] = affine_backward(dF,cache['a_' + str(self.num_layers)])

                grads['W' + str(self.num_layers)] += self.reg * self.params['W' + str(self.num_layers)]


                for i in np.arange(self.num_layers - 2) + 1:

                    dout_a = relu_backward(dout_r, cache['r_' + str(self.num_layers - i)])

                    dout_r , grads['W' + str(self.num_layers - i)], grads['b' + str(self.num_layers - i)] = affine_backward(dout_a, cache['a_' + str(self.num_layers - i)])

                    grads['W' + str(self.num_layers - i)] += self.reg * self.params['W' + str(self.num_layers - i)]


                dout_a = relu_backward(dout_r, cache['r_' + str(1)])

                dx , grads['W' + str(1)], grads['b' + str(1)] = affine_backward(dout_a, cache['a_' + str(1)])

                grads['W' + str(1)] += self.reg * self.params['W' + str(1)]



        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads


def affine_batch_relu_dropout_forward(x, w, b, gamma, beta, bn_param, dropout_param):
    """
    Convenience layer that perorms an affine transform followed by a ReLU

    Inputs:
    - x: Input to the affine layer
    - w, b: Weights for the affine layer

    Returns a tuple of:
    - out: Output from the ReLU
    - cache: Object to give to the backward pass
    """
    out_a, cache_a = affine_forward(x, w, b)
    out_b, cache_b = batchnorm_forward(out_a, gamma, beta, bn_param)
    out_r, cache_r = relu_forward(out_b)
    out_d, cache_d = dropout_forward(out_r,dropout_param)
    cache = (cache_a, cache_b, cache_r, cache_d)

    return out_d, cache

def affine_batch_relu_forward(x, w, b, gamma, beta, bn_param):
    """
    Convenience layer that perorms an affine transform followed by a ReLU

    Inputs:
    - x: Input to the affine layer
    - w, b: Weights for the affine layer

    Returns a tuple of:
    - out: Output from the ReLU
    - cache: Object to give to the backward pass
    """
    out_a, cache_a = affine_forward(x, w, b)
    out_b, cache_b = batchnorm_forward(out_a, gamma, beta, bn_param)
    out_r, cache_r = relu_forward(out_b)
    cache = (cache_a, cache_b, cache_r)

    return out_r, cache


def affine_batch_relu_dropout_backward(dout, cache):
    """
    Backward pass for the affine-relu convenience layer
    """
    cache_a, cache_b, cache_r, cache_d = cache

    dout_r = dropout_backward(dout, cache_d)
    dout_b = relu_backward(dout_r, cache_r)
    dout_a, dgamma, dbeta = batchnorm_backward(dout_b, cache_b)
    dx, dw, db = affine_backward(dout_a, cache_a)

    #da = relu_backward(dout, relu_cache)
    #dx, dw, db = affine_backward(da, fc_cache)
    return dx, dw, db, dgamma, dbeta

def affine_batch_relu_backward(dout, cache):
    """
    Backward pass for the affine-relu convenience layer
    """
    cache_a, cache_b, cache_r = cache

    dout_b = relu_backward(dout, cache_r)
    dout_a, dgamma, dbeta = batchnorm_backward(dout_b, cache_b)
    dx, dw, db = affine_backward(dout_a, cache_a)

    #da = relu_backward(dout, relu_cache)
    #dx, dw, db = affine_backward(da, fc_cache)
    return dx, dw, db, dgamma, dbeta

from layers import *
import numpy as np
import matplotlib.pyplot as plt

class MLQP_NN(object):
    def __init__(self, hidden_dims, input_dim=2, num_classes=1,
                 weight_scale=1e-2, dtype=np.float32 ,fun ='lms'):
        """
        Initialize a new MLQPNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - dtype:float32 is faster but less accurate, so you should use
          float64 for numeric gradient checking.
        """
        self.num_layers = 1 + len(hidden_dims)
        self.fun = fun
        self.dtype = dtype
        self.params = {}
        #cache for momentum
        self.cache = {}

        #ToDo:initialize parameters
        self.params['U1'] = weight_scale * np.random.randn(hidden_dims[0],input_dim)
        self.params['V1'] = weight_scale * np.random.randn(hidden_dims[0],input_dim)
        self.params['b1'] = np.zeros(hidden_dims[0])
        self.cache['U1'] = np.zeros([hidden_dims[0],input_dim])
        self.cache['V1'] = np.zeros([hidden_dims[0],input_dim])
        self.cache['b1'] = np.zeros(hidden_dims[0])

        for i in range(1, self.num_layers - 1):
            self.params['U' + str(i + 1)] = weight_scale * np.random.randn(hidden_dims[i], hidden_dims[i-1])
            self.params['V' + str(i + 1)] = weight_scale * np.random.randn(hidden_dims[i], hidden_dims[i-1])
            self.params['b' + str(i + 1)] = np.zeros(hidden_dims[i])
            self.cache['U' + str(i + 1)] = np.zeros([hidden_dims[i], hidden_dims[i-1]])
            self.cache['V' + str(i + 1)] = np.zeros([hidden_dims[i], hidden_dims[i-1]])
            self.cache['b' + str(i + 1)] = np.zeros(hidden_dims[i])


        self.params['U' + str(self.num_layers)] = weight_scale * np.random.randn(num_classes,hidden_dims[self.num_layers - 2])
        self.params['V' + str(self.num_layers)] = weight_scale * np.random.randn(num_classes,hidden_dims[self.num_layers - 2])
        self.params['b' + str(self.num_layers)] = np.zeros(num_classes)
        self.cache['U' + str(self.num_layers)] = np.zeros([num_classes,hidden_dims[self.num_layers - 2]])
        self.cache['V' + str(self.num_layers)] = np.zeros([num_classes,hidden_dims[self.num_layers - 2]])
        self.cache['b' + str(self.num_layers)] = np.zeros(num_classes)
        # print self.params['W1'].shape,self.params['W2'].shape,self.params['W3'].shape

        # Cast all parameters to the correct datatype
        for k, v in self.params.iteritems():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        X = X.astype(self.dtype)
        mode = 'test' if y is None else 'train'
        # ToDo:mlpd forward pass
        out = {}
        cache = {}
        out['out0'] = X
        for i in range(1, self.num_layers+1):
            out['out' + str(i)], cache['cache' + str(i)] = mlqp_forward(out['out' + str(i - 1)],
                                        self.params['U'+str(i)],self.params['V'+str(i)],self.params['b'+str(i)])
        scores = out['out'+str(self.num_layers)]

        # If test mode,return early
        if mode == 'test':
            return scores

        # ToDo:mlpd backward pass
        loss, grads = 0.0, {}
        if(self.fun == 'lms'):
            loss,dx = lms_loss(scores,y)
        if(self.fun == 'cEnt'):
            loss,dx = cEnt_loss(scores,y)

        for i in range(self.num_layers, 0, -1):
            temp = np.copy(dx)
            dx, dv, du, db = mlqp_backward(temp, cache['cache' + str(i)])
            grads['U' + str(i)] = du
            grads['V' + str(i)] = dv
            grads['b' + str(i)] = db
        return loss, grads

    def train(self,X,y,learning_rate=1e-4,batch_size = None,epoch = 5,verbose = False,mom_decay = 0.9):
        if(batch_size == None):
            mode = 'online'
            batch_size = 1
        else:
            mode = 'batch'

        loss_his = []
        num_iter = int(X.shape[1]/batch_size)
        for i in range(0,epoch):
            if(verbose):
                if(i % 100 == 0):
                    print 'epoch %d validation accuracy:%.3f' % (i,self.check_accuracy(X, y))

            for j in range(num_iter):
                if(mode == 'batch'):
                    mask = np.random.choice(X.shape[1],batch_size)
                    X_batch = X[:,mask]
                    y_batch = y[mask]
                else:
                    X_batch = X[:,j]
                    y_batch = y[j]

                loss,grads = self.loss(X_batch,y_batch)

                if(j == num_iter-1):
                    loss_his.append(loss)
                for p, w in self.params.iteritems():
                    #print p
                    self.cache[p] = mom_decay * self.cache[p]  - learning_rate * grads[p]
                    self.params[p] += self.cache[p]
        return loss_his

    def check_accuracy(self,X,y):
        scores = self.loss(X)
        scores[ scores < 0.5] = 0
        scores[ scores >= 0.5] = 1
        return np.mean(scores == y)

    def draw(self,grid=0.1):
        X = np.mgrid[-3.5:3.5:grid, -3.5:3.5:grid]
        size = np.prod(X.shape)
        X = X.reshape(2,size/2)
        scores = self.loss(X)
        scores[scores < 0.5] = 0
        scores[scores >= 0.5] = 1

        #draw hyperplane
        for i in range(scores.shape[1]):
            if(scores[0,i] == 0):
                style = 'yo'
            else:
                style = 'ko'
            plt.plot(X[0,i],X[1,i],style)

        #draw points
        data_train = np.loadtxt('two_spiral_train.txt')
        data_train = data_train.T
        X = data_train[:2, :]
        y = data_train[2, :]
        # print X.shape
        # print y.shape
        for i in range(X.shape[1]):
            if (y[i] == 0):
                style = 'ro'
            else:
                style = 'bo'
            plt.plot(X[0, i], X[1, i], style)
        plt.show()

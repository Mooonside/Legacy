import numpy as np
from MLQP_NN import MLQP_NN
import matplotlib.pyplot as plt
from layers import *
import time

data_train = np.loadtxt('two_spiral_train.txt')
data_train = data_train.T
X= data_train[:2,:]
y = data_train[2,:]

net = MLQP_NN(hidden_dims = [10], input_dim = 2, num_classes = 1,
              weight_scale = 1, dtype = np.float64,fun='lms')

tic = time.time()
loss_his = net.train(X,y,learning_rate = 1e-3,batch_size = 20,
                     epoch = 30000,verbose = True)
toc = time.time()

scores = net.loss(X)
print 'final train accuracy:%.3f' % net.check_accuracy(X,y)

data_test = np.loadtxt('two_spiral_test.txt')
data_test = data_test.T
X_test = data_test[:2,:]
y_test = data_test[2,:]
print 'validation:%.3f' % net.check_accuracy(X_test,y_test)

print 'run time:%.3f s' % (toc-tic)

plt.title('Training loss')
plt.plot(loss_his,'o')
plt.xlabel('Iteration')
plt.show()

net.draw()

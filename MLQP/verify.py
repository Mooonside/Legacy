import numpy as np
from MLQP_NN import MLQP_NN
import matplotlib.pyplot as plt
from layers import *

# for verifying : single layer MLQP to separate xnor
X = np.asarray([[0,0,1,1],[0,1,0,1]])
y = np.asarray([0,1,1,0])
# X = np.asarray([[0.3,0.3,0.7,0.7],[0.3,0.7,0.7,0.3]])
# y = np.asarray([0, 1, 0, 1])

net = MLQP_NN(hidden_dims=[10], input_dim = 2, num_classes = 1,
              weight_scale=1e-2, dtype=np.float32)

scores = net.loss(X)
print 'intial', scores

loss_his = net.train(X,y,learning_rate = 0.2,batch_size = 4,epoch = 5000)
scores = net.loss(X)
print 'later', scores

plt.title('Training loss')
plt.plot(loss_his,'o')
plt.xlabel('Iteration')
plt.show()
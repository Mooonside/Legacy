import numpy as np

def linear_forward(x, w):
  out = w.dot(x)
  cache = (x, w)
  return out, cache

def linear_backward(dout, cache):
  x, w= cache
  dx = w.T.dot(dout)
  dw = dout.dot(x.T)
  return dx, dw

def bias_forward(x, b):
  out = (x.T + b).T
  return out

def bias_backward(dout):
  dx = dout
  db = np.sum(dout,axis=1)
  return dx,db

def add_forward(x, y):
  out = x + y
  return out

def add_backward(dout):
  dx = dout
  dy = dout
  return dx,dy

def quad_forward(x):
  out = x**2
  cache = (x)
  return out, cache

def quad_backward(dout,cache):
  x = cache
  dx = 2*x*dout
  return dx

def sigmoid_forward(x):
  out = 1 / (np.e**(-x) + 1)
  cache = (x)
  return out, cache

def sigmoid_backward(dout,cache):
  x = cache
  dx = np.e**(-x)/((1+np.e**(-x))**2)
  dx *= dout
  return dx

def mlqp_forward(x,u,v,b):
  out1,cache1 = linear_forward(x,v)
  out2,cache2 = quad_forward(x)
  out3,cache3 = linear_forward(out2,u)
  out4 = add_forward(out1,out3)
  out5 = bias_forward(out4,b)
  out6,cache6 = sigmoid_forward(out5)
  cache = (cache1,cache2,cache3,cache6)
  return out6, cache

def mlqp_backward(dout,cache):
    cache1,cache2,cache3,cache6 = cache
    dout5 = sigmoid_backward(dout,cache6)
    dout4,db = bias_backward(dout5)
    dout1,dout3 = add_backward(dout4)
    dx,dv = linear_backward(dout1,cache1)
    dout2,du = linear_backward(dout3,cache3)
    dx += quad_backward(dout2,cache2)
    return dx, dv, du, db

def lms_loss(s,y):
  loss_vec = (y - s)**2
  loss = np.mean(loss_vec) / 2
  ds = - 2 * (y - s)/loss_vec.shape[1]
  return loss,ds

def cEnt_loss(s,y):
  ds_vec = -y/s + (1-y)/(1-s)
  ds = ds_vec / ds_vec.shape[1]
  loss_vec =  y*np.log(s) + (1-y)*np.log(1-s)
  loss = -np.mean(loss_vec)
  return loss,ds

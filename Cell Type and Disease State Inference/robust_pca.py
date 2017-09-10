from __future__ import division
import numpy as np

def rel_error(x, y):
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))

def robust_pca(M,p1=1,p2=1):
    '''
    :param M: 
    :param p1: penalty factor for sparse(L1 norm) 
    :param p2: penalty factor for low-rank(F norm) 
    :return: 
    '''
    # intialize them randomly
    S = np.random.randn(*M.shape)
    Y = np.random.randn(*M.shape)
    L = np.random.randn(*M.shape)

    k = 1
    error = 1
    while(k < 40 and error > 1e-10):
        #update L_k
        X = M - S + Y
        U,E,V = np.linalg.svd(X,full_matrices=False)
        # print U.shape,E.shape,V.shape
        E -= 1/p2
        E[E < 0] = 0
        # using svt to update L to L_k
        L = (U.dot(np.diag(E))).dot(V)
        #update S_k
        X = M - L + Y
        S = np.copy(X)
        S[X >= p1/p2] -= p1/p2
        S[X <= -p1/p2] += p1/p2
        S[np.abs(X) <= p1/p2] = 0
        #update y_k
        Y += p2 * (M - L - S)
        #update iteraive times k
        k += 1
        error = rel_error(M,L+S)
        print error
    return L,S

def main():
    a = np.linspace(1,10,10)
    b = 2 * a
    c = np.vstack([a,b])
    robust_pca(c)

if __name__ == '__main__':
    main()
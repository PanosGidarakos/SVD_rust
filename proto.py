### First Github . https://github.com/emchinn/Bidiagonalization/blob/master/Golub-Kahan.ipynb
import numpy as np
np.set_printoptions(suppress=True, precision=4)

def set_lowVal_zero(X):
    low_values_indices = abs(X) < 9e-15   # where values are low
    X[low_values_indices] = 0             # all low values set to 0
    return X

def Householder(x, i):
    alpha = -np.sign(x[i]) * np.linalg.norm(x)
    e = np.zeros(len(x)); e[i] = 1.0
    
    v = (x - alpha * e)
    w = v / np.linalg.norm(v)
    P = np.eye(len(x)) - 2 * np.outer(w, w.T)
    
    return P


def Golub_Kahan_Diagonal(X):
    col = X.shape[1]
    row = X.shape[0]
    
    J = X.copy()

    for i in range(col - 2):
        # column
        h = np.zeros(len(J[:, i]))
        h[i:] = J[i:, i]
        P = Householder(h, i)
        J = set_lowVal_zero(P @ J)
#         print(J, '\n')
        # row
        h = np.zeros(len(J[i, :]))
        h[i+1:] = J[i, i+1:] 
        Q = Householder(h, i+1)
        J = set_lowVal_zero(J @ Q)
#         print(J, '\n')
    return J
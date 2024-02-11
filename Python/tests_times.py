import time

import numpy as np
from numpy.linalg import svd as numpy_svd
from scipy.linalg import svd as scipy_svd
from G_R_svd import clear_svd as g_r_svd
from non_exact_svd import simplified_svd_2d as n_e_svd
import matplotlib
matplotlib.use('Agg')  # Set the backend to Agg for non-GUI environments
import matplotlib.pyplot as plt

def time_svd(func, A):
    start = time.time()
    _ = func(A)
    end = time.time()
    return end - start

def compare_svd_performance(steps):
    t_custom, t_numpy, t_scipy, t_2d = [], [], [], []
    
    for i in steps:
        A = np.random.randn(i, i)
        t_custom.append(time_svd(g_r_svd, A))  # Assuming clear_svd is defined
        t_numpy.append(time_svd(numpy_svd, A))
        t_scipy.append(time_svd(scipy_svd, A))
        t_2d.append(time_svd(n_e_svd, A))  # Assuming svd_2d is defined

    plt.plot(steps, t_custom, label='G_R SVD')
    plt.plot(steps, t_numpy, label='NumPy SVD')
    plt.plot(steps, t_scipy, label='SciPy SVD')
    plt.plot(steps, t_2d, label='N_E SVD')  # Assuming you have a custom 2D SVD implementation
    plt.xlabel('Matrix Size')
    plt.ylabel('Time (seconds)')
    plt.legend()
    plt.title('SVD Running Times Comparison')
    plt.show()
    plt.savefig('timestimestimes.png')

if __name__ == '__main__':
    steps = range(25, 500, 25)
    compare_svd_performance(steps)

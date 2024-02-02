import numpy as np

def householder_vector(x):
    """
    Create the householder vector that will zero out all but the first component of x.
    """
    normx = np.linalg.norm(x)
    sigma = -np.sign(x[0]) * normx
    u = x.copy()
    u[0] -= sigma
    return u / np.linalg.norm(u), sigma

def householder_transformation(v):
    """
    Create the Householder matrix given the Householder vector.
    """
    return np.eye(len(v)) - 2 * np.outer(v, v)

def bidiagonalize(A):
    m, n = A.shape
    U = np.eye(m)
    V = np.eye(n)
    B = A.copy()

    for k in range(n):
        # Create the Householder matrix for column k
        x = B[k:, k]
        v, beta = householder_vector(x)
        Qk = np.eye(m)
        Qk[k:, k:] = householder_transformation(v)
        # Update B, U
        B = Qk @ B
        U = U @ Qk.T

        if k < n - 2:
            # Create the Householder matrix for row k
            x = B[k, k+1:]
            v, beta = householder_vector(x)
            Pk = np.eye(n)
            Pk[k+1:, k+1:] = householder_transformation(v)
            # Update B, V
            B = B @ Pk.T
            V = V @ Pk

    # Since B is supposed to be upper bidiagonal, zero out any non-bidiagonal elements
    for i in range(m):
        for j in range(n):
            if i > j or i < j-1:
                B[i, j] = 0
   
    return B, U, V


def givens_rotation(a, b):
    """
    Compute matrix entries for Givens rotation.
    """
    if b == 0:
        c = 1
        s = 0
    else:
        if np.abs(b) > np.abs(a):
            tau = -a / b
            s = 1 / np.sqrt(1 + tau**2)
            c = s * tau
        else:
            tau = -b / a
            c = 1 / np.sqrt(1 + tau**2)
            s = c * tau
    return c, s

def apply_givens_rotation(B, i, k, c, s):
    """
    Apply Givens rotation to matrix B.
    """
    G = np.eye(len(B))
    G[[i, k], [i, k]] = c
    G[i, k] = s
    G[k, i] = -s
    return G.T @ B @ G

# def svd_step(B, U, V, eps):
#     """
#     Perform one step of the SVD algorithm.
#     """
#     m, n = B.shape
#     for i in range(min(m, n) - 1):
#         if np.abs(B[i, i+1]) <= eps * (np.abs(B[i, i]) + np.abs(B[i+1, i+1])):
#             B[i, i+1] = 0

#     # Determine the smallest p and largest q such that B can be blocked
#     p = 0
#     while p < n and B[p, p] != 0:
#         p += 1
#     q = n - 1
#     while q >= 0 and B[q, q] == 0:
#         q -= 1
#     q = n - q

#     if q == n:
#         # B is diagonal
#         return np.diag(B), U, V
#     else:
#         for i in range(p, n - q - 1):
#             if B[i, i+1] == 0:
#                 c, s = givens_rotation(B[i, i], B[i+1, i])
#                 G = apply_givens_rotation(B, i, i+1, c, s)
#                 B = G @ B @ G.T
#                 U = U @ G.T
#                 V = V @ G
#     return B, U, V

import numpy as np

def svd_step(B, U, V, eps):
    """
    Perform multiple steps of the SVD algorithm.
    """
    m, n = B.shape

    # Define a maximum number of iterations (adjust as needed)
    max_iterations = 100000

    for iteration in range(max_iterations):
        for i in range(min(m, n) - 1):
            if np.abs(B[i, i+1]) <= eps * (np.abs(B[i, i]) + np.abs(B[i+1, i+1])):
                B[i, i+1] = 0

        # Determine the smallest p and largest q such that B can be blocked
        p = 0
        while p < n and B[p, p] != 0:
            p += 1
        q = n - 1
        while q >= 0 and B[q, q] == 0:
            q -= 1
        q = n - q

        if q == n:
            # B is diagonal
            return np.diag(B), U, V
        else:
            for i in range(p, n - q - 1):
                if B[i, i+1] == 0:
                    c, s = givens_rotation(B[i, i], B[i+1, i])
                    G = apply_givens_rotation(B, i, i+1, c, s)
                    
                    B = G @ B @ G.T
                    U = U @ G.T
                    V = V @ G

    # If max_iterations reached without convergence, you may want to handle this case appropriately.
    print("SVD algorithm did not converge within the maximum number of iterations.")
    return np.diag(B), U, V


def svd(A, eps=1e-20):
    """
    Compute the Singular Value Decomposition of A using the Golub–Reinsch algorithm.
    """
    B, U, V = bidiagonalize(A)
    
    Sigma, U, V = svd_step(B, U, V, eps)
    return Sigma, U, V





# Define necessary functions for the full SVD algorithm as per the provided algorithms
import numpy as np
def householder_transformation(a):
    """
    Creates a Householder transformation matrix that will zero out all but the
    first component of the vector 'a'.
    """
    e = np.zeros_like(a)
    e[0] = np.linalg.norm(a)
    v = a + np.sign(a[0]) * e
    v = v / np.linalg.norm(v)
    H = np.eye(a.shape[0]) - 2 * np.outer(v, v)
    return H

def householder_bidiagonalization(A):
    """
    Reduces a matrix A to upper bidiagonal form using Householder transformations.
    """
    m, n = A.shape
    U = np.eye(m)
    V = np.eye(n)
    B = A.copy()

    for i in range(n):
        # Left transformation
        H = householder_transformation(B[i:, i])
        B[i:, :] = np.dot(H, B[i:, :])
        U[:, i:] = np.dot(U[:, i:], H)
       
        if i < n - 1:
            # Right transformation
            H = householder_transformation(B[i, i+1:])
            B[:, i+1:] = np.dot(B[:, i+1:], H.T)
            V[i+1:, :] = np.dot(H, V[i+1:, :])

    return U, B, V.T

def golub_kahan_svd_step(B, epsilon=1e-10):
    """
    Performs one step of the Golub-Kahan SVD bidiagonalization algorithm.
    """
    m, n = B.shape
    Q = np.eye(n)
    P = np.eye(m)

    for k in range(n-1):
        # Determine Givens rotation
        a = B[k, k]
        b = B[k, k+1]
        c, s = givens_rotation(a, b)
        G = np.array([[c, s], [-s, c]])

        # Apply Givens rotation to B, Q, P
        B[k:k+2, :] = np.dot(G.T, B[k:k+2, :])  # Apply to B
        Q[:, k:k+2] = np.dot(Q[:, k:k+2], G)  # Apply to Q

        if k < n - 2:
            a = B[k, k+1]
            b = B[k+1, k+1]
            c, s = givens_rotation(a, b)
            G = np.array([[c, s], [-s, c]])

            B[:, k+1:k+3] = np.dot(B[:, k+1:k+3], G)  # Apply to B
            P[k+1:k+3, :] = np.dot(G.T, P[k+1:k+3, :])  # Apply to P

    return B, Q, P

# def svd(A, epsilon=1e-10):
#     """
#     Computes the Singular Value Decomposition of matrix A using the
#     Golub-Reinsch algorithm.
#     """
#     U, B, V_T = householder_bidiagonalization(A)
#     print('diag')
#     print(U)
#     print(B)
#     print(V_T)
   
#     # Golub-Reinsch iterations
#     while True:
#         # Check for convergence
#         off_diagonal = np.sqrt(np.sum(B[-1, :-1]**2))
#         if off_diagonal <= epsilon:
#             break
       
#         # Apply Golub-Kahan SVD step
#         B, Q, P = golub_kahan_svd_step(B)
#         U = np.dot(U, P)
#         V_T = np.dot(Q.T, V_T)

#     # Sort singular values and corresponding vectors
#     s = np.diag(B)
#     sort_indices = np.argsort(s)[::-1]
#     s = s[sort_indices]
#     U = U[:, sort_indices]
#     V_T = V_T[sort_indices, :]

#     return U, np.diag(s), V_T

import numpy as np

def svd(A, max_steps=1000, epsilon=1e-10):
    """
    Computes the Singular Value Decomposition of matrix A using the
    Golub-Reinsch algorithm.

    Parameters:
    - A: Input matrix
    - max_steps: Maximum number of Golub-Kahan SVD steps to perform
    - epsilon: Convergence threshold

    Returns:
    - U: Left singular vectors
    - S: Singular values
    - V_T: Right singular vectors (transposed)
    """
    U, B, V_T = householder_bidiagonalization(A)

    # Golub-Reinsch iterations
    step_count = 0
    while True:
        # Check for convergence
        off_diagonal = np.sqrt(np.sum(B[-1, :-1]**2))
        if off_diagonal <= epsilon or (max_steps is not None and step_count >= max_steps):
            break

        # Apply Golub-Kahan SVD step
        B, Q, P = golub_kahan_svd_step(B)
        U = np.dot(U, P)
        V_T = np.dot(Q.T, V_T)
        step_count += 1

    # Sort singular values and corresponding vectors
    s = np.diag(B)
    sort_indices = np.argsort(s)[::-1]
    s = s[sort_indices]
    U = U[:, sort_indices]
    V_T = V_T[sort_indices, :]

    return U, np.diag(s), V_T






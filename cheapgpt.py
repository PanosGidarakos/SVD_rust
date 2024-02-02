import numpy as np

def householder_bidiagonalization(A):
    m, n = A.shape
    B = np.copy(A)
    U = np.eye(m)
    V = np.eye(n)

    for k in range(min(m, n)):
        # Householder matrix Qk
        x = B[k:, k]
        v = np.zeros_like(x)
        v[0] = np.sign(x[0]) * np.linalg.norm(x) + x[0]
        v[1:] = x[1:]
        v = v / np.linalg.norm(v)

        Qk = np.eye(m)
        Qk[k:, k:] -= 2.0 * np.outer(v, v)

        # Update matrices
        B = np.dot(Qk, B)
        U = np.dot(U, Qk)

        if k < n - 1:
            # Householder matrix Pk+1
            y = B[k, k+1:]
            w = np.zeros_like(y)
            w[1:] = y[1:]
            norm_w = np.linalg.norm(w)
            if norm_w != 0:
                w = w / norm_w

                Pk1 = np.eye(n)
                Pk1[k+1:, k+1:] -= 2.0 * np.outer(w, w)

                # Update matrices
                B = np.dot(B, Pk1)
                V = np.dot(Pk1, V)

    return B, U, V

import numpy as np

def golub_reinsch_svd(A, epsilon=1e-20):
    m, n = A.shape
    B, U, V = householder_bidiagonalization(A)
#     print('Householder')
#     print(B,U,V)

    while True:
        # Step 2a: Check for small off-diagonal elements
        for i in range(n - 1):
            if np.abs(B[i, i + 1]) <= epsilon * (np.abs(B[i, i]) + np.abs(B[i + 1, i + 1])):
                B[i, i + 1] = 0

        # Step 2b: Block diagonalization
        p, q = block_diagonalize(B, epsilon)

        # Step 2c: Check if q = n
        if q == n:
            Sigma = np.diag(B.diagonal())

            return Sigma, U, V

        # Step 2d: Apply Givens rotations or Algorithm 1c
        if B[p + 1, p] == 0:
            # Apply Givens rotations to make B2,2 upper bidiagonal
            B, V = givens_rotations(B, V, p, n - q - 1)
        else:
            # Apply Algorithm 1c
            B, U, V = algorithm_1c(n, B, U, V, p, q, epsilon)

def block_diagonalize(B, epsilon):
    p = 0
    while p < B.shape[0] - 1 and B[p, p + 1] != 0:
        p += 1

    q = p + 1
    while q < B.shape[1] and B[q, q] != 0:
        q += 1

    return p, q

def givens_rotations(B, V, p, num_rotations):
    for _ in range(num_rotations):
        for i in range(p, B.shape[0] - 1):
            G = np.eye(B.shape[0])
            c, s = givens_rotation_parameters(B[i, i], B[i + 1, i])
            G[i:i+2, i:i+2] = np.array([[c, -s], [s, c]])
            B = np.dot(G.T, np.dot(B, G))
            V = np.dot(V, G)
    return B, V

def givens_rotation_parameters(a, b):
    if b == 0:
        c = 1
        s = 0
    else:
        if np.abs(b) > np.abs(a):
            r = -a / b
            s = 1 / np.sqrt(1 + r**2)
            c = s * r
        else:
            r = -b / a
            c = 1 / np.sqrt(1 + r**2)
            s = c * r
    return c, s

def algorithm_1c(n, B, U, V, p, q, epsilon):
    # Implement Algorithm 1c here (omitted for brevity)
    pass



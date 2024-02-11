import numpy as np
def simplified_random_unit_vector(n):
    unnormalized = np.random.normal(0, 1, n)
    norm = np.linalg.norm(unnormalized)
    return unnormalized / norm

def simplified_svd_1d(A, epsilon=1e-10):
    n, m = A.shape
    x = simplified_random_unit_vector(min(n, m))
    B = np.dot(A.T, A) if n > m else np.dot(A, A.T)
    last_v = None
    current_v = x
    iterations = 0
    while last_v is None or np.abs(np.dot(current_v, last_v)) < 1 - epsilon:
        iterations += 1
        last_v = current_v
        current_v = np.dot(B, last_v)
        current_v /= np.linalg.norm(current_v)
    return current_v

def simplified_svd_2d(A, k=None, epsilon=1e-10):
    A = np.array(A, dtype=float)
    n, m = A.shape
    if k==None:
        k=int(min(n,m)/2)
    svd_so_far = []
    k = min(k, min(n, m))
    for _ in range(k):
        matrix_for_1d = A.copy()
        for sigma, u, v in svd_so_far:
            matrix_for_1d -= sigma * np.outer(u, v)
        if n > m:
            v = simplified_svd_1d(matrix_for_1d, epsilon)
            u_unnormalized = np.dot(A, v)
        else:
            u = simplified_svd_1d(matrix_for_1d, epsilon)
            u_unnormalized = np.dot(A.T, u)
        sigma = np.linalg.norm(u_unnormalized)
        u_or_v = u_unnormalized / sigma
        if n > m:
            svd_so_far.append((sigma, u_or_v, v))
        else:
            svd_so_far.append((sigma, u, u_or_v))
    singular_values, us, vs = zip(*svd_so_far)  # Unpack the SVD components
    return np.array(singular_values), np.array(us).T, np.array(vs)

if __name__ == '__main__':
    A = np.random.random((4, 4))
    S, U, Vt = simplified_svd_2d(A.copy())
    print("U:\n", U)
    print("S:\n", np.diag(S))
    print("V^T:\n", Vt)

import numpy as np
from numpy.linalg import svd as np_svd
import matplotlib.pyplot as plt


def main(steps, percent_removed):
    # B = np.diag(np.diag(B))  # take only diagonal values and discard others
    matrices_results = []
    for I in steps:
        results = []
        matrices_results.append((I, results))

        A = np.random.randn(I, I)
       
        U, S, Vt = np_svd(A)
        # print(S)
        R = 0
        for j in range(S.shape[0]):
            if S[j] == 0:
                break
            R += 1
        print(f"{S.shape=} {R=}")
        for perc in percent_removed:
            w = S.shape[0]
            cols2remove = round(w * (perc / 100))
            r = R-cols2remove
            U2 = U[..., 0:r]
            S2 = S[0:r]
            Vt2 = Vt[0:r, ...]
            A2 = U2@np.diag(S2)@Vt2
            err = np.linalg.norm(A - A2)
            print(f"{perc=}% {err=}")
            results.append(err)
    for size, errors in matrices_results:
        plt.plot(percent_removed, errors, label=str(size))
    plt.xlabel('Percent Removed (%)')
    plt.ylabel('error (norm)')
    plt.legend()
    plt.show()
    plt.savefig('timestimestimes.png', transparent=True)


if __name__ == '__main__':
    main(
        steps=range(500, 3001, 500),
        # percent_removed=[5, 10, 25, 50],
        percent_removed=[5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 99, 100],
    )

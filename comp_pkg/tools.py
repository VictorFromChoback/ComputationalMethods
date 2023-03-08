import numpy as np


def matrix_multiplication(A: np.ndarray, B: np.ndarray):
    assert A.shape[1] == B.shape[0]
    both_size: int = A.shape[1]
    result: np.ndarray = np.zeros((A.shape[0], B.shape[1]))
    for i in range(A.shape[0]):
        for j in range(B.shape[1]):
            for r in range(both_size):
                result[i, j] += A[i, r].copy() * B[r, j].copy()
    return result


def scalar_product(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) == 0:
        return 0
    return matrix_multiplication(x.reshape((-1, len(x))), np.reshape(y, (len(y), -1)))

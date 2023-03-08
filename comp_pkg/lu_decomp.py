import typing as tp

import numpy as np

from .solver import AbstractSolver
from .tools import matrix_multiplication, scalar_product


class LU_Solver(AbstractSolver):

    def __init__(self, A: np.ndarray, b: np.ndarray) -> tp.NoReturn:
        self._A = A
        self._b = b
        self._dim = A.shape[0]
    
    def _lu(self):
        U: np.ndarray = self._A.copy().astype("float64")
        L: np.ndarray = np.eye(self._dim).astype("float64")
        Q: np.ndarray = np.eye(self._dim)
        for k in range(self._dim - 1):
            best_ind: int = np.argmax(np.abs(U[k, k:]))
            best_ind = best_ind + k
            U[:, k], U[:, best_ind] = U[:, best_ind].copy(), U[:, k].copy()
            L[:k, k], L[:k, best_ind] = L[:k, best_ind].copy(), L[:k, k].copy()  
            Q[:, k], Q[:, best_ind] = Q[:, best_ind].copy(), Q[:, k].copy()
            for j in range(k + 1, self._dim):
                L[j, k] = U[j, k].copy() / U[k, k].copy()
                U[j, k:] -= L[j, k].copy() * U[k, k:].copy()

        return L, U, Q

    def _solve_downside(self, L: np.ndarray, right_size: np.ndarray) -> np.ndarray:
        solution: np.ndarray = np.zeros(self._dim)
        for j in range(self._dim):
            prefix = scalar_product(solution[:j], L[j, :j])
            solution[j] = right_size[j] - prefix
        return solution

    def _solve_upside(self, U: np.ndarray, right_size: np.ndarray) -> np.ndarray:
        solution: np.ndarray = np.zeros(self._dim)
        for j in range(self._dim - 1, -1, -1):
            prefix = scalar_product(U[j, j + 1:], solution[j + 1:])
            solution[j] = (right_size[j] - prefix) / U[j, j]
        return solution 

    def solve(self) -> tp.Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        L, U, Q = self._lu()

        solution_L: np.ndarray = self._solve_downside(L, self._b)
        solution_U: np.ndarray = self._solve_upside(U, solution_L)
        solution: np.ndarray = matrix_multiplication(Q, solution_U.reshape((self._dim, -1))).flatten()

        return L, U, Q, solution

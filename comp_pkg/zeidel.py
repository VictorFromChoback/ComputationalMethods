import typing as tp

import numpy as np
from tqdm.notebook import tqdm_notebook

from .tools import matrix_multiplication, scalar_product


class ZeidelSolver:

    def _decompose(self):
        self._U = np.triu(self._A, 1)

    def _solve_down(self, right_side: np.ndarray) -> np.ndarray:
        solution: np.ndarray = np.ones(self._dim)
        for k in range(self._dim):
            solution[k] = (right_side[k] - scalar_product(solution[:k], self._A[k, :k])) / self._A[k, k]
        return solution

    def __init__(self, A, b):
        self._A = A
        self._b = b
        self._dim = len(b)
        self._decompose()

    def solve(self, iterations: int, tol: float = 1e-3) -> tp.Tuple[np.ndarray, tp.List[float]]:
        solution = np.ones(self._dim)
        trace = []
        for _ in tqdm_notebook(range(iterations)):
            right_side: np.ndarray = matrix_multiplication(-self._U, solution.reshape((self._dim, -1)))
            right_side += self._b.reshape((self._dim, -1))
            new_solution = self._solve_down(right_side.flatten())
            tol_ = np.linalg.norm(matrix_multiplication(self._A, solution.reshape((self._dim, -1))).flatten() - self._b)
            trace.append(solution)
            if tol_ < tol:
                return new_solution, trace
            solution = new_solution

        return solution, trace

import typing as tp

import numpy as np

from .solver import AbstractSolver
from .types import PointOrNpArray, ValueOrNpArray, WithErrors, EPSILON_SINGLE, EPSILON_DOUBLE


class GornerScheme(AbstractSolver):
    """
    Gorner scheme algorithm described at
    https://en.wikipedia.org/wiki/Horner%27s_method
    """

    def __init__(self, coefficients: tp.Iterable[float]):
        self._coef = coefficients
        self._deg = len(coefficients)
    
    def _step(self, current_b: ValueOrNpArray, step_number: int, point: PointOrNpArray) -> ValueOrNpArray:
        return current_b * point + self._coef[-(1 + step_number)]

    def solve(self, 
              point: PointOrNpArray,
              with_erros: bool = False,
              precision: str = "single") -> tp.Union[ValueOrNpArray, WithErrors]:
        if precision == "single":
            eps = EPSILON_SINGLE
        else:
            eps = EPSILON_DOUBLE
        current_b_value: ValueOrNpArray = self._coef[-1]
        for step_number in range(1, self._deg):
            current_b_value = self._step(current_b_value, step_number, point)
        if with_erros:
            delta: ValueOrNpArray = 2 * self._deg * eps * np.abs((point - 2.0) ** 9)
            return (
                current_b_value,
                current_b_value - delta,
                current_b_value + delta
            )
        return current_b_value

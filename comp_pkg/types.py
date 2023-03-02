import typing as tp

import numpy as np


EPSILON_SINGLE = 2 ** (-24)
EPSILON_DOUBLE = 2 ** (-52)

PointOrNpArray = tp.Union[float, np.ndarray]
ValueOrNpArray = tp.Union[float, np.ndarray]

WithErrors = tp.Tuple[ValueOrNpArray, ValueOrNpArray, ValueOrNpArray]

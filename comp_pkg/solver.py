from abc import ABC, abstractmethod


class AbstractSolver(ABC):

    @abstractmethod
    def solve(self, *args, **kwargs):
        pass

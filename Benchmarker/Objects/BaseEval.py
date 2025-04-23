from pydantic import BaseModel
from abc import ABC, abstractmethod
import typing

class BaseEval(ABC):
    @abstractmethod
    def evaluate(self, modelAnswer: str, trueAnswer: str) -> float:
        pass
import sympy as sp
from sympy.parsing.latex import parse_latex
import re
from re import Pattern
from typing import Union

from .BaseEval import BaseEval

class LatexEval(BaseEval):
    def __init__(self, parsing_regex: Union[Pattern, None] = None):
            self.regex = regex

    def evaluate(self, modelAnswer: str, trueAnswer: str) -> float:

       if self.regex:
            modelMatches = re.search(self.regex, modelAnswer)
            trueMatches = re.search(self.regex, trueAnswer)

            if modelMatches and trueMatches:
               modelAnswer = modelMatches.group(1)
               trueAnswer = trueMatches.group(1)
            else:
               return 0

       modelSym = parse_latex(modelAnswer) 
       trueSym = parse_latex(trueAnswer) 
       
       print("Model Answer: " + modelAnswer + " True Answer " + trueAnswer)

       modelSym = sp.simplify(modelSym)
       trueSym = sp.simplify(trueSym)

       if modelSym.equals(trueSym):
           return 1
       else:
           return 0


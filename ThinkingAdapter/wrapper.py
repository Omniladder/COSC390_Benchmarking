from .adapter import ReasoningAdapter

from .prompts import output_prompt
from langchain_core.output_parsers import StrOutputParser

class ReasoningWrapper():
    
    def __init__(self, models, output_model, tree_depth, tree_width):

        self.reasoner = ReasoningAdapter(models=models, tree_depth=tree_depth, tree_width=tree_width)
        self.output_model = output_model


    def invoke(self, prompt):
        
        reason_model = output_prompt | self.output_model | StrOutputParser()

        reasoning = self.reasoner.invoke(prompt=prompt)

        response = reason_model.invoke({"reason": reasoning, "prompt" :prompt})

        return response
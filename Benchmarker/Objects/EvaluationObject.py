from datasets import load_dataset
import pandas as pd
from typing import Type
from langchain.llms.base import BaseLLM
from langchain_core.prompts import PromptTemplate

from Models.DataColumnModel import DataColumnModel
from Objects.BaseEval import BaseEval


class EvaluationObject():

    base_prompt = """
        System Message: 
        {System}

        Context: 
        {Context}

        Question:
        {Question}
    """
    template = PromptTemplate(input_variables=["System", "Context", "Question"], template=base_prompt)

    def __init__(self, dataset, dataName: str, dataColumns: DataColumnModel, evalStrat: Type[BaseEval]) -> None:
        self.datasetName = dataName
        self.data = dataset
        self.dataColumns = dataColumns
        self.evalStrat = evalStrat
    
    def get_init_data_frame(self) -> pd.DataFrame:
        return pd.DataFrame({'Dataset': [], 'System' : [],'Context': [],'Question': [], 'Model': [], 'Answer': [], 'Score': []})

    def evaluate_model(self, model: Type[BaseLLM], modelName : str, outputData: pd.DataFrame, systemMessage: str | None = None) -> pd.DataFrame:

        # Actual Chain we Call
        modelChain = self.template | model

        #loads system prompt for these questions
        if systemMessage:
            systemMessage = systemMessage
        elif self.dataColumns.systemPrompt:
            systemMessage = self.data[0][self.dataColumns.systemPrompt]
        else:
            systemMessage = None
        
        newrows = []
        
        for index, problem in enumerate(self.data):

            # Gets and stores data about question
            if self.dataColumns.contextColumn:
                context = problem[self.dataColumns.contextColumn]
            else:
                context = None

            question = problem[self.dataColumns.questionColumn]
            answer = problem[self.dataColumns.answerColumn]

            #Invokes LLM for response
            response = modelChain.invoke({"System" : systemMessage, "Context" : context, "Question" : question})

            #Scores based on given method
            eval = self.evalStrat.evaluate(response, answer)

            print(f"::COMPLETED:: Model {modelName}: Question {index} / {len(self.data)}: Dataset: {self.datasetName} Score: {eval}")

            if(index % 100 == 0):
                print(f"Question Response {response} \n \n Question Answer: {' '.join(answer["text"])} \n \n")

            #Combines all data
            newrow = {'Dataset': [self.datasetName], 'System' : [systemMessage],'Context': [context],'Question': [question], 'Model': [modelName], 'Answer': [answer], 'Score': [eval]}
            newrows.append(newrow)

        #Combines all Data
        outputData = pd.concat([outputData, pd.DataFrame(newrows)], ignore_index=True)

        return outputData



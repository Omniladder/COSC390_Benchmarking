from llama_index.core.evaluation import CorrectnessEvaluator
from llama_index.llms.openai import OpenAI
from langchain_ollama.llms import OllamaLLM
from langchain_openai import ChatOpenAI
import pandas as pd
import time
import os

"""

"""


from ThinkingAdapter.wrapper import ReasoningWrapper

from collaborate_funcs import Collaboration
from model_interface import ModelInterface

print("imports done")

api_key = os.environ["OPENAI_API_KEY"]
#mistral = OllamaLLM(model="mistral")
llama1 = OllamaLLM(model="llama3.1")
llama2 = OllamaLLM(model="llama3.1")
llama3 = OllamaLLM(model="llama3.1")
llama4 = OllamaLLM(model="llama3.1")

openai = ChatOpenAI(model="gpt-4o-mini", api_key=api_key)

reasoning_models = {
    "mistral": openai, # langchain base  model
    "deepseek": openai# langchain base  model
}

reasoner  = ReasoningWrapper(models=[llama1, llama2, llama3], output_model=llama4, tree_depth=3, tree_width=2)

'''
model_interface = ModelInterface(
    reasoning_models=reasoning_models,
    output_model=openai # langchain base chat model
)
'''
'''
mistral = OllamaLLM(model="mistral")
print("models declared") 

'''


json_data = pd.read_json("leetcodecomplete.jsonl", lines=True)

print("data loaded")

'''
collaboration = Collaboration(
    reasoning_model_ids=["mistral", "deepseek"], # list of model ids such as ["gpt", "deepseek"]
    model_interface=model_interface
)
'''

# print(json_data)
# for column_index, column in enumerate(json_data["instruction"]):


output_data = pd.DataFrame()
for json_index in range(len(json_data[:100])):
    try:
        print(json_index)
    
        eval_model = OpenAI("gpt-4o-mini")
        evaluator = CorrectnessEvaluator(llm=eval_model)
    
        prompt = f"""
            Instruction:
            {json_data["instruction"][json_index]}
        
            Input:
            {json_data["input"][json_index]}"""
    
        print(json_data["output"][json_index])
    
        start_time = time.perf_counter()
        #model_output = llama.invoke(prompt)
        #model_output = mistral.invoke(prompt)
        #model_output = openai.invoke(prompt)
        model_output = reasoner.invoke(prompt=prompt)
        '''
        model_output = collaboration.collaborate_code_2(
            prompt=prompt
        )
        '''
        end_time = time.perf_counter()
    
        result = evaluator.evaluate(
            query=prompt,
            response=model_output,
            reference=json_data["output"][json_index],
        )
    
        execution_time = end_time - start_time
        print(f"Execution time: {execution_time} seconds")
        print(model_output)
    
        output_data = pd.concat(
            [
                pd.DataFrame(
                    {
                        "input": [prompt],
                        "output": [model_output],
                        "correctness": [result.score],
                        "time": [execution_time],
                    }
                ),
                output_data,
            ]
        )
    except Exception as e:
        print(e)
        continue
    output_data.to_csv("./llama_results_ToT.csv", index=False)

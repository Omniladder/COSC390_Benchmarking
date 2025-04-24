from llama_index.core.evaluation import CorrectnessEvaluator
from llama_index.llms.openai import OpenAI
from langchain_ollama.llms import OllamaLLM
from langchain_openai import OpenAI as OpenAI_Model
import pandas as pd
import time
import os

from ThinkingAdapter.wrapper import ReasoningWrapper

from collaborate_funcs import Collaboration
from model_interface import ModelInterface

print("imports done")


reasoning_models = {
    "mistral": OllamaLLM(model="mistral"), # langchain base  model
    "deepseek": OllamaLLM(model="deepseek-v2")# langchain base  model
}

model_interface = ModelInterface(
    reasoning_models=reasoning_models,
    output_model=OllamaLLM(model="llama3.1") # langchain base chat model
)
mistral = OllamaLLM(model="mistral")
print("models declared")

json_data = pd.read_json("leetcodecomplete.jsonl", lines=True)

print("data loaded")

collaboration = Collaboration(
    reasoning_model_ids=["mistral", "deepseek"], # list of model ids such as ["gpt", "deepseek"]
    model_interface=model_interface
)

# print(json_data)
# for column_index, column in enumerate(json_data["instruction"]):

api_key = os.environ["OPENAI_API_KEY"]
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
        # model_output = llama.invoke(prompt)
        model_output = mistral.invoke(prompt)
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
    except:
        continue
    output_data.to_csv("./llama_results_linear.csv", index=False)

from langchain_ollama.llms import OllamaLLM
from wrapper import ReasoningWrapper
import pandas as pd
import time
from codebleu import calc_codebleu
import json


import ast
import difflib
import re

def strip_markdown_code_block(text):
    match = re.search(r"```python\n(.*?)\n```", text, re.DOTALL)
    return match.group(1) if match else None

def get_ast_score(prompt, answer, model):
    current_prompt = f"""
        Code the following prompt in python
    
    
        {prompt}
    """

    code_output = llama.invoke(current_prompt)
        
    output = (strip_markdown_code_block(output))
    code_output = (strip_markdown_code_block(code_output))
        
    tree1 = ast.dump(ast.parse((output)))
    tree2 = ast.dump(ast.parse((code_output)))
        
    similarity = difflib.SequenceMatcher(Noneone, tree1, tree2).ratio()
    return float(similarity)


llama = OllamaLLM(model="llama3.1")
deepseek = OllamaLLM(model="deepseek-v2")
mistral = OllamaLLM(model="mistral")

models = [llama, deepseek, mistral]

reasoner = ReasoningWrapper(models=models, output_model=llama, tree_depth=3, tree_width=2)

df = pd.read_json("hf://datasets/greengerong/leetcode/leetcode-train.jsonl", lines=True)

df = df.iloc[[0]]
print(df.columns)

'''
printrow = lambda row: print(row.columns)
df.apply(printrow, axis=1)
'''

df["llama_output"] = df.apply(lambda row: get_ast_score(row["content"], row["python"], llama))
df["llama_reason_output"] = df.apply(lambda row: get_ast_score(row["content"], row["python"], reasoning), axis=1)

print("AST Similarity:", similarity)

print(df)

print('importing')
import os
import json
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
from datasets import load_dataset
import evaluate
import re
print('done')

# Load benchmark dataset
print('load dataset')
dataset = load_dataset("gsm8k", "main", split="test")
print('loaded')

# Load evaluation metrics
print('eval load bleu')
bleu = evaluate.load("bleu")
print('done')
print('eval load rouge')
rouge = evaluate.load("rouge")
print('done')

def evaluate_model(model_name, dataset, num_samples=50):
    print(f"Evaluating model: {model_name}")
    print("Loading model...")
    llm = OllamaLLM(model=model_name)
    print("Model loaded")
    print("Forming prompt")
    prompt = ChatPromptTemplate.from_template("""
        You will be given a question. Please respond with the answer formatted inside a section labeled 'solution'. 

        For example:

        Question: What is 2 + 2?
        Answer: \solution{{4}}

        Now, answer the following question:
        Question: {question}

        Answer: \solution{{}}""")
    print(prompt)
    print("Prompt formed")
    print("Creating chain")
    chain = prompt | llm
    print("Chain created")

    pattern = r"{(.*?)}"

    scores = {"bleu": [], "rouge": [], "exact_match": []}
    

    for example in dataset.shuffle().select(range(num_samples)):
        print("Getting response")
        response = chain.invoke({"question": example["question"]})
        print("Question: " + example["question"])
        # response = re.findall(pattern, response)[0] #In theory parses
        print(f"\n\nRESPONSE\n\n{response}")
        reference = example["answer"]
        print(f"\n\nREFERENCE\n\n{reference}")
        
        # Calculate metrics
        print("getting scores for bleu")
        scores["bleu"].append(bleu.compute(
            predictions=[response],
            references=[reference]
        )["bleu"])
        
        print("getting scores for rouge")
        scores["rouge"].append(rouge.compute(
            predictions=[response],
            references=[reference]
        )["rougeL"])
        
        scores["exact_match"].append(
            response.strip() == reference.strip()
    )
    
    return {k: sum(v)/len(v) for k, v in scores.items()}

if __name__ == "__main__":
    print('initializing')
    llama_3 = {}
    ds = {}
    mistral = {}
    # Run evaluation
    for model in ["llama3.2", "deepseek-r1", "mistral"]:
        print(f"Evaluating {model}...")
        results = evaluate_model(model, dataset)

        print(f"Results for {model}:")
        print(results)
        print("\n" + "="*50 + "\n")

        if model == "llama3.2":
            llama_3["model"] = model
            llama_3["results"] = results
        elif model == "deepseek-r1":
            ds["model"] = model
            ds["results"] = results
        elif model == "mistral":
            mistral["model"] = model
            mistral["results"] = results
    
    with open('llama_results.json', 'w') as llama_file:
        json.dump(llama_3, llama_file)
    with open('deepseek_results.json', 'w') as deepseek_file:
        json.dump(ds, deepseek_file)
    with open('mistral_results.json', 'w') as mistral_file:
        json.dump(mistral, mistral_file)
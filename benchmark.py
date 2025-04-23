
from langchain_ollama.llms import OllamaLLM
from ThinkingAdapterScuffedVersion.model_interface import ModelInterface
from ThinkingAdapterScuffedVersion.collaboration_simple import collaborate


model_interface = ModelInterface()

llama = OllamaLLM(model="llama3.1")
deepseek = OllamaLLM(model="deepseek-v2")
mistral = OllamaLLM(model="mistral")


model_interface.add_reasoning_model("llama3.1", llama)
model_interface.add_reasoning_model("deepseek-v2", deepseek)
model_interface.add_reasoning_model("mistral", mistral)

model_interface.set_output_model(llama)

user_input = """
Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
Create a solution in python for the input asked.

### Input:
The algorithm initializes a set containing the elements from the banned list. It then iterates from `n` down to `1` and checks the following conditions:

1. If the current integer `i` is not in the banned set.
2. If the maxSum minus the current sum is greater than or equal to `i`.

If both conditions are met, it adds the integer to the sum and increments the count of chosen integers. After the loop finishes, the algorithm returns the count.

The algorithm efficiently finds the maximum number of integers that meet the conditions by iterating in decreasing order, and the usage of set ensures constant time lookup for banned elements.#
"""

for token in collaborate(["llama3.1", "deepseek-v2", "mistral"], model_interface=model_interface, user_input=user_input):
    print(token,end="",flush=True)
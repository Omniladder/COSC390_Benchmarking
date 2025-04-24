import pandas as pd
import matplotlib.pyplot as plt

llama_data = pd.read_csv("./llama_results.csv")
reason_data = pd.read_csv("./openai_results.csv")
linear_data = pd.read_csv("./llama_results_linear.csv")

'''
for i in range(100):

    if(llama_data["correctness"][:i].mean() < reason_data["correctness"][:i].mean()):
        print(i)

    
print(llama_data["correctness"][:100].mean())
print(reason_data["correctness"][:100].mean())
'''


#llama_data = llama_data[llama_data["correctness"] > 3]
#reason_data = reason_data[reason_data["correctness"] > 3]


columns = ["llama", "reason_adapter", "linear_adapter"]
model_scores = [llama_data["correctness"].mean(), reason_data["correctness"].mean(), linear_data["correctness"].mean()]



plt.bar(columns, model_scores)

plt.title("Model Correctness Score")
plt.xlabel("LLM Model")
plt.ylabel("Llama Index Score")

plt.savefig("./correctness_graph.png")
plt.show()



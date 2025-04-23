#imports for local stuff
from Models.DataColumnModel import DataColumnModel
from Objects.CosEval import CosEval
from Objects.LatexEval import LatexEval
from Objects.EvaluationObject import EvaluationObject

from datasets import load_dataset
from langchain_ollama.llms import OllamaLLM
from sentence_transformers import SentenceTransformer

print(DataColumnModel)

#Dataset
dataset_name = "HuggingFaceH4/MATH-500"
ds = load_dataset(dataset_name)
ds = ds["test"].select(range(1))

#Structure Object For SQuAD Dataset
dataStructure = DataColumnModel(questionColumn="problem", answerColumn="answer")

#Sets up Cosine Eval Objects
embeddingModel = SentenceTransformer('all-MiniLM-L6-v2')
#evalStrategy = CosEval(embeddingModel)
evalStrategy = LatexEval(parsing_regex=r"\\hbox{(.*)?}")

#Actual Object to do Evaluations on
evalEngine = EvaluationObject(dataset=ds, dataName=dataset_name, dataColumns=dataStructure, evalStrat=evalStrategy)

data = evalEngine.get_init_data_frame()

systemMessage = """
    You are a general question-answering model. You will be given a context (passage) and a question. 
    Answer the question based on the context. If the question cannot be answered from the context, respond with "No answer."
    Keep answers short, too the point and direct smallest possible answer attempt to use direct writing from text
""".strip()

model = OllamaLLM(model="llama3.1", temperature=0)
data = evalEngine.evaluate_model(model, "llama3.1", data, systemMessage)

model = OllamaLLM(model="mistral", temperature=0)
data = evalEngine.evaluate_model(model, "mistral", data, systemMessage)

model = OllamaLLM(model="deepseek-v2", temperature=0)
data = evalEngine.evaluate_model(model, "Deepseek", data, systemMessage)

data.to_csv("./llama_test.csv", index=False)

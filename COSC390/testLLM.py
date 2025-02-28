from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM

template = """Question: {question}

Answer: Let's think step by step."""

prompt = ChatPromptTemplate.from_template(template)

modelList = [OllamaLLM(model="llama3.2"), OllamaLLM(model="deepseek-r1"), OllamaLLM(model="mistral")]


for model in modelList:
    chain = prompt | model

    response = chain.invoke({"question": "What is LangChain?"})

    print(response)
    print("\n\n\n\n\n")
from typing import List, Dict, Any, Generator
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
import time
from model_interface import ModelInterface
from langchain_core.runnables import Runnable, RunnablePassthrough
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from prompts.reasoning_prompt import REASONING_PROMPT
from prompts.chat_prompt import CHAT_PROMPT
from prompts.chat_prompt import CHAT_PROMPT_CODE_COLLABORATION
from prompts.chat_prompt import CHAT_PROMPT_CODE_NO_COLLABORATION
from prompts.chat_prompt import CHAT_PROMPT_CODE_COLLABORATION_2
from prompts.reasoning_prompt import REASONING_PROMPT_CODE
from langchain_core.output_parsers import StrOutputParser

def collaborate(
    reasoning_model_ids: List[str],
    model_interface: ModelInterface,
    user_input: str,
) -> Generator[str, None, None]:
    """
    Perform a collaborative reasoning task with true parallel execution.
    
    Args:
        reasoning_model_ids: List of model IDs to use for reasoning
        model_interface: ModelInterface object containing the configured models
        user_input: User message to process
        
    Yields:
        Tokens from the reasoning models and final chat response
    """
    # Get models from interface
    reasoning_1 = model_interface.reasoning_models[reasoning_model_ids[0]]
    reasoning_2 = model_interface.reasoning_models[reasoning_model_ids[1]]
    
    # Create chains
    reasoning_1_chain = _create_reasoning_chain(reasoning_1)
    reasoning_2_chain = _create_reasoning_chain(reasoning_2)
    chat_chain = _create_chat_chain(model_interface.output_model)
    
    # Buffer for collecting output
    model1_buffer = []
    model2_buffer = []
    both_done = threading.Event()
    model1_done = threading.Event()
    model2_done = threading.Event()
    
    # Define worker functions that will run in parallel
    def process_model1():
        try:
            for chunk in reasoning_1_chain.stream({"message": user_input, "reasoning": ""}):
                if hasattr(chunk, 'content') and chunk.content:
                    model1_buffer.append(chunk.content)
            model1_done.set()
            if model2_done.is_set():
                both_done.set()
        except Exception as e:
            print(f"Error in model 1: {str(e)}")
            model1_done.set()
    
    def process_model2():
        try:
            # Start immediately but we'll only output after model1 is done
            for chunk in reasoning_2_chain.stream({"message": user_input, "reasoning": "".join(model1_buffer)}):
                if hasattr(chunk, 'content') and chunk.content:
                    model2_buffer.append(chunk.content)
            model2_done.set()
            if model1_done.is_set():
                both_done.set()
        except Exception as e:
            print(f"Error in model 2: {str(e)}")
            model2_done.set()
    
    # Start both models in separate threads
    with ThreadPoolExecutor(max_workers=2) as executor:
        executor.submit(process_model1)
        executor.submit(process_model2)
        
        # Start yielding results from model 1 immediately
        yield "<thinking>\n"
        
        # Wait for each token to be available from model 1
        last_idx = 0
        while not model1_done.is_set() or last_idx < len(model1_buffer):
            # Check if we have new tokens to yield
            if last_idx < len(model1_buffer):
                for i in range(last_idx, len(model1_buffer)):
                    yield model1_buffer[i]
                last_idx = len(model1_buffer)
            else:
                # Small sleep to avoid tight polling
                import time
                time.sleep(0.01)
        
        # Now yield from model 2, which might still be generating
        yield "\n\n"
        
        last_idx = 0
        while not model2_done.is_set() or last_idx < len(model2_buffer):
            # Check if we have new tokens to yield
            if last_idx < len(model2_buffer):
                for i in range(last_idx, len(model2_buffer)):
                    yield model2_buffer[i]
                last_idx = len(model2_buffer)
            else:
                # Small sleep to avoid tight polling
                import time
                time.sleep(0.01)
    
    yield "\n</thinking>\n"
    
    # Now generate the final response with all reasoning
    reasoning_output = "".join(model1_buffer) + "\n\n" + "".join(model2_buffer)
    for chunk in chat_chain.stream({"reasoning": reasoning_output, "user_query": user_input}):
        if hasattr(chunk, 'content') and chunk.content:
            yield chunk.content

def no_collaboration(
    model: BaseChatModel,
    user_input: str,
) -> Generator[str, None, None]:
    """
    Perform a single model reasoning task.
    """
    chat_chain = _create_chat_chain(model)
    full_response = chat_chain.invoke({"reasoning": "", "user_query": user_input})

def _create_reasoning_chain(model: BaseChatModel) -> Runnable:
    """
    Create a reasoning chain that takes a message and prior reasoning.
    """
    prompt = ChatPromptTemplate.from_template(REASONING_PROMPT)
    return prompt | model


def _create_chat_chain(model: BaseChatModel) -> Runnable:
    """
    Create a chat chain that takes reasoning and user query.
    """
    prompt = ChatPromptTemplate.from_template(CHAT_PROMPT)
    return prompt | model

def create_streaming_chain(model: BaseChatModel, variables: Dict[str, str], template: str) -> Runnable:
    """
    Create a custom streaming chain with a specific template and variables.
    """
    prompt = ChatPromptTemplate.from_template(template)
    
    if variables:
        chain = RunnablePassthrough.assign(**variables) | prompt | model
    else:
        chain = prompt | model
        
    return chain

def _create_reasoning_chain_code(
    model: BaseChatModel,
) -> Runnable:
    """
    Create a reasoning chain that takes a problem and prior reasoning.
    """
    prompt = ChatPromptTemplate.from_template(REASONING_PROMPT_CODE)
    return prompt | model | StrOutputParser()

def _create_chat_chain_code_collaboration(
    model: BaseChatModel,
) -> Runnable:
    """
    Create a chat chain that takes reasoning and user query.
    """
    prompt = ChatPromptTemplate.from_template(CHAT_PROMPT_CODE_COLLABORATION_2)
    return prompt | model | StrOutputParser()

def _create_chat_chain_code_no_collaboration(
    model: BaseChatModel,
) -> Runnable:
    """
    Create a chat chain that takes reasoning and user query.
    """
    prompt = ChatPromptTemplate.from_template(CHAT_PROMPT_CODE_NO_COLLABORATION)
    return prompt | model | StrOutputParser()

def no_collaboration_code_chat(
    model: BaseChatModel,
    title: str,
    description: str,
) -> Generator[str, None, None]:
    chat_chain = _create_chat_chain_code_no_collaboration(model)
    full_response = chat_chain.invoke({"title": title, "description": description})
    return full_response


def collaborate_code(
    reasoning_model_ids: List[str],
    model_interface: ModelInterface,
    title: str,
    description: str,
) -> Generator[str, None, None]:
    """
    Perform a collaborative reasoning task with true parallel execution.
    
    Args:
        reasoning_model_ids: List of model IDs to use for reasoning
        model_interface: ModelInterface object containing the configured models
        title: Title of the problem
        description: Description of the problem
        
    Yields:
        Tokens from the reasoning models and final chat response
    """
    # Get models from interface
    reasoning_1 = model_interface.reasoning_models[reasoning_model_ids[0]]
    reasoning_2 = model_interface.reasoning_models[reasoning_model_ids[1]]
    
    # Create chains
    reasoning_1_chain = _create_reasoning_chain_code(reasoning_1)
    reasoning_2_chain = _create_reasoning_chain_code(reasoning_2)
    chat_chain = _create_chat_chain_code_collaboration(model_interface.output_model)
    
    reasoning_1_output = reasoning_1_chain.invoke({"problem": f"Title: {title}\n\nDescription: {description}", "reasoning": ""})
    reasoning_2_output = reasoning_2_chain.invoke({"problem": f"Title: {title}\n\nDescription: {description}", "reasoning": reasoning_1_output})
    
    # Now generate the final response with all reasoning
    reasoning_output = reasoning_1_output + "\n\n" + reasoning_2_output
    final_output = chat_chain.invoke({
        "title": title,
        "description": description,
        "reasoning": reasoning_output
    })
    return final_output


class Collaboration:
    def __init__(self, reasoning_model_ids: List[str], model_interface: ModelInterface):
        # Get models from interface
        self.reasoning_1 = model_interface.reasoning_models[reasoning_model_ids[0]]
        self.reasoning_2 = model_interface.reasoning_models[reasoning_model_ids[1]]
        self.reasoning_1_chain = _create_reasoning_chain_code(self.reasoning_1)
        self.reasoning_2_chain = _create_reasoning_chain_code(self.reasoning_2)
        
        self.chat_chain = _create_chat_chain_code_collaboration(model_interface.output_model)
        

    def collaborate_code_2(
        self,
        prompt: str,
    ) -> Generator[str, None, None]:
        """
        Perform a collaborative reasoning task with true parallel execution.
        
        Args:
            reasoning_model_ids: List of model IDs to use for reasoning
            model_interface: ModelInterface object containing the configured models
            title: Title of the problem
            description: Description of the problem
            
        Yields:
            Tokens from the reasoning models and final chat response
        """

        reasoning_1_output = self.reasoning_1_chain.invoke(
            {
                "prompt": prompt,
                "reasoning": ""
            }
        )
        reasoning_2_output = self.reasoning_2_chain.invoke(
            {
                "prompt": prompt,
                "reasoning": reasoning_1_output
            }
        )
        
        # Now generate the final response with all reasoning
        reasoning_output = reasoning_1_output + "\n\n" + reasoning_2_output
        final_output = self.chat_chain.invoke({
            "prompt": prompt,
            "reasoning": reasoning_output
        })
        
        return final_output

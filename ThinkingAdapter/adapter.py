from typing import Any, Dict, Iterator, List, Mapping, Optional

from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.llms import BaseLLM
from langchain_core.outputs import GenerationChunk
from langchain_core.outputs import Generation, LLMResult
from langchain_core.output_parsers import StrOutputParser
import asyncio

from .prompts import topic_prompt, reason_prompt

import tracemalloc
tracemalloc.start()

class ReasoningAdapter():
    lock = asyncio.Lock()

    def __init__(self, models: List[BaseLLM], tree_width: int|None = None, tree_depth: int = 3):
        if(not tree_width):
            self.tree_width = len(models)
        else:
            self.tree_width = tree_width

        self.models= models
        self.tree_depth: int = tree_depth
    """The number of characters from the last message of the prompt to be echoed."""

    async def _build_reasoning(self, prompt:str, topic: str, starting_index: int = 0) -> str:
        shifted_models = self.models[:-1 * starting_index] + self.models[-1 * starting_index:]
        reasoning = []
        reason = ''

        for index in range(self.tree_depth):
            model = shifted_models[index % len(self.models)]
            model_chain = reason_prompt | model | StrOutputParser()
            reasoning.append(await model_chain.ainvoke({"prompt" : prompt, "topic": topic, "reason" : reason}))
            reason = '\n'.join(reasoning)
        
        return reason

   

    async def ainvoke(
        self,
        prompt: str,
    ) -> str:
        topics = []
        reason_outputs = []

        async def process_reasoning(index: int):
            model = self.models[index]
            
            async with self.lock:
                topic_chain = topic_prompt | model | StrOutputParser()
                new_topic = topic_chain.invoke({"prompt": prompt, "topics": topics})
                topics.append(new_topic)
            return await self._build_reasoning(prompt, new_topic, starting_index=index)
            
        reason_outputs = await asyncio.gather(*[process_reasoning(index) for index in range(self.tree_width)])
        reasoning = "\n\n".join(reason_outputs)

        reasoning = f"Topics explored:\n- " + "\n- ".join(topics) + "\n\nReasoning:\n" + reasoning
        
        return reasoning

    def invoke(self, prompt: str):
        return asyncio.run(self.ainvoke(prompt=prompt))


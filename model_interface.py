from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field
import logging
from langchain_ollama.llms import OllamaLLM
from langchain_core.language_models.chat_models import BaseChatModel

class ModelInterface(BaseModel):
    """A simple container for language models used in the collaborative reasoning system.
    
    This class stores references to language models for both reasoning and output
    generation, with minimal logic for model access and identification.
    """
    
    reasoning_models: Dict[str, Any] = Field(
        default_factory=dict,
        description="Dictionary of reasoning models {model_id: model}"
    )
    
    output_model: Optional[Any] = Field(
        None,
        description="Model used for final output generation"
    )
    
    class Config:
        arbitrary_types_allowed = True
    
    def add_reasoning_model(self, model_id: str, model: Any) -> None:
        """Add a reasoning model to the interface.
        
        Args:
            model_id: Unique identifier for the model
            model: LangChain chat model
        """
        self.reasoning_models[model_id] = model
    
    def get_reasoning_model(self, model_id: str) -> Optional[Any]:
        """Get a specific reasoning model by ID.
        
        Args:
            model_id: Identifier of the model to retrieve
            
        Returns:
            The model or None if not found
        """
        return self.reasoning_models.get(model_id)
    
    def set_output_model(self, model: Any) -> None:
        """
        Set the output model.
        
        Args:
            model: LangChain chat model for output generation
        """
        self.output_model = model
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the configured models.
        
        Returns:
            Dictionary with model information
        """
        return {
            "reasoning_models": list(self.reasoning_models.keys()),
            "output_model": "configured" if self.output_model else "not configured",
            "total_models": len(self.reasoning_models) + (1 if self.output_model else 0)
        } 
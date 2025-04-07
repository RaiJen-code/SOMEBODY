"""
Large Language Model (LLM) module for SOMEBODY using Ollama
"""

import logging
import os
import json
import requests
from typing import List, Dict, Any, Optional
from utils.common import timeit

logger = logging.getLogger(__name__)

class OllamaLLM:
    """Language model using Ollama API"""
    
    def __init__(self, model_name: str = "llama3"):
        """Initialize Ollama LLM
        
        Args:
            model_name: Name of the model to use in Ollama
        """
        self.model_name = model_name
        self.base_url = "http://localhost:11434/api"
        self.loaded = False
        logger.info(f"Initialized Ollama LLM with model: {model_name}")
    
    def load_model(self) -> bool:
        """Check if model is available in Ollama
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Check if model exists by calling Ollama API
            response = requests.get(f"{self.base_url}/tags")
            
            if response.status_code != 200:
                logger.error(f"Failed to get models from Ollama: {response.text}")
                return False
                
            models = response.json().get("models", [])
            model_names = [model.get("name") for model in models]
            
            if self.model_name in model_names:
                logger.info(f"Model {self.model_name} is available in Ollama")
                self.loaded = True
                return True
            else:
                logger.warning(f"Model {self.model_name} not found in Ollama")
                logger.info(f"Available models: {', '.join(model_names)}")
                logger.info(f"Attempting to pull model {self.model_name}...")
                
                # Try to pull the model
                pull_response = requests.post(
                    f"{self.base_url}/pull",
                    json={"name": self.model_name}
                )
                
                if pull_response.status_code == 200:
                    logger.info(f"Successfully pulled model {self.model_name}")
                    self.loaded = True
                    return True
                else:
                    logger.error(f"Failed to pull model {self.model_name}: {pull_response.text}")
                    return False
                    
        except requests.exceptions.ConnectionError:
            logger.error("Failed to connect to Ollama. Is Ollama running?")
            return False
        except Exception as e:
            logger.error(f"Error checking model availability: {str(e)}")
            return False
    
    @timeit
    def generate_response(self, prompt: str, max_tokens: int = 1000, temperature: float = 0.7) -> str:
        """Generate text response based on prompt
        
        Args:
            prompt: Text prompt for generation
            max_tokens: Maximum number of tokens to generate
            temperature: Temperature for sampling (higher = more random)
            
        Returns:
            Generated text response
        """
        if not self.loaded:
            if not self.load_model():
                return "I'm sorry, but I'm having trouble accessing my language model. Please check if Ollama is running."
        
        try:
            logger.info(f"Generating response for prompt: {prompt[:50]}...")
            
            # Call Ollama API for completion
            response = requests.post(
                f"{self.base_url}/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "num_predict": max_tokens,
                        "temperature": temperature
                    }
                }
            )
            
            if response.status_code != 200:
                logger.error(f"Error generating response: {response.text}")
                return "I'm sorry, but I encountered an error generating a response."
                
            result = response.json()
            generated_text = result.get("response", "")
            
            logger.info(f"Generated response: {generated_text[:50]}...")
            return generated_text
            
        except requests.exceptions.ConnectionError:
            logger.error("Failed to connect to Ollama. Is Ollama running?")
            return "I'm sorry, but I can't connect to my language model right now. Please check if Ollama is running."
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return f"I apologize, but I encountered an error: {str(e)}"
    
    @timeit
    def embedding(self, text: str) -> List[float]:
        """Generate embedding for text
        
        Args:
            text: Input text to embed
            
        Returns:
            Vector embedding as list of floats
        """
        if not self.loaded:
            if not self.load_model():
                logger.error("Failed to load model for embedding")
                # Return a zero vector as fallback
                return [0.0] * 10
        
        try:
            # Call Ollama API for embeddings
            response = requests.post(
                f"{self.base_url}/embeddings",
                json={
                    "model": self.model_name,
                    "prompt": text
                }
            )
            
            if response.status_code != 200:
                logger.error(f"Error generating embedding: {response.text}")
                # Return a zero vector as fallback
                return [0.0] * 10
                
            result = response.json()
            embedding = result.get("embedding", [])
            
            logger.info(f"Generated embedding with {len(embedding)} dimensions")
            return embedding
            
        except requests.exceptions.ConnectionError:
            logger.error("Failed to connect to Ollama. Is Ollama running?")
            return [0.0] * 10
        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            return [0.0] * 10
    
    def is_loaded(self) -> bool:
        """Check if model is loaded
        
        Returns:
            True if model is loaded, False otherwise
        """
        return self.loaded
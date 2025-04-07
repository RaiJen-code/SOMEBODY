"""
Script to test LLM functionality
"""

import os
import sys
import logging

# Add parent directory to path so we can import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm.model import OllamaLLM

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def test_ollama_llm():
    """Test Ollama LLM functionality"""
    try:
        # Initialize LLM
        logger.info("Initializing Ollama LLM")
        llm = OllamaLLM(model_name="llama3")  # Use the smallest/fastest model
        
        # Load model
        logger.info("Loading model")
        if not llm.load_model():
            logger.error("Failed to load model")
            return
        
        # Test generation
        test_prompt = "Explain what is an Orange Pi 5 in one paragraph:"
        logger.info(f"Testing generation with prompt: {test_prompt}")
        
        response = llm.generate_response(test_prompt, max_tokens=200)
        logger.info(f"Generated response: {response}")
        
        # Test embedding
        test_text = "Orange Pi 5"
        logger.info(f"Testing embedding with text: {test_text}")
        
        embedding = llm.embedding(test_text)
        logger.info(f"Generated embedding with {len(embedding)} dimensions")
        logger.info(f"First 5 dimensions: {embedding[:5]}")
        
        logger.info("Tests completed successfully")
        
    except Exception as e:
        logger.error(f"Error testing Ollama LLM: {str(e)}")

if __name__ == "__main__":
    test_ollama_llm()
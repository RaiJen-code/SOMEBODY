"""
Test script for vision integration with brain
"""

import os
import sys
import logging
import tempfile

# Add parent directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

from integration.brain import SomebodyBrain
from vision.camera import Camera

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def test_vision_brain_integration():
    """Test vision integration with brain"""
    
    try:
        # Initialize brain
        logger.info("Initializing brain...")
        brain = SomebodyBrain()
        
        if not brain.initialize():
            logger.error("Failed to initialize brain")
            return
        
        # Capture image
        logger.info("Capturing image from camera...")
        image_data = brain.camera.capture_image()
        
        if not image_data:
            logger.error("Failed to capture image")
            return
        
        logger.info(f"Captured image: {len(image_data)} bytes")
        
        # Process image with query
        query = "What do you see in this image?"
        logger.info(f"Processing image with query: {query}")
        
        result = brain.process_image(image_data, query)
        
        if "error" in result:
            logger.error(f"Error processing image: {result['error']}")
            return
        
        # Display analysis result
        logger.info(f"Image dimensions: {result.get('dimensions', {})}")
        logger.info(f"Face count: {result.get('face_count', 0)}")
        logger.info(f"Image description: {result.get('description', 'No description')}")
        
        # Display LLM response
        if "response" in result:
            logger.info(f"LLM response: {result['response']}")
            
            # Speak the response
            brain.speak(result['response'])
        
        logger.info("Vision brain integration test completed successfully")
        
    except Exception as e:
        logger.error(f"Error testing vision brain integration: {str(e)}")

if __name__ == "__main__":
    test_vision_brain_integration()
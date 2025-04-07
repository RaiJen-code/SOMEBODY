"""
Test script for speech integration with brain
"""

import os
import sys
import logging
import time

# Add parent directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

from integration.brain import SomebodyBrain

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def test_speech_brain_interaction():
    """Test speech interaction with brain"""
    
    try:
        # Initialize brain
        logger.info("Initializing brain...")
        brain = SomebodyBrain()
        
        if not brain.initialize():
            logger.error("Failed to initialize brain")
            return
        
        # Test text-to-speech
        logger.info("Testing text-to-speech...")
        test_text = "Hello, I am SOMEBODY, your personal assistant. I'm now ready to help you."
        brain.speak(test_text)
        
        # Test listening and responding
        logger.info("Testing listen and respond (speak into your microphone)...")
        logger.info("Recording will start in 3 seconds...")
        
        # Countdown
        for i in range(3, 0, -1):
            logger.info(f"{i}...")
            time.sleep(1)
            
        logger.info("Start speaking now!")
        
        # Listen and respond
        result = brain.listen_and_respond(listen_duration=5.0)
        
        logger.info(f"Interaction result: {result}")
        
        # Another test with direct text input
        logger.info("Testing direct text processing...")
        response = brain.process_text("What can you help me with?")
        
        logger.info(f"Response: {response}")
        
        # Speak the response
        brain.speak(response)
        
        logger.info("Tests completed")
        
    except Exception as e:
        logger.error(f"Error testing speech brain integration: {str(e)}")

if __name__ == "__main__":
    test_speech_brain_interaction()
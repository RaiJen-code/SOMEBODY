"""
Test script for vision components
"""

import os
import sys
import logging
import tempfile

# Add parent directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

from vision.camera import Camera
from vision.analyzer import VisionAnalyzer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def test_vision_components():
    """Test camera and vision analyzer"""
    
    try:
        # Test camera
        logger.info("Testing camera...")
        camera = Camera()
        
        # Open camera
        if not camera.open():
            logger.error("Failed to open camera")
            return
        
        # Capture image
        logger.info("Capturing image...")
        image_data = camera.capture_image()
        
        if not image_data:
            logger.error("Failed to capture image")
            return
        
        logger.info(f"Captured image: {len(image_data)} bytes")
        
        # Save image
        temp_image_path = os.path.join(tempfile.gettempdir(), "test_camera.jpg")
        camera.save_image(image_data, temp_image_path)
        
        logger.info(f"Saved image to {temp_image_path}")
        
        # Close camera
        camera.close()
        
        # Test vision analyzer
        logger.info("Testing vision analyzer...")
        analyzer = VisionAnalyzer()
        
        if not analyzer.load_model():
            logger.error("Failed to load vision analyzer")
            return
        
        # Analyze captured image
        logger.info("Analyzing image...")
        analysis = analyzer.analyze_image(image_data)
        
        logger.info(f"Image analysis: {analysis}")
        
        # Print description
        if "description" in analysis:
            logger.info(f"Image description: {analysis['description']}")
        
        logger.info("Vision component tests completed successfully")
        
    except Exception as e:
        logger.error(f"Error testing vision components: {str(e)}")

if __name__ == "__main__":
    test_vision_components()
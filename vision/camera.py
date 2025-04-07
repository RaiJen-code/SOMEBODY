"""
Camera module for SOMEBODY using OpenCV
"""

import logging
import os
import time
import tempfile
import numpy as np
from typing import Optional, Tuple, Union

logger = logging.getLogger(__name__)

class Camera:
    """Camera interface for capturing images using OpenCV"""
    
    def __init__(self, camera_index: int = 0):
        """Initialize camera
        
        Args:
            camera_index: Index of the camera to use
        """
        self.camera_index = camera_index
        self.camera = None
        self.cv2 = None  # Will be imported dynamically
        logger.info(f"Initialized camera with index {camera_index}")
    
    def open(self) -> bool:
        """Open camera connection
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Import OpenCV here to avoid importing it at module level
            import cv2
            self.cv2 = cv2
            
            logger.info(f"Opening camera {self.camera_index}")
            self.camera = cv2.VideoCapture(self.camera_index)
            
            if not self.camera.isOpened():
                logger.error(f"Failed to open camera {self.camera_index}")
                return False
                
            logger.info(f"Camera {self.camera_index} opened successfully")
            return True
        except ImportError:
            logger.error("OpenCV not installed. Cannot open camera.")
            return False
        except Exception as e:
            logger.error(f"Error opening camera {self.camera_index}: {str(e)}")
            return False
            
    def close(self) -> bool:
        """Close camera connection
        
        Returns:
            True if successful, False otherwise
        """
        try:
            if self.camera:
                logger.info("Closing camera")
                self.camera.release()
                self.camera = None
            return True
        except Exception as e:
            logger.error(f"Error closing camera: {str(e)}")
            return False
            
    def capture_image(self) -> Optional[bytes]:
        """Capture image from camera
        
        Returns:
            Image data as bytes, or None if failed
        """
        try:
            if not self.camera:
                if not self.open():
                    return None
            
            # Give camera time to adjust
            time.sleep(0.5)
                    
            logger.info("Capturing image")
            ret, frame = self.camera.read()
            
            if not ret:
                logger.error("Failed to capture image")
                return None
                
            # Convert to bytes
            _, buffer = self.cv2.imencode('.jpg', frame)
            image_bytes = buffer.tobytes()
            
            logger.info(f"Captured image: {len(image_bytes)} bytes")
            return image_bytes
        except Exception as e:
            logger.error(f"Error capturing image: {str(e)}")
            return None
    
    def save_image(self, image_data: bytes, file_path: str) -> bool:
        """Save image data to file
        
        Args:
            image_data: Image data as bytes
            file_path: Path to save image
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with open(file_path, 'wb') as f:
                f.write(image_data)
            logger.info(f"Saved image to {file_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving image: {str(e)}")
            return False
            
    def capture_and_save(self, file_path: str) -> bool:
        """Capture image and save to file
        
        Args:
            file_path: Path to save image
            
        Returns:
            True if successful, False otherwise
        """
        image_data = self.capture_image()
        if image_data:
            return self.save_image(image_data, file_path)
        return False
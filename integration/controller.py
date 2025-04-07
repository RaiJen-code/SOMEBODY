"""
SOMEBODY Controller - Manages brain operations
"""

import logging
import threading
import os
import numpy as np
from typing import Optional, Dict, Any

from integration.brain import SomebodyBrain
from config.settings import DATA_DIR

logger = logging.getLogger(__name__)

class SomebodyController:
    """Controller class for managing SOMEBODY brain"""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        """Singleton pattern to ensure only one controller instance exists"""
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(SomebodyController, cls).__new__(cls)
                cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Initialize controller (only once due to singleton pattern)"""
        if not self._initialized:
            logger.info("Initializing SOMEBODY controller")
            self.brain = SomebodyBrain()
            self.state_dir = os.path.join(DATA_DIR, "state")
            self._initialized = True
    
    def initialize(self) -> bool:
        """Initialize brain
        
        Returns:
            True if successful, False otherwise
        """
        return self.brain.initialize()
    
    def process_text(self, text: str) -> str:
        """Process text input
        
        Args:
            text: Input text
            
        Returns:
            Response text
        """
        return self.brain.process_text(text)
    
    def process_audio(self, audio_data: bytes) -> Dict[str, Any]:
        """Process audio input
        
        Args:
            audio_data: Audio data as bytes
            
        Returns:
            Dictionary with response data
        """
        return self.brain.process_audio(audio_data)
    
    def process_image(self, image_data: bytes, query: Optional[str] = None) -> Dict[str, Any]:
        """Process image
        
        Args:
            image_data: Image data as bytes
            query: Optional text query about the image
            
        Returns:
            Dictionary with analysis results
        """
        return self.brain.process_image(image_data, query)
    
    def capture_and_analyze_image(self, query: Optional[str] = None) -> Dict[str, Any]:
        """Capture and analyze image
        
        Args:
            query: Optional text query about the image
            
        Returns:
            Dictionary with analysis results
        """
        return self.brain.capture_and_analyze_image(query)
    
    def listen_and_respond(self, listen_duration: float = 5.0) -> Dict[str, Any]:
        """Listen for speech, process it, and respond verbally
        
        Args:
            listen_duration: Duration to listen in seconds
            
        Returns:
            Dictionary with results of interaction
        """
        return self.brain.listen_and_respond(listen_duration)

    def listen(self, duration: float = 5.0) -> Dict[str, Any]:
        """Listen through microphone and transcribe speech
        
        Args:
            duration: Recording duration in seconds
            
        Returns:
            Dictionary with transcribed text and other info
        """
        return self.brain.listen(duration)

    def speak(self, text: str) -> bool:
        """Speak text using text-to-speech
        
        Args:
            text: Text to speak
            
        Returns:
            True if successful, False otherwise
        """
        return self.brain.speak(text)
    
    # Untuk controller.py
    def train_yolo_model(self, dataset_path: str, epochs: int = 50) -> Dict[str, Any]:
        """Train custom YOLO model for electronic components
        
        Args:
            dataset_path: Path to dataset in YOLO format
            epochs: Number of training epochs
            
        Returns:
            Dictionary with training results
        """
        return self.brain.train_yolo_model(dataset_path, epochs)
    
    def detect_objects(self, image_data: bytes, confidence: float = 0.25) -> Dict[str, Any]:
        """Detect objects in image using YOLO
        
        Args:
            image_data: Image data as bytes
            confidence: Confidence threshold (0-1)
            
        Returns:
            Dictionary with detection results
        """
        return self.brain.detect_objects(image_data, confidence)
    
    def save_state(self) -> bool:
        """Save controller state
        
        Returns:
            True if successful, False otherwise
        """
        return self.brain.save_state(self.state_dir)
    
    def load_state(self) -> bool:
        """Load controller state
        
        Returns:
            True if successful, False otherwise
        """
        return self.brain.load_state(self.state_dir)
    
    def shutdown(self) -> None:
        """Clean shutdown of controller"""
        logger.info("Shutting down SOMEBODY controller")
        # Save state before shutdown
        self.save_state()
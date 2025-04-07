"""
Wake word detection module for SOMEBODY
"""

import logging
import os
import time
import threading
import pvporcupine
import pyaudio
import numpy as np
from typing import Optional, Callable

logger = logging.getLogger(__name__)

class WakeWordDetector:
    """Detector for wake word 'Hey SOMEBODY'"""

    # def __init__(self, 
    #              keyword_paths: Optional[list] = None,
    #              keywords: Optional[list] = None,
    #              access_key: Optional[str] = None,
    #              callback: Optional[Callable] = None):
    #     """Initialize wake word detector
        
    #     Args:
    #         keyword_paths: Optional paths to custom keyword model files
    #         keywords: List of built-in keywords to detect ("jarvis", "computer", etc.)
    #         access_key: Picovoice access key (optional for built-in keywords)
    #         callback: Function to call when wake word is detected
    #     """
    #     self.keyword_paths = keyword_paths
    #     self.keywords = keywords or ["jarvis"]  # Use "jarvis" as a proxy for "SOMEBODY"
    #     self.access_key = access_key
    #     self.callback = callback
    #     self.porcupine = None
    #     self.audio = None
    #     self.stream = None
    #     self._is_running = False
    #     self._thread = None
    #     logger.info(f"Wake word detector initialized with keywords: {self.keywords}")
    def __init__(self, 
                keyword_paths: Optional[list] = None,
                keywords: Optional[list] = None,
                access_key: Optional[str] = None,
                callback: Optional[Callable] = None):
        """Initialize wake word detector
        
        Args:
            keyword_paths: Optional paths to custom keyword model files
            keywords: List of built-in keywords to detect ("jarvis", "computer", etc.)
            access_key: Picovoice access key (optional for built-in keywords)
            callback: Function to call when wake word is detected
        """
        # Load access key from environment if not provided
        from dotenv import load_dotenv
        import os
        
        load_dotenv()
        self.access_key = access_key or os.getenv("PICOVOICE_ACCESS_KEY")
        
        # Check for custom keyword model
        model_path = os.path.join("data", "models", "wake_word", "SOMEBODY_en_windows_v3_0_0.ppn")
        if os.path.exists(model_path):
            self.keyword_paths = [model_path]
            self.keywords = None
            logger.info(f"Using custom wake word model: {model_path}")
        else:
            self.keyword_paths = keyword_paths
            self.keywords = keywords or ["jarvis"]
            logger.info(f"Using built-in keywords: {self.keywords}")
            
        self.callback = callback
        self.porcupine = None
        self.audio = None
        self.stream = None
        self._is_running = False
        self._thread = None
        
    def initialize(self) -> bool:
        """Initialize porcupine engine
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Create Porcupine engine
            self.porcupine = pvporcupine.create(
                access_key=self.access_key,
                keyword_paths=self.keyword_paths,
                keywords=self.keywords,
                sensitivities=[0.7] * (len(self.keywords) if self.keywords is not None else 0) +
                              [0.7] * (len(self.keyword_paths) if self.keyword_paths is not None else 0)
            )
            
            # Initialize PyAudio
            self.audio = pyaudio.PyAudio()
            
            logger.info("Wake word detector initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Error initializing wake word detector: {str(e)}")
            self.cleanup()
            return False
            
    def start(self) -> bool:
        """Start listening for wake word in background thread
        
        Returns:
            True if started successfully, False otherwise
        """
        if self._is_running:
            logger.warning("Wake word detector is already running")
            return True
            
        if not self.porcupine:
            if not self.initialize():
                return False
        
        try:
            # Open microphone stream
            self.stream = self.audio.open(
                rate=self.porcupine.sample_rate,
                channels=1,
                format=pyaudio.paInt16,
                input=True,
                frames_per_buffer=self.porcupine.frame_length
            )
            
            self._is_running = True
            self._thread = threading.Thread(target=self._listen)
            self._thread.daemon = True
            self._thread.start()
            
            logger.info("Wake word detector started")
            return True
        except Exception as e:
            logger.error(f"Error starting wake word detector: {str(e)}")
            self.cleanup()
            return False
    
    def _listen(self):
        """Background thread to listen for wake word"""
        try:
            while self._is_running:
                # Read audio frame
                pcm = self.stream.read(self.porcupine.frame_length, exception_on_overflow=False)
                if not pcm:
                    logger.warning("Received empty audio frame")
                    continue
                try:
                    pcm = np.frombuffer(pcm, dtype=np.int16)
                except Exception as e:
                    logger.error(f"Error processing audio frame: {str(e)}")
                    continue
                
                # Process frame with Porcupine
                keyword_index = self.porcupine.process(pcm)
                
                # If wake word detected
                if keyword_index >= 0:
                    if self.keywords:
                        detected_keyword = self.keywords[keyword_index] if keyword_index < len(self.keywords) else "custom_keyword"
                    else:
                        # Get keyword name from custom model path
                        detected_keyword = os.path.splitext(os.path.basename(
                            self.keyword_paths[keyword_index]
                        ))[0].split('_')[0]
                    logger.debug(f"Wake word index: {keyword_index}, Keyword: {detected_keyword}")
                    logger.info(f"Wake word detected: {detected_keyword}")
                    
                    # Call callback if provided
                    if self.callback:
                        self.callback(detected_keyword)
        except Exception as e:
            if self._is_running:  # Only log if we're supposed to be running
                logger.error(f"Error in wake word detection loop: {str(e)}")
        finally:
            logger.info("Wake word detection loop stopped")
    
    def stop(self):
        """Stop wake word detection"""
        self._is_running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        self.cleanup()
        logger.info("Wake word detector stopped")
    
    def cleanup(self):
        """Release resources"""
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None
            
        if self.audio:
            self.audio.terminate()
            self.audio = None
            
        if self.porcupine:
            self.porcupine.delete()
            self.porcupine = None
        
    def is_running(self) -> bool:
        """Check if detector is running
        
        Returns:
            True if running, False otherwise
        """
        return self._is_running
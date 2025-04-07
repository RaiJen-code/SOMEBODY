"""
Speech recognition module for SOMEBODY using Whisper
"""

import logging
import os
import tempfile
import numpy as np
from typing import Optional, Union
from utils.common import timeit

logger = logging.getLogger(__name__)

class WhisperRecognizer:
    """Speech recognition using OpenAI's Whisper model"""
    
    def __init__(self, model_name: str = "tiny"):
        """Initialize Whisper recognizer
        
        Args:
            model_name: Whisper model name ("tiny", "base", "small", "medium", "large")
        """
        self.model_name = model_name
        self.model = None
        self.loaded = False
        logger.info(f"Initialized Whisper recognizer with model: {model_name}")
    
    def load_model(self) -> bool:
        """Load the Whisper model
        
        Returns:
            True if successful, False otherwise
        """
        try:
            import whisper
            logger.info(f"Loading Whisper model: {self.model_name}")
            self.model = whisper.load_model(self.model_name)
            self.loaded = True
            logger.info(f"Whisper model {self.model_name} loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Error loading Whisper model: {str(e)}")
            return False
    
    @timeit
    def transcribe(self, audio_data: Union[str, bytes, np.ndarray]) -> str:
        """Transcribe audio to text
        
        Args:
            audio_data: Audio data as file path, bytes, or numpy array
            
        Returns:
            Transcribed text
        """
        if not self.loaded:
            if not self.load_model():
                return "Failed to load speech recognition model."
        
        try:
            # Handle different types of audio data
            if isinstance(audio_data, str):
                # It's a file path
                audio_path = audio_data
                logger.info(f"Using audio file path: {audio_path}")
                
                # Check if file exists
                if not os.path.exists(audio_path):
                    error_msg = f"Audio file does not exist: {audio_path}"
                    logger.error(error_msg)
                    return f"Error: {error_msg}"
                    
                # Check file size
                file_size = os.path.getsize(audio_path)
                logger.info(f"Audio file size: {file_size} bytes")
                
                if file_size == 0:
                    error_msg = f"Audio file is empty: {audio_path}"
                    logger.error(error_msg)
                    return f"Error: {error_msg}"
                
            elif isinstance(audio_data, bytes):
                # It's audio data as bytes, save to temporary file
                logger.info(f"Converting {len(audio_data)} bytes to temporary file")
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                    temp_file.write(audio_data)
                    audio_path = temp_file.name
                logger.info(f"Saved audio bytes to temporary file: {audio_path}")
                
            elif isinstance(audio_data, np.ndarray):
                # It's a numpy array, save to temporary file
                logger.info(f"Converting numpy array of shape {audio_data.shape} to temporary file")
                import soundfile as sf
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                    sf.write(temp_file.name, audio_data, 16000)  # Assuming 16kHz sample rate
                    audio_path = temp_file.name
                logger.info(f"Saved numpy array to temporary file: {audio_path}")
                
            else:
                error_msg = f"Unsupported audio data type: {type(audio_data)}"
                logger.error(error_msg)
                return f"Error: {error_msg}"
            
            logger.info(f"Transcribing audio file: {audio_path}")
            
            # Transcribe with Whisper
            result = self.model.transcribe(audio_path)
            transcribed_text = result["text"].strip()
            
            # Clean up temporary file if created
            if isinstance(audio_data, (bytes, np.ndarray)) and os.path.exists(audio_path):
                logger.info(f"Removing temporary file: {audio_path}")
                os.unlink(audio_path)
            
            logger.info(f"Transcription result: {transcribed_text[:100]}...")
            return transcribed_text
            
        except Exception as e:
            logger.error(f"Error transcribing audio: {str(e)}", exc_info=True)
            return f"Error in transcription: {str(e)}"
    
    def is_loaded(self) -> bool:
        """Check if model is loaded
        
        Returns:
            True if model is loaded, False otherwise
        """
        return self.loaded
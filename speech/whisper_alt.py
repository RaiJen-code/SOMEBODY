"""
Alternative Whisper speech recognition that doesn't rely on FFmpeg
"""

import os
import numpy as np
import logging
import tempfile
import soundfile as sf
from typing import Union

logger = logging.getLogger(__name__)

def load_audio_file(file_path: str, sr: int = 16000) -> np.ndarray:
    """Load audio file using soundfile instead of FFmpeg
    
    Args:
        file_path: Path to audio file
        sr: Target sample rate
        
    Returns:
        Audio data as float32 numpy array normalized to [-1, 1]
    """
    try:
        logger.info(f"Loading audio file with soundfile: {file_path}")
        
        # Read audio file
        audio_data, sample_rate = sf.read(file_path)
        
        # Convert to mono if stereo
        if len(audio_data.shape) > 1:
            audio_data = audio_data.mean(axis=1)
        
        # Resample if needed
        if sample_rate != sr:
            # Simple resampling (not the best quality but works for testing)
            # For better resampling consider using librosa or scipy
            duration = len(audio_data) / sample_rate
            new_length = int(duration * sr)
            indices = np.linspace(0, len(audio_data) - 1, new_length)
            audio_data = np.interp(indices, np.arange(len(audio_data)), audio_data)
        
        # Normalize to [-1, 1]
        if audio_data.dtype == np.int16:
            audio_data = audio_data.astype(np.float32) / 32768.0
        elif audio_data.dtype == np.int32:
            audio_data = audio_data.astype(np.float32) / 2147483648.0
        elif audio_data.dtype == np.uint8:
            audio_data = (audio_data.astype(np.float32) - 128) / 128.0
        
        logger.info(f"Audio loaded successfully, shape: {audio_data.shape}, sample rate: {sr}")
        return audio_data
        
    except Exception as e:
        logger.error(f"Error loading audio file: {str(e)}")
        # Return empty array if loading fails
        return np.zeros(0, dtype=np.float32)

class WhisperAlternative:
    """Alternative implementation that loads audio using soundfile"""
    
    def __init__(self, model_name: str = "tiny"):
        """Initialize Whisper recognizer
        
        Args:
            model_name: Whisper model name ("tiny", "base", "small", "medium", "large")
        """
        self.model_name = model_name
        self.model = None
        self.loaded = False
        logger.info(f"Initialized Whisper alternative with model: {model_name}")
    
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
    
    def transcribe(self, audio_path: str) -> str:
        """Transcribe audio to text
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Transcribed text
        """
        if not self.loaded:
            if not self.load_model():
                return "Failed to load speech recognition model."
        
        try:
            logger.info(f"Transcribing audio file: {audio_path}")
            
            # Check if file exists
            if not os.path.exists(audio_path):
                error_msg = f"Audio file does not exist: {audio_path}"
                logger.error(error_msg)
                return f"Error: {error_msg}"
                
            # Load audio using our custom function
            audio_data = load_audio_file(audio_path)
            
            if len(audio_data) == 0:
                error_msg = "Failed to load audio data"
                logger.error(error_msg)
                return f"Error: {error_msg}"
            
            # Transcribe directly with audio array
            logger.info(f"Transcribing audio data with shape {audio_data.shape}")
            result = self.model.transcribe(audio_data)
            transcribed_text = result["text"].strip()
            
            logger.info(f"Transcription result: {transcribed_text[:100]}...")
            return transcribed_text
            
        except Exception as e:
            logger.error(f"Error transcribing audio: {str(e)}")
            return f"Error in transcription: {str(e)}"
    
    def is_loaded(self) -> bool:
        """Check if model is loaded
        
        Returns:
            True if model is loaded, False otherwise
        """
        return self.loaded
"""
Audio utilities for speech processing
"""

import logging
import wave
import numpy as np
from typing import Tuple, Optional

logger = logging.getLogger(__name__)

def load_audio_file(file_path: str) -> Tuple[np.ndarray, int]:
    """Load audio file and return audio data and sample rate
    
    Args:
        file_path: Path to audio file
        
    Returns:
        Tuple containing audio data as numpy array and sample rate
    """
    try:
        with wave.open(file_path, 'rb') as wf:
            # Get audio parameters
            n_channels = wf.getnchannels()
            sample_width = wf.getsampwidth()
            sample_rate = wf.getframerate()
            n_frames = wf.getnframes()
            
            # Read audio data
            audio_data = wf.readframes(n_frames)
            
            # Convert to numpy array
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            
            # Reshape for multi-channel audio
            if n_channels > 1:
                audio_array = audio_array.reshape(-1, n_channels)
            
            logger.info(f"Loaded audio file: {file_path}, sample rate: {sample_rate}")
            return audio_array, sample_rate
    except Exception as e:
        logger.error(f"Error loading audio file {file_path}: {str(e)}")
        # Return empty array and standard sample rate
        return np.array([]), 16000

def save_audio_file(file_path: str, audio_data: np.ndarray, sample_rate: int, n_channels: int = 1) -> bool:
    """Save audio data to WAV file
    
    Args:
        file_path: Output file path
        audio_data: Audio data as numpy array
        sample_rate: Sample rate in Hz
        n_channels: Number of channels
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Make sure directory exists
        import os
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        
        # Convert to int16 if needed
        if audio_data.dtype != np.int16:
            audio_data = (audio_data * 32767).astype(np.int16)
        
        with wave.open(file_path, 'wb') as wf:
            wf.setnchannels(n_channels)
            wf.setsampwidth(2)  # 2 bytes for int16
            wf.setframerate(sample_rate)
            wf.writeframes(audio_data.tobytes())
            
        logger.info(f"Saved audio file: {file_path}")
        return True
    except Exception as e:
        logger.error(f"Error saving audio file {file_path}: {str(e)}")
        return False
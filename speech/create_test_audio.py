"""
Utility to create test audio file
"""

import os
import numpy as np
import soundfile as sf
import tempfile
import logging

logger = logging.getLogger(__name__)

def create_test_audio_file():
    """Create a simple test audio file with sine wave
    
    Returns:
        Path to created audio file
    """
    try:
        # Generate a simple sine wave
        sample_rate = 16000
        duration = 3  # seconds
        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        
        # Create a 440 Hz sine wave
        audio_data = np.sin(2 * np.pi * 440 * t) * 0.5
        
        # Convert to int16
        audio_data = (audio_data * 32767).astype(np.int16)
        
        # Create temporary file
        temp_dir = tempfile.gettempdir()
        test_audio_path = os.path.join(temp_dir, "test_tone.wav")
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(os.path.abspath(test_audio_path)), exist_ok=True)
        
        # Save to file
        sf.write(test_audio_path, audio_data, sample_rate)
        
        # Verify file
        if os.path.exists(test_audio_path):
            file_size = os.path.getsize(test_audio_path)
            logger.info(f"Created test audio file: {test_audio_path}, size: {file_size} bytes")
            return test_audio_path
        else:
            logger.error(f"Failed to create test audio file")
            return None
    except Exception as e:
        logger.error(f"Error creating test audio file: {str(e)}")
        return None
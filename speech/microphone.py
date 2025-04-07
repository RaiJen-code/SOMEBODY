"""
Microphone utility for capturing audio input
"""

import logging
import tempfile
import numpy as np
import os
import soundfile as sf
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

class Microphone:
    """Interface for capturing audio from microphone"""
    
    def __init__(self, device_index: Optional[int] = None, sample_rate: int = 16000):
        """Initialize microphone
        
        Args:
            device_index: Index of audio input device
            sample_rate: Audio sample rate in Hz
        """
        self.device_index = device_index
        self.sample_rate = sample_rate
        self.stream = None
        self.pyaudio = None
        logger.info(f"Initialized microphone with device index: {device_index}, sample rate: {sample_rate}Hz")
    
    def list_devices(self) -> list:
        """List available audio input devices
        
        Returns:
            List of audio device info
        """
        try:
            import pyaudio
            p = pyaudio.PyAudio()
            
            devices = []
            for i in range(p.get_device_count()):
                device_info = p.get_device_info_by_index(i)
                if device_info.get('maxInputChannels') > 0:  # Input device
                    devices.append({
                        'index': i,
                        'name': device_info.get('name'),
                        'channels': device_info.get('maxInputChannels')
                    })
            
            p.terminate()
            return devices
        except ImportError:
            logger.error("PyAudio not installed. Cannot list audio devices.")
            return []
        except Exception as e:
            logger.error(f"Error listing audio devices: {str(e)}")
            return []
    
    def open(self) -> bool:
        """Open audio stream
        
        Returns:
            True if successful, False otherwise
        """
        try:
            import pyaudio
            
            self.pyaudio = pyaudio.PyAudio()
            self.stream = self.pyaudio.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=self.sample_rate,
                input=True,
                input_device_index=self.device_index,
                frames_per_buffer=1024
            )
            
            logger.info(f"Opened microphone stream with device index: {self.device_index}")
            return True
        except ImportError:
            logger.error("PyAudio not installed. Cannot open microphone.")
            return False
        except Exception as e:
            logger.error(f"Error opening microphone: {str(e)}")
            return False
    
    def close(self) -> bool:
        """Close audio stream
        
        Returns:
            True if successful, False otherwise
        """
        try:
            if self.stream:
                self.stream.stop_stream()
                self.stream.close()
                
            if self.pyaudio:
                self.pyaudio.terminate()
                
            self.stream = None
            self.pyaudio = None
            
            logger.info("Closed microphone stream")
            return True
        except Exception as e:
            logger.error(f"Error closing microphone: {str(e)}")
            return False
    
    def record(self, duration: float = 5.0) -> Tuple[np.ndarray, int]:
        """Record audio for specified duration
        
        Args:
            duration: Recording duration in seconds
            
        Returns:
            Tuple of (audio data as numpy array, sample rate)
        """
        try:
            import pyaudio
            import numpy as np
            
            if not self.stream:
                if not self.open():
                    return np.array([]), self.sample_rate
            
            logger.info(f"Recording audio for {duration} seconds")
            
            frames = []
            for i in range(0, int(self.sample_rate / 1024 * duration)):
                data = self.stream.read(1024, exception_on_overflow=False)
                frames.append(data)
            
            # Convert to numpy array
            audio_data = np.frombuffer(b''.join(frames), dtype=np.int16)
            
            logger.info(f"Recorded {len(audio_data)} samples at {self.sample_rate}Hz")
            return audio_data, self.sample_rate
            
        except Exception as e:
            logger.error(f"Error recording audio: {str(e)}")
            return np.array([]), self.sample_rate
    
    def save_recording(self, audio_data: np.ndarray, file_path: str) -> bool:
        """Save recorded audio to file
        
        Args:
            audio_data: Audio data as numpy array
            file_path: Path to save audio file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            import soundfile as sf
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
            
            logger.info(f"Saving audio to {file_path}")
            sf.write(file_path, audio_data, self.sample_rate)
            
            # Verify file was created
            if os.path.exists(file_path):
                file_size = os.path.getsize(file_path)
                logger.info(f"Audio file saved successfully: {file_path}, size: {file_size} bytes")
                return True
            else:
                logger.error(f"Failed to verify audio file: {file_path}")
                return False
                
        except Exception as e:
            logger.error(f"Error saving audio: {str(e)}")
            return False
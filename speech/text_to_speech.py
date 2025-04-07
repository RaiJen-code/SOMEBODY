"""
Text-to-Speech module for SOMEBODY using pyttsx3
"""

import logging
import os
import tempfile
from typing import Optional, Union

logger = logging.getLogger(__name__)

class PyttsxTTS:
    """Text-to-speech using pyttsx3"""
    
    def __init__(self, voice_id: Optional[str] = None, rate: int = 200):
        """Initialize pyttsx3 text-to-speech
        
        Args:
            voice_id: ID of voice to use (None for default)
            rate: Speech rate (words per minute)
        """
        self.voice_id = voice_id
        self.rate = rate
        self.engine = None
        self.initialized = False
        logger.info(f"Initialized pyttsx3 TTS with voice ID: {voice_id}, rate: {rate}")
    
    def initialize(self) -> bool:
        """Initialize TTS engine
        
        Returns:
            True if successful, False otherwise
        """
        try:
            import pyttsx3
            
            logger.info("Initializing pyttsx3 TTS engine")
            self.engine = pyttsx3.init()
            
            # Set properties
            self.engine.setProperty('rate', self.rate)
            
            if self.voice_id:
                self.engine.setProperty('voice', self.voice_id)
            
            self.initialized = True
            logger.info("pyttsx3 TTS engine initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Error initializing pyttsx3 TTS engine: {str(e)}")
            return False
    
    def get_available_voices(self) -> list:
        """Get list of available voices
        
        Returns:
            List of voice info dictionaries
        """
        if not self.initialized:
            if not self.initialize():
                return []
        
        try:
            voices = []
            for voice in self.engine.getProperty('voices'):
                voices.append({
                    'id': voice.id,
                    'name': voice.name,
                    'languages': voice.languages,
                    'gender': voice.gender
                })
            
            return voices
        except Exception as e:
            logger.error(f"Error getting available voices: {str(e)}")
            return []
    
    def synthesize(self, text: str) -> bytes:
        """Synthesize text to audio
        
        Args:
            text: Text to synthesize
            
        Returns:
            Audio data as bytes
        """
        if not self.initialized:
            if not self.initialize():
                return b""
        
        try:
            logger.info(f"Synthesizing text: {text[:50]}...")
            
            # Create a temporary file to save audio
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_path = temp_file.name
            
            # Save speech to file
            self.engine.save_to_file(text, temp_path)
            self.engine.runAndWait()
            
            # Read file into bytes
            with open(temp_path, 'rb') as f:
                audio_data = f.read()
            
            # Clean up
            os.unlink(temp_path)
            
            logger.info(f"Synthesized {len(audio_data)} bytes of audio")
            return audio_data
            
        except Exception as e:
            logger.error(f"Error synthesizing speech: {str(e)}")
            return b""
    
    def speak(self, text: str) -> bool:
        """Speak text directly through speakers
        
        Args:
            text: Text to speak
            
        Returns:
            True if successful, False otherwise
        """
        try:
            import pyttsx3
            
            logger.info(f"Speaking text: {text[:50]}...")
            engine = pyttsx3.init()
            engine.setProperty('rate', self.rate)
            if self.voice_id:
                engine.setProperty('voice', self.voice_id)
            engine.say(text)
            engine.runAndWait()
            return True
        except Exception as e:
            logger.error(f"Error speaking text: {str(e)}")
            return False
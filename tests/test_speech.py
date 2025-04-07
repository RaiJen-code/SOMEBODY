"""
Script to test Speech components
"""

import os
import sys
import logging
import tempfile
import time

# Add parent directory to path so we can import modules
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

from speech.recognition import WhisperRecognizer
from speech.text_to_speech import PyttsxTTS
from speech.microphone import Microphone
from speech.create_test_audio import create_test_audio_file

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def test_speech_components():
    """Test speech recognition and text-to-speech"""
    try:
        # Part 1: Create a test audio file instead of recording
        logger.info("Creating test audio file...")
        test_audio_path = create_test_audio_file()
        
        if not test_audio_path or not os.path.exists(test_audio_path):
            logger.error("Failed to create test audio file")
            # Try with microphone recording
            test_microphone_recording()
        else:
            logger.info(f"Test audio file created: {test_audio_path}")
            
            # Part 2: Test speech recognition
            test_speech_recognition(test_audio_path)
                
    except Exception as e:
        logger.error(f"Error testing speech components: {str(e)}")
        # Always test TTS even if other parts fail
        test_tts("This is a fallback test of the text to speech system.")

def test_microphone_recording():
    """Test microphone recording"""
    try:
        logger.info("Testing microphone recording...")
        mic = Microphone()
        
        # List available devices
        devices = mic.list_devices()
        logger.info(f"Available audio input devices: {devices}")
        
        # Record audio
        logger.info("Recording 5 seconds of audio...")
        audio_data, sample_rate = mic.record(duration=5.0)
        
        if len(audio_data) == 0:
            logger.error("Failed to record audio")
            # Skip to text-to-speech test if recording fails
            test_tts("This is a test of the text to speech system.")
            return
        
        # Save recording - use an absolute path in the temporary directory
        temp_dir = tempfile.gettempdir()
        temp_audio_path = os.path.join(temp_dir, "test_recording.wav")
        
        # Save the recording
        success = mic.save_recording(audio_data, temp_audio_path)
        if not success:
            logger.error(f"Failed to save recording to {temp_audio_path}")
            # Skip to text-to-speech test if saving fails
            test_tts("This is a test of the text to speech system.")
            return
            
        logger.info(f"Saved recording to {temp_audio_path}")
        
        # Verify the file exists
        if not os.path.isfile(temp_audio_path):
            logger.error(f"Audio file doesn't exist at {temp_audio_path}")
            # Skip to text-to-speech test if file doesn't exist
            test_tts("This is a test of the text to speech system.")
            return
        
        # Test speech recognition with the recording
        test_speech_recognition(temp_audio_path)
            
    except Exception as e:
        logger.error(f"Error testing microphone recording: {str(e)}")
        # Fallback to text-to-speech test
        test_tts("This is a fallback test of the text to speech system.")

def test_speech_recognition(audio_path):
    """Test speech recognition with the given audio file"""
    try:
        logger.info("Testing speech recognition...")
        recognizer = WhisperRecognizer(model_name="tiny")
        
        if not recognizer.load_model():
            logger.error("Failed to load speech recognition model")
            # Skip to text-to-speech test if model loading fails
            test_tts("This is a test of the text to speech system.")
            return
        
        # Print detailed file info
        file_info = f"File path: {audio_path}, exists: {os.path.exists(audio_path)}"
        if os.path.exists(audio_path):
            file_info += f", size: {os.path.getsize(audio_path)} bytes"
        logger.info(f"Audio file details: {file_info}")
        
        # Allow file system to complete writing
        time.sleep(1)
        
        # Transcribe audio file
        try:
            logger.info(f"Transcribing audio file: {audio_path}")
            transcription = recognizer.transcribe(audio_path)
            logger.info(f"Transcription result: {transcription}")
            
            # Use transcription for TTS
            test_tts(f"I heard: {transcription}")
        except Exception as e:
            logger.error(f"Error during transcription: {str(e)}")
            # Default text if transcription fails
            test_tts("This is a test of the text to speech system.")
            
    except Exception as e:
        logger.error(f"Error testing speech recognition: {str(e)}")
        # Fallback to text-to-speech test
        test_tts("This is a fallback test of the text to speech system.")

def test_tts(text):
    """Test text-to-speech with the given text"""
    try:
        # Test text-to-speech
        logger.info("Testing text-to-speech...")
        tts = PyttsxTTS()
        
        if not tts.initialize():
            logger.error("Failed to initialize text-to-speech engine")
            return
        
        # List available voices
        voices = tts.get_available_voices()
        logger.info(f"Available TTS voices: {voices}")
        
        # Speak directly
        logger.info(f"Speaking: {text}")
        tts.speak(text)
        
        # Synthesize to file
        audio_data = tts.synthesize(text)
        
        if len(audio_data) == 0:
            logger.error("Failed to synthesize speech")
            return
        
        # Save synthesized audio
        temp_dir = tempfile.gettempdir()
        temp_tts_path = os.path.join(temp_dir, "test_tts.wav")
        with open(temp_tts_path, 'wb') as f:
            f.write(audio_data)
        logger.info(f"Saved synthesized speech to {temp_tts_path}")
        
        logger.info("Speech component tests completed successfully")
        
    except Exception as e:
        logger.error(f"Error testing text-to-speech: {str(e)}")

if __name__ == "__main__":
    test_speech_components()
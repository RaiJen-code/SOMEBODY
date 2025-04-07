"""
SOMEBODY Brain - Core integration of all components
"""

import logging
import os
from typing import Optional, Dict, Any, Union
import tempfile
import numpy as np

from llm.model import OllamaLLM
from llm.memory import ConversationMemory
from llm.prompt import get_template
from speech.recognition import WhisperRecognizer
from speech.text_to_speech import PyttsxTTS
from speech.microphone import Microphone
from vision.camera import Camera
from vision.analyzer import VisionAnalyzer
from speech.wake_word import WakeWordDetector
from utils.internet import InternetSearch
from config.settings import USE_INTERNET, GOOGLE_API_KEY, GOOGLE_CSE_ID
from config.settings import (
    OLLAMA_MODEL, 
    SPEECH_RECOGNITION_MODEL_PATH,
    TTS_MODEL_PATH,
    VISION_MODEL_PATH,
    CAMERA_INDEX
)

logger = logging.getLogger(__name__)

class SomebodyBrain:
    """Main brain class that integrates all components"""
    
    def __init__(self):
        """Initialize SOMEBODY brain"""
        logger.info("Initializing SOMEBODY brain")
        
        # Initialize components
        self.llm = OllamaLLM(model_name=OLLAMA_MODEL)
        self.speech_recognizer = WhisperRecognizer(model_name="tiny")
        self.tts = PyttsxTTS()
        self.microphone = Microphone()
        self.camera = Camera(camera_index=CAMERA_INDEX)
        self.vision_analyzer = VisionAnalyzer(model_path=VISION_MODEL_PATH)
        
        # Initialize memory
        self.memory = ConversationMemory()
        
        # State tracking
        self.initialized = False
        # Add wake word detector
        self.wake_word_detector = WakeWordDetector(
            keywords=["SOMEBODY"],  # Gunakan "jarvis" sebagai proxy untuk "SOMEBODY"
            callback=self.on_wake_word_detected
        )
        # Initialize internet search if enabled
        self.internet = InternetSearch(
            google_api_key=GOOGLE_API_KEY,
            google_cse_id=GOOGLE_CSE_ID
        ) if USE_INTERNET else None
        
    def initialize(self) -> bool:
        """Initialize all components
        
        Returns:
            True if initialization successful, False otherwise
        """
        try:
            logger.info("Initializing SOMEBODY components")
            
            # Initialize LLM (most important component)
            llm_loaded = self.llm.load_model()
            if not llm_loaded:
                logger.error("Failed to load LLM model - SOMEBODY cannot function without it")
                return False
            
            # Try to initialize other components, but continue even if they fail
            try:
                self.speech_recognizer.load_model()
            except Exception as e:
                logger.warning(f"Failed to initialize speech recognizer: {str(e)}")
                
            try:
                self.tts.initialize()
            except Exception as e:
                logger.warning(f"Failed to initialize text-to-speech: {str(e)}")
                
            try:
                self.vision_analyzer.load_model()
            except Exception as e:
                logger.warning(f"Failed to initialize vision analyzer: {str(e)}")

            try:
                self.wake_word_detector.initialize()
            except Exception as e:
                logger.warning(f"Failed to initialize wake word detector: {str(e)}")

            self.initialized = True
            logger.info("SOMEBODY brain initialization complete")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing SOMEBODY brain: {str(e)}")
            return False
    
    def on_wake_word_detected(self, keyword):
        """Called when wake word is detected"""
        logger.info(f"Wake word '{keyword}' detected")
        
        # Start listening for a command
        return self.listen_and_respond(listen_duration=5.0)

    def start_wake_word_detection(self):
        """Start listening for wake word"""
        if not self.initialized:
            if not self.initialize():
                return False
        
        return self.wake_word_detector.start()

    def stop_wake_word_detection(self):
        """Stop listening for wake word"""
        return self.wake_word_detector.stop()

    def process_text(self, text: str) -> str:
        """Process text input and generate response
        
        Args:
            text: Input text from user
            
        Returns:
            Response text
        """
        if not self.initialized:
            if not self.initialize():
                return "I'm sorry, I'm having trouble initializing my systems. Please try again later."
        
        logger.info(f"Processing text input: {text[:50]}...")
        
        # Add user message to memory
        self.memory.add_message("user", text)
        
        # Generate prompt using template
        template = get_template("base_conversation")
        context = "You are running on an Orange Pi 5 and have access to speech recognition, text-to-speech, and computer vision capabilities."
        
        # Check if it's an internet search query
        if USE_INTERNET and (
            text.lower().startswith("search for") or 
            text.lower().startswith("find information about") or
            text.lower().startswith("look up") or
            "what is" in text.lower() or
            "who is" in text.lower() or
            "tell me about" in text.lower()
        ):
            # Extract search query
            search_query = text
            for prefix in ["search for", "find information about", "look up", "tell me about"]:
                if text.lower().startswith(prefix):
                    search_query = text[len(prefix):].strip()
                    break
            
            # Perform search
            search_results = self.internet_search(search_query)
            
            # Format response with search results
            if search_results.get("wikipedia") and search_results["wikipedia"].get("success"):
                wiki_data = search_results["wikipedia"]
                wiki_response = f"Here's what I found about '{search_query}' on Wikipedia:\n\n"
                wiki_response += f"**{wiki_data['title']}**\n\n{wiki_data['summary']}\n\n"
                
                # Add link
                if "url" in wiki_data:
                    wiki_response += f"Source: {wiki_data['url']}"
                    
                # Add to memory
                self.memory.add_message("user", text)
                self.memory.add_message("assistant", wiki_response)
                
                return wiki_response
                
            # Fall back to regular LLM response if no good search results

        # Include recent conversation history if available
        if len(self.memory.messages) > 1:
            history = self.memory.format_for_prompt(max_messages=5)
            context += f"\n\nRecent conversation:\n{history}"
        
        prompt = template.format(
            context=context,
            user_input=text
        )
        
        # Generate response from LLM
        response = self.llm.generate_response(prompt)
        
        # Add assistant response to memory
        self.memory.add_message("assistant", response)
        
        return response
    
    def listen(self, duration: float = 5.0) -> Dict[str, Any]:
        """Listen through microphone and transcribe speech
        
        Args:
            duration: Recording duration in seconds
            
        Returns:
            Dictionary with transcribed text and other info
        """
        if not self.initialized:
            if not self.initialize():
                return {"error": "Initialization failed"}
        
        logger.info(f"Listening for {duration} seconds...")
        
        try:
            # Record audio
            audio_data, sample_rate = self.microphone.record(duration=duration)
            
            if len(audio_data) == 0:
                error_msg = "Failed to record audio"
                logger.error(error_msg)
                return {"error": error_msg, "transcribed_text": ""}
            
            # Save recording to temporary file
            temp_file = os.path.join(tempfile.gettempdir(), "somebody_recording.wav")
            self.microphone.save_recording(audio_data, temp_file)
            
            # Transcribe audio
            transcribed_text = self.speech_recognizer.transcribe(temp_file)
            
            logger.info(f"Transcribed text: {transcribed_text}")
            
            return {
                "transcribed_text": transcribed_text,
                "audio_file": temp_file,
                "audio_data": audio_data,
                "sample_rate": sample_rate
            }
        except Exception as e:
            logger.error(f"Error listening: {str(e)}")
            return {"error": str(e), "transcribed_text": ""}

    def speak(self, text: str) -> bool:
        """Speak text using text-to-speech
        
        Args:
            text: Text to speak
            
        Returns:
            True if successful, False otherwise
        """
        if not self.initialized:
            if not self.initialize():
                return False
        
        logger.info(f"Speaking: {text[:50]}...")
        
        try:
            # Use TTS to speak text
            return self.tts.speak(text)
        except Exception as e:
            logger.error(f"Error speaking: {str(e)}")
            return False

    def synthesize_speech(self, text: str) -> bytes:
        """Synthesize speech from text
        
        Args:
            text: Text to synthesize
            
        Returns:
            Audio data as bytes
        """
        if not self.initialized:
            if not self.initialize():
                return b""
        
        logger.info(f"Synthesizing speech: {text[:50]}...")
        
        try:
            # Use TTS to synthesize speech
            return self.tts.synthesize(text)
        except Exception as e:
            logger.error(f"Error synthesizing speech: {str(e)}")
            return b""
    
    def process_audio(self, audio_data: Union[bytes, np.ndarray, str]) -> Dict[str, Any]:
        """Process audio input and generate response
        
        Args:
            audio_data: Audio data as bytes, numpy array, or file path
            
        Returns:
            Dictionary with text and audio response
        """
        if not self.initialized:
            if not self.initialize():
                return {"error": "Initialization failed"}
        
        logger.info("Processing audio input")
        
        try:
            # Transcribe audio to text
            if isinstance(audio_data, (bytes, np.ndarray)):
                # For bytes or numpy array, save to temp file first
                temp_file = os.path.join(tempfile.gettempdir(), "somebody_input.wav")
                if isinstance(audio_data, bytes):
                    with open(temp_file, 'wb') as f:
                        f.write(audio_data)
                else:
                    import soundfile as sf
                    sf.write(temp_file, audio_data, 16000)  # Assuming 16kHz sample rate
                
                transcribed_text = self.speech_recognizer.transcribe(temp_file)
            else:
                # Assume it's a file path
                transcribed_text = self.speech_recognizer.transcribe(audio_data)
            
            if not transcribed_text or transcribed_text.startswith("Error:"):
                error_msg = f"Failed to transcribe audio: {transcribed_text}"
                logger.error(error_msg)
                return {
                    "error": error_msg,
                    "transcribed_text": "",
                    "response_text": "I couldn't understand what you said.",
                    "response_audio": self.synthesize_speech("I couldn't understand what you said.")
                }
            
            # Process the transcribed text
            response_text = self.process_text(transcribed_text)
            
            # Synthesize text to speech
            response_audio = self.synthesize_speech(response_text)
            
            return {
                "transcribed_text": transcribed_text,
                "response_text": response_text,
                "response_audio": response_audio
            }
        except Exception as e:
            logger.error(f"Error processing audio: {str(e)}")
            error_msg = "I'm sorry, there was an error processing your audio."
            return {
                "error": str(e),
                "transcribed_text": "",
                "response_text": error_msg,
                "response_audio": self.synthesize_speech(error_msg)
            }
    
        # Untuk brain.py
    def train_yolo_model(self, dataset_path: str, epochs: int = 50) -> Dict[str, Any]:
        """Train custom YOLO model for electronic components
        
        Args:
            dataset_path: Path to dataset in YOLO format
            epochs: Number of training epochs
            
        Returns:
            Dictionary with training results
        """
        if not self.initialized:
            if not self.initialize():
                return {"error": "Initialization failed"}
        
        try:
            # Train custom model
            result = self.vision_analyzer.train_custom_model(dataset_path, epochs)
            
            # Add success/failure message to memory
            if result.get("success", False):
                self.memory.add_message("user", "[Permintaan pelatihan model YOLO kustom]")
                self.memory.add_message(
                    "assistant", 
                    f"Model YOLO kustom untuk komponen elektronik berhasil dilatih dan disimpan di {result.get('model_path')}"
                )
            
            return result
        except Exception as e:
            logger.error(f"Error training YOLO model: {str(e)}")
            return {"error": str(e)}

    def detect_objects(self, image_data: bytes, confidence: float = 0.25) -> Dict[str, Any]:
        """Detect objects in image using YOLO
        
        Args:
            image_data: Image data as bytes
            confidence: Confidence threshold (0-1)
            
        Returns:
            Dictionary with detection results
        """
        if not self.initialized:
            if not self.initialize():
                return {"error": "Initialization failed"}
        
        try:
            # Detect objects with YOLO
            result = self.vision_analyzer.detect_objects(image_data)
            
            # If there's a query and we have detections, process with LLM
            if "error" not in result:
                # Add information to memory
                detection_summary = result.get("description", "Analisis gambar selesai.")
                self.memory.add_message("user", "[Permintaan deteksi objek dengan YOLO]")
                self.memory.add_message("assistant", detection_summary)
                
                logger.info(f"Object detection result: {len(result.get('detections', []))} objek terdeteksi")
            
            return result
        except Exception as e:
            logger.error(f"Error detecting objects: {str(e)}")
            return {"error": str(e)}

    def listen_and_respond(self, listen_duration: float = 5.0) -> Dict[str, Any]:
        """Listen for speech, process it, and respond verbally
        
        Args:
            listen_duration: Duration to listen in seconds
            
        Returns:
            Dictionary with results of interaction
        """
        if not self.initialized:
            if not self.initialize():
                return {"error": "Initialization failed"}
        
        logger.info(f"Starting listen and respond cycle (listening for {listen_duration}s)...")
        
        try:
            # Listen for speech
            listen_result = self.listen(duration=listen_duration)
            
            transcribed_text = listen_result.get("transcribed_text", "")
            
            if not transcribed_text or "error" in listen_result:
                error_msg = "I couldn't understand what you said."
                logger.warning(error_msg)
                self.speak(error_msg)
                return {
                    "error": listen_result.get("error", "Transcription failed"),
                    "transcribed_text": transcribed_text,
                    "response_text": error_msg
                }
            
            # Process the transcribed text
            response_text = self.process_text(transcribed_text)
            
            # Speak the response
            self.speak(response_text)
            
            return {
                "transcribed_text": transcribed_text,
                "response_text": response_text
            }
        except Exception as e:
            logger.error(f"Error in listen and respond cycle: {str(e)}")
            error_msg = "I'm sorry, there was an error processing your request."
            self.speak(error_msg)
            return {
                "error": str(e),
                "response_text": error_msg
            }
    
    def process_image(self, image_data: bytes, query: Optional[str] = None) -> Dict[str, Any]:
        """Process image and generate analysis
        
        Args:
            image_data: Image data as bytes
            query: Optional text query about the image
            
        Returns:
            Dictionary with analysis results
        """
        if not self.initialized:
            if not self.initialize():
                return {"error": "Initialization failed"}
        
        logger.info("Processing image")
        
        try:
            # Analyze the image
            analysis = self.vision_analyzer.analyze_image(image_data)
            
            if "error" in analysis:
                logger.error(f"Error analyzing image: {analysis['error']}")
                return analysis
            
            # Save the image to a temporary file (for debugging)
            temp_image_path = os.path.join(tempfile.gettempdir(), "somebody_image.jpg")
            with open(temp_image_path, 'wb') as f:
                f.write(image_data)
            
            # If there's a query, process it with the LLM
            if query:
                # Create a detailed scene description for the LLM
                scene_description = analysis.get("description", "No description available.")
                
                # Add more details about detected faces if any
                if analysis.get("face_count", 0) > 0:
                    scene_description += f"\nDetected {analysis['face_count']} faces in the image."
                
                # Add image dimensions
                if "dimensions" in analysis:
                    dims = analysis["dimensions"]
                    scene_description += f"\nImage dimensions: {dims.get('width', 0)}x{dims.get('height', 0)} pixels."
                
                # Create prompt for LLM
                template = get_template("image_analysis")
                prompt = template.format(
                    image_description=scene_description,
                    user_input=query
                )
                
                # Generate response from LLM
                response = self.llm.generate_response(prompt)
                analysis["response"] = response
                
                # Add to memory
                self.memory.add_message("user", f"[Image analysis request]: {query}")
                self.memory.add_message("assistant", response)
                
                logger.info(f"Generated response for image query: {response[:100]}...")
            
            return analysis
        except Exception as e:
            logger.error(f"Error processing image: {str(e)}")
            return {"error": str(e)}
    
    def capture_and_analyze_image(self, query: Optional[str] = None) -> Dict[str, Any]:
        """Capture image from camera and analyze it
        
        Args:
            query: Optional text query about the image
            
        Returns:
            Dictionary with analysis results
        """
        if not self.initialized:
            if not self.initialize():
                return {"error": "Initialization failed"}
        
        logger.info("Capturing and analyzing image")
        
        try:
            # Capture image
            image_data = self.camera.capture_image()
            if not image_data:
                error_msg = "Failed to capture image"
                logger.error(error_msg)
                return {"error": error_msg}
            
            # Analyze the image
            return self.process_image(image_data, query)
        except Exception as e:
            logger.error(f"Error capturing and analyzing image: {str(e)}")
            return {"error": str(e)}
    
    def save_state(self, directory: str) -> bool:
        """Save brain state
        
        Args:
            directory: Directory to save state
            
        Returns:
            True if successful, False otherwise
        """
        try:
            os.makedirs(directory, exist_ok=True)
            
            # Save conversation memory
            memory_path = os.path.join(directory, "conversation.json")
            self.memory.save(memory_path)
            
            logger.info(f"Saved SOMEBODY brain state to {directory}")
            return True
        except Exception as e:
            logger.error(f"Error saving brain state: {str(e)}")
            return False
    
    def load_state(self, directory: str) -> bool:
        """Load brain state
        
        Args:
            directory: Directory to load state from
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if not os.path.exists(directory):
                logger.warning(f"State directory {directory} not found")
                return False
            
            # Load conversation memory
            memory_path = os.path.join(directory, "conversation.json")
            if os.path.exists(memory_path):
                loaded_memory = ConversationMemory.load(memory_path)
                if loaded_memory:
                    self.memory = loaded_memory
            
            logger.info(f"Loaded SOMEBODY brain state from {directory}")
            return True
        except Exception as e:
            logger.error(f"Error loading brain state: {str(e)}")
            return False
        
    def internet_search(self, query: str) -> Dict[str, Any]:
        """Perform internet search
        
        Args:
            query: Search query
            
        Returns:
            Search results
        """
        if not self.initialized:
            if not self.initialize():
                return {"error": "Initialization failed"}
        
        if not self.internet:
            return {"error": "Internet search is disabled"}
        
        logger.info(f"Performing internet search for: {query}")
        return self.internet.search(query)

    def get_weather(self, location: str) -> Dict[str, Any]:
        """Get weather information
        
        Args:
            location: Location name
            
        Returns:
            Weather information
        """
        if not self.initialized:
            if not self.initialize():
                return {"error": "Initialization failed"}
        
        if not self.internet:
            return {"error": "Internet search is disabled"}
        
        logger.info(f"Getting weather for: {location}")
        return self.internet.get_current_weather(location)
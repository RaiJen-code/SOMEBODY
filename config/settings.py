"""
Configuration settings for SOMEBODY
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Base directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Application settings
APP_NAME = "SOMEBODY"
APP_VERSION = "0.1.0"
DEBUG = os.getenv("DEBUG", "False").lower() in ("true", "1", "t")

# Paths
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(DATA_DIR, "models")
LOGS_DIR = os.path.join(BASE_DIR, "logs")

# Make sure directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

# API settings
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))

# # LLM settings
# LLM_MODEL = os.getenv("LLM_MODEL", "llama")
# LLM_MODEL_PATH = os.getenv("LLM_MODEL_PATH", os.path.join(MODELS_DIR, "llm"))
# Ollama settings
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api")

# Speech settings
SPEECH_RECOGNITION_MODEL = os.getenv("SPEECH_RECOGNITION_MODEL", "whisper-tiny")
SPEECH_RECOGNITION_MODEL_PATH = os.getenv(
    "SPEECH_RECOGNITION_MODEL_PATH", 
    os.path.join(MODELS_DIR, "speech", "recognition")
)
TTS_MODEL = os.getenv("TTS_MODEL", "piper")
TTS_MODEL_PATH = os.getenv(
    "TTS_MODEL_PATH", 
    os.path.join(MODELS_DIR, "speech", "tts")
)

# Vision settings
VISION_MODEL = os.getenv("VISION_MODEL", "yolo")
VISION_MODEL_PATH = os.getenv(
    "VISION_MODEL_PATH", 
    os.path.join(MODELS_DIR, "vision")
)
CAMERA_INDEX = int(os.getenv("CAMERA_INDEX", "0"))

# # Internet connection
# USE_INTERNET = os.getenv("USE_INTERNET", "False").lower() in ("true", "1", "t")

# Internet search settings
USE_INTERNET = os.getenv("USE_INTERNET", "False").lower() in ("true", "1", "t")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID", "")
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY", "")
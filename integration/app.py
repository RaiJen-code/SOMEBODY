"""
Core integration server for SOMEBODY components
"""

import logging
import os
import base64
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, Form
from dotenv import load_dotenv
from utils.internet import InternetSearch
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, Dict, Any

from integration.controller import SomebodyController

logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="SOMEBODY API", version="0.1.0")

# Initialize controller
controller = SomebodyController()

class TextRequest(BaseModel):
    """Request model for text input"""
    text: str

class TextResponse(BaseModel):
    """Response model for text output"""
    text: str
    
class ImageAnalysisRequest(BaseModel):
    """Request model for image analysis"""
    image_data: str  # Base64 encoded image
    query: Optional[str] = None
    
@app.on_event("startup")
async def startup_event():
    """Initialize components on startup"""
    load_dotenv()  # Load environment variables
    logger.info("Starting up SOMEBODY API")
    controller.initialize()
    controller.load_state()
    
@app.on_event("shutdown")
async def shutdown_event():
    """Clean shutdown"""
    logger.info("Shutting down SOMEBODY API")
    controller.shutdown()
    
@app.get("/")
async def root():
    """Root endpoint to check if service is running"""
    return {"status": "online", "service": "SOMEBODY Integration API"}

@app.post("/process/text", response_model=TextResponse)
async def process_text(request: TextRequest):
    """Process text input and return a response"""
    logger.info(f"Received text request: {request.text[:50]}...")
    response = controller.process_text(request.text)
    return TextResponse(text=response)

@app.post("/process/image")
async def process_image(
    image: UploadFile = File(...),
    query: Optional[str] = Form(None)
):
    """Process uploaded image and return analysis"""
    logger.info(f"Received image analysis request with query: {query}")
    
    # Read image data
    image_data = await image.read()
    
    # Process image
    result = controller.process_image(image_data, query)
    
    return result

@app.post("/capture/image")
async def capture_image(query: Optional[str] = Form(None)):
    """Capture image from camera and return analysis"""
    logger.info(f"Received camera capture request with query: {query}")
    
    # Capture and analyze image
    result = controller.capture_and_analyze_image(query)
    
    return result

@app.post("/test/llm")
async def test_llm(request: TextRequest):
    """Test LLM with a direct prompt"""
    logger.info(f"Testing LLM with prompt: {request.text[:50]}...")
    
    # Get controller and use LLM directly
    response = controller.brain.llm.generate_response(request.text)
    
    return {
        "prompt": request.text,
        "response": response
    }

# Endpoint baru untuk interaksi suara
@app.post("/speech/listen")
async def listen(duration: float = Form(5.0)):
    """Listen for speech and return transcription"""
    logger.info(f"Received listen request with duration: {duration}s")
    
    # Listen for speech
    result = controller.listen(duration)
    
    # If there's audio data, convert to base64
    if "audio_data" in result and isinstance(result["audio_data"], np.ndarray):
        # Remove audio_data from result (too large to send in JSON)
        del result["audio_data"]
    
    return result

@app.post("/speech/speak")
async def speak(request: TextRequest):
    """Speak text using text-to-speech"""
    logger.info(f"Received speak request: {request.text[:50]}...")
    
    # Speak text
    success = controller.speak(request.text)
    
    return {"success": success}

@app.post("/speech/listen_and_respond")
async def listen_and_respond(duration: float = Form(5.0)):
    """Listen for speech, process it, and respond verbally"""
    logger.info(f"Received listen and respond request with duration: {duration}s")
    
    # Listen and respond
    result = controller.listen_and_respond(duration)
    
    return result

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time communication"""
    await websocket.accept()
    logger.info("WebSocket connection established")
    
    try:
        while True:
            # Receive message from client
            data = await websocket.receive_json()
            
            # Process based on message type
            message_type = data.get("type", "text")
            
            if message_type == "text":
                text = data.get("text", "")
                logger.info(f"Received text via WebSocket: {text[:50]}...")
                
                # Process text
                response = controller.process_text(text)
                
                # Send response
                await websocket.send_json({
                    "type": "text",
                    "text": response
                })
                
            elif message_type == "audio":
                # Decode base64 audio data
                audio_data = base64.b64decode(data.get("audio", ""))
                logger.info("Received audio via WebSocket")
                
                # Process audio
                result = controller.process_audio(audio_data)
                
                # Encode response audio to base64
                if "response_audio" in result and result["response_audio"]:
                    result["response_audio"] = base64.b64encode(result["response_audio"]).decode("utf-8")
                
                # Send response
                await websocket.send_json({
                    "type": "audio_response",
                    **result
                })
                
            elif message_type == "image":
                # Decode base64 image data
                image_data = base64.b64decode(data.get("image", ""))
                query = data.get("query")
                logger.info(f"Received image via WebSocket with query: {query}")
                
                # Process image
                result = controller.process_image(image_data, query)
                
                # Send response
                await websocket.send_json({
                    "type": "image_analysis",
                    **result
                })
                
            elif message_type == "capture_image":
                query = data.get("query")
                logger.info(f"Received capture image request via WebSocket with query: {query}")
                
                # Capture and analyze image
                result = controller.capture_and_analyze_image(query)
                
                # Send response
                await websocket.send_json({
                    "type": "image_analysis",
                    **result
                })
                
            elif message_type == "listen":
                duration = float(data.get("duration", 5.0))
                logger.info(f"Received listen request via WebSocket with duration: {duration}s")
                
                # Listen for speech
                result = controller.listen(duration)
                
                # Remove audio_data if present (too large for WebSocket)
                if "audio_data" in result:
                    del result["audio_data"]
                
                # Send response
                await websocket.send_json({
                    "type": "listen_response",
                    **result
                })
                
            elif message_type == "speak":
                text = data.get("text", "")
                logger.info(f"Received speak request via WebSocket: {text[:50]}...")
                
                # Speak text
                success = controller.speak(text)
                
                # Send response
                await websocket.send_json({
                    "type": "speak_response",
                    "success": success
                })
                
            elif message_type == "listen_and_respond":
                duration = float(data.get("duration", 5.0))
                logger.info(f"Received listen and respond request via WebSocket with duration: {duration}s")
                
                # Listen and respond
                result = controller.listen_and_respond(duration)
                
                # Send response
                await websocket.send_json({
                    "type": "listen_and_respond_response",
                    **result
                })
                
            else:
                logger.warning(f"Unknown message type: {message_type}")
                await websocket.send_json({
                    "type": "error",
                    "message": f"Unknown message type: {message_type}"
                })
            
    except WebSocketDisconnect:
        logger.info("WebSocket connection closed")
    except Exception as e:
        logger.error(f"Error in WebSocket connection: {str(e)}")

@app.get("/health")
async def health_check():
    """Check health status of all components"""
    try:
        # Check components status
        llm_status = controller.brain.llm.is_loaded() 
        speech_status = controller.brain.speech_recognizer.is_loaded()
        tts_status = hasattr(controller.brain.tts, 'initialized') and controller.brain.tts.initialized
        vision_status = controller.brain.vision_analyzer.is_loaded()
        
        return {
            "status": "online",
            "llm": llm_status,
            "speech": speech_status,
            "tts": tts_status,
            "vision": vision_status,
            "internet": False  # Not implemented yet
        }
    except Exception as e:
        logger.error(f"Health check error: {str(e)}")
        return {
            "status": "error",
            "message": str(e)
        }

@app.post("/brain/process")
async def process_brain(input_data: dict):
    """Process text input through the brain"""
    text = input_data.get("input", "")
    if not text:
        return {"error": "No input provided"}
    
    response = controller.process_text(text)
    return {"response": response}

@app.post("/system/reset")
async def reset_system():
    """Reset all systems"""
    try:
        # Re-initialize controller
        controller.initialize()
        return {"status": "success", "message": "Systems reset successfully"}
    except Exception as e:
        logger.error(f"Error resetting systems: {str(e)}")
        return {"status": "error", "message": str(e)}

@app.post("/system/power")
async def power_system():
    """Simulate power down of the system"""
    try:
        # Just a simulation of powering down
        controller.shutdown()
        return {"status": "success", "message": "System powering down"}
    except Exception as e:
        logger.error(f"Error powering system: {str(e)}")
        return {"status": "error", "message": str(e)}
@app.post("/brain/process")
async def process_brain(input_data: dict):
    """Process text input through the brain"""
    text = input_data.get("input", "")
    if not text:
        return {"error": "No input provided"}
    
    try:
        response = controller.process_text(text)
        return {"response": response}
    except Exception as e:
        logger.error(f"Error processing text: {str(e)}")
        return {"error": str(e)}

@app.post("/system/reset")
async def reset_system():
    """Reset all systems"""
    try:
        # Re-initialize controller
        controller.initialize()
        return {"status": "success", "message": "Systems reset successfully"}
    except Exception as e:
        logger.error(f"Error resetting systems: {str(e)}")
        return {"status": "error", "message": str(e)}

@app.post("/system/power")
async def power_system():
    """Simulate power down of the system"""
    try:
        # Just a simulation of powering down
        controller.shutdown()
        return {"status": "success", "message": "System powering down"}
    except Exception as e:
        logger.error(f"Error powering system: {str(e)}")
        return {"status": "error", "message": str(e)}

@app.post("/wake_word/start")
async def start_wake_word():
    """Start wake word detection"""
    try:
        success = controller.brain.start_wake_word_detection()
        if success:
            return {"status": "success", "message": "Wake word detection started"}
        else:
            return {"status": "error", "message": "Failed to start wake word detection"}
    except Exception as e:
        logger.error(f"Error starting wake word detection: {str(e)}")
        return {"status": "error", "message": str(e)}

@app.post("/wake_word/stop")
async def stop_wake_word():
    """Stop wake word detection"""
    try:
        controller.brain.stop_wake_word_detection()
        return {"status": "success", "message": "Wake word detection stopped"}
    except Exception as e:
        logger.error(f"Error stopping wake word detection: {str(e)}")
        return {"status": "error", "message": str(e)}

@app.post("/search")
async def internet_search(query: str = Form(...)):
    """Perform internet search"""
    try:
        results = controller.brain.internet_search(query)
        return results
    except Exception as e:
        logger.error(f"Error performing internet search: {str(e)}")
        return {"error": str(e)}

@app.post("/weather")
async def get_weather(location: str = Form(...)):
    """Get weather information"""
    try:
        results = controller.brain.get_weather(location)
        return results
    except Exception as e:
        logger.error(f"Error getting weather: {str(e)}")
        return {"error": str(e)}

@app.post("/system/internet")
async def toggle_internet(data: dict):
    """Toggle internet connection on/off"""
    try:
        enabled = data.get("enabled", False)
        
        # Update settings
        global USE_INTERNET
        USE_INTERNET = enabled
        
        # Reinitialize internet component if needed
        if enabled and not controller.brain.internet:
            controller.brain.internet = InternetSearch(
                google_api_key=os.environ.get("GOOGLE_API_KEY"),
                google_cse_id=os.environ.get("GOOGLE_CSE_ID")
            )
        elif not enabled:
            controller.brain.internet = None
            
        return {"status": "success", "internet_enabled": enabled}
    except Exception as e:
        logger.error(f"Error toggling internet: {str(e)}")
        return {"status": "error", "message": str(e)}
    
@app.post("/detect/objects")
async def detect_objects(
    image: UploadFile = File(...),
    confidence: float = Form(0.25)
):
    """Detect objects in uploaded image using YOLO"""
    logger.info(f"Received object detection request with confidence threshold: {confidence}")
    
    # Read image data
    image_data = await image.read()
    
    # Detect objects
    result = controller.detect_objects(image_data, confidence)
    
    return result

@app.post("/train/yolo")
async def train_yolo_model(
    dataset_path: str = Form(...),
    epochs: int = Form(50)
):
    """Train custom YOLO model for electronic components
    
    Args:
        dataset_path: Path to dataset in YOLO format
        epochs: Number of training epochs
    """
    logger.info(f"Received YOLO training request: dataset={dataset_path}, epochs={epochs}")
    
    # Train model
    result = controller.train_yolo_model(dataset_path, epochs)
    
    return result
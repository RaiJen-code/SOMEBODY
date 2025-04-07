import sys
import os
import time
import threading
import requests
from datetime import datetime
import tempfile
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QPushButton, QTextEdit, QLineEdit, 
                            QLabel, QFrame, QSplitter, QScrollArea)
from PyQt5.QtGui import QFont, QIcon, QPixmap, QColor, QPainter, QPen, QBrush
from PyQt5.QtCore import Qt, QSize, QTimer, QThread, pyqtSignal, QUrl
from PyQt5.QtMultimedia import QCamera, QCameraInfo, QMediaRecorder, QAudioRecorder, QCameraImageCapture
from PyQt5.QtMultimediaWidgets import QCameraViewfinder

# Impor konfigurasi dari modul config yang sudah ada
from config.settings import API_HOST, API_PORT

import logging
logger = logging.getLogger(__name__)

# Backend API URL (menggunakan konfigurasi yang sudah ada)
#API_URL = f"http://{API_HOST}:{API_PORT}"
API_URL = f"http://localhost:{API_PORT}"

class AudioRecorderThread(QThread):
    """Thread for recording audio and sending to speech recognition"""
    finished = pyqtSignal(str)
    
    def __init__(self, parent=None, duration=5.0):
        super().__init__(parent)
        self.is_recording = False
        self.duration = duration
        
    def run(self):
        self.is_recording = True
        try:
            # Call API to listen
            response = requests.post(
                f"{API_URL}/speech/listen_and_respond",
                data={"duration": self.duration}
            )
            
            if response.status_code == 200:
                result = response.json()
                text = result.get("transcribed_text", "")
                self.finished.emit(text)
            else:
                self.finished.emit("Error processing speech")
        except Exception as e:
            self.finished.emit(f"Error: {str(e)}")
        finally:
            self.is_recording = False

class ChatMessage(QFrame):
    """Custom widget for displaying chat messages"""
    def __init__(self, text, is_user=False, parent=None):
        super().__init__(parent)
        self.setFrameShape(QFrame.StyledPanel)
        self.setStyleSheet(
            "background-color: #313244;" if is_user else "background-color: #45475A;"
        )
        
        layout = QVBoxLayout(self)
        
        # Time label
        time_label = QLabel(datetime.now().strftime("%H:%M:%S"))
        time_label.setStyleSheet("color: #7F849C; font-size: 10px;")
        time_label.setAlignment(Qt.AlignRight if is_user else Qt.AlignLeft)
        
        # Message text
        message = QLabel(text)
        message.setStyleSheet("color: #CDD6F4; font-size: 12px;")
        message.setWordWrap(True)
        
        layout.addWidget(time_label)
        layout.addWidget(message)
        
        # Adjust alignment based on sender
        self.setMaximumWidth(400)
        if is_user:
            self.setContentsMargins(50, 5, 10, 5)
        else:
            self.setContentsMargins(10, 5, 50, 5)

class CircularButton(QPushButton):
    """Custom circular button for voice activation"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(40, 40)
        self.active = False
        self.setToolTip("Klik untuk mengaktifkan input suara")
    
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Draw outer circle
        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(QColor("#F5C2E7")))
        painter.drawEllipse(2, 2, 36, 36)
        
        # Draw inner circle, color depends on active state
        if self.active:
            painter.setBrush(QBrush(QColor("#F38BA8")))
        else:
            painter.setBrush(QBrush(QColor("#89B4FA")))
        painter.drawEllipse(10, 10, 20, 20)
            
        # Draw microphone icon
        painter.setPen(QPen(QColor("#1E1E2E"), 2))
        # Simple microphone shape - vertical line
        painter.drawLine(20, 15, 20, 25)
        # Microphone head
        painter.drawArc(15, 15, 10, 5, 0, 180 * 16)
    
    def toggle_active(self):
        self.active = not self.active
        self.update()

class StatusIndicator(QWidget):
    """Status indicator for system components"""
    def __init__(self, label_text, status=True, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(5, 2, 5, 2)
        
        self.status_indicator = QLabel()
        self.status_indicator.setFixedSize(12, 12)
        self.status_indicator.setObjectName("status_indicator")
        self.set_status(status)
        
        self.label = QLabel(label_text)
        self.label.setStyleSheet("color: #CDD6F4; font-size: 12px;")
        
        layout.addWidget(self.status_indicator)
        layout.addWidget(self.label)
        layout.addStretch()
    
    def set_status(self, status):
        if status:
            self.status_indicator.setStyleSheet(
                "background-color: #A6E3A1; border-radius: 6px;"
            )
        else:
            self.status_indicator.setStyleSheet(
                "background-color: #F38BA8; border-radius: 6px;"
            )

class SomebodyMainWindow(QMainWindow):
    """Main window for SOMEBODY application"""
    def __init__(self):
        super().__init__()
        self.initUI()
        
        # Initialize camera if available
        self.initialize_camera()
        
        # Initialize threads
        self.audio_thread = AudioRecorderThread()
        self.audio_thread.finished.connect(self.on_speech_recognized)
        
        # Stored last captured image path
        self.last_captured_image_path = None
        
        # Check backend status
        self.check_backend_status()
    
    def initUI(self):
        self.setWindowTitle("SOMEBODY - Rangga Prasetya")
        self.setMinimumSize(800, 600)
        self.setStyleSheet("background-color: #1E1E2E;")
        
        # Main layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)
        
        # Status indicator untuk digunakan nanti
        self.mic_status = QLabel("Voice Input")
        self.mic_status.setStyleSheet("color: #94E2D5; font-size: 12px;")
        
        # Header area
        header_frame = QFrame()
        header_frame.setObjectName("header_frame")
        header_frame.setStyleSheet("background-color: #313244; border-radius: 10px;")
        header_layout = QHBoxLayout(header_frame)
        
        logo_label = QLabel("SOMEBODY")
        logo_label.setStyleSheet("color: #CDD6F4; font-size: 24px; font-weight: bold;")
        
        subtitle_label = QLabel("Aku temen kamu sampe kapanpun!")
        subtitle_label.setStyleSheet("color: #94E2D5; font-size: 12px;")
        
        self.system_status = QLabel("System Active")
        self.system_status.setStyleSheet("color: #CDD6F4; font-size: 14px;")
        
        self.voice_button = CircularButton()
        self.voice_button.clicked.connect(self.toggle_voice_recording)
        
        header_layout.addWidget(logo_label)
        header_layout.addWidget(subtitle_label)
        header_layout.addStretch()
        header_layout.addWidget(self.system_status)
        header_layout.addWidget(self.mic_status)
        header_layout.addWidget(self.voice_button)
        
        # Content area
        content_layout = QHBoxLayout()
        
        # Left side - conversation
        conversation_frame = QFrame()
        conversation_frame.setObjectName("conversation_frame")
        conversation_frame.setStyleSheet("background-color: #313244; border-radius: 10px;")
        conversation_layout = QVBoxLayout(conversation_frame)
        
        # Conversation history in a scroll area
        self.chat_area = QWidget()
        self.chat_layout = QVBoxLayout(self.chat_area)
        self.chat_layout.addStretch()
        
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setWidget(self.chat_area)
        self.scroll_area.setStyleSheet("background-color: #1E1E2E; border-radius: 5px;")
        
        # Input area
        input_layout = QHBoxLayout()
        
        self.message_input = QLineEdit()
        self.message_input.setPlaceholderText("Type a message...")
        self.message_input.setStyleSheet(
            "background-color: #1E1E2E; color: #CDD6F4; padding: 8px; border-radius: 5px;"
        )
        self.message_input.returnPressed.connect(self.send_message)
        
        # Tombol mikrofon
        self.mic_button = QPushButton()
        self.mic_button.setFixedSize(30, 30)
        self.mic_button.setStyleSheet(
            "background-color: #F5C2E7; border-radius: 15px;"
        )
        self.mic_button.setToolTip("Klik untuk berbicara")
        self.mic_button.clicked.connect(self.toggle_voice_recording)
        
        self.send_button = QPushButton()
        self.send_button.setFixedSize(30, 30)
        self.send_button.setStyleSheet(
            "background-color: #89B4FA; border-radius: 15px;"
        )
        self.send_button.clicked.connect(self.send_message)
        
        input_layout.addWidget(self.message_input)
        input_layout.addWidget(self.mic_button)
        input_layout.addWidget(self.send_button)
        
        conversation_layout.addWidget(self.scroll_area)
        conversation_layout.addLayout(input_layout)
        
        # Right side - functionality modules
        right_layout = QVBoxLayout()
        
        # Camera module
        camera_frame = QFrame()
        camera_frame.setObjectName("camera_frame")
        camera_frame.setStyleSheet("background-color: #313244; border-radius: 10px;")
        camera_layout = QVBoxLayout(camera_frame)
        
        self.viewfinder = QCameraViewfinder()
        self.viewfinder.setStyleSheet("background-color: #1E1E2E; border-radius: 5px;")
        
        camera_controls = QHBoxLayout()
        
        self.capture_button = QPushButton("Capture")
        self.capture_button.setObjectName("capture_button")
        self.capture_button.setStyleSheet(
            "background-color: #89B4FA; color: #1E1E2E; font-weight: bold; border-radius: 5px;"
        )
        self.capture_button.clicked.connect(self.capture_image)
        
        self.analyze_button = QPushButton("Analyze")
        self.analyze_button.setObjectName("analyze_button")
        self.analyze_button.setStyleSheet(
            "background-color: #F5C2E7; color: #1E1E2E; font-weight: bold; border-radius: 5px;"
        )
        self.analyze_button.clicked.connect(self.analyze_image)
        
        camera_controls.addWidget(self.capture_button)
        camera_controls.addWidget(self.analyze_button)
        # --- Tambahkan setelah camera_controls.addWidget(self.analyze_button) ---
        self.detect_objects_button = QPushButton("Detect Objects")
        self.detect_objects_button.setObjectName("detect_objects_button")
        self.detect_objects_button.setStyleSheet(
            "background-color: #B4BEFE; color: #1E1E2E; font-weight: bold; border-radius: 5px;"
        )
        self.detect_objects_button.clicked.connect(self.detect_objects)
        camera_controls.addWidget(self.detect_objects_button)
        
        camera_layout.addWidget(self.viewfinder)
        camera_layout.addLayout(camera_controls)
        
        # Status module
        status_frame = QFrame()
        status_frame.setObjectName("status_frame")
        status_frame.setStyleSheet("background-color: #313244; border-radius: 10px;")
        status_layout = QVBoxLayout(status_frame)
        
        status_title = QLabel("System Status")
        status_title.setStyleSheet("color: #CDD6F4; font-size: 14px; font-weight: bold;")
        
        # Status indicators
        self.llm_status = StatusIndicator("LLM Engine: Active")
        self.speech_status = StatusIndicator("Speech Recognition: Ready")
        self.tts_status = StatusIndicator("Text-to-Speech: Ready")
        self.vision_status = StatusIndicator("Vision Module: Ready")
        self.internet_status = StatusIndicator("Internet: Not Connected", False)
        
        # Toggle buttons
        self.internet_toggle = QPushButton("Enable Internet")
        self.internet_toggle.setObjectName("internet_button")
        self.internet_toggle.setCheckable(True)
        self.internet_toggle.setChecked(False)
        self.internet_toggle.setStyleSheet(
            "background-color: #F5C2E7; color: #1E1E2E; font-weight: bold; border-radius: 5px;"
        )
        self.internet_toggle.clicked.connect(self.toggle_internet)
        
        status_buttons = QHBoxLayout()
        
        wake_word_toggle = QPushButton("Enable Wake Word")
        wake_word_toggle.setObjectName("wake_word_button")
        wake_word_toggle.setCheckable(True)
        wake_word_toggle.setStyleSheet(
            "background-color: #89B4FA; color: #1E1E2E; font-weight: bold; border-radius: 5px;"
        )
        wake_word_toggle.clicked.connect(self.toggle_wake_word)
        
        reset_button = QPushButton("Reset")
        reset_button.setObjectName("reset_button")
        reset_button.setStyleSheet(
            "background-color: #89B4FA; color: #1E1E2E; font-weight: bold; border-radius: 5px;"
        )
        reset_button.clicked.connect(self.reset_system)
        
        power_button = QPushButton("Power")
        power_button.setObjectName("power_button")
        power_button.setStyleSheet(
            "background-color: #F38BA8; color: #1E1E2E; font-weight: bold; border-radius: 5px;"
        )
        power_button.clicked.connect(self.power_toggle)
        
        # Add components to status_layout
        status_layout.addWidget(status_title)
        status_layout.addWidget(self.llm_status)
        status_layout.addWidget(self.speech_status)
        status_layout.addWidget(self.tts_status)
        status_layout.addWidget(self.vision_status)
        status_layout.addWidget(self.internet_status)
        status_layout.addWidget(self.internet_toggle)
        status_layout.addStretch()
        
        # Add buttons to status_buttons layout
        status_buttons.addWidget(wake_word_toggle)
        status_buttons.addWidget(reset_button)
        status_buttons.addWidget(power_button)
        
        # Add status_buttons to status_layout
        status_layout.addLayout(status_buttons)
        
        # Add frames to right_layout
        right_layout.addWidget(camera_frame)
        right_layout.addWidget(status_frame)
        
        # Add all main components to content layout
        content_layout.addWidget(conversation_frame, 2)
        content_layout.addLayout(right_layout, 1)
        
        # Add everything to main layout
        main_layout.addWidget(header_frame)
        main_layout.addLayout(content_layout)
        
        # Add welcome message
        welcome_message = (
            "Hello! I'm SOMEBODY, your personal assistant. How can I help you today?\n\n"
            "Tips: Click the circle button at the top right to activate voice input, "
            "or use the text field below to type your messages."
        )
        self.add_assistant_message(welcome_message)

    def initialize_camera(self):
        """Initialize camera if available"""
        try:
            available_cameras = QCameraInfo.availableCameras()
            if available_cameras:
                self.camera = QCamera(available_cameras[0])
                self.camera.setViewfinder(self.viewfinder)
                
                # Setup image capture
                self.image_capture = QCameraImageCapture(self.camera)
                self.image_capture.imageCaptured.connect(self.on_image_captured)
                
                # Add on_image_saved method
                self.image_capture.imageSaved.connect(self.on_image_saved)

                self.camera.start()
                self.vision_status.set_status(True)
                logger.info("Camera initialized successfully")
            else:
                self.vision_status.set_status(False)
                logger.warning("No cameras available")
        except Exception as e:
            self.vision_status.set_status(False)
            logger.error(f"Error initializing camera: {str(e)}")
    
    def on_image_saved(self, id, path):
        """Called when an image has been saved to a file"""
        self.last_captured_image_path = path
        logger.debug(f"Image saved to: {path}")
        
    def check_backend_status(self):
        """Check if backend services are running"""
        try:
            response = requests.get(f"{API_URL}/health")
            if response.status_code == 200:
                services = response.json()
                self.llm_status.set_status(services.get("llm", False))
                self.speech_status.set_status(services.get("speech", False))
                self.tts_status.set_status(services.get("tts", False))
                self.vision_status.set_status(services.get("vision", False))
                self.internet_status.set_status(services.get("internet", False))
            else:
                # Set all to false if backend unreachable
                self.llm_status.set_status(False)
                self.speech_status.set_status(False)
                self.tts_status.set_status(False)
                self.vision_status.set_status(False)
                self.internet_status.set_status(False)
        except Exception:
            # Set all to false if backend unreachable
            self.llm_status.set_status(False)
            self.speech_status.set_status(False)
            self.tts_status.set_status(False)
            self.vision_status.set_status(False)
            self.internet_status.set_status(False)
        
        # Schedule next check
        QTimer.singleShot(10000, self.check_backend_status)
    
    def add_user_message(self, text):
        """Add a user message to the chat area"""
        message = ChatMessage(text, is_user=True)
        self.chat_layout.addWidget(message)
        self.scroll_to_bottom()
    
    def add_assistant_message(self, text):
        """Add an assistant message to the chat area"""
        message = ChatMessage(text, is_user=False)
        self.chat_layout.addWidget(message)
        self.scroll_to_bottom()
        
        # Also speak the response
        self.speak_text(text)
    
    def scroll_to_bottom(self):
        """Scroll the chat area to the bottom"""
        # Fixed version that checks if scroll_area has a vertical scrollbar
        if hasattr(self, 'scroll_area') and hasattr(self.scroll_area, 'verticalScrollBar'):
            QTimer.singleShot(100, lambda: 
                self.scroll_area.verticalScrollBar().setValue(
                    self.scroll_area.verticalScrollBar().maximum()
                )
            )
    
    def send_message(self):
        """Send a message from the input field"""
        text = self.message_input.text().strip()
        if not text:
            return
        
        self.add_user_message(text)
        self.message_input.clear()
        
        # Send to LLM API
        self.process_input(text)
    
    def process_input(self, text):
        """Process text input and get response from LLM"""
        try:
            # Check if it's explicitly a search query
            if text.lower().startswith("search for") or text.lower().startswith("find"):
                # Indicate we're searching
                self.add_assistant_message(f"Searching for information about '{text}'...")
                
            # Send to LLM API (which will handle search internally)
            response = requests.post(
                f"{API_URL}/brain/process",
                json={"input": text}
            )
            if response.status_code == 200:
                result = response.json().get("response", "I'm having trouble processing that.")
                self.add_assistant_message(result)
            else:
                self.add_assistant_message("Sorry, I encountered an error processing your request.")
        except Exception as e:
            self.add_assistant_message(f"Sorry, I couldn't connect to my brain. Error: {str(e)}")
    
    # def toggle_voice_recording(self):
    #     """Toggle voice recording on/off"""
    #     if self.audio_thread.is_recording:
    #         # Already recording, do nothing
    #         return
        
    #     self.voice_button.toggle_active()
    #     self.add_assistant_message("Listening...")
        
    #     # Start recording in a thread
    #     self.audio_thread.start()

    def toggle_voice_recording(self):
        if self.audio_thread.is_recording:
            return
        
        self.voice_button.toggle_active()
        if self.voice_button.active:
            self.mic_status.setText("Listening...")
            self.mic_status.setStyleSheet("color: #F38BA8; font-size: 12px; font-weight: bold;")
        else:
            self.mic_status.setText("Voice Input")
            self.mic_status.setStyleSheet("color: #94E2D5; font-size: 12px;")
        
        self.add_assistant_message("Listening...")
        self.audio_thread.start()
    
    def on_speech_recognized(self, text):
        """Handle recognized speech"""
        self.voice_button.toggle_active()
        self.mic_status.setText("Voice Input") 
        self.mic_status.setStyleSheet("color: #94E2D5; font-size: 12px;")
        if text:
            self.add_user_message(text)
            self.process_input(text)
        else:
            self.add_assistant_message("I didn't catch that. Could you try again?")
    
    def speak_text(self, text):
        """Use text-to-speech to speak the given text"""
        try:
            requests.post(
                f"{API_URL}/speech/speak",
                json={"text": text}
            )
        except Exception as e:
            # Log error but continue silently
            logger.error(f"Error using text-to-speech: {str(e)}")
    
    def capture_image(self):
        """Capture an image from the camera"""
        try:
            # Create temporary file
            temp_dir = tempfile.gettempdir()
            self.last_captured_image_path = os.path.join(temp_dir, "somebody_capture.jpg")
            
            # Simplified approach - capture directly from viewfinder
            if hasattr(self, 'camera') and self.camera:
                pixmap = self.viewfinder.grab()
                pixmap.save(self.last_captured_image_path)
                self.add_assistant_message(f"Image captured. Would you like me to analyze it?")
                logger.info(f"Image captured and saved to {self.last_captured_image_path}")
            else:
                self.add_assistant_message("Camera is not available")
                logger.warning("Camera is not available for capturing")
        except Exception as e:
            self.add_assistant_message(f"Error capturing image: {str(e)}")
            logger.error(f"Error capturing image: {str(e)}")
    
    def on_image_captured(self, id, preview):
        """Called when an image has been captured"""
        self.add_assistant_message("Image captured. Would you like me to analyze it?")
    
    def on_capture_error(self, id, error, errorString):
        """Called when there's an error capturing an image"""
        self.add_assistant_message(f"Error capturing image: {errorString}")
    
    def analyze_image(self):
        """Analyze the current camera view"""
        try:
            if not hasattr(self, 'last_captured_image_path') or not os.path.exists(self.last_captured_image_path):
                # If no image was captured previously, try to capture one now
                self.capture_image()
                QTimer.singleShot(2000, self.analyze_image)  # Try again after 2 seconds
                return
                
            self.add_assistant_message("Analyzing image...")
            
            # Read the image file
            with open(self.last_captured_image_path, 'rb') as image_file:
                files = {'image': ('image.jpg', image_file, 'image/jpeg')}
                data = {'query': 'What do you see in this image?'}
                
                # Send to API
                response = requests.post(f"{API_URL}/process/image", files=files, data=data)
                
                if response.status_code == 200:
                    result = response.json()
                    description = result.get('description', 'No description available')
                    
                    if 'response' in result:
                        analysis_result = result['response']
                    else:
                        analysis_result = description
                        
                    self.add_assistant_message(analysis_result)
                else:
                    self.add_assistant_message("Error analyzing image. The API returned an unexpected response.")
        except Exception as e:
            self.add_assistant_message(f"Error analyzing image: {str(e)}")
        # --- Tambahkan setelah fungsi analyze_image ---
    def detect_objects(self):
        """Detect objects in the current image using YOLO"""
        try:
            if not hasattr(self, 'last_captured_image_path') or not os.path.exists(self.last_captured_image_path):
                # If no image was captured previously, try to capture one now
                self.capture_image()
                QTimer.singleShot(2000, self.detect_objects)  # Try again after 2 seconds
                return
                
            self.add_assistant_message("Mendeteksi objek dalam gambar...")
            
            # Read the image file
            with open(self.last_captured_image_path, 'rb') as image_file:
                files = {'image': ('image.jpg', image_file, 'image/jpeg')}
                data = {'confidence': 0.25}  # Default confidence threshold
                
                # Send to API
                response = requests.post(f"{API_URL}/detect/objects", files=files, data=data)
                
                if response.status_code == 200:
                    result = response.json()
                    
                    if 'error' in result:
                        self.add_assistant_message(f"Error mendeteksi objek: {result['error']}")
                        return
                    
                    # Get detection results
                    detections = result.get('detections', [])
                    custom_detections = result.get('custom_detections', [])
                    description = result.get('description', 'Tidak ada deskripsi tersedia')
                    
                    # Format detection results
                    detected_objects = []
                    for det in detections:
                        detected_objects.append(f"{det['class_name']} ({det['confidence']:.2f})")
                    
                    detected_electronics = []
                    for det in custom_detections:
                        detected_electronics.append(f"{det['class_name']} ({det['confidence']:.2f})")
                    
                    # Create response message
                    if detections or custom_detections:
                        result_message = description
                        
                        # If annotated image available, save it
                        if 'annotated_image_base64' in result and result['annotated_image_base64']:
                            try:
                                import base64
                                img_data = base64.b64decode(result['annotated_image_base64'])
                                annotated_img_path = os.path.join(tempfile.gettempdir(), "somebody_annotated.jpg")
                                with open(annotated_img_path, 'wb') as f:
                                    f.write(img_data)
                                
                                result_message += f"\n\nGambar dengan anotasi objek telah disimpan di {annotated_img_path}"
                            except Exception as e:
                                logger.error(f"Error menyimpan gambar anotasi: {str(e)}")
                    else:
                        result_message = "Tidak ada objek terdeteksi dalam gambar."
                    
                    # Display result
                    self.add_assistant_message(result_message)
                else:
                    self.add_assistant_message("Error mendeteksi objek. API mengembalikan respons yang tidak terduga.")
        except Exception as e:
            self.add_assistant_message(f"Error mendeteksi objek: {str(e)}")
            
    def reset_system(self):
        """Reset all systems"""
        try:
            response = requests.post(f"{API_URL}/system/reset")
            self.add_assistant_message("All systems have been reset.")
            self.check_backend_status()
        except Exception as e:
            self.add_assistant_message(f"Failed to reset systems: {str(e)}")
    
    def power_toggle(self):
        """Power down or restart the system"""
        try:
            response = requests.post(f"{API_URL}/system/power")
            self.add_assistant_message("System is shutting down. Goodbye!")
            QTimer.singleShot(3000, self.close)
        except Exception as e:
            self.add_assistant_message(f"Failed to shut down systems: {str(e)}")
    
    def toggle_wake_word(self):
        """Toggle wake word detection on/off"""
        sender = self.sender()
        try:
            if sender.isChecked():
                # Turn on wake word detection
                response = requests.post(f"{API_URL}/wake_word/start")
                if response.status_code == 200:
                    sender.setText("Disable Wake Word")
                    self.add_assistant_message("Wake word detection enabled. Say 'Hey SOMEBODY' to activate me.")
                else:
                    sender.setChecked(False)
                    self.add_assistant_message("Failed to enable wake word detection.")
            else:
                # Turn off wake word detection
                response = requests.post(f"{API_URL}/wake_word/stop")
                if response.status_code == 200:
                    sender.setText("Enable Wake Word")
                    self.add_assistant_message("Wake word detection disabled.")
                else:
                    sender.setChecked(True)
                    self.add_assistant_message("Failed to disable wake word detection.")
        except Exception as e:
            sender.setChecked(False)
            self.add_assistant_message(f"Error toggling wake word detection: {str(e)}")

    def toggle_internet(self):
        """Toggle internet connection on/off"""
        sender = self.sender()
        try:
            if sender.isChecked():
                # Turn on internet
                response = requests.post(f"{API_URL}/system/internet", json={"enabled": True})
                if response.status_code == 200:
                    sender.setText("Disable Internet")
                    self.internet_status.set_status(True)
                    self.add_assistant_message("Internet connection enabled. I can now search for information online.")
                else:
                    sender.setChecked(False)
                    self.add_assistant_message("Failed to enable internet connection.")
            else:
                # Turn off internet
                response = requests.post(f"{API_URL}/system/internet", json={"enabled": False})
                if response.status_code == 200:
                    sender.setText("Enable Internet")
                    self.internet_status.set_status(False)
                    self.add_assistant_message("Internet connection disabled.")
                else:
                    sender.setChecked(True)
                    self.add_assistant_message("Failed to disable internet connection.")
        except Exception as e:
            sender.setChecked(False)
            self.add_assistant_message(f"Error toggling internet connection: {str(e)}")
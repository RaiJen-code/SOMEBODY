"""
Vision analyzer module for SOMEBODY
"""

import logging
import os
from typing import Dict, List, Optional, Any
import tempfile
import base64

logger = logging.getLogger(__name__)

class VisionAnalyzer:
    """Base class for vision analysis"""
    
    def __init__(self, model_path: Optional[str] = None, 
                 yolo_model: str = "yolov8n.pt",
                 custom_yolo_path: Optional[str] = None):
        """Initialize vision analyzer
        
        Args:
            model_path: Path to the vision model (optional)
            yolo_model: Name of YOLO model to use
            custom_yolo_path: Path to custom YOLO model for electronic components
        """
        self.model_path = model_path
        self.model = None
        self.cv2 = None  # Will be imported dynamically
        self.np = None   # Will be imported dynamically
        
        # YOLO detector
        self.yolo_detector = None
        self.yolo_model_name = yolo_model
        self.custom_yolo_path = custom_yolo_path
        
        logger.info("Vision analyzer initialized")
        
    def load_model(self) -> bool:
        """Load vision dependencies and model
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Import OpenCV and NumPy here to avoid importing at module level
            import cv2
            import numpy as np
            
            self.cv2 = cv2
            self.np = np
            
            logger.info("Vision dependencies loaded successfully")
            
            # Initialize YOLO detector
            from vision.yolo_detector import YOLODetector
            self.yolo_detector = YOLODetector(
                model_name=self.yolo_model_name,
                custom_model_path=self.custom_yolo_path
            )
            
            # Initialize the YOLO detector
            if not self.yolo_detector.initialize():
                logger.warning("Failed to initialize YOLO detector")
            else:
                logger.info("YOLO detector initialized successfully")
            
            return True
        except ImportError as e:
            logger.error(f"Error loading vision dependencies: {str(e)}")
            return False
        except Exception as e:
            logger.error(f"Error loading vision model: {str(e)}")
            return False
            
    def analyze_image(self, image_data: bytes) -> Dict[str, Any]:
        """Analyze image and return results
        
        Args:
            image_data: Image data as bytes
            
        Returns:
            Dictionary containing analysis results
        """
        if not self.cv2 or not self.np:
            if not self.load_model():
                return {"error": "Failed to load vision dependencies"}
        
        try:
            # Convert bytes to image
            nparr = self.np.frombuffer(image_data, self.np.uint8)
            img = self.cv2.imdecode(nparr, self.cv2.IMREAD_COLOR)
            
            if img is None:
                logger.error("Failed to decode image")
                return {"error": "Failed to decode image"}
            
            # Basic image information
            height, width, channels = img.shape
            
            # Convert to grayscale for analysis
            gray = self.cv2.cvtColor(img, self.cv2.COLOR_BGR2GRAY)
            
            # Get average brightness
            brightness = self.cv2.mean(gray)[0]
            
            # Detect edges
            edges = self.cv2.Canny(gray, 100, 200)
            edge_percentage = self.np.count_nonzero(edges) / (height * width) * 100
            
            # Detect faces (if available)
            face_cascade = self.cv2.CascadeClassifier(self.cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            face_count = len(faces)
            
            # Generate a simple scene description based on basic features
            description = f"Gambar dengan dimensi {width}x{height} piksel. "
            description += f"Memiliki {channels} channel warna. "
            description += f"Kecerahan rata-rata {brightness:.1f}/255. "
            
            if face_count > 0:
                description += f"Terdeteksi {face_count} {'wajah' if face_count == 1 else 'wajah'} pada gambar. "
            
            if edge_percentage > 10:
                description += "Gambar mengandung banyak tepi dan mungkin kompleks atau detail. "
            elif edge_percentage > 5:
                description += "Gambar memiliki jumlah detail yang sedang. "
            else:
                description += "Gambar tampak relatif sederhana dengan sedikit tepi. "
            
            # Convert the processed image to base64 for visualization (optional)
            _, buffer = self.cv2.imencode('.jpg', edges)
            edge_image_base64 = base64.b64encode(buffer).decode('utf-8')
            
            # Run YOLO object detection if available
            yolo_results = {}
            if self.yolo_detector and self.yolo_detector.is_loaded():
                try:
                    logger.info("Running YOLO object detection")
                    yolo_results = self.yolo_detector.detect_objects(img)
                    
                    # If YOLO detection was successful, add the description
                    if yolo_results.get("success", False):
                        description += "\n\n" + yolo_results.get("description", "")
                except Exception as e:
                    logger.error(f"Error running YOLO detection: {str(e)}")
            else:
                logger.warning("YOLO detector not available for object detection")
                
            # Combine results
            results = {
                "dimensions": {"width": width, "height": height, "channels": channels},
                "brightness": brightness,
                "edge_percentage": edge_percentage,
                "face_count": face_count,
                "faces": [{"x": int(x), "y": int(y), "width": int(w), "height": int(h)} for (x, y, w, h) in faces],
                "description": description,
                "edge_image_base64": edge_image_base64
            }
            
            # Add YOLO results if available
            if yolo_results.get("success", False):
                results.update({
                    "yolo_detections": yolo_results.get("detections", []),
                    "custom_detections": yolo_results.get("custom_detections", []),
                    "annotated_image_base64": yolo_results.get("annotated_image_base64", "")
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Error analyzing image: {str(e)}")
            return {"error": str(e)}
    
    def detect_objects(self, image_data: bytes) -> Dict[str, Any]:
        """Detect objects in image using YOLO
        
        Args:
            image_data: Image data as bytes
            
        Returns:
            Dictionary with detection results
        """
        if not self.yolo_detector:
            if not self.load_model():
                return {"error": "Failed to load YOLO detector"}
        
        try:
            # Run YOLO detection
            return self.yolo_detector.detect_objects(image_data)
        except Exception as e:
            logger.error(f"Error detecting objects: {str(e)}")
            return {"error": str(e)}
    
    def is_loaded(self) -> bool:
        """Check if model is loaded
        
        Returns:
            True if model is loaded, False otherwise
        """
        return self.cv2 is not None and self.np is not None

    def train_custom_model(self, dataset_path: str, epochs: int = 50) -> Dict[str, Any]:
        """Train custom YOLO model for electronic components
        
        Args:
            dataset_path: Path to dataset in YOLO format
            epochs: Number of training epochs
            
        Returns:
            Dictionary with training results
        """
        if not self.yolo_detector:
            if not self.load_model():
                return {"error": "Failed to load YOLO detector"}
                
        try:
            # Run training
            results = self.yolo_detector.train_custom_model(
                dataset_path=dataset_path,
                epochs=epochs
            )
            
            # Update custom model path if training was successful
            if results.get("success", False):
                self.custom_yolo_path = results.get("model_path")
            
            return results
        except Exception as e:
            logger.error(f"Error training custom model: {str(e)}")
            return {"error": str(e)}
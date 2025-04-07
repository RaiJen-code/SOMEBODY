"""
YOLO object detection module for SOMEBODY
"""

import logging
import os
import tempfile
import time
from typing import Dict, List, Optional, Any, Union
import numpy as np
import cv2
import base64

logger = logging.getLogger(__name__)

class YOLODetector:
    """Object detection using YOLO models"""
    
    def __init__(self, model_name: str = "yolov8n.pt", custom_model_path: Optional[str] = None):
        """Initialize YOLO detector
        
        Args:
            model_name: Name of the pre-trained YOLO model to use
            custom_model_path: Path to custom model for specific detection tasks
        """
        self.model_name = model_name
        self.custom_model_path = custom_model_path
        self.model = None
        self.custom_model = None
        self.is_initialized = False
        
        # Create directory for custom models if it doesn't exist
        self.models_dir = os.path.join("data", "models", "yolo")
        os.makedirs(self.models_dir, exist_ok=True)
        
        logger.info(f"YOLO detector initialized with model: {model_name}")
        
    def initialize(self) -> bool:
        """Initialize YOLO models
        
        Returns:
            True if initialization successful, False otherwise
        """
        try:
            from ultralytics import YOLO
            
            # Load general YOLO model
            logger.info(f"Loading YOLO model: {self.model_name}")
            self.model = YOLO(self.model_name)
            
            # Load custom model if specified
            if self.custom_model_path and os.path.exists(self.custom_model_path):
                logger.info(f"Loading custom YOLO model: {self.custom_model_path}")
                self.custom_model = YOLO(self.custom_model_path)
            
            self.is_initialized = True
            logger.info("YOLO models loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing YOLO detector: {str(e)}")
            return False
            
    def detect_objects(self, image_data: Union[bytes, np.ndarray, str], 
                       conf_threshold: float = 0.25) -> Dict[str, Any]:
        """Detect objects in image
        
        Args:
            image_data: Image data as bytes, numpy array, or file path
            conf_threshold: Confidence threshold for detections (0-1)
            
        Returns:
            Dictionary with detection results
        """
        if not self.is_initialized:
            if not self.initialize():
                return {"error": "Failed to initialize YOLO detector"}
        
        try:
            # Convert input to numpy array if needed
            if isinstance(image_data, bytes):
                # Convert bytes to numpy array
                nparr = np.frombuffer(image_data, np.uint8)
                image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            elif isinstance(image_data, str):
                # Load image from file path
                image = cv2.imread(image_data)
            else:
                # Assume it's already a numpy array
                image = image_data
                
            if image is None:
                return {"error": "Failed to decode image"}
                
            # Save original image dimensions
            height, width = image.shape[:2]
                
            # Run detection with general model
            logger.info("Running object detection with general YOLO model")
            results = self.model(image, conf=conf_threshold)[0]
            
            # Parse results
            detections = []
            
            for i, (box, score, cls) in enumerate(zip(results.boxes.xyxy.cpu().numpy(), 
                                                   results.boxes.conf.cpu().numpy(),
                                                   results.boxes.cls.cpu().numpy())):
                x1, y1, x2, y2 = box.astype(int)
                class_id = int(cls)
                class_name = results.names[class_id]
                confidence = float(score)
                
                detections.append({
                    "id": i,
                    "class_id": class_id,
                    "class_name": class_name,
                    "confidence": confidence,
                    "box": {"x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2)},
                    "width": int(x2 - x1),
                    "height": int(y2 - y1)
                })
            
            # Try with custom model if available
            custom_detections = []
            if self.custom_model:
                logger.info("Running object detection with custom YOLO model")
                custom_results = self.custom_model(image, conf=conf_threshold)[0]
                
                for i, (box, score, cls) in enumerate(zip(custom_results.boxes.xyxy.cpu().numpy(), 
                                                     custom_results.boxes.conf.cpu().numpy(),
                                                     custom_results.boxes.cls.cpu().numpy())):
                    x1, y1, x2, y2 = box.astype(int)
                    class_id = int(cls)
                    class_name = custom_results.names[class_id]
                    confidence = float(score)
                    
                    custom_detections.append({
                        "id": i,
                        "class_id": class_id,
                        "class_name": class_name,
                        "confidence": confidence,
                        "box": {"x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2)},
                        "width": int(x2 - x1),
                        "height": int(y2 - y1)
                    })
            
            # Create annotated image
            annotated_image = self._annotate_image(image, detections + custom_detections)
            
            # Convert annotated image to base64
            _, buffer = cv2.imencode('.jpg', annotated_image)
            annotated_image_base64 = base64.b64encode(buffer).decode('utf-8')
            
            # Generate description
            description = self._generate_description(detections, custom_detections, width, height)
            
            return {
                "success": True,
                "detections": detections,
                "custom_detections": custom_detections,
                "description": description,
                "annotated_image_base64": annotated_image_base64,
                "dimensions": {"width": width, "height": height}
            }
            
        except Exception as e:
            logger.error(f"Error detecting objects: {str(e)}")
            return {"error": str(e)}
            
    def _annotate_image(self, image: np.ndarray, detections: List[Dict[str, Any]]) -> np.ndarray:
        """Annotate image with bounding boxes and labels
        
        Args:
            image: Input image as numpy array
            detections: List of detection dictionaries
            
        Returns:
            Annotated image as numpy array
        """
        annotated = image.copy()
        
        for detection in detections:
            # Extract box coordinates
            x1 = detection["box"]["x1"]
            y1 = detection["box"]["y1"]
            x2 = detection["box"]["x2"]
            y2 = detection["box"]["y2"]
            
            # Generate random color based on class ID for consistency
            color = self._get_color_by_class_id(detection["class_id"])
            
            # Draw bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            
            # Prepare label text
            label = f"{detection['class_name']} {detection['confidence']:.2f}"
            
            # Draw label background
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(annotated, (x1, y1 - text_size[1] - 5), (x1 + text_size[0], y1), color, -1)
            
            # Draw label text
            cv2.putText(annotated, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
        return annotated
        
    def _get_color_by_class_id(self, class_id: int) -> tuple:
        """Generate consistent color for a class ID
        
        Args:
            class_id: Class ID number
            
        Returns:
            BGR color tuple
        """
        # Generate colors based on class ID for consistency
        # Using HSV colorspace to get vibrant colors
        hue = (class_id * 30) % 180
        hsv = np.array([[[hue, 255, 255]]], dtype=np.uint8)
        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)[0][0]
        # Convert to BGR for OpenCV
        return (int(rgb[2]), int(rgb[1]), int(rgb[0]))
        
    def _generate_description(self, general_detections: List[Dict[str, Any]], 
                             custom_detections: List[Dict[str, Any]],
                             width: int, height: int) -> str:
        """Generate human-readable description of detections
        
        Args:
            general_detections: List of general model detections
            custom_detections: List of custom model detections
            width: Image width
            height: Image height
            
        Returns:
            Description text
        """
        # Start with image dimensions
        description = f"Gambar berukuran {width}x{height} piksel. "
        
        # General detections
        if general_detections:
            # Count objects by class
            class_counts = {}
            for det in general_detections:
                class_name = det["class_name"]
                class_counts[class_name] = class_counts.get(class_name, 0) + 1
            
            # Format the counts into description
            objects_list = []
            for class_name, count in class_counts.items():
                objects_list.append(f"{count} {class_name}")
            
            # Add to description
            description += f"Terdeteksi {len(general_detections)} objek umum: {', '.join(objects_list)}. "
        else:
            description += "Tidak terdeteksi objek umum. "
        
        # Custom detections (elektronik)
        if custom_detections:
            # Count objects by class
            class_counts = {}
            for det in custom_detections:
                class_name = det["class_name"]
                class_counts[class_name] = class_counts.get(class_name, 0) + 1
            
            # Format the counts into description
            objects_list = []
            for class_name, count in class_counts.items():
                objects_list.append(f"{count} {class_name}")
            
            # Add to description
            description += f"Terdeteksi {len(custom_detections)} komponen elektronik: {', '.join(objects_list)}. "
        else:
            description += "Tidak terdeteksi komponen elektronik khusus. "
            
        return description
        
    def is_loaded(self) -> bool:
        """Check if model is loaded
        
        Returns:
            True if model is loaded, False otherwise
        """
        return self.is_initialized
        
    def train_custom_model(self, 
                           dataset_path: str,
                           epochs: int = 50,
                           batch_size: int = 16,
                           img_size: int = 640) -> Dict[str, Any]:
        """Train a custom YOLO model on electronic components
        
        Args:
            dataset_path: Path to dataset in YOLO format
            epochs: Number of training epochs
            batch_size: Batch size for training
            img_size: Image size for training
            
        Returns:
            Dictionary with training results
        """
        if not self.is_initialized:
            if not self.initialize():
                return {"error": "Failed to initialize YOLO detector"}
        
        try:
            # Set output directory for trained model
            output_dir = os.path.join(self.models_dir, "custom")
            os.makedirs(output_dir, exist_ok=True)
            
            logger.info(f"Training custom YOLO model on dataset: {dataset_path}")
            logger.info(f"Training parameters: epochs={epochs}, batch_size={batch_size}, img_size={img_size}")
            
            # Start training
            results = self.model.train(
                data=dataset_path,
                epochs=epochs,
                batch=batch_size,
                imgsz=img_size,
                project=output_dir,
                name="electronic_components"
            )
            
            # Get path to best model
            best_model_path = os.path.join(output_dir, "electronic_components", "weights", "best.pt")
            
            if os.path.exists(best_model_path):
                # Update custom model path and load it
                self.custom_model_path = best_model_path
                self.custom_model = self.model.__class__(best_model_path)
                
                logger.info(f"Custom model trained and loaded from: {best_model_path}")
                
                return {
                    "success": True,
                    "model_path": best_model_path,
                    "message": "Custom model trained successfully"
                }
            else:
                return {
                    "success": False,
                    "error": "Training completed but model file not found"
                }
            
        except Exception as e:
            logger.error(f"Error training custom model: {str(e)}")
            return {"error": str(e)}
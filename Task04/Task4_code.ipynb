import cv2
import numpy as np
from tensorflow.keras.models import load_model
import logging
from typing import Optional, Tuple, Any

# -----------------------------
# Configuration
# -----------------------------
CONFIG = {
    'accum_weight': 0.5,
    'roi_coordinates': (10, 350, 225, 590),  # top, right, bottom, left
    'calibration_frames': 30,
    'threshold_value': 25,
    'gaussian_blur': (7, 7),
    'image_size': (100, 120),  # width, height for model input
    'prediction_interval': 10,
    'frame_resolution': (700, 700)
}

GESTURE_CLASSES = {
    0: "Blank",
    1: "OK", 
    2: "Thumbs Up",
    3: "Thumbs Down",
    4: "Punch",
    5: "High Five"
}

# -----------------------------
# Background Subtraction Class
# -----------------------------
class BackgroundSubtractor:
    def __init__(self, accum_weight: float = 0.5):
        self.bg: Optional[np.ndarray] = None
        self.accum_weight = accum_weight
    
    def run_avg(self, image: np.ndarray) -> None:
        if self.bg is None:
            self.bg = image.copy().astype("float")
            return
        cv2.accumulateWeighted(image, self.bg, self.accum_weight)
    
    def segment(self, image: np.ndarray, threshold: int = 25) -> Optional[np.ndarray]:
        if self.bg is None:
            return None
            
        diff = cv2.absdiff(self.bg.astype("uint8"), image)
        _, thresholded = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
        
        # Clean up the image
        kernel = np.ones((3, 3), np.uint8)
        thresholded = cv2.morphologyEx(thresholded, cv2.MORPH_OPEN, kernel)
        thresholded = cv2.morphologyEx(thresholded, cv2.MORPH_CLOSE, kernel)
        
        # INVERT THE IMAGE: Hand will be black, background will be white
        thresholded = cv2.bitwise_not(thresholded)
        
        return thresholded

# -----------------------------
# Gesture Recognizer Class
# -----------------------------
class GestureRecognizer:
    def __init__(self, model_path: str):
        self.model = self._load_model(model_path)
        self.background_subtractor = BackgroundSubtractor(CONFIG['accum_weight'])
        self.frame_count = 0
        self.is_calibrated = False
        
    def _load_model(self, model_path: str) -> Any:
        try:
            model = load_model(model_path)
            print("Model loaded successfully")
            return model
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # If the image is inverted (hand black, background white), invert it back for the model
        # Most models expect hand to be white and background black
        image = cv2.bitwise_not(image)
        
        image = cv2.resize(image, CONFIG['image_size'])
        image = image.astype('float32') / 255.0
        image = image.reshape(1, *CONFIG['image_size'][::-1], 1)
        return image
    
    def predict_gesture(self, image: np.ndarray) -> str:
        if self.model is None:
            return "No Model"
            
        try:
            preprocessed = self.preprocess_image(image)
            prediction = self.model.predict(preprocessed, verbose=0)
            predicted_class = np.argmax(prediction)
            confidence = np.max(prediction)
            
            if confidence < 0.7:
                return "Uncertain"
                
            return GESTURE_CLASSES.get(predicted_class, "Unknown")
        except Exception as e:
            return "Error"

# -----------------------------
# Main Application
# -----------------------------
class HandGestureApp:
    def __init__(self):
        self.camera = None
        self.recognizer = None
        self.top, self.right, self.bottom, self.left = CONFIG['roi_coordinates']
        
    def initialize_camera(self) -> bool:
        self.camera = cv2.VideoCapture(0)
        if not self.camera.isOpened():
            print("Cannot open camera")
            return False
        return True
    
    def run(self) -> None:
        if not self.initialize_camera():
            return
            
        try:
            self.recognizer = GestureRecognizer("hand_gesture_recog_model.h5")
        except:
            print("Failed to load model")
            return
        
        print("Starting gesture recognition. Press 'q' to quit, 'r' to recalibrate.")
        
        predicted_gesture = "Place hand in box"
        
        while True:
            grabbed, frame = self.camera.read()
            if not grabbed:
                break
            
            # Resize and flip frame
            frame = cv2.resize(frame, CONFIG['frame_resolution'])
            frame = cv2.flip(frame, 1)
            
            # Extract ROI
            roi = frame[self.top:self.bottom, self.right:self.left]
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, CONFIG['gaussian_blur'], 0)
            
            # Calibration phase
            if not self.recognizer.is_calibrated:
                self.recognizer.background_subtractor.run_avg(gray)
                self.recognizer.frame_count += 1
                
                if self.recognizer.frame_count == 1:
                    print("Calibrating... Please wait.")
                elif self.recognizer.frame_count == CONFIG['calibration_frames']:
                    self.recognizer.is_calibrated = True
                    print("Calibration complete!")
                    predicted_gesture = "Show your gesture"
            else:
                # Get the inverted black and white hand image (hand will be black, background white)
                thresholded = self.recognizer.background_subtractor.segment(gray)
                
                if thresholded is not None:
                    # Display the inverted image in the ROI area
                    thresholded_bgr = cv2.cvtColor(thresholded, cv2.COLOR_GRAY2BGR)
                    frame[self.top:self.bottom, self.right:self.left] = thresholded_bgr
                    
                    # Predict gesture at intervals
                    if self.recognizer.frame_count % CONFIG['prediction_interval'] == 0:
                        predicted_gesture = self.recognizer.predict_gesture(thresholded)
                
                self.recognizer.frame_count += 1
            
            # Draw the box
            cv2.rectangle(frame, (self.left, self.top), (self.right, self.bottom), (0, 255, 0), 2)
            
            # Display gesture text
            cv2.putText(frame, f"Gesture: {predicted_gesture}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            # Show calibration status
            if not self.recognizer.is_calibrated:
                cv2.putText(frame, "Calibrating...", (10, 70), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
            # Show the frame
            cv2.imshow("Hand Gesture Recognition", frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                self.recognizer.background_subtractor.bg = None
                self.recognizer.frame_count = 0
                self.recognizer.is_calibrated = False
                predicted_gesture = "Recalibrating..."
                print("Recalibrating...")
        
        self.cleanup()
    
    def cleanup(self) -> None:
        if self.camera:
            self.camera.release()
        cv2.destroyAllWindows()
        print("Application closed")

# -----------------------------
# Main execution
# -----------------------------
if __name__ == "__main__":
    app = HandGestureApp()
    app.run()

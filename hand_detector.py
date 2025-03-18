import cv2
import mediapipe as mp
import numpy as np

class HandDetector:
    def __init__(self, static_mode=False, max_hands=2, detection_confidence=0.5, tracking_confidence=0.5):
       
        self.static_mode = static_mode
        self.max_hands = max_hands
        self.detection_confidence = detection_confidence
        self.tracking_confidence = tracking_confidence
        
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=static_mode,
            max_num_hands=max_hands,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
    def find_hands(self, img, draw=True):
      
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        self.results = self.hands.process(img_rgb)
        
        if self.results.multi_hand_landmarks and draw:
            for hand_landmarks in self.results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    img,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style()
                )
        
        return img, self.results
    
    def find_positions(self, img, hand_index=0):
      
        h, w, c = img.shape
        landmark_list = []
        bbox = [0, 0, 0, 0]  # x_min, y_min, x_max, y_max
        
        # Check if any hands detected
        if not self.results.multi_hand_landmarks:
            return landmark_list, bbox
        
        # Check if requested hand index exists
        if hand_index >= len(self.results.multi_hand_landmarks):
            return landmark_list, bbox
        
        # Extract hand landmarks
        hand = self.results.multi_hand_landmarks[hand_index]
        
        # Initialize bounding box coordinates
        x_min, y_min = w, h
        x_max, y_max = 0, 0
        
        # Process each landmark
        for id, lm in enumerate(hand.landmark):
            # Convert normalized coordinates to pixel coordinates
            cx, cy, cz = int(lm.x * w), int(lm.y * h), lm.z
            landmark_list.append([id, cx, cy, cz])
            
            # Update bounding box
            x_min = min(x_min, cx)
            y_min = min(y_min, cy)
            x_max = max(x_max, cx)
            y_max = max(y_max, cy)
        
        # Add padding to bounding box
        padding = 20
        bbox = [
            max(0, x_min - padding),
            max(0, y_min - padding),
            min(w, x_max + padding),
            min(h, y_max + padding)
        ]
        
        return landmark_list, bbox
    
    def draw_bounding_box(self, img, bbox, hand_label="Hand"):
        
        x_min, y_min, x_max, y_max = bbox
        
        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        
        cv2.putText(img, hand_label, (x_min, y_min - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return img
    
    def get_hand_type(self, idx=0):
        if not self.results.multi_hand_landmarks:
            return None
            
        if idx >= len(self.results.multi_handedness):
            return None
            
        # Get hand type from classification
        hand_type = self.results.multi_handedness[idx].classification[0].label
        return hand_type

if __name__ == "__main__":
    detector = HandDetector()
    
    cap = cv2.VideoCapture(0)
    
    while True:
        success, img = cap.read()
        if not success:
            break
            
        img, results = detector.find_hands(img)
        
        lm_list, bbox = detector.find_positions(img, hand_index=0)
        
        if lm_list:
            hand_type = detector.get_hand_type(0)
            img = detector.draw_bounding_box(img, bbox, hand_type)
            
            if len(lm_list) > 8: 
                cv2.circle(img, (lm_list[8][1], lm_list[8][2]), 15, (255, 0, 255), cv2.FILLED)
        
        cv2.imshow("Hand Detector", img)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()
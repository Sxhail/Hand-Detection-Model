import cv2
import numpy as np
import time
from src.hand_detector import HandDetector
from src.feature_extractor import FeatureExtractor

def main():
    cap = cv2.VideoCapture(0)
    detector = HandDetector(detection_confidence=0.7)
    extractor = FeatureExtractor()
    
    prev_time = 0
    curr_time = 0
    
    while True:
        success, img = cap.read()
        if not success:
            print("Failed to capture image from camera")
            break
            
        img, results = detector.find_hands(img)
        
        landmarks, bbox = detector.find_positions(img, hand_index=0)
        
        if landmarks:
            hand_type = detector.get_hand_type(0)
            
            img = detector.draw_bounding_box(img, bbox, hand_type)
            
            features = extractor.extract_features(landmarks)
            
            if features:
                finger_states = features['finger_states']
                fingers_text = f"Fingers: {finger_states}"
                cv2.putText(img, fingers_text, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                for idx, is_extended in enumerate(finger_states):
                    if is_extended:
                        fingertip_idx = extractor.fingertips[idx]
                        if fingertip_idx < len(landmarks):
                            x, y = landmarks[fingertip_idx][1], landmarks[fingertip_idx][2]
                            cv2.circle(img, (x, y), 10, (0, 255, 255), cv2.FILLED)
        
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
        prev_time = curr_time
        
        cv2.putText(img, f"FPS: {int(fps)}", (10, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow("Number of fingers held up", img)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
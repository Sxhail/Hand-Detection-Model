import numpy as np
import math

class FeatureExtractor:
    def __init__(self):
        self.fingertips = [4, 8, 12, 16, 20]  
        self.finger_bases = [2, 5, 9, 13, 17]  

    def extract_features(self, landmarks):
        if not landmarks:
            return None

        lm_array = np.array(landmarks)
        wrist = lm_array[0, 1:3]  

        features = {}

        features['fingertip_positions'] = self._get_normalized_positions(lm_array, self.fingertips, wrist)
        features['finger_states'] = self._get_finger_states(lm_array)
        features['finger_angles'] = self._get_finger_angles(lm_array)
        features['fingertip_distances'] = self._get_fingertip_distances(lm_array)
        features['hand_orientation'] = self._get_hand_orientation(lm_array)
        features['vector'] = self._create_feature_vector(features)

        return features

    def _get_normalized_positions(self, lm_array, landmark_indices, reference):
        positions = []
        for idx in landmark_indices:
            if idx < len(lm_array):
                point = lm_array[idx, 1:3]
                dx, dy = point - reference
                if 12 < len(lm_array):  
                    middle_tip = lm_array[12, 1:3]
                    hand_size = np.linalg.norm(middle_tip - reference)
                    if hand_size > 0:
                        dx, dy = dx / hand_size, dy / hand_size
                positions.append([dx, dy])
            else:
                positions.append([0, 0])
        return positions

    def _get_finger_states(self, lm_array):
        states = []
        for i in range(5):
            if i == 0:  
                if self.fingertips[i] < len(lm_array) and self.finger_bases[i] < len(lm_array):
                    tip = lm_array[self.fingertips[i], 1:3]
                    base = lm_array[self.finger_bases[i], 1:3]
                    mid = lm_array[3, 1:3]  
                    v1 = base - mid
                    v2 = tip - mid
                    angle = self._angle_between(v1, v2)
                    states.append(1 if angle > 0.7 else 0)  
                else:
                    states.append(0)
            else:
                if self.fingertips[i] < len(lm_array) and self.finger_bases[i] < len(lm_array):
                    tip_y = lm_array[self.fingertips[i], 2]
                    base_y = lm_array[self.finger_bases[i], 2]
                    states.append(1 if tip_y < base_y else 0)  
                else:
                    states.append(0)
        return states

    def _get_finger_angles(self, lm_array):
        angles = []
        mcp_joints = [1, 5, 9, 13, 17]
        wrist = lm_array[0, 1:3]
        vectors = []
        for mcp in mcp_joints:
            if mcp < len(lm_array):
                knuckle = lm_array[mcp, 1:3]
                vector = knuckle - wrist
                vectors.append(vector)
            else:
                vectors.append(np.array([0, 0]))
        for i in range(len(vectors) - 1):
            angle = self._angle_between(vectors[i], vectors[i+1])
            angles.append(angle)
        return angles

    def _get_fingertip_distances(self, lm_array):
        distances = []
        for i in range(len(self.fingertips)):
            for j in range(i+1, len(self.fingertips)):
                if self.fingertips[i] < len(lm_array) and self.fingertips[j] < len(lm_array):
                    tip1 = lm_array[self.fingertips[i], 1:3]
                    tip2 = lm_array[self.fingertips[j], 1:3]
                    distance = np.linalg.norm(tip1 - tip2)
                    if 0 < len(lm_array) and 12 < len(lm_array):
                        wrist = lm_array[0, 1:3]
                        middle_tip = lm_array[12, 1:3]
                        hand_size = np.linalg.norm(middle_tip - wrist)
                        if hand_size > 0:
                            distance = distance / hand_size
                    distances.append(distance)
                else:
                    distances.append(0)
        return distances

    def _get_hand_orientation(self, lm_array):
        if 5 < len(lm_array) and 17 < len(lm_array):
            index_mcp = lm_array[5, 1:3]
            pinky_mcp = lm_array[17, 1:3]
            palm_vector = pinky_mcp - index_mcp
            angle = math.atan2(palm_vector[1], palm_vector[0])
            return angle
        return 0

    def _angle_between(self, v1, v2):
        if np.linalg.norm(v1) * np.linalg.norm(v2) == 0:
            return 0
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        cos_angle = min(1.0, max(-1.0, cos_angle))  
        return np.arccos(cos_angle)

    def _create_feature_vector(self, features):
        vector = []
        for pos in features['fingertip_positions']:
            vector.extend(pos)
        vector.extend(features['finger_states'])
        vector.extend(features['finger_angles'])
        vector.extend(features['fingertip_distances'])
        vector.append(features['hand_orientation'])
        return np.array(vector)

if __name__ == "__main__":
    import cv2
    from hand_detector import HandDetector

    detector = HandDetector()
    extractor = FeatureExtractor()

    cap = cv2.VideoCapture(0)

    while True:
        success, img = cap.read()
        if not success:
            break

        img, results = detector.find_hands(img)

        landmarks, bbox = detector.find_positions(img)

        if landmarks:
            features = extractor.extract_features(landmarks)

            if features:
                finger_states = features['finger_states']
                fingers_text = f"Fingers: {finger_states}"
                cv2.putText(img, fingers_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                            0.7, (0, 255, 0), 2)

        cv2.imshow("Feature Extractor", img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

import mediapipe as mp
import cv2 # Though not directly used in constructor, good to have for image ops later
import numpy as np # For type hinting and potential array manipulations

class HandFeatureExtractor:
    def __init__(self, static_image_mode: bool = False, max_num_hands: int = 2, 
                 min_detection_confidence: float = 0.5, min_tracking_confidence: float = 0.5):
        """
        Initializes the Hand Feature Extractor using MediaPipe Hands.

        Args:
            static_image_mode (bool): Whether to treat the input images as a batch of static
                                      images unrelated to each other.
            max_num_hands (int): Maximum number of hands to detect.
            min_detection_confidence (float): Minimum confidence value ([0.0, 1.0]) for hand
                                              detection to be considered successful.
            min_tracking_confidence (float): Minimum confidence value ([0.0, 1.0]) for hand
                                               landmarks to be considered tracked successfully.
        """
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=static_image_mode,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles # For more styled drawing options

        print("HandFeatureExtractor initialized with MediaPipe Hands.")

    def extract_features(self, image_rgb: np.ndarray) -> tuple[list[list[dict]] | None, any]:
        """
        Extracts hand landmarks from an RGB image.

        Args:
            image_rgb (np.ndarray): The input image in RGB format.

        Returns:
            tuple[list[list[dict]] | None, any]: 
                - A list of lists of landmark dictionaries if hands are detected. 
                  Each inner list represents a hand, and each dictionary contains 
                  'x', 'y', 'z' for a landmark. Returns None if no hands are detected.
                - The original results object from MediaPipe Hands for potential further use 
                  (e.g., for drawing or accessing handedness).
        """
        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image_rgb.flags.writeable = False
        results = self.hands.process(image_rgb)
        image_rgb.flags.writeable = True # Back to writeable

        all_hands_landmarks = []
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                current_hand_data = []
                for landmark in hand_landmarks.landmark:
                    current_hand_data.append({
                        'x': landmark.x,
                        'y': landmark.y,
                        'z': landmark.z,
                        # 'visibility': landmark.visibility # Optional
                    })
                all_hands_landmarks.append(current_hand_data)
            
            if not all_hands_landmarks: # Should not happen if results.multi_hand_landmarks is true
                return None, results 
            return all_hands_landmarks, results
        
        return None, results # No hands detected or no landmarks within them

    def draw_landmarks_on_image(self, image_rgb_or_bgr: np.ndarray, results_from_process: any) -> None:
        """
        Draws the hand landmarks and connections on the input image.
        The image is modified in place.

        Args:
            image_rgb_or_bgr (np.ndarray): The image on which to draw. 
                                           Can be RGB or BGR (OpenCV default).
                                           MediaPipe drawing utilities handle colors correctly.
            results_from_process (any): The results object obtained from self.hands.process().
                                        This object contains `multi_hand_landmarks`.
        """
        if results_from_process.multi_hand_landmarks:
            for hand_landmarks in results_from_process.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    image=image_rgb_or_bgr,
                    landmark_list=hand_landmarks,
                    connections=self.mp_hands.HAND_CONNECTIONS,
                    landmark_drawing_spec=self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    connection_drawing_spec=self.mp_drawing_styles.get_default_hand_connections_style()
                )

    def close(self) -> None:
        """
        Closes the MediaPipe Hands instance and releases resources.
        """
        print("Closing MediaPipe Hands in HandFeatureExtractor.")
        self.hands.close()

    def __del__(self) -> None:
        """
        Destructor to ensure MediaPipe Hands is closed.
        """
        self.close()

if __name__ == '__main__':
    # Basic test: try to initialize the feature extractor
    try:
        print("Attempting to initialize HandFeatureExtractor...")
        extractor = HandFeatureExtractor()
        print("HandFeatureExtractor initialized successfully.")
        # extractor.hands.close() # Important to close MediaPipe Hands when done, will be handled in a release method later
    except Exception as e:
        print(f"An error occurred during HandFeatureExtractor initialization: {e}")

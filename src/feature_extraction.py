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


class BodyPoseExtractor:
    def __init__(self, static_image_mode: bool = False, model_complexity: int = 1,
                 smooth_landmarks: bool = True, enable_segmentation: bool = False, # Added enable_segmentation
                 smooth_segmentation: bool = True, # Added smooth_segmentation
                 min_detection_confidence: float = 0.5, min_tracking_confidence: float = 0.5):
        """
        Initializes the Body Pose Extractor using MediaPipe Pose.

        Args:
            static_image_mode (bool): Whether to treat input images as static or a video stream.
            model_complexity (int): Complexity of the pose landmark model: 0, 1, or 2.
            smooth_landmarks (bool): Whether to filter landmarks across frames to reduce jitter.
            enable_segmentation (bool): Whether to predict segmentation mask.
            smooth_segmentation (bool): Whether to filter segmentation mask across frames.
            min_detection_confidence (float): Minimum confidence for person detection.
            min_tracking_confidence (float): Minimum confidence for landmark tracking.
        """
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=static_image_mode,
            model_complexity=model_complexity,
            smooth_landmarks=smooth_landmarks,
            enable_segmentation=enable_segmentation,
            smooth_segmentation=smooth_segmentation,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        # Re-use mp_drawing and mp_drawing_styles if they are already module-level attributes 
        # or ensure they are accessible. For simplicity, let's assume mp.solutions.drawing_utils
        # will be used directly or re-assigned if needed in drawing method.
        # For this step, just initializing self.pose is key.
        # self.mp_drawing = mp.solutions.drawing_utils # Already available via mp.solutions
        # self.mp_drawing_styles = mp.solutions.drawing_styles # Already available via mp.solutions

        print("BodyPoseExtractor initialized with MediaPipe Pose.")

    def extract_features(self, image_rgb: np.ndarray) -> tuple[list[dict] | None, any]:
        """
        Extracts body pose landmarks from an RGB image.

        Args:
            image_rgb (np.ndarray): The input image in RGB format.

        Returns:
            tuple[list[dict] | None, any]: 
                - A list of landmark dictionaries if pose is detected. 
                  Each dictionary contains 'x', 'y', 'z', and 'visibility' for a landmark.
                  Returns None if no pose is detected.
                - The original results object from MediaPipe Pose.
        """
        image_rgb.flags.writeable = False
        results = self.pose.process(image_rgb)
        image_rgb.flags.writeable = True

        pose_landmarks_data = []
        if results.pose_landmarks:
            for landmark in results.pose_landmarks.landmark:
                pose_landmarks_data.append({
                    'x': landmark.x,
                    'y': landmark.y,
                    'z': landmark.z,
                    'visibility': landmark.visibility
                })
            
            if not pose_landmarks_data: # Should not happen if results.pose_landmarks is true
                 return None, results
            return pose_landmarks_data, results
        
        return None, results # No pose detected

    def draw_landmarks_on_image(self, image_rgb_or_bgr: np.ndarray, results_from_process: any) -> None:
        """
        Draws the pose landmarks and connections on the input image.
        The image is modified in place.

        Args:
            image_rgb_or_bgr (np.ndarray): The image on which to draw.
            results_from_process (any): The results object obtained from self.pose.process().
                                        This object contains `pose_landmarks`.
        """
        if results_from_process.pose_landmarks:
            mp.solutions.drawing_utils.draw_landmarks( # Use mp.solutions.drawing_utils directly
                image=image_rgb_or_bgr,
                landmark_list=results_from_process.pose_landmarks,
                connections=self.mp_pose.POSE_CONNECTIONS,
                # For styling, one might need mp.solutions.drawing_styles
                # landmark_drawing_spec=mp.solutions.drawing_styles.get_default_pose_landmarks_style() # Example
            )

    def close(self) -> None:
        """
        Closes the MediaPipe Pose instance and releases resources.
        """
        print("Closing MediaPipe Pose in BodyPoseExtractor.")
        self.pose.close()

    def __del__(self) -> None:
        """
        Destructor to ensure MediaPipe Pose is closed.
        """
        self.close()

if __name__ == '__main__':
    # Basic test: try to initialize the feature extractor
    try:
        print("Attempting to initialize HandFeatureExtractor...")
        hand_extractor = HandFeatureExtractor()
        print("HandFeatureExtractor initialized successfully.")
        # hand_extractor.hands.close() # Important to close MediaPipe Hands when done, will be handled in a release method later
    except Exception as e:
        print(f"An error occurred during HandFeatureExtractor initialization: {e}")
    
    # A simple test for this new class:
    try:
        print("Attempting to initialize BodyPoseExtractor...")
        body_extractor = BodyPoseExtractor()
        print("BodyPoseExtractor initialized successfully.")
        # body_extractor.pose.close() # To be handled by a proper close method
    except Exception as e:
        print(f"An error occurred during BodyPoseExtractor initialization: {e}")

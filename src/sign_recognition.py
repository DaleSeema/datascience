import numpy as np # For type hinting feature_sequence
from typing import List, Optional, Dict, Any # 'Any' for raw MediaPipe results for now
from dataclasses import dataclass, field # To define FrameFeatures


# --- Landmark Selection Constants ---
# For hands (21 landmarks), we'll use all.

# For pose (33 landmarks), let's select some upper body and arm landmarks.
POSE_LANDMARK_INDICES = [
    0,  # Nose
    11, 12,  # Shoulders
    13, 14,  # Elbows
    15, 16,  # Wrists
    23, 24,  # Hips
] # Total: 8 landmarks 

# For face: select a fixed number of initial landmarks.
NUM_FACE_LANDMARKS_TO_USE = 50 

# Constants for number of coordinates per landmark type
HAND_COORDS = 3 # x, y, z
POSE_COORDS = 4 # x, y, z, visibility
FACE_COORDS = 3 # x, y, z


# Data Structure for Frame Features (remains the same)
@dataclass
class FrameFeatures:
    """
    Holds all extracted features for a single frame.
    Each feature list contains dictionaries of x, y, z (and visibility for pose).
    - hand_0_landmarks: Landmarks for the first detected hand.
    - hand_1_landmarks: Landmarks for the second detected hand (optional).
    - pose_landmarks: Landmarks for body pose.
    - face_landmarks: Landmarks for the face mesh.
    All are optional and default to None or empty list if not detected.
    """
    hand_0_landmarks: Optional[List[Dict[str, float]]] = field(default_factory=list)
    hand_1_landmarks: Optional[List[Dict[str, float]]] = field(default_factory=list)
    pose_landmarks: Optional[List[Dict[str, float]]] = field(default_factory=list)
    face_landmarks: Optional[List[Dict[str, float]]] = field(default_factory=list)
    # raw_mediapipe_results: Optional[Dict[str, Any]] = field(default_factory=dict) # If needed later


# Function for Feature Aggregation (Updated with Normalization)
def aggregate_frame_features(frame_features_data: FrameFeatures) -> Optional[np.ndarray]:
    """
    Aggregates selected raw landmark data from hands, pose, and face for a single frame
    into a single, flat feature vector. Implements padding for missing/incomplete features
    and normalizes hand and pose landmarks.

    Normalization:
    - Hand landmarks are normalized relative to their respective wrist landmark (index 0).
    - Pose landmarks (selected by POSE_LANDMARK_INDICES) are normalized relative to the
      midpoint of the left and right shoulders.

    Missing or Incomplete Data Handling:
    - If a hand feature set (hand_0_landmarks, hand_1_landmarks) is missing or does not
      contain the expected 21 landmarks, its corresponding section in the output vector
      is zero-padded (21 landmarks * 3 coordinates). This padding occurs *before* normalization
      would have been applied, meaning the padded values are absolute zeros.
    - If pose_landmarks are missing or do not contain the expected 33 landmarks (which are
      required to safely calculate the shoulder midpoint and access selected indices),
      the section for selected pose landmarks is zero-padded (len(POSE_LANDMARK_INDICES) * 4 coordinates).
      Similar to hands, this padding is of absolute zeros.
    - If face_landmarks are missing or do not contain at least NUM_FACE_LANDMARKS_TO_USE (out of an expected 478),
      the section for selected face landmarks is zero-padded (NUM_FACE_LANDMARKS_TO_USE * 3 coordinates).
    This zero-padding strategy ensures a consistent feature vector length for the model.

    Args:
        frame_features_data (FrameFeatures): An object containing all raw landmark data for the frame.

    Returns:
        Optional[np.ndarray]: A flat NumPy array representing the combined feature vector
                              for the frame.
    """
    aggregated_features = []
    
    # --- Hand 0 Normalization (relative to its own wrist - landmark 0) ---
    hand_0_origin_landmark_index = 0 # Wrist
    if frame_features_data.hand_0_landmarks and len(frame_features_data.hand_0_landmarks) == 21:
        origin_h0 = frame_features_data.hand_0_landmarks[hand_0_origin_landmark_index]
        origin_h0_x = origin_h0.get('x', 0.0)
        origin_h0_y = origin_h0.get('y', 0.0)
        origin_h0_z = origin_h0.get('z', 0.0)
        for i in range(21): # All 21 landmarks
            landmark = frame_features_data.hand_0_landmarks[i]
            aggregated_features.append(landmark.get('x', 0.0) - origin_h0_x)
            aggregated_features.append(landmark.get('y', 0.0) - origin_h0_y)
            aggregated_features.append(landmark.get('z', 0.0) - origin_h0_z)
    else:
        # Current strategy: Zero-padding for missing or incomplete hand data.
        # Alternatives for missing data:
        # 1. Mean imputation (replace with average landmark values from a training set).
        # 2. Special flag values (e.g., -1, but requires model to handle them).
        # 3. More complex imputation techniques (e.g., KNN imputation).
        # For now, zero-padding ensures fixed vector length and simplicity.
        # Note: Since normalization is relative, padding with zeros here means these
        #       features effectively represent the origin if normalization had occurred.
        aggregated_features.extend([0.0] * (21 * HAND_COORDS))

    # --- Hand 1 Normalization (relative to its own wrist - landmark 0) ---
    hand_1_origin_landmark_index = 0 # Wrist
    if frame_features_data.hand_1_landmarks and len(frame_features_data.hand_1_landmarks) == 21:
        origin_h1 = frame_features_data.hand_1_landmarks[hand_1_origin_landmark_index]
        origin_h1_x = origin_h1.get('x', 0.0)
        origin_h1_y = origin_h1.get('y', 0.0)
        origin_h1_z = origin_h1.get('z', 0.0)
        for i in range(21): # All 21 landmarks
            landmark = frame_features_data.hand_1_landmarks[i]
            aggregated_features.append(landmark.get('x', 0.0) - origin_h1_x)
            aggregated_features.append(landmark.get('y', 0.0) - origin_h1_y)
            aggregated_features.append(landmark.get('z', 0.0) - origin_h1_z)
    else:
        # Current strategy: Zero-padding for missing or incomplete hand data.
        # See comments in Hand 0 section for alternatives.
        aggregated_features.extend([0.0] * (21 * HAND_COORDS))

    # --- Pose Normalization (selected landmarks relative to shoulder midpoint) ---
    if frame_features_data.pose_landmarks and len(frame_features_data.pose_landmarks) == 33:
        left_shoulder = frame_features_data.pose_landmarks[11] 
        right_shoulder = frame_features_data.pose_landmarks[12]
           
        shoulder_mid_x = (left_shoulder.get('x', 0.0) + right_shoulder.get('x', 0.0)) / 2
        shoulder_mid_y = (left_shoulder.get('y', 0.0) + right_shoulder.get('y', 0.0)) / 2
        shoulder_mid_z = (left_shoulder.get('z', 0.0) + right_shoulder.get('z', 0.0)) / 2

        for index in POSE_LANDMARK_INDICES:
            landmark = frame_features_data.pose_landmarks[index]
            aggregated_features.append(landmark.get('x', 0.0) - shoulder_mid_x)
            aggregated_features.append(landmark.get('y', 0.0) - shoulder_mid_y)
            aggregated_features.append(landmark.get('z', 0.0) - shoulder_mid_z)
            aggregated_features.append(landmark.get('visibility', 0.0)) # Keep visibility as is
    else:
        # Current strategy: Zero-padding for missing or incomplete pose data.
        # This applies if the full 33 landmarks are not available, which are needed
        # for reliable shoulder midpoint calculation and indexing.
        # Alternatives are similar to those for hand data (mean imputation, flags, etc.).
        # Zero-padding ensures a fixed vector length.
        aggregated_features.extend([0.0] * (len(POSE_LANDMARK_INDICES) * POSE_COORDS))

    # --- Face (first NUM_FACE_LANDMARKS_TO_USE landmarks, no normalization in this step) ---
    # MediaPipe FaceMesh with refine_landmarks=True provides 478 landmarks.
    if frame_features_data.face_landmarks and len(frame_features_data.face_landmarks) == 478:
        for i in range(NUM_FACE_LANDMARKS_TO_USE):
            landmark = frame_features_data.face_landmarks[i]
            aggregated_features.extend([landmark.get('x', 0), landmark.get('y', 0), landmark.get('z', 0)])
    else:
        # Current strategy: Zero-padding for missing or incomplete face data.
        # This applies if the expected 478 landmarks (from which the first
        # NUM_FACE_LANDMARKS_TO_USE are taken) are not available.
        # Alternatives are similar to other feature types.
        # Zero-padding maintains fixed vector length.
        aggregated_features.extend([0.0] * (NUM_FACE_LANDMARKS_TO_USE * FACE_COORDS))
       
    return np.array(aggregated_features, dtype=np.float32)


# Class for Buffering Feature Sequences (remains the same)
class FeatureSequenceBuffer:
    def __init__(self, sequence_length: int = 30, buffer_overlap: int = 15):
        if sequence_length <= 0:
            raise ValueError("Sequence length must be positive.")
        if buffer_overlap < 0 or buffer_overlap >= sequence_length:
            raise ValueError("Buffer overlap must be non-negative and less than sequence length.")
        self.sequence_length = sequence_length
        self.buffer_overlap = buffer_overlap
        self.feature_buffer: List[np.ndarray] = []

    def add_frame_features(self, aggregated_features: Optional[np.ndarray]) -> Optional[List[np.ndarray]]:
        if aggregated_features is None:
            return None 
        self.feature_buffer.append(aggregated_features)
        if len(self.feature_buffer) == self.sequence_length:
            current_sequence = list(self.feature_buffer) 
            if self.buffer_overlap > 0:
                self.feature_buffer = self.feature_buffer[-self.buffer_overlap:]
            else:
                self.feature_buffer = []
            return current_sequence
        return None 

    def get_current_buffer_state(self) -> List[np.ndarray]:
        return list(self.feature_buffer)

    def clear_buffer(self) -> None:
        self.feature_buffer = []
        print("FeatureSequenceBuffer cleared.")


class SignRecognitionModel:
    def __init__(self, model_path: Optional[str] = None, num_expected_features: Optional[int] = None):
        self.model_path = model_path
        self.num_expected_features = num_expected_features
        self.model = None 
        if self.model_path:
            self.load_model(self.model_path)
        print(f"SignRecognitionModel initialized. Model path: {self.model_path}, Expected features: {self.num_expected_features}")

    def load_model(self, model_path: str) -> None:
        print(f"Attempting to load model from: {model_path}")
        if "dummy_model" in model_path: 
            self.model = "dummy_trained_model_object" 
            print("Dummy model loaded successfully.")
        else:
            print("Model loading not implemented yet, or model path is not a dummy path.")

    def predict(self, feature_sequence: List[np.ndarray]) -> Optional[str]:
        if not feature_sequence:
            print("Warning: Empty feature sequence received for prediction.")
            return None
        if self.num_expected_features and feature_sequence and len(feature_sequence[0]) != self.num_expected_features:
            print(f"Warning: Feature vector dimension mismatch. Expected {self.num_expected_features}, got {len(feature_sequence[0])}")
        print(f"Predicting sign for a sequence of {len(feature_sequence)} frames...")
        if self.model: 
            if len(feature_sequence) > 10: 
                return "DUMMY_SIGN_A"
            else:
                return "DUMMY_SIGN_B"
        else:
            print("Sign prediction not implemented yet without a loaded model. Returning default.")
            return "NO_MODEL_DUMMY_SIGN"


if __name__ == '__main__':
    print("Testing SignRecognitionModel...")
    print("\nTest 1: Initialize without model path")
    model_no_path = SignRecognitionModel(num_expected_features=100) 
    dummy_sequence_1 = [np.random.rand(100) for _ in range(15)] 
    prediction_1 = model_no_path.predict(dummy_sequence_1)
    print(f"Prediction 1: {prediction_1}") 

    print("\nTest 2: Initialize with a dummy model path")
    model_dummy_path = SignRecognitionModel(model_path="dummy_model_directory/model.pth", num_expected_features=50)
    dummy_sequence_2 = [np.random.rand(50) for _ in range(5)] 
    prediction_2 = model_dummy_path.predict(dummy_sequence_2)
    print(f"Prediction 2: {prediction_2}") 
    dummy_sequence_3 = [np.random.rand(50) for _ in range(20)] 
    prediction_3 = model_dummy_path.predict(dummy_sequence_3)
    print(f"Prediction 3: {prediction_3}") 
    
    print("\nTest 3: Initialize with a non-dummy model path")
    model_other_path = SignRecognitionModel(model_path="actual_model_path/model.hf", num_expected_features=75)
    dummy_sequence_4 = [np.random.rand(75) for _ in range(10)]
    prediction_4 = model_other_path.predict(dummy_sequence_4)
    print(f"Prediction 4: {prediction_4}") 
    print("\nSignRecognitionModel tests complete.")

    print("\nTesting FrameFeatures and aggregation...")
    expected_len_selected = (21 * HAND_COORDS) + \
                            (21 * HAND_COORDS) + \
                            (len(POSE_LANDMARK_INDICES) * POSE_COORDS) + \
                            (NUM_FACE_LANDMARKS_TO_USE * FACE_COORDS)
    print(f"Expected length (selected landmarks): {expected_len_selected}")

    # Test with full data
    ff_data_test_full = FrameFeatures(
        hand_0_landmarks=[{'x': 0.5, 'y': 0.5, 'z': 0.1, 'visibility': 1.0}] * 21, # Wrist at 0.5,0.5,0.1
        hand_1_landmarks=[{'x': 1.5, 'y': 1.5, 'z': 0.2, 'visibility': 1.0}] * 21, # Wrist at 1.5,1.5,0.2
        pose_landmarks=[{'x': 0.0, 'y': 0.0, 'z': 0.0, 'visibility': 1.0}] * 33,   # Shoulders will be at 0,0,0
        face_landmarks=[{'x': 0.7, 'y': 0.7, 'z': 0.3, 'visibility': 1.0}] * 478
    )
    # Adjust specific shoulder landmarks for pose normalization test
    ff_data_test_full.pose_landmarks[11] = {'x': -0.1, 'y': 0.2, 'z': 0.0, 'visibility': 1.0} # Left shoulder
    ff_data_test_full.pose_landmarks[12] = {'x': 0.1, 'y': 0.2, 'z': 0.0, 'visibility': 1.0}  # Right shoulder
    # Shoulder midpoint will be (0, 0.2, 0)

    aggregated_vector_test = aggregate_frame_features(ff_data_test_full)
    if aggregated_vector_test is not None:
        print(f"Test aggregated vector length (with selection & normalization): {len(aggregated_vector_test)}")
        if len(aggregated_vector_test) == expected_len_selected:
            print("Aggregated vector length matches new expected length.")
            # Check a few normalized values conceptually
            # Hand 0, first landmark (wrist) should be 0,0,0
            print(f"  Hand 0, Wrist (normalized): {aggregated_vector_test[0:3]}") 
            # Pose, first selected landmark (Nose, index 0) relative to shoulder mid (0, 0.2, 0)
            # If Nose was at (0,0,0), it should become (0, -0.2, 0) + visibility
            nose_original = ff_data_test_full.pose_landmarks[0]
            print(f"  Pose, Nose (normalized x,y,z): {aggregated_vector_test[126:129]}, Original Nose: ({nose_original.get('x')}, {nose_original.get('y')}, {nose_original.get('z')})")
        else:
            print(f"Error: Aggregated vector length ({len(aggregated_vector_test)}) does NOT match new expected length ({expected_len_selected}).")
    else:
        print("Test aggregation returned None, which is unexpected with current padding and full data.")

    # Test with missing data
    ff_data_test_missing = FrameFeatures(
        hand_0_landmarks=[{'x': 0.1, 'y': 0.1, 'z': 0.1}] * 21,
        pose_landmarks=[{'x':0.3, 'y':0.3, 'z':0.3, 'visibility':1.0}] * 33
    )
    aggregated_vector_missing_test = aggregate_frame_features(ff_data_test_missing)
    print(f"Expected length (with padding for missing): {expected_len_selected}")
    if aggregated_vector_missing_test is not None:
        print(f"Test aggregated vector length (with missing data): {len(aggregated_vector_missing_test)}")
        if len(aggregated_vector_missing_test) == expected_len_selected:
            print("Aggregated vector length with missing data matches new expected length (due to padding).")
        else:
            print(f"Error: Aggregated vector length with missing data ({len(aggregated_vector_missing_test)}) does NOT match new expected length ({expected_len_selected}).")
    else:
        print("Test aggregation with missing data returned None, which is unexpected with current padding.")

    print("\nTesting FeatureSequenceBuffer...")
    buffer = FeatureSequenceBuffer(sequence_length=5, buffer_overlap=2)
    test_features = [np.array([i]*(expected_len_selected)) for i in range(10)] 
    print(f"Buffer settings: length={buffer.sequence_length}, overlap={buffer.buffer_overlap}")
    for i, feat in enumerate(test_features):
        print(f"Adding feature {i} (length {len(feat)})...")
        sequence = buffer.add_frame_features(feat)
        if sequence:
            print(f"  >> Sequence formed: {len(sequence)} frames. First feat of seq starts with: {sequence[0][:2]}...") 
        current_state = buffer.get_current_buffer_state()
        print(f"  Buffer state: {len(current_state)} frames. Last feat in buffer starts with: {current_state[-1][:2] if current_state else 'Empty'}")
    
    print("Clearing buffer...")
    buffer.clear_buffer()
    current_state = buffer.get_current_buffer_state()
    print(f"Buffer state after clear: {len(current_state)} frames. {'Empty' if not current_state else 'Error: Not Empty'}")

import cv2
import numpy as np # Ensure numpy is imported
from video_capture import VideoCapture
from feature_extraction import HandFeatureExtractor, BodyPoseExtractor, FaceMeshExtractor
# New imports for sign recognition components
from sign_recognition import FrameFeatures, aggregate_frame_features, FeatureSequenceBuffer, SignRecognitionModel

def run_live_feed():
    video_source_index = 0
    video_stream = None
    hand_extractor = None
    body_extractor = None
    face_extractor = None
    # Sign recognition components
    sequence_buffer = None
    sign_model = None

    try:
        video_stream = VideoCapture(video_source_index)
        hand_extractor = HandFeatureExtractor()
        body_extractor = BodyPoseExtractor()
        face_extractor = FaceMeshExtractor()
        
        # Initialize sign recognition components
        # The num_expected_features must match the output of aggregate_frame_features
        # The num_expected_features must match the output of aggregate_frame_features in sign_recognition.py
        # Current calculation based on selected landmarks:
        # Hand 0 (21 landmarks * 3 coords) = 63
        # Hand 1 (21 landmarks * 3 coords) = 63
        # Pose (8 selected landmarks * 4 coords) = 32
        # Face (50 selected landmarks * 3 coords) = 150
        # Total = 63 + 63 + 32 + 150 = 408
        AGGREGATED_FEATURE_LENGTH = 408
        sequence_buffer = FeatureSequenceBuffer(sequence_length=30, buffer_overlap=15) # e.g., 30 frames, 15 overlap
        sign_model = SignRecognitionModel(num_expected_features=AGGREGATED_FEATURE_LENGTH,
                                          model_path="dummy_model_directory/model.pth") # Optionally test dummy loading

        print("All components initialized successfully.")

    except Exception as e:
        print(f"An error occurred during initialization: {e}")
        # Perform cleanup for any components that were successfully initialized
        if video_stream: video_stream.release()
        if hand_extractor: hand_extractor.close()
        if body_extractor: body_extractor.close()
        if face_extractor: face_extractor.close()
        return

    print("Displaying live feed with all landmarks and conceptual sign prediction. Press 'q' to quit.")
    feature_sequence_for_prediction = None # Initialize outside the condition for clarity
    while True:
        success, frame = video_stream.get_frame()
        if not success or frame is None:
            print("Failed to retrieve frame or end of stream. Exiting.")
            break

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # --- Feature Extraction ---
        hand_landmarks_data, hand_results = hand_extractor.extract_features(image_rgb)
        pose_landmarks_data, pose_results = body_extractor.extract_features(image_rgb)
        face_landmarks_data, face_results = face_extractor.extract_features(image_rgb)

        # --- Feature Aggregation & Sequencing ---
        current_frame_obj = FrameFeatures()
        if hand_landmarks_data: # hand_landmarks_data is list of hands; each hand is list of dicts
            if len(hand_landmarks_data) > 0:
                current_frame_obj.hand_0_landmarks = hand_landmarks_data[0]
            if len(hand_landmarks_data) > 1:
                current_frame_obj.hand_1_landmarks = hand_landmarks_data[1]
        
        current_frame_obj.pose_landmarks = pose_landmarks_data if pose_landmarks_data else []
        current_frame_obj.face_landmarks = face_landmarks_data if face_landmarks_data else []
        
        aggregated_vector = aggregate_frame_features(current_frame_obj)
        
        predicted_sign_text = "SIGN: ..." # Default text
        if aggregated_vector is not None:
            # Check vector length if num_expected_features was set in model
            if sign_model.num_expected_features and len(aggregated_vector) != sign_model.num_expected_features:
                print(f"Warning: Aggregated vector length ({len(aggregated_vector)}) does not match model's expected ({sign_model.num_expected_features}). Prediction might be unreliable.")

            feature_sequence_for_prediction = sequence_buffer.add_frame_features(aggregated_vector)
            
            if feature_sequence_for_prediction:
                predicted_sign = sign_model.predict(feature_sequence_for_prediction)
                if predicted_sign:
                    # print(f"Predicted Sign: {predicted_sign}") # Print to console
                    predicted_sign_text = f"SIGN: {predicted_sign}" 
        
        # --- Drawing Landmarks (on BGR frame) ---
        hand_extractor.draw_landmarks_on_image(frame, hand_results)
        body_extractor.draw_landmarks_on_image(frame, pose_results)
        face_extractor.draw_landmarks_on_image(frame, face_results)
        
        # Display predicted sign on frame
        cv2.putText(frame, predicted_sign_text, (10, frame.shape[0] - 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow("Live Feed - All Landmarks & Sign Prediction - Press 'q' to quit", frame)

        key_press = cv2.waitKey(1) & 0xFF
        if key_press == ord('q'):
            print("'q' key pressed. Exiting live feed.")
            break

    print("Releasing resources...")
    if video_stream: video_stream.release()
    if hand_extractor: hand_extractor.close()
    if body_extractor: body_extractor.close()
    if face_extractor: face_extractor.close()
    # No close method for SignRecognitionModel or FeatureSequenceBuffer as defined
    cv2.destroyAllWindows()
    print("Video feed stopped and resources released.")

if __name__ == "__main__":
    run_live_feed()

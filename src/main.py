import cv2
import numpy as np
from video_capture import VideoCapture
from feature_extraction import HandFeatureExtractor, BodyPoseExtractor, FaceMeshExtractor
from sign_recognition import FrameFeatures, aggregate_frame_features, FeatureSequenceBuffer, SignRecognitionModel
from tts import TextToSpeech # <<< NEW IMPORT

def run_live_feed():
    video_source_index = 0
    video_stream = None
    hand_extractor = None
    body_extractor = None
    face_extractor = None
    sequence_buffer = None
    sign_model = None
    tts_engine = None # <<< INITIALIZE TTS ENGINE VARIABLE

    try:
        video_stream = VideoCapture(video_source_index)
        hand_extractor = HandFeatureExtractor()
        body_extractor = BodyPoseExtractor()
        face_extractor = FaceMeshExtractor()
        
        AGGREGATED_FEATURE_LENGTH = 408 # From previous step
        sequence_buffer = FeatureSequenceBuffer(sequence_length=30, buffer_overlap=15)
        # Initialize SignRecognitionModel without a dummy model path for now to avoid "dummy model loaded" prints
        # We'll rely on its num_expected_features for the warning logic if needed.
        sign_model = SignRecognitionModel(num_expected_features=AGGREGATED_FEATURE_LENGTH) 
        
        tts_engine = TextToSpeech() # <<< INITIALIZE TTS ENGINE
        print("All components initialized successfully (including TTS).")

    except Exception as e:
        print(f"An error occurred during initialization: {e}")
        if video_stream: video_stream.release()
        if hand_extractor: hand_extractor.close()
        if body_extractor: body_extractor.close()
        if face_extractor: face_extractor.close()
        # No explicit close for tts_engine needed here based on current tts.py
        return

    print("Displaying live feed with all landmarks and conceptual sign prediction/speech. Press 'q' to quit.")
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
        if hand_landmarks_data:
            if len(hand_landmarks_data) > 0: current_frame_obj.hand_0_landmarks = hand_landmarks_data[0]
            if len(hand_landmarks_data) > 1: current_frame_obj.hand_1_landmarks = hand_landmarks_data[1]
        current_frame_obj.pose_landmarks = pose_landmarks_data if pose_landmarks_data else []
        current_frame_obj.face_landmarks = face_landmarks_data if face_landmarks_data else []
        aggregated_vector = aggregate_frame_features(current_frame_obj)
        
        predicted_sign_text_display = "SIGN: ..." # Default for display
        
        if aggregated_vector is not None:
            if sign_model.num_expected_features and len(aggregated_vector) != sign_model.num_expected_features:
                print(f"Warning: Aggregated vector length ({len(aggregated_vector)}) does not match model's expected ({sign_model.num_expected_features}). Prediction might be unreliable.")
                pass # Continue, but prediction may be off
            feature_sequence_for_prediction = sequence_buffer.add_frame_features(aggregated_vector)
            
            if feature_sequence_for_prediction:
                predicted_sign = sign_model.predict(feature_sequence_for_prediction)
                if predicted_sign:
                    predicted_sign_text_display = f"SIGN: {predicted_sign}"
                    # >>> SPEAK THE PREDICTED SIGN <<<
                    if tts_engine and tts_engine.engine: # Check if tts_engine was successfully initialized
                        print(f"Speaking: {predicted_sign}") # Log to console
                        tts_engine.speak(predicted_sign) 
                    # <<< END SPEAKING BLOCK >>>
        
        # --- Drawing Landmarks ---
        hand_extractor.draw_landmarks_on_image(frame, hand_results)
        body_extractor.draw_landmarks_on_image(frame, pose_results)
        face_extractor.draw_landmarks_on_image(frame, face_results)
        
        cv2.putText(frame, predicted_sign_text_display, (10, frame.shape[0] - 30), 
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
    # No explicit close for tts_engine needed here
    cv2.destroyAllWindows()
    print("Video feed stopped and resources released.")

if __name__ == "__main__":
    run_live_feed()

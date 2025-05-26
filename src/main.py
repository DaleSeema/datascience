import cv2
from video_capture import VideoCapture
# Updated import to include FaceMeshExtractor
from feature_extraction import HandFeatureExtractor, BodyPoseExtractor, FaceMeshExtractor 

def run_live_feed():
    video_source_index = 0
    video_stream = None
    hand_extractor = None
    body_extractor = None
    face_extractor = None # Initialize to None

    try:
        video_stream = VideoCapture(video_source_index)
        print(f"Successfully opened video source: {video_source_index}")
    except IOError as e:
        print(f"Error: Could not open video source {video_source_index}. {e}")
        return
    except Exception as e:
        print(f"An unexpected error occurred during camera initialization: {e}")
        return

    try:
        hand_extractor = HandFeatureExtractor()
        print("HandFeatureExtractor initialized successfully.")
    except Exception as e:
        print(f"An unexpected error occurred during HandFeatureExtractor initialization: {e}")
        if video_stream: video_stream.release()
        return

    try:
        body_extractor = BodyPoseExtractor()
        print("BodyPoseExtractor initialized successfully.")
    except Exception as e:
        print(f"An unexpected error occurred during BodyPoseExtractor initialization: {e}")
        if video_stream: video_stream.release()
        if hand_extractor: hand_extractor.close()
        return

    try:
        face_extractor = FaceMeshExtractor() # Initialize FaceMeshExtractor
        print("FaceMeshExtractor initialized successfully.")
    except Exception as e:
        print(f"An unexpected error occurred during FaceMeshExtractor initialization: {e}")
        if video_stream: video_stream.release()
        if hand_extractor: hand_extractor.close()
        if body_extractor: body_extractor.close()
        return

    print("Displaying live feed with hand, pose, and face landmark detection. Press 'q' to quit.")
    while True:
        success, frame = video_stream.get_frame()

        if not success:
            print("Failed to retrieve frame from the camera. Exiting.")
            break
        
        if frame is None:
            print("Retrieved a None frame despite success flag. Exiting.")
            break

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Hand landmark processing
        hand_landmarks_data, hand_results = hand_extractor.extract_features(image_rgb)
        if hand_landmarks_data:
            pass # Optional print/debug
        hand_extractor.draw_landmarks_on_image(frame, hand_results)

        # Body pose landmark processing
        pose_landmarks_data, pose_results = body_extractor.extract_features(image_rgb)
        if pose_landmarks_data:
            pass # Optional print/debug
        body_extractor.draw_landmarks_on_image(frame, pose_results)

        # Face mesh landmark processing
        face_landmarks_data, face_results = face_extractor.extract_features(image_rgb)
        if face_landmarks_data:
            pass # Optional print/debug
        face_extractor.draw_landmarks_on_image(frame, face_results) # Draw on BGR frame

        cv2.imshow("Live Feed - All Landmarks - Press 'q' to quit", frame) # Updated window title

        key_press = cv2.waitKey(1) & 0xFF
        if key_press == ord('q'):
            print("'q' key pressed. Exiting live feed.")
            break

    print("Releasing resources...")
    if video_stream: video_stream.release()
    if hand_extractor: hand_extractor.close()
    if body_extractor: body_extractor.close()
    if face_extractor: face_extractor.close() # Close FaceMeshExtractor
    cv2.destroyAllWindows()
    print("Video feed stopped and resources released.")

if __name__ == "__main__":
    run_live_feed()

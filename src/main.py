import cv2
from video_capture import VideoCapture
from feature_extraction import HandFeatureExtractor # Added import

def run_live_feed():
    video_source_index = 0
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
        hand_extractor = HandFeatureExtractor() # Initialize HandFeatureExtractor
        print("HandFeatureExtractor initialized successfully.")
    except Exception as e:
        print(f"An unexpected error occurred during HandFeatureExtractor initialization: {e}")
        video_stream.release() # Release video stream if hand extractor fails
        return

    print("Displaying live feed with hand landmark detection. Press 'q' to quit.")
    while True:
        success, frame = video_stream.get_frame()

        if not success:
            print("Failed to retrieve frame from the camera. Exiting.")
            break
        
        if frame is None:
            print("Retrieved a None frame despite success flag. Exiting.")
            break

        # Convert the BGR image to RGB for MediaPipe.
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the image and extract hand landmarks.
        # `landmarks_data` will be our list of lists of dicts if hands are found.
        # `results` is the raw MediaPipe results object, useful for drawing.
        landmarks_data, results = hand_extractor.extract_features(image_rgb)

        if landmarks_data:
            # For debugging, print the number of hands detected or some landmark info
            # print(f"Detected {len(landmarks_data)} hand(s).")
            # For example, to print x,y of the wrist of the first detected hand:
            # if len(landmarks_data[0]) > 0:
            #     print(f"Hand 1, Wrist (x,y): {landmarks_data[0][0]['x']:.2f}, {landmarks_data[0][0]['y']:.2f}")
            pass # Keep console clean for now, can uncomment above for debugging

        # Draw the hand annotations on the original BGR frame.
        hand_extractor.draw_landmarks_on_image(frame, results)

        cv2.imshow("Live Video Feed - Hand Landmarks - Press 'q' to quit", frame)

        key_press = cv2.waitKey(1) & 0xFF
        if key_press == ord('q'):
            print("'q' key pressed. Exiting live feed.")
            break

    # Cleanup
    print("Releasing resources...")
    video_stream.release()
    hand_extractor.close() # Close MediaPipe Hands
    cv2.destroyAllWindows()
    print("Video feed stopped and resources released.")

if __name__ == "__main__":
    run_live_feed()

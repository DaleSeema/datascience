import cv2
# This import assumes that when main.py is run, the 'src' directory is effectively
# the current working directory or on the PYTHONPATH, so it can find video_capture.py
# If main.py is run as 'python src/main.py' from the project root, this import will work.
from video_capture import VideoCapture 

def run_live_feed():
    video_source_index = 0 # Default camera
    try:
        video_stream = VideoCapture(video_source_index)
        print(f"Successfully opened video source: {video_source_index}")
    except IOError as e:
        print(f"Error: Could not open video source {video_source_index}. {e}")
        print("Please check your camera connection and ensure the source index is correct.")
        return
    except Exception as e:
        print(f"An unexpected error occurred during camera initialization: {e}")
        return

    print("Displaying live feed. Press 'q' to quit.")
    while True:
        success, frame = video_stream.get_frame()

        if not success:
            print("Failed to retrieve frame from the camera. Exiting.")
            break
        
        if frame is None: # Should not happen if success is True, but good for robustness
            print("Retrieved a None frame despite success flag. Exiting.")
            break

        cv2.imshow("Live Video Feed - Press 'q' to quit", frame)

        # Wait for a key press for 1 millisecond.
        # If 'q' is pressed, exit the loop.
        key_press = cv2.waitKey(1) & 0xFF
        if key_press == ord('q'):
            print("'q' key pressed. Exiting live feed.")
            break
        # Optional: handle other key presses or events here

    # Cleanup
    video_stream.release()
    cv2.destroyAllWindows()
    print("Video feed stopped and resources released.")

if __name__ == "__main__":
    run_live_feed()

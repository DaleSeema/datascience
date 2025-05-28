import cv2
import numpy as np # Import numpy for type hinting, though not directly used in this step

class VideoCapture:
    def __init__(self, source: int = 0):
        """
        Initializes the video capture object.

        Args:
            source (int): The camera source index (default is 0 for the default camera).
        """
        self.source = source
        self.cap = cv2.VideoCapture(self.source)

        if not self.cap.isOpened():
            raise IOError(f"Cannot open video source: {self.source}")
        else:
            print(f"Video source {self.source} opened successfully.")
    
    def get_frame(self) -> tuple[bool, np.ndarray | None]:
        """
        Reads a single frame from the video source.

        Returns:
            tuple[bool, np.ndarray | None]: A tuple containing:
                - success (bool): True if a frame was successfully read, False otherwise.
                - frame (np.ndarray | None): The captured frame as a NumPy array if successful, 
                                             None otherwise.
        """
        if self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                return True, frame
            else:
                print("Warning: Failed to retrieve frame.")
                return False, None
        else:
            print("Warning: Video source is not open. Cannot get frame.")
            return False, None

    def release(self) -> None:
        """
        Releases the video capture object.
        """
        if self.cap.isOpened():
            self.cap.release()
            print(f"Video source {self.source} released.")

    def __del__(self) -> None:
        """
        Destructor to ensure the video capture is released when the object is deleted.
        """
        self.release()

if __name__ == '__main__':
    # Basic test: try to initialize the camera
    try:
        print("Attempting to initialize default camera...")
        cap_test = VideoCapture(0)
        print("VideoCapture initialized. Releasing camera.")
        cap_test.cap.release() # Directly release for this basic test
        print("Camera released.")
    except IOError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

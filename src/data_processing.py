import os
import cv2 # For VideoCapture, though not directly used in constructor
import numpy as np
import pandas as pd # For reading metadata TSV later
from typing import List, Optional, Dict, Any

# Assuming these are in src/ and correctly importable
from .feature_extraction import HandFeatureExtractor, BodyPoseExtractor, FaceMeshExtractor
from .sign_recognition import FrameFeatures, aggregate_frame_features
# Note: FeatureSequenceBuffer might not be directly used here if we save per-clip sequences directly

# Constants for feature aggregation (should match what's in sign_recognition.py for consistency)
# These could also be imported or configured if they become more dynamic.
HAND_COORDS = 3
POSE_COORDS = 4
FACE_COORDS = 3
POSE_LANDMARK_INDICES = [0, 11, 12, 13, 14, 15, 16, 23, 24] # 8 landmarks
NUM_FACE_LANDMARKS_TO_USE = 50

class DatasetProcessor:
    def __init__(self,
                 processed_clips_dir: str,
                 metadata_tsv_path: str,
                 features_output_dir: str,
                 feature_extractor_config: Optional[Dict[str, Any]] = None):
        """
        Initializes the DatasetProcessor.

        Args:
            processed_clips_dir (str): Path to the directory containing processed video clips.
            metadata_tsv_path (str): Path to the OpenASL metadata TSV file.
            features_output_dir (str): Path to the directory where extracted features will be saved.
            feature_extractor_config (Optional[Dict[str, Any]]): Configuration for feature extractors
                                                                 if needed (e.g., confidence thresholds).
        """
        self.processed_clips_dir = processed_clips_dir
        self.metadata_tsv_path = metadata_tsv_path
        self.features_output_dir = features_output_dir

        # Ensure output directory exists
        os.makedirs(self.features_output_dir, exist_ok=True)

        # Initialize feature extractors
        # For now, use default configurations. Config could be passed for more flexibility.
        config = feature_extractor_config if feature_extractor_config else {}
        try:
            self.hand_extractor = HandFeatureExtractor(**config.get('hands', {}))
            self.pose_extractor = BodyPoseExtractor(**config.get('pose', {}))
            self.face_extractor = FaceMeshExtractor(**config.get('face', {})) # refine_landmarks=True is default
            print("Feature extractors (Hand, Pose, Face) initialized successfully.")
        except Exception as e:
            print(f"Error initializing feature extractors: {e}")
            # Depending on desired robustness, might re-raise or handle
            raise  # Re-raise for now, as processing can't continue without them

        # Placeholder for metadata, will be loaded in a separate method
        self.df_metadata: Optional[pd.DataFrame] = None
        
        print(f"DatasetProcessor initialized.")
        print(f"  Processed clips input dir: {self.processed_clips_dir}")
        print(f"  Metadata TSV path: {self.metadata_tsv_path}")
        print(f"  Features output dir: {self.features_output_dir}")

    def run_feature_extraction_pipeline(self, max_videos_to_process: Optional[int] = None) -> None:
        """
        Runs the full feature extraction pipeline for the dataset.
        It loads metadata, iterates through video clips, processes each clip,
        and saves the extracted features and labels.

        Args:
            max_videos_to_process (Optional[int]): If set, limits the number of videos
                                                   processed from the dataset. Useful for testing.
        """
        print("Starting feature extraction pipeline...")
        if not self.load_metadata() or self.df_metadata is None:
            print("Halting pipeline: Metadata not loaded successfully.")
            return

        processed_data_manifest = []
        num_processed = 0
        num_successfully_extracted = 0

        print(f"Iterating through {len(self.df_metadata)} entries in metadata...")
        for index, row in self.df_metadata.iterrows():
            if max_videos_to_process is not None and num_processed >= max_videos_to_process:
                print(f"Reached max_videos_to_process limit: {max_videos_to_process}")
                break
            
            num_processed += 1
            
            video_id_from_meta = row.get('video_id') 
            text_label = row.get('text') # Assuming 'text' column for labels
            # OpenASL specific columns that might be useful for filename construction
            uuid = row.get('uuid')
            name_idx = row.get('name_idx')


            if pd.isna(text_label): 
                 print(f"Warning: Skipping row {index} (video_id: {video_id_from_meta or uuid}) due to missing text_label.")
                 continue

            clip_filename_stem = None
            if uuid is not None and name_idx is not None:
                 # Try to convert name_idx to integer if it's not already, for consistent formatting
                try:
                    clip_filename_stem = f"{uuid}_{int(name_idx)}"
                except ValueError:
                    clip_filename_stem = f"{uuid}_{name_idx}" # Use as is if not convertible
            elif video_id_from_meta is not None: 
                clip_filename_stem = str(video_id_from_meta)
            
            if not clip_filename_stem:
                print(f"Warning: Could not determine filename stem for row {index}. Skipping.")
                continue

            potential_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm'] 
            video_file_path = None
            for ext in potential_extensions:
                path_try = os.path.join(self.processed_clips_dir, clip_filename_stem + ext)
                if os.path.exists(path_try):
                    video_file_path = path_try
                    break
            
            if not video_file_path:
                print(f"Warning: Video file for stem '{clip_filename_stem}' (row {index}) not found in {self.processed_clips_dir} with common extensions. Skipping.")
                continue

            print(f"Processing ({num_processed}/{len(self.df_metadata)}): {video_file_path} -> Label: '{str(text_label)[:30]}...'")
            
            feature_sequence = self.process_video_clip(video_file_path)

            if feature_sequence and len(feature_sequence) > 0:
                feature_file_name = f"{clip_filename_stem}.npy"
                feature_file_path = os.path.join(self.features_output_dir, feature_file_name)
                try:
                    # Ensure feature_sequence is a list of np.ndarray before converting to object array
                    np_feature_sequence = np.array([np.asarray(frame_features, dtype=np.float32) for frame_features in feature_sequence], dtype=object)
                    np.save(feature_file_path, np_feature_sequence)
                    
                    processed_data_manifest.append({
                        'feature_file': feature_file_path,
                        'video_id': clip_filename_stem, 
                        'label': text_label,
                        'num_frames': len(feature_sequence),
                        'feature_dim': feature_sequence[0].shape[0] if feature_sequence and len(feature_sequence[0]) > 0 else 0
                    })
                    num_successfully_extracted += 1
                except Exception as e:
                    print(f"Error saving features for {clip_filename_stem}: {e}")
            else:
                print(f"Warning: No features extracted for {video_file_path}. Skipping save.")

        print(f"\nFeature extraction pipeline finished.")
        print(f"Total entries considered from metadata: {num_processed}") 
        print(f"Successfully extracted features for: {num_successfully_extracted} videos.")

        if processed_data_manifest:
            manifest_df = pd.DataFrame(processed_data_manifest)
            manifest_path = os.path.join(self.features_output_dir, "feature_manifest.csv")
            try:
                manifest_df.to_csv(manifest_path, index=False)
                print(f"Manifest of processed features saved to: {manifest_path}")
            except Exception as e:
                print(f"Error saving manifest: {e}")
        else:
            print("No features were successfully processed to create a manifest.")

    def process_video_clip(self, video_file_path: str) -> Optional[List[np.ndarray]]:
        """
        Processes a single video clip to extract a sequence of aggregated feature vectors.

        Args:
            video_file_path (str): The full path to the video clip file.

        Returns:
            Optional[List[np.ndarray]]: A list of NumPy arrays, where each array is the
                                        aggregated feature vector for a single frame of the clip.
                                        Returns None if the video cannot be opened or processed.
        """
        if not os.path.exists(video_file_path):
            print(f"Warning: Video file not found: {video_file_path}")
            return None

        cap = cv2.VideoCapture(video_file_path)
        if not cap.isOpened():
            print(f"Warning: Could not open video file: {video_file_path}")
            return None

        clip_feature_sequence = []
        frame_count = 0
        try:
            while True:
                success, frame = cap.read()
                if not success:
                    break # End of video or error

                frame_count += 1
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Extract features
                hand_landmarks_data, _ = self.hand_extractor.extract_features(image_rgb)
                pose_landmarks_data, _ = self.pose_extractor.extract_features(image_rgb)
                face_landmarks_data, _ = self.face_extractor.extract_features(image_rgb)
                # The second return value (raw MediaPipe results) is ignored here

                # Populate FrameFeatures object
                current_frame_obj = FrameFeatures()
                if hand_landmarks_data:
                    if len(hand_landmarks_data) > 0:
                        current_frame_obj.hand_0_landmarks = hand_landmarks_data[0]
                    if len(hand_landmarks_data) > 1:
                        current_frame_obj.hand_1_landmarks = hand_landmarks_data[1]
                current_frame_obj.pose_landmarks = pose_landmarks_data if pose_landmarks_data else []
                current_frame_obj.face_landmarks = face_landmarks_data if face_landmarks_data else []
                
                # Aggregate features for the current frame
                aggregated_vector = aggregate_frame_features(current_frame_obj)
                
                if aggregated_vector is not None: # Should always be a vector due to padding
                    clip_feature_sequence.append(aggregated_vector)
                else:
                    # This else block might be redundant if aggregate_frame_features always returns a vector
                    print(f"Warning: aggregate_frame_features returned None for a frame in {video_file_path}. Frame {frame_count}")
                    # Decide on padding strategy if this can happen (e.g., use expected length)
                    # For now, this path should ideally not be hit.

        except Exception as e:
            print(f"An error occurred while processing video {video_file_path}, frame {frame_count}: {e}")
            cap.release()
            return None # Or return clip_feature_sequence for partial results
        finally:
            cap.release()

        if not clip_feature_sequence:
            print(f"Warning: No features extracted from video {video_file_path} (e.g., empty video or all frames failed processing).")
            return None
            
        # print(f"Processed {video_file_path}: {len(clip_feature_sequence)} frames, vector dim: {clip_feature_sequence[0].shape if clip_feature_sequence else 'N/A'}")
        return clip_feature_sequence

    def load_metadata(self) -> bool:
        """
        Loads the video metadata from the TSV file into a pandas DataFrame.
        The TSV file is expected to have columns like 'video_id', 'text', 
        'start_time', 'end_time', etc. The exact column names will depend on
        the OpenASL TSV format. We'll assume common names for now or select all.

        Returns:
            bool: True if metadata was loaded successfully, False otherwise.
        """
        try:
            print(f"Loading metadata from: {self.metadata_tsv_path}")
            # Specify separator as tab.
            # The DtypeWarning from user's previous log suggests issues with column 6.
            # We can try to address it by:
            # 1. Setting low_memory=False (as suggested by warning)
            # 2. Specifying dtypes for problematic columns if known (e.g., {6: str})
            # For now, let's use low_memory=False.
            self.df_metadata = pd.read_csv(self.metadata_tsv_path, sep='\t', low_memory=False)
            
            # Basic validation: Check if DataFrame is empty or essential columns exist
            if self.df_metadata.empty:
                print("Warning: Metadata file loaded an empty DataFrame.")
                return False
            
            # Example: Check for a few expected column names (these might need adjustment based on actual TSV)
            # Common columns in such datasets: 'ID', 'Text', 'StartTime', 'EndTime', 'SignerID', 'SourceURL'
            # The OpenASL TSV seems to have columns like:
            # 'video_id', 'signer_id', 'start_frame', 'end_frame', 'text', 'url', 'split'
            # Let's check for 'video_id' and 'text' as essential ones.
            expected_cols = ['video_id', 'text'] 
            missing_cols = [col for col in expected_cols if col not in self.df_metadata.columns]
            if missing_cols:
                print(f"Warning: Metadata missing essential columns: {missing_cols}. Found columns: {list(self.df_metadata.columns)}")
                # Depending on strictness, might return False or just warn.
                # For now, let's be a bit lenient and just warn if 'video_id' or 'text' is missing,
                # but it will likely cause issues later.
            
            print(f"Metadata loaded successfully. Shape: {self.df_metadata.shape}")
            # print("First 5 rows of metadata:\n", self.df_metadata.head()) # For debugging
            return True
            
        except FileNotFoundError:
            print(f"Error: Metadata file not found at {self.metadata_tsv_path}")
            self.df_metadata = None
            return False
        except pd.errors.EmptyDataError:
            print(f"Error: Metadata file at {self.metadata_tsv_path} is empty.")
            self.df_metadata = None
            return False
        except Exception as e:
            print(f"An unexpected error occurred while loading metadata: {e}")
            self.df_metadata = None
            return False

    def close_extractors(self) -> None:
        """Closes all initialized feature extractors."""
        print("Closing feature extractors...")
        if hasattr(self, 'hand_extractor') and self.hand_extractor:
            self.hand_extractor.close()
        if hasattr(self, 'pose_extractor') and self.pose_extractor:
            self.pose_extractor.close()
        if hasattr(self, 'face_extractor') and self.face_extractor:
            self.face_extractor.close()
        print("Feature extractors closed.")

    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close_extractors()


if __name__ == '__main__':
    print("Testing DatasetProcessor initialization...")
    # Create dummy directories and files for testing
    DUMMY_CLIPS_DIR = "dummy_processed_clips"
    DUMMY_METADATA_FILE = "dummy_metadata.tsv"
    DUMMY_FEATURES_OUT_DIR = "dummy_features_output"

    os.makedirs(DUMMY_CLIPS_DIR, exist_ok=True)
    os.makedirs(DUMMY_FEATURES_OUT_DIR, exist_ok=True)
    # Create a dummy tsv file
    dummy_tsv_content = "video_id\ttext_label\tstart_time\tend_time\n"
    with open(DUMMY_METADATA_FILE, 'w') as f:
        f.write(dummy_tsv_content)

    try:
        # Using 'with' statement to test __enter__ and __exit__ (for closing extractors)
        with DatasetProcessor(
            processed_clips_dir=DUMMY_CLIPS_DIR,
            metadata_tsv_path=DUMMY_METADATA_FILE,
            features_output_dir=DUMMY_FEATURES_OUT_DIR
        ) as processor:
            print("DatasetProcessor initialized within 'with' statement.")
            metadata_loaded_successfully = processor.load_metadata() # <<< CALL NEW METHOD
            print(f"Metadata loaded successfully flag: {metadata_loaded_successfully}")
            if metadata_loaded_successfully and processor.df_metadata is not None:
                print(f"  Number of entries in metadata: {len(processor.df_metadata)}")
                print(f"  Metadata columns: {list(processor.df_metadata.columns)}")
            else:
                print("  Metadata DataFrame is None or loading failed.")

            print("\n--- Testing process_video_clip (with a non-existent file) ---")
            non_existent_video_features = processor.process_video_clip("non_existent_video.mp4")
            if non_existent_video_features is None:
                print("Correctly handled non-existent video: returned None.")
            else:
                print("Error: processing non-existent video did not return None.")
            
            # Note: A more thorough test for process_video_clip would require creating a dummy video file.
            # This can be done with cv2.VideoWriter but adds complexity to this unit test.
            # For now, testing the non-existent file path is a basic check.
            
            print("\n--- Attempting to run feature extraction pipeline (will likely find no real videos in dummy setup) ---")
            # Update dummy_metadata.tsv to have a 'text' column for the test and a plausible video_id
            # This will test the loop and file searching logic.
            # The actual video won't exist, so process_video_clip should handle it.
            if metadata_loaded_successfully and processor.df_metadata is not None:
                # Create a slightly more useful dummy metadata for the pipeline test
                test_pipeline_metadata_content = (
                    "video_id\ttext\tuuid\tname_idx\n"
                    "dummy_video_01\tHELLO WORLD\tdummy_uuid_01\t0\n"
                    "dummy_video_02\tGOODBYE\tdummy_uuid_02\t1\n"
                    "non_existent_video_id\tTESTING MISSING\tnon_existent_uuid\t0\n"
                )
                with open(DUMMY_METADATA_FILE, 'w') as f:
                    f.write(test_pipeline_metadata_content)
                
                # Reload metadata for the pipeline test
                processor.load_metadata() 
                
                # Create one dummy file to test the successful path partially
                dummy_video_file_for_pipeline = os.path.join(DUMMY_CLIPS_DIR, "dummy_uuid_01_0.mp4")
                if not os.path.exists(dummy_video_file_for_pipeline):
                    # Create an empty file, process_video_clip will fail to open it with cv2
                    # but it will pass the os.path.exists check.
                    open(dummy_video_file_for_pipeline, 'a').close() 
                    print(f"Created dummy video file: {dummy_video_file_for_pipeline} for pipeline test.")


                processor.run_feature_extraction_pipeline(max_videos_to_process=3) # Test with a limit

                # Clean up the specific dummy video file created for this test
                if os.path.exists(dummy_video_file_for_pipeline):
                    os.remove(dummy_video_file_for_pipeline)


        print("DatasetProcessor test finished (after 'with' statement, extractors should be closed).")

    except Exception as e:
        print(f"An error occurred during DatasetProcessor test: {e}")
    finally:
        # Clean up dummy files and directories
        if os.path.exists(DUMMY_METADATA_FILE):
            os.remove(DUMMY_METADATA_FILE)
        if os.path.exists(DUMMY_CLIPS_DIR):
            # Note: rmdir only works on empty directories. If tests create files in it, more complex cleanup is needed.
            # For this init test, it should be empty.
            try:
                os.rmdir(DUMMY_CLIPS_DIR)
            except OSError: # If not empty (e.g. due to other tests or manual additions)
                 print(f"Warning: Could not remove dummy_clips_dir '{DUMMY_CLIPS_DIR}' as it might not be empty.")
        if os.path.exists(DUMMY_FEATURES_OUT_DIR):
            try:
                os.rmdir(DUMMY_FEATURES_OUT_DIR)
            except OSError:
                 print(f"Warning: Could not remove dummy_features_out_dir '{DUMMY_FEATURES_OUT_DIR}' as it might not be empty.")
        print("Dummy files/dirs cleanup attempted.")

import argparse
import os
import sys

# Add src directory to Python path to allow importing modules from src
# This assumes the script is in 'scripts/' and 'src/' is a sibling directory.
# Adjust if your project structure is different.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(SCRIPT_DIR, '..', 'src')
sys.path.append(SRC_DIR)

try:
    from data_processing import DatasetProcessor
except ImportError as e:
    print(f"Error importing DatasetProcessor: {e}")
    print(f"Ensure 'src' directory is in Python path: {SRC_DIR} (added to sys.path).")
    print("Also ensure data_processing.py and its dependencies are in src/.")
    sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Process a dataset of video clips to extract features.")
    parser.add_argument("--clips_dir", required=True, type=str,
                        help="Path to the directory containing processed video clips.")
    parser.add_argument("--metadata_file", required=True, type=str,
                        help="Path to the metadata TSV file (e.g., openasl-v1.0.tsv or a filtered version).")
    parser.add_argument("--output_dir", required=True, type=str,
                        help="Path to the directory where extracted features and manifest will be saved.")
    parser.add_argument("--max_videos", type=int, default=None,
                        help="Optional: Maximum number of videos to process from the dataset (for testing).")
    
    # Add other arguments if needed, e.g., for feature_extractor_config
    # parser.add_argument("--detection_confidence", type=float, default=0.5, help="Min detection confidence for MediaPipe models.")

    args = parser.parse_args()

    print("Starting Dataset Processing Script...")
    print(f"  Video Clips Directory: {args.clips_dir}")
    print(f"  Metadata File: {args.metadata_file}")
    print(f"  Output Features Directory: {args.output_dir}")
    if args.max_videos is not None:
        print(f"  Processing at most {args.max_videos} videos.")

    # Basic validation of paths
    if not os.path.isdir(args.clips_dir):
        print(f"Error: Video clips directory not found: {args.clips_dir}")
        sys.exit(1)
    if not os.path.isfile(args.metadata_file):
        print(f"Error: Metadata file not found: {args.metadata_file}")
        sys.exit(1)
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Ensured output directory exists: {args.output_dir}")

    try:
        # Use 'with' statement for automatic cleanup of extractors
        with DatasetProcessor(
            processed_clips_dir=args.clips_dir,
            metadata_tsv_path=args.metadata_file,
            features_output_dir=args.output_dir
            # feature_extractor_config can be added here if CLI args are added for it
        ) as processor:
            processor.run_feature_extraction_pipeline(max_videos_to_process=args.max_videos)
        
        print("Dataset processing completed successfully.")

    except Exception as e:
        print(f"An error occurred during the dataset processing pipeline: {e}")
        # Consider more detailed error logging or re-raising for debugging
        # import traceback
        # traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()

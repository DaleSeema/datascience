import os
import sys
import subprocess
import json
from tqdm import tqdm
import argparse
import pandas as pd
import shutil # For shutil.which
import math # For math.ceil or just int casting

# Make sure the download_video function (using yt-dlp) from the previous response is defined above main()
# def download_video(vids, target_dir, downloader_executable):
#     # ... (this function remains the same as the yt-dlp version provided before)
#     # It should use downloader_executable, which will be yt-dlp
#     for url_or_id in tqdm(vids):
#         output_template = os.path.join(target_dir, "%(id)s.%(ext)s")
#         cmd_list = [
#             downloader_executable, url_or_id,
#             "-f", "bestvideo+bestaudio[ext=m4a]/bestvideo+bestaudio/best",
#             "--merge-output-format", "mp4",
#             "--no-check-certificate", "--restrict-filenames",
#             "-o", output_template
#         ]
#         print(f"Executing: {' '.join(cmd_list)}")
#         try:
#             result = subprocess.run(cmd_list, capture_output=True, text=True, check=False, timeout=300) # Added timeout
#             if result.returncode != 0:
#                 print(f"Warning: yt-dlp failed for {url_or_id} (Return Code: {result.returncode})")
#                 # print(f"Stdout: {result.stdout.strip()[:200]}...") # Print snippet of stdout
#                 # print(f"Stderr: {result.stderr.strip()[:200]}...") # Print snippet of stderr
#             # else:
#             #     print(f"Successfully processed {url_or_id}")
#         except subprocess.TimeoutExpired:
#             print(f"Timeout expired while trying to download {url_or_id}")
#         except Exception as e:
#             print(f"An error occurred while trying to download {url_or_id}: {e}")
#     return

def main():
    parser = argparse.ArgumentParser(description='download video', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--tsv', type=str, required=True, help='data tsv file')
    parser.add_argument('--dest', type=str, required=True, help='dest dir')
    parser.add_argument('--slurm', action='store_true', help='slurm or not')
    parser.add_argument('--nshard', type=int, default=100, help='number of slurm jobs to launch in total')
    parser.add_argument('--slurm-argument', type=str, default='{"slurm_array_parallelism":100,"slurm_partition":"speech-cpu","timeout_min":240,"slurm_mem":"16g"}', help='slurm arguments')
    
    # New argument for percentage
    parser.add_argument('--percentage', type=float, default=1.0, help='Percentage of videos to download (e.g., 0.1 for 10%). Default is 1.0 (all).')

    args = parser.parse_args()
    
    tsv_fn = args.tsv
    df = pd.read_csv(tsv_fn, sep='\t')
    all_yids_from_tsv = sorted(list(set(df['yid']))) # Get all unique video IDs
    original_yids_count = len(all_yids_from_tsv)

    # --- MODIFICATION FOR PARTIAL DOWNLOAD ---
    download_percentage = args.percentage
    
    if not (0 < download_percentage <= 1.0):
        print("Warning: Percentage must be between 0 (exclusive) and 1.0 (inclusive). Defaulting to 1.0 (all videos).")
        download_percentage = 1.0

    if download_percentage < 1.0:
        num_to_download = math.ceil(original_yids_count * download_percentage) # Use math.ceil to get at least 1 if percentage is small
        # Or use int() and handle the case where it becomes 0 for small percentages but non-empty list
        # num_to_download = int(original_yids_count * download_percentage)
        # if num_to_download == 0 and original_yids_count > 0:
        #     num_to_download = 1 # Ensure at least one video if possible
            
        yids = all_yids_from_tsv[:num_to_download] # Take the first 'num_to_download' videos
        print(f"Original total videos available: {original_yids_count}")
        print(f"Selected {len(yids)} videos (approx. {download_percentage*100:.1f}%) for download based on the first entries in the sorted list.")
    else:
        yids = all_yids_from_tsv
        print(f"Selected all {original_yids_count} videos for download.")
    # --- END MODIFICATION ---

    if not yids:
        print("No video IDs selected for download. Exiting.")
        return

    print(f"Attempting to download {len(yids)} videos into {args.dest}")
    os.makedirs(args.dest, exist_ok=True)

    downloader_executable = "yt-dlp"
    if not shutil.which(downloader_executable):
        print(f"Error: '{downloader_executable}' not found in PATH.")
        print("Please ensure yt-dlp is installed and accessible (e.g., pip install -U yt-dlp).")
        sys.exit(1)
    print(f"Using downloader: {shutil.which(downloader_executable)}")

    if not args.slurm:
        download_video(yids, args.dest, downloader_executable)
    else:
        try:
            import submitit # Make sure this is installed if using slurm
        except ImportError:
            print("Error: 'submitit' library is not installed. Please install it to use SLURM features (e.g., pip install submitit).")
            sys.exit(1)
            
        executor = submitit.AutoExecutor(folder=os.path.join(args.dest, 'submitit_logs'))
        params = json.loads(args.slurm_argument)
        executor.update_parameters(**params)
        print(f"Launching slurm jobs with arguments: {params}...")
        
        batch_yids_for_slurm = []
        # Adjust nshard if the number of videos is very small
        effective_nshard = min(args.nshard, len(yids)) if len(yids) > 0 else 1
        
        num_per_shard = (len(yids) + effective_nshard - 1) // effective_nshard if effective_nshard > 0 else 0
        
        if num_per_shard > 0 :
            for i in range(0, len(yids), num_per_shard):
                batch_yids_for_slurm.append(yids[i: i + num_per_shard])
        
        if not batch_yids_for_slurm and yids: # If yids is not empty but batch is (e.g. num_per_shard was 0)
             batch_yids_for_slurm.append(yids) # Put all in one batch
        elif not batch_yids_for_slurm and not yids:
            print("No videos to process with SLURM.")
            jobs = []
        else:
            print(f"Distributing {len(yids)} videos into {len(batch_yids_for_slurm)} SLURM jobs.")
            jobs = executor.map_array(download_video, batch_yids_for_slurm, [args.dest] * len(batch_yids_for_slurm), [downloader_executable] * len(batch_yids_for_slurm))
            
            print(f"Waiting for {len(jobs)} Slurm jobs to complete...")
            for job_idx, job in enumerate(tqdm(jobs, desc="Slurm Jobs")):
                try:
                    job.result()
                except Exception as e:
                    print(f"A slurm job (index {job_idx}) failed: {e}")
        
    # Verification process (operates on the 'yids' subset)
    missing, n_complete = [], 0
    print(f"Verifying downloaded videos for the {len(yids)} selected IDs...")
    for yid_check in tqdm(yids, desc="Verifying videos"):
        expected_filename_base = yid_check # Adapt if yid_check is a full URL
        dest_fn = os.path.join(args.dest, f"{expected_filename_base}.mp4")
        
        if os.path.isfile(dest_fn) and os.path.getsize(dest_fn) > 0: # Check if file exists and is not empty
            n_complete += 1
        else:
            missing.append(yid_check)
    
    print(f"{n_complete}/{len(yids)} videos from the selection found successfully in {args.dest}.")
    if len(missing) > 0:
        fn = os.path.join(args.dest, "missing_selected_videos.txt")
        with open(fn, "w") as fo:
            fo.write("\n".join(missing) + "\n")
        print(f"List of {len(missing)} potentially undownloaded/missing videos (from the selection) saved in {fn}.")
    return

if __name__ == '__main__':
    # Ensure download_video function is defined (copied from previous response, using yt-dlp)
    # Example:
    def download_video(vids, target_dir, downloader_executable):
        for url_or_id in tqdm(vids, desc="Downloading batch"):
            output_template = os.path.join(target_dir, "%(id)s.%(ext)s")
            cmd_list = [
                downloader_executable, url_or_id,
                "-f", "bestvideo+bestaudio[ext=m4a]/bestvideo+bestaudio/best",
                "--merge-output-format", "mp4",
                "--no-check-certificate", "--restrict-filenames",
                "-o", output_template,
                "--socket-timeout", "30", # Timeout for connection
                "--retries", "3" # Number of retries
            ]
            # print(f"Executing: {' '.join(cmd_list)}") # Can be verbose
            try:
                # Increased timeout for the subprocess run itself, useful if a download hangs
                result = subprocess.run(cmd_list, capture_output=True, text=True, check=False, timeout=600) # 10 min timeout for a single video
                if result.returncode != 0:
                    tqdm.write(f"Warning: yt-dlp failed for {url_or_id} (Code: {result.returncode}). Stderr: {result.stderr.strip()[:200]}...")
            except subprocess.TimeoutExpired:
                tqdm.write(f"Timeout expired (10 min) while trying to download {url_or_id}")
            except Exception as e:
                tqdm.write(f"An error occurred while trying to download {url_or_id}: {e}")
        return

    main()

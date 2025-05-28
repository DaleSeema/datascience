import os
import json
import cv2
import sys
import subprocess
import shutil
import tempfile
import argparse
import numpy as np
from tqdm import tqdm
import pandas as pd
import traceback # For detailed error logging

def crop_resize(imgs, bbox, target_size):
    """
    Crops and resizes a list of images based on a bounding box.
    The bounding box is squared before cropping.
    """
    x0_orig, y0_orig, x1_orig, y1_orig = bbox[0], bbox[1], bbox[2], bbox[3]

    if x1_orig <= x0_orig or y1_orig <= y0_orig:
        tqdm.write(f"Warning: Invalid bbox dimensions [{x0_orig},{y0_orig},{x1_orig},{y1_orig}]. Returning no ROIs.")
        return []

    if not imgs:
        tqdm.write("Warning: crop_resize called with empty image list.")
        return []

    # Center of the original bbox
    center_x = (x0_orig + x1_orig) / 2
    center_y = (y0_orig + y1_orig) / 2

    # Make the bbox square by taking the larger dimension
    orig_width = x1_orig - x0_orig
    orig_height = y1_orig - y0_orig
    side_length = max(orig_width, orig_height)

    # New squared bbox coordinates
    x0 = center_x - side_length / 2
    y0 = center_y - side_length / 2
    x1 = center_x + side_length / 2
    y1 = center_y + side_length / 2
    
    # Round to nearest integers for cropping
    x0, y0, x1, y1 = int(round(x0)), int(round(y0)), int(round(x1)), int(round(y1))

    rois = []
    img_h_orig_shape, img_w_orig_shape = imgs[0].shape[:2] # Dimensions of the source frames

    for img in imgs:
        # Calculate padding needed if the squared bbox extends outside image dimensions
        pad_left = max(0, -x0)
        pad_top = max(0, -y0)
        pad_right = max(0, x1 - img_w_orig_shape)
        pad_bottom = max(0, y1 - img_h_orig_shape)

        expanded_img = cv2.copyMakeBorder(img, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=(0,0,0))

        # Adjust bbox coordinates for the padded image
        crop_x0_padded = x0 + pad_left
        crop_y0_padded = y0 + pad_top
        crop_x1_padded = x1 + pad_left
        crop_y1_padded = y1 + pad_top
        
        # Ensure crop coordinates are valid
        if crop_y1_padded <= crop_y0_padded or crop_x1_padded <= crop_x0_padded:
            tqdm.write(f"Warning: Invalid ROI after padding and squaring [{crop_y0_padded}:{crop_y1_padded}, {crop_x0_padded}:{crop_x1_padded}]. Skipping frame.")
            continue
            
        roi = expanded_img[crop_y0_padded:crop_y1_padded, crop_x0_padded:crop_x1_padded]

        if roi.size == 0:
            tqdm.write(f"Warning: ROI is empty. Original bbox {bbox}, "
                       f"Squared bbox [{x0},{y0},{x1},{y1}], "
                       f"Padded crop [{crop_y0_padded}:{crop_y1_padded}, {crop_x0_padded}:{crop_x1_padded}]. Skipping frame.")
            continue
        
        try:
            resized_roi = cv2.resize(roi, (target_size, target_size), interpolation=cv2.INTER_AREA)
            rois.append(resized_roi)
        except cv2.error as e_cv2_resize:
            tqdm.write(f"Warning: cv2.resize failed for ROI. Size: {roi.shape}, Error: {e_cv2_resize}. Skipping frame.")
            continue
            
    return rois

def write_video_ffmpeg(rois, target_path, ffmpeg_exe='ffmpeg', fps=25):
    """
    Writes a list of ROI images to a video file using ffmpeg.
    Fixes the "I/O operation on closed file" error.
    """
    if not rois:
        tqdm.write(f"No ROIs to write for {target_path}. Skipping video creation.")
        return

    parent_dir = os.path.dirname(target_path)
    if parent_dir: 
        os.makedirs(parent_dir, exist_ok=True)
    
    tmp_write_dir = ""
    try:
        tmp_write_dir = tempfile.mkdtemp()
        decimals = 8
        list_fn = os.path.join(tmp_write_dir, "framelist.txt")
        
        frames_successfully_written_to_list = 0
        # Store paths of successfully written frames for the framelist
        valid_frame_paths_for_ffmpeg_list = []

        for i in range(len(rois)):
            if rois[i] is None or rois[i].size == 0:
                tqdm.write(f"Warning: ROI image {i} for {target_path} is empty or None. Skipping this frame.")
                continue
            
            frame_filename = str(i).zfill(decimals) + '.png'
            frame_path = os.path.join(tmp_write_dir, frame_filename)
            
            try:
                write_success = cv2.imwrite(frame_path, rois[i])
                if not write_success:
                    tqdm.write(f"Warning: Failed to write frame {i} to {frame_path} using cv2.imwrite. Skipping this frame.")
                    continue
            except Exception as e_cv2:
                tqdm.write(f"Warning: Error during cv2.imwrite for frame {i} ({frame_path}): {e_cv2}. Skipping this frame.")
                continue

            # Use POSIX paths with forward slashes in the list file for ffmpeg
            ffmpeg_frame_path = frame_path.replace(os.sep, '/')
            valid_frame_paths_for_ffmpeg_list.append(ffmpeg_frame_path)
            frames_successfully_written_to_list += 1
        
        if frames_successfully_written_to_list == 0:
            tqdm.write(f"No frames were successfully prepared for {target_path}. Video not created.")
            return

        with open(list_fn, 'w') as f_list:
            for i in range(frames_successfully_written_to_list):
                f_list.write(f"file '{valid_frame_paths_for_ffmpeg_list[i]}'\n")
                f_list.write(f"duration {1/fps}\n")
            
            # Append the last frame file path again for persistence, if any frames were written
            if valid_frame_paths_for_ffmpeg_list:
                 f_list.write(f"file '{valid_frame_paths_for_ffmpeg_list[-1]}'\n")
        # f_list is now closed

        if os.path.isfile(target_path):
            try:
                os.remove(target_path)
            except OSError as e_rm:
                tqdm.write(f"Warning: Could not remove existing target file {target_path}: {e_rm}")
        
        cmd_write_video = [
            ffmpeg_exe, 
            "-f", "concat",
            "-safe", "0", 
            "-i", list_fn,
            "-an",
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-r", str(fps),
            "-crf", "20",
            "-y",
            target_path
        ]
        
        pipe = subprocess.run(cmd_write_video, capture_output=True, text=True, check=False, timeout=120) # 2 min timeout
        if pipe.returncode != 0:
            tqdm.write(f"ffmpeg failed to write video {target_path}. FFmpeg stderr (first 500 chars): {pipe.stderr[:500]}...")
    
    except subprocess.TimeoutExpired:
        tqdm.write(f"ffmpeg command timed out while writing video {target_path}.")
    except Exception as e:
        tqdm.write(f"Unexpected error in write_video_ffmpeg for {target_path}: {e}")
        tqdm.write(traceback.format_exc())
    finally:
        if tmp_write_dir and os.path.isdir(tmp_write_dir):
            shutil.rmtree(tmp_write_dir)
    return

def get_clip(input_video_dir, output_video_dir, tsv_fn, bbox_fn, rank, nshard, target_size=224, ffmpeg='ffmpeg'):
    os.makedirs(output_video_dir, exist_ok=True)

    print(f"INFO: Scanning {input_video_dir} for available .mp4 files...")
    available_yids_on_disk = set()
    try:
        for f_name in os.listdir(input_video_dir):
            if f_name.endswith('.mp4'):
                full_file_path = os.path.join(input_video_dir, f_name)
                if os.path.isfile(full_file_path) and os.path.getsize(full_file_path) > 0:
                    yid = f_name[:-4]
                    available_yids_on_disk.add(yid)
    except FileNotFoundError:
        print(f"ERROR: Input video directory not found: {input_video_dir}")
        return
    
    if not available_yids_on_disk:
        print(f"INFO: No non-empty .mp4 files found in {input_video_dir}. Exiting.")
        return
    print(f"INFO: Found {len(available_yids_on_disk)} non-empty .mp4 files in {input_video_dir}.")

    print(f"INFO: Loading TSV file: {tsv_fn}")
    try:
        df = pd.read_csv(tsv_fn, sep='\t', low_memory=False)
    except FileNotFoundError:
        print(f"ERROR: TSV file not found: {tsv_fn}")
        return
        
    print(f"INFO: Loading Bbox JSON file: {bbox_fn}")
    try:
        with open(bbox_fn, 'r') as f:
            vid2bbox = json.load(f)
    except FileNotFoundError:
        print(f"ERROR: Bbox JSON file not found: {bbox_fn}")
        return
    except json.JSONDecodeError:
        print(f"ERROR: Bbox JSON file {bbox_fn} is not valid JSON.")
        return

    items_to_process = []
    print(f"INFO: Filtering {len(df)} TSV entries against {len(available_yids_on_disk)} available videos and bbox data...")
    
    for index, row in tqdm(df.iterrows(), total=len(df), desc="Filtering TSV entries"):
        try:
            vid = str(row['vid'])
            yid = str(row['yid'])
            start_time = row['start']
            end_time = row['end']
        except KeyError as e:
            tqdm.write(f"Warning: Missing expected column {e} in TSV row {index}. Skipping row.")
            continue

        if yid in available_yids_on_disk:
            if vid not in vid2bbox:
                continue
            
            bbox_data = vid2bbox[vid]
            if not isinstance(bbox_data, list) or len(bbox_data) != 4:
                continue
            items_to_process.append([vid, yid, start_time, end_time, bbox_data])
    
    if not items_to_process:
        print(f"INFO: No processable video items found after filtering. Exiting.")
        return
    
    total_processable_items = len(items_to_process)
    print(f"INFO: Found {total_processable_items} items that are available locally and have valid bbox data.")

    if nshard <= 0: nshard = 1
    if rank < 0: rank = 0
    
    num_per_shard = (total_processable_items + nshard - 1) // nshard
    start_index = num_per_shard * rank
    end_index = min(num_per_shard * (rank + 1), total_processable_items)
    shard_items = items_to_process[start_index:end_index]
    
    actual_items_in_shard = len(shard_items)
    print(f"INFO: Rank {rank}/{nshard}: This shard will process {actual_items_in_shard} videos (indices {start_index}-{end_index-1} from {total_processable_items} total processable).")

    if not shard_items:
        print(f"INFO: Rank {rank}/{nshard}: No items assigned to this shard. Exiting.")
        return

    for vid, yid, start_time, end_time, bbox_coords in tqdm(shard_items, desc=f"Processing Shard {rank}"):
        input_video_whole = os.path.join(input_video_dir, yid + '.mp4')
        output_video = os.path.join(output_video_dir, vid + '.mp4')

        if os.path.isfile(output_video):
            continue

        if not os.path.isfile(input_video_whole):
            tqdm.write(f"CRITICAL_WARNING: Input video {input_video_whole} (YID: {yid}) expected but not found! Skipping.")
            continue
        
        tmp_clip_dir = "" 
        try:
            tmp_clip_dir = tempfile.mkdtemp()
            input_video_clip_path = os.path.join(tmp_clip_dir, 'tmp_clip.mp4')
            
            cmd_extract = [
                ffmpeg, 
                '-ss', str(start_time), 
                '-to', str(end_time), 
                '-i', input_video_whole, 
                '-c:v', 'libx264', '-crf', '20', '-an', '-y',              
                input_video_clip_path
            ]
            result_extract = subprocess.run(cmd_extract, capture_output=True, text=True, check=False, timeout=300)
            
            if result_extract.returncode != 0:
                tqdm.write(f"ffmpeg clip extraction failed for {input_video_whole} (Vid: {vid}). Stderr: {result_extract.stderr[:200]}...")
                continue 

            if not os.path.isfile(input_video_clip_path) or os.path.getsize(input_video_clip_path) == 0:
                tqdm.write(f"Extracted clip {input_video_clip_path} is missing or empty for {input_video_whole} (Vid: {vid}). Skipping.")
                continue

            cap = cv2.VideoCapture(input_video_clip_path)
            frames_origin = []
            while cap.isOpened(): 
                ret, frame = cap.read()
                if not ret: break
                frames_origin.append(frame)
            cap.release()

            if not frames_origin:
                tqdm.write(f"No frames extracted by OpenCV from {input_video_clip_path} (Vid: {vid}). Skipping.")
                continue
            
            img_h, img_w = frames_origin[0].shape[:2]
            # Ensure bbox_coords are float for multiplication before int conversion
            abs_x0 = float(bbox_coords[0]) * img_w
            abs_y0 = float(bbox_coords[1]) * img_h
            abs_x1 = float(bbox_coords[2]) * img_w
            abs_y1 = float(bbox_coords[3]) * img_h
            current_abs_bbox = [abs_x0, abs_y0, abs_x1, abs_y1] # Pass float to crop_resize for its rounding
            
            rois = crop_resize(frames_origin, current_abs_bbox, target_size)
            
            if not rois: 
                tqdm.write(f"No ROIs generated by crop_resize for Vid: {vid}. Skipping writing video.")
                continue
            write_video_ffmpeg(rois, output_video, ffmpeg_exe=ffmpeg, fps=25)
        
        except subprocess.TimeoutExpired:
            tqdm.write(f"ffmpeg command timed out for {input_video_whole} (Vid: {vid}).")
        except Exception as e:
            tqdm.write(f"An unexpected error occurred while processing Vid: {vid} (YID: {yid}): {e}")
            tqdm.write(traceback.format_exc())
        finally:
            if tmp_clip_dir and os.path.isdir(tmp_clip_dir): 
                shutil.rmtree(tmp_clip_dir)
    return

def main():
    parser = argparse.ArgumentParser(description='Process videos: extract clips, crop, and resize.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--tsv', type=str, required=True, help='Data TSV file (e.g., openasl-v1.0.tsv)')
    parser.add_argument('--bbox', type=str, required=True, help='Bounding box JSON file (e.g., bbox-v1.0.json)')
    parser.add_argument('--raw', type=str, required=True, help='Directory containing raw downloaded MP4 videos')
    parser.add_argument('--output', type=str, required=True, help='Output directory for processed video clips')
    parser.add_argument('--ffmpeg', type=str, default='ffmpeg', help='Path to ffmpeg executable')
    parser.add_argument('--target-size', type=int, default=224, help='Target size for cropped video frames (e.g., 224 for 224x224)')

    parser.add_argument('--slurm', action='store_true', help='Enable SLURM for distributed processing (requires submitit setup)')
    parser.add_argument('--nshard', type=int, default=1, help='Total number of shards. If >1 without --slurm, also use --rank.')
    parser.add_argument('--rank', type=int, default=0, help='Current shard rank/ID (0 to nshard-1). Used if --nshard > 1.')

    args = parser.parse_args()

    if args.slurm:
        print("INFO: SLURM mode enabled.")
        print(f"INFO: This job instance is Rank {args.rank} of {args.nshard} total shards.")
        if args.rank < 0 or args.rank >= args.nshard:
            print(f"ERROR: Invalid Rank {args.rank} for Nshard {args.nshard}. Rank must be 0 to Nshard-1. Exiting.")
            sys.exit(1)
        get_clip(args.raw, args.output, args.tsv, args.bbox, 
                 rank=args.rank, nshard=args.nshard, 
                 target_size=args.target_size, ffmpeg=args.ffmpeg)
    else:
        if args.nshard > 1:
            print(f"INFO: Non-SLURM mode. Processing Shard {args.rank} of {args.nshard}.")
            if args.rank < 0 or args.rank >= args.nshard:
                print(f"ERROR: Invalid Rank {args.rank} for Nshard {args.nshard}. Rank must be 0 to Nshard-1. Exiting.")
                sys.exit(1)
            get_clip(args.raw, args.output, args.tsv, args.bbox, 
                     rank=args.rank, nshard=args.nshard, 
                     target_size=args.target_size, ffmpeg=args.ffmpeg)
        else: 
            print("INFO: Non-SLURM mode. Processing all data as a single shard (Rank 0 of 1).")
            get_clip(args.raw, args.output, args.tsv, args.bbox, 
                     rank=0, nshard=1, 
                     target_size=args.target_size, ffmpeg=args.ffmpeg)
    
    print("INFO: Processing finished.")
    return

if __name__ == '__main__':
    main()

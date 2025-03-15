import sys
import logging
import time

if sys.version_info[0] < 3 and sys.version_info[1] < 2:
	raise Exception("Must be using >= Python 3.2")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("preprocess.log")
    ]
)
logger = logging.getLogger("Wav2Lip-Preprocess")

from os import listdir, path

if not path.isfile('face_detection/detection/sfd/s3fd.pth'):
	logger.error("Missing s3fd model file")
	raise FileNotFoundError('Save the s3fd model to face_detection/detection/sfd/s3fd.pth \
							before running this script!')

import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import argparse, os, cv2, traceback, subprocess
from tqdm import tqdm
from glob import glob
import audio
from hparams import hparams as hp

import face_detection
import torch

parser = argparse.ArgumentParser()

parser.add_argument('--ngpu', help='Number of GPUs across which to run in parallel', default=1, type=int)
parser.add_argument('--batch_size', help='Single GPU Face detection batch size', default=32, type=int)
parser.add_argument("--data_root", help="Root folder of the LRS2 dataset", required=True)
parser.add_argument("--preprocessed_root", help="Root folder of the preprocessed dataset", required=True)
parser.add_argument("--verbose", help="Enable verbose logging", action='store_true')
parser.add_argument("--log_interval", help="How often to log progress", default=10, type=int)
parser.add_argument("--profile_memory", help="Profile GPU memory usage", action='store_true')

args = parser.parse_args()

# Set more detailed logging for verbose mode
if args.verbose:
    logger.setLevel(logging.DEBUG)

# Initialize the face detection models on their respective GPUs
logger.info(f"Initializing {args.ngpu} face detectors on GPUs...")
fa = []
for id in range(args.ngpu):
    try:
        logger.debug(f"Initializing face detector on GPU {id}")
        detector = face_detection.FaceAlignment(face_detection.LandmarksType._2D, 
                                               flip_input=False, 
                                               device=f'cuda:{id}')
        fa.append(detector)
        logger.debug(f"Successfully initialized detector on GPU {id}")
    except Exception as e:
        logger.error(f"Failed to initialize face detector on GPU {id}: {e}")
        raise

# Command templates
template = 'ffmpeg -loglevel panic -y -i {} -strict -2 {}'
# template2 = 'ffmpeg -hide_banner -loglevel panic -threads 1 -y -i {} -async 1 -ac 1 -vn -acodec pcm_s16le -ar 16000 {}'

def process_video_file(vfile, args, gpu_id):
    start_time = time.time()
    logger.info(f"Processing video: {vfile} on GPU {gpu_id}")
    
    video_stream = cv2.VideoCapture(vfile)
    if not video_stream.isOpened():
        logger.error(f"Could not open video file: {vfile}")
        return
    
    fps = video_stream.get(cv2.CAP_PROP_FPS)
    frame_count = int(video_stream.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(video_stream.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video_stream.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    logger.info(f"Video specs: {width}x{height}, {fps} fps, {frame_count} frames")
    
    frames = []
    frame_idx = 0
    while True:
        still_reading, frame = video_stream.read()
        if not still_reading:
            video_stream.release()
            break
        frames.append(frame)
        frame_idx += 1
        if frame_idx % args.log_interval == 0:
            logger.debug(f"Read {frame_idx}/{frame_count} frames from {vfile}")
    
    logger.info(f"Finished reading {len(frames)} frames from {vfile}")
    
    vidname = os.path.basename(vfile).split('.')[0]
    dirname = vfile.split('/')[-2]

    fulldir = path.join(args.preprocessed_root, dirname, vidname)
    os.makedirs(fulldir, exist_ok=True)
    
    logger.debug(f"Processing {len(frames)} frames in batches of {args.batch_size}")
    batches = [frames[i:i + args.batch_size] for i in range(0, len(frames), args.batch_size)]
    
    face_count = 0
    saved_count = 0
    missed_frames = []
    
    i = -1
    batch_idx = 0
    for fb in batches:
        batch_start = time.time()
        logger.debug(f"Processing batch {batch_idx+1}/{len(batches)}, {len(fb)} frames")
        
        try:
            # Use no_grad context to avoid keeping computational history
            with torch.no_grad():
                # Convert batch to tensor
                batch_tensor = np.asarray(fb)
                
                # Get face detections
                preds = fa[gpu_id].get_detections_for_batch(batch_tensor)
                
                # Count successfully detected faces
                batch_faces = sum(1 for f in preds if f is not None)
                face_count += batch_faces
                
                logger.debug(f"Batch {batch_idx+1}: detected faces in {batch_faces}/{len(fb)} frames")
                
                # Process each prediction
                for j, f in enumerate(preds):
                    i += 1
                    if f is None:
                        missed_frames.append(i)
                        continue
                        
                    try:
                        x1, y1, x2, y2 = f
                        face_width = x2 - x1
                        face_height = y2 - y1
                        
                        if args.verbose and (i % args.log_interval == 0):
                            logger.debug(f"Frame {i}: face {x1},{y1} to {x2},{y2} ({face_width}x{face_height})")
                        
                        # Ensure the face is a reasonable size
                        if face_width < 10 or face_height < 10:
                            logger.warning(f"Frame {i}: detected face too small ({face_width}x{face_height}), skipping")
                            missed_frames.append(i)
                            continue
                        
                        face_img = fb[j][y1:y2, x1:x2]
                        save_path = path.join(fulldir, f'{i}.jpg')
                        cv2.imwrite(save_path, face_img)
                        saved_count += 1
                    except Exception as e:
                        logger.error(f"Error saving face for frame {i}: {e}")
                        missed_frames.append(i)
                
              
        except Exception as e:
            logger.error(f"Error detecting faces in batch {batch_idx}: {e}")
        
     
        batch_time = time.time() - batch_start
        logger.debug(f"Batch {batch_idx+1} took {batch_time:.2f}s ({len(fb)/batch_time:.2f} fps)")
        batch_idx += 1
    
    total_time = time.time() - start_time
    logger.info(f"Finished processing {vfile}: detected {face_count} faces in {len(frames)} frames")
    logger.info(f"Saved {saved_count} face crops to {fulldir} ({saved_count/len(frames)*100:.1f}%)")
    logger.info(f"Processing time: {total_time:.2f}s ({len(frames)/total_time:.2f} fps)")
    
    if missed_frames:
        logger.warning(f"Missed faces in {len(missed_frames)} frames: {missed_frames[:5]}...")
        
    # Final GPU cleanup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return saved_count, len(frames)

def process_audio_file(vfile, args):
    logger.info(f"Extracting audio from {vfile}")
    start_time = time.time()

    vidname = os.path.basename(vfile).split('.')[0]
    dirname = vfile.split('/')[-2]

    fulldir = path.join(args.preprocessed_root, dirname, vidname)
    os.makedirs(fulldir, exist_ok=True)

    wavpath = path.join(fulldir, 'audio.wav')

    command = template.format(vfile, wavpath)
    logger.debug(f"Running command: {command}")
    
    try:
        subprocess.call(command, shell=True)
        
        if os.path.exists(wavpath):
            audio_size = os.path.getsize(wavpath) / (1024 * 1024)  # MB
            logger.info(f"Audio extraction successful: {wavpath} ({audio_size:.2f} MB)")
        else:
            logger.error(f"Audio extraction failed: {wavpath} does not exist")
            
        # Load audio to verify it's valid
        if os.path.exists(wavpath):
            try:
                sample_rate, audio_data = audio.load_wav(wavpath, hp.sample_rate)
                duration = len(audio_data) / sample_rate
                logger.info(f"Audio duration: {duration:.2f}s, sample rate: {sample_rate}Hz")
            except Exception as e:
                logger.error(f"Failed to validate audio file {wavpath}: {e}")
    except Exception as e:
        logger.error(f"Error during audio extraction: {e}")
        
    elapsed = time.time() - start_time
    logger.info(f"Audio extraction completed in {elapsed:.2f}s")

def mp_handler(job):
    vfile, args, gpu_id = job
    try:
        logger.info(f"Starting job for {vfile} on GPU {gpu_id}")
        
        # Set CUDA device for this thread
        if torch.cuda.is_available():
            torch.cuda.set_device(gpu_id)
            logger.debug(f"Set CUDA device to GPU {gpu_id}")
        
        stats = process_video_file(vfile, args, gpu_id)
        
        # Release GPU memory after processing file
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return vfile, stats
    except KeyboardInterrupt:
        # Clean up on interrupt
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.warning(f"Job for {vfile} interrupted")
        exit(0)
    except Exception as e:
        # Clean up on error
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.error(f"Error processing {vfile}: {str(e)}")
        traceback.print_exc()
        return vfile, (0, 0)
        
def main(args):
    logger.info(f"Started preprocessing pipeline for {args.data_root} with {args.ngpu} GPUs")
    logger.info(f"Using batch size of {args.batch_size} frames per GPU")
    logger.info(f"Output directory: {args.preprocessed_root}")

    # Ensure output directory exists
    os.makedirs(args.preprocessed_root, exist_ok=True)
    
    # Find all video files
    filelist = glob(path.join(args.data_root, '**/*.mp4'), recursive=True)
    logger.info(f"Found {len(filelist)} video files for processing")
    
    if len(filelist) == 0:
        logger.error(f"No video files found in {args.data_root}")
        return

    # Set up process pool and distribute work
    start_time = time.time()
    logger.info(f"Starting face detection across {args.ngpu} GPUs...")
    
    jobs = [(vfile, args, i % args.ngpu) for i, vfile in enumerate(filelist)]
    
    # Keep track of statistics
    total_frames = 0
    total_faces = 0
    failed_videos = []
    
    # If profiling is enabled, log memory statistics
    if args.verbose and torch.cuda.is_available():
        for i in range(args.ngpu):
            torch.cuda.reset_peak_memory_stats(i)
    
    with ThreadPoolExecutor(args.ngpu) as p:
        futures = [p.submit(mp_handler, j) for j in jobs]
        for r in tqdm(as_completed(futures), total=len(futures)):
            try:
                vfile, (faces, frames) = r.result()
                if frames > 0:
                    total_frames += frames
                    total_faces += faces
                    logger.debug(f"Completed {vfile}: {faces}/{frames} faces detected ({faces/frames*100:.1f}%)")
                else:
                    failed_videos.append(vfile)
            except Exception as e:
                logger.error(f"Error getting result: {e}")

    face_time = time.time() - start_time
    logger.info(f"Face detection completed in {face_time:.2f}s")
    logger.info(f"Processed {total_frames} frames with {total_faces} faces detected ({total_faces/max(1, total_frames)*100:.1f}%)")
    
    if failed_videos:
        logger.warning(f"Failed to process {len(failed_videos)} videos")
        if args.verbose:
            for v in failed_videos:
                logger.warning(f"Failed: {v}")

    # Print memory statistics if verbose
    if args.verbose and torch.cuda.is_available():
        for i in range(args.ngpu):
            peak_memory = torch.cuda.max_memory_allocated(i) / (1024 * 1024)
            logger.debug(f"GPU {i} peak memory usage: {peak_memory:.2f} MB")

    logger.info('Starting audio extraction...')
    audio_start_time = time.time()

    audio_files_processed = 0
    audio_files_failed = []
    
    for vfile in tqdm(filelist):
        try:
            process_audio_file(vfile, args)
            audio_files_processed += 1
        except KeyboardInterrupt:
            logger.warning("Processing interrupted by user")
            exit(0)
        except Exception as e:
            logger.error(f"Error extracting audio from {vfile}: {e}")
            audio_files_failed.append(vfile)
            continue

    audio_time = time.time() - audio_start_time
    logger.info(f"Audio extraction completed in {audio_time:.2f}s")
    logger.info(f"Processed {audio_files_processed} audio files")
    
    if audio_files_failed:
        logger.warning(f"Failed to extract audio from {len(audio_files_failed)} videos")
        if args.verbose:
            for v in audio_files_failed:
                logger.warning(f"Failed audio: {v}")
    
    total_time = time.time() - start_time
    logger.info(f"Preprocessing completed in {total_time:.2f}s")
    logger.info(f"Results saved to {args.preprocessed_root}")

if __name__ == '__main__':
    # Initialize memory profiling if requested
    if args.profile_memory and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        
    main(args)
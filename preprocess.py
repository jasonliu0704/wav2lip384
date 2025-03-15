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

# Add maximum resolution constant at the top of the file after imports
MAX_WIDTH = 2048
MAX_HEIGHT = 1080

def process_video_file(vfile, args, gpu_id):
    start_time = time.time()
    logger.info(f"Processing video: {vfile} on GPU {gpu_id}")
    
    try:
        # Safer video opening with proper error handling
        video_stream = cv2.VideoCapture(vfile)
        if not video_stream.isOpened():
            logger.error(f"Could not open video file: {vfile}")
            return 0, 0
        
        fps = video_stream.get(cv2.CAP_PROP_FPS)
        frame_count = int(video_stream.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(video_stream.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video_stream.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Check if resolution exceeds maximum allowed
        needs_resize = width > MAX_WIDTH or height > MAX_HEIGHT
        if needs_resize:
            # Calculate new dimensions maintaining aspect ratio
            scale_factor = min(MAX_WIDTH / width, MAX_HEIGHT / height)
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
            logger.info(f"Video resolution ({width}x{height}) exceeds maximum allowed. Resizing to {new_width}x{new_height}")
        else:
            new_width, new_height = width, height
            
        logger.info(f"Video specs: {width}x{height}, {fps} fps, {frame_count} frames")
        if needs_resize:
            logger.info(f"Working resolution: {new_width}x{new_height}")
        
        # Early validation of video parameters
        if width <= 0 or height <= 0 or frame_count <= 0:
            logger.error(f"Invalid video dimensions or frame count: {width}x{height}, {frame_count} frames")
            return 0, 0
        
        frames = []
        frame_idx = 0
        
        # Read frames with memory tracking
        mem_check_interval = 100
        while True:
            if torch.cuda.is_available() and frame_idx % mem_check_interval == 0:
                # Check memory usage and exit gracefully if close to limit
                if torch.cuda.memory_allocated(gpu_id) > 0.9 * torch.cuda.get_device_properties(gpu_id).total_memory:
                    logger.warning(f"GPU {gpu_id} memory nearly exhausted. Processing first {len(frames)} frames only.")
                    break
                    
            still_reading, frame = video_stream.read()
            if not still_reading:
                break
                
            # Validate frame dimensions
            if frame is None or frame.size == 0:
                logger.warning(f"Empty frame at position {frame_idx}, skipping")
                frame_idx += 1
                continue
            
            # Resize if needed (maintaining aspect ratio)
            if needs_resize:
                frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
                
            frames.append(frame)
            frame_idx += 1
            
            if frame_idx % args.log_interval == 0:
                logger.debug(f"Read {frame_idx}/{frame_count} frames from {vfile}")
        
        # Always close the video
        video_stream.release()
        
        # Sanity check on frames
        if not frames:
            logger.error(f"No valid frames extracted from {vfile}")
            return 0, 0
            
        logger.info(f"Finished reading {len(frames)} frames from {vfile}")
        
        # Create output directory
        vidname = os.path.basename(vfile).split('.')[0]
        dirname = vfile.split('/')[-2]
        fulldir = path.join(args.preprocessed_root, dirname, vidname)
        os.makedirs(fulldir, exist_ok=True)
        
        # Determine optimal batch size based on available memory
        if torch.cuda.is_available():
            # Estimate memory per frame (rough approximation)
            est_mem_per_frame = width * height * 3 * 4  # bytes (assuming float32)
            avail_mem = torch.cuda.get_device_properties(gpu_id).total_memory * 0.7  # Use 70% of memory
            dynamic_batch_size = max(1, int(avail_mem / est_mem_per_frame))
            batch_size = min(args.batch_size, dynamic_batch_size)
            logger.debug(f"Using dynamic batch size: {batch_size} (original: {args.batch_size})")
        else:
            batch_size = args.batch_size
            
        # Create batches with memory considerations
        batches = []
        for i in range(0, len(frames), batch_size):
            end_idx = min(i + batch_size, len(frames))
            batches.append(frames[i:end_idx])
        
        face_count = 0
        saved_count = 0
        missed_frames = []
        
        # Process each batch with more robust error handling
        frame_idx = -1
        for batch_idx, fb in enumerate(batches):
            try:
                batch_start = time.time()
                batch_size = len(fb)  # Store batch size before any processing
                logger.debug(f"Processing batch {batch_idx+1}/{len(batches)}, {batch_size} frames")
                
                # Check if device is still available
                if torch.cuda.is_available() and not torch.cuda.device_count() > gpu_id:
                    logger.error(f"GPU {gpu_id} no longer available")
                    break
                
                # Process the batch with error handling
                try:
                    # Convert batch to numpy array with explicit type
                    batch_tensor = np.array(fb, dtype=np.uint8)
                    
                    # Safe batch processing with exception handling
                    with torch.no_grad():
                        try:
                            # Get face detections with timeout protection
                            preds = fa[gpu_id].get_detections_for_batch(batch_tensor)
                        except RuntimeError as e:
                            if "out of memory" in str(e).lower():
                                # Handle OOM error
                                logger.error(f"GPU {gpu_id} out of memory processing batch {batch_idx}")
                                torch.cuda.empty_cache()
                                
                                # Try with smaller batch by splitting
                                if len(fb) > 1:
                                    logger.info(f"Retrying with smaller batches")
                                    mid = len(fb) // 2
                                    batches.extend([fb[:mid], fb[mid:]])
                                    continue
                                else:
                                    # Can't split further
                                    logger.error(f"Can't process even a single frame. Skipping batch.")
                                    
                                # Add all frames to missed list
                                for _ in range(len(fb)):
                                    frame_idx += 1
                                    missed_frames.append(frame_idx)
                                continue
                            else:
                                # Other runtime error
                                raise
                                
                    # Verify batch prediction results - should match batch size
                    if preds is None:
                        logger.error(f"Detector returned None for batch {batch_idx}")
                        # Add all frames to missed list
                        for _ in range(len(fb)):
                            frame_idx += 1
                            missed_frames.append(frame_idx)
                        continue
                        
                    if len(preds) != len(fb):
                        logger.error(f"Prediction count {len(preds)} doesn't match batch size {len(fb)}")
                        # Resize preds to match batch size
                        if len(preds) < len(fb):
                            preds = list(preds) + [None] * (len(fb) - len(preds))
                        else:
                            preds = preds[:len(fb)]
                    
                    # Count successful detections
                    batch_faces = sum(1 for f in preds if f is not None)
                    face_count += batch_faces
                    logger.debug(f"Batch {batch_idx+1}: detected faces in {batch_faces}/{len(fb)} frames")
                    
                    # Process each prediction
                    for j, f in enumerate(preds):
                        frame_idx += 1
                        
                        # Skip if no face detected
                        if f is None:
                            missed_frames.append(frame_idx)
                            continue
                            
                        try:
                            # Bounds check on predictions
                            x1, y1, x2, y2 = map(int, f)
                            
                            # Validate face coordinates
                            if x1 < 0: x1 = 0
                            if y1 < 0: y1 = 0
                            if x2 >= fb[j].shape[1]: x2 = fb[j].shape[1] - 1
                            if y2 >= fb[j].shape[0]: y2 = fb[j].shape[0] - 1
                            
                            face_width = x2 - x1
                            face_height = y2 - y1
                            
                            # Skip tiny faces
                            if face_width < 10 or face_height < 10 or x1 >= x2 or y1 >= y2:
                                logger.warning(f"Frame {frame_idx}: detected face too small or invalid ({face_width}x{face_height}), skipping")
                                missed_frames.append(frame_idx)
                                continue
                            
                            # Safe image crop with bounds checking
                            face_img = fb[j][y1:y2, x1:x2].copy()  # Copy to ensure memory safety
                            
                            # Verify crop worked
                            if face_img.size == 0:
                                logger.warning(f"Empty face crop for frame {frame_idx}")
                                missed_frames.append(frame_idx)
                                continue
                                
                            # Save face image
                            save_path = path.join(fulldir, f'{frame_idx}.jpg')
                            success = cv2.imwrite(save_path, face_img)
                            
                            if not success:
                                logger.warning(f"Failed to save image for frame {frame_idx}")
                                missed_frames.append(frame_idx)
                                continue
                                
                            saved_count += 1
                        except Exception as e:
                            logger.error(f"Error saving face for frame {frame_idx}: {str(e)}")
                            missed_frames.append(frame_idx)
                            continue
                            
                except Exception as e:
                    logger.error(f"Error in batch {batch_idx} processing: {str(e)}")
                    # Safely increment frame index for missed frames
                    for _ in range(len(fb)):
                        frame_idx += 1
                        missed_frames.append(frame_idx)
                        
                # Explicitly release batch memory
                del fb
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
                batch_time = time.time() - batch_start
                # Use stored batch_size instead of accessing fb which is now deleted
                logger.debug(f"Batch {batch_idx+1} took {batch_time:.2f}s ({batch_size/batch_time:.2f} fps)")
                
            except Exception as e:
                logger.error(f"Unhandled error in batch {batch_idx}: {str(e)}")
                traceback.print_exc()
                # Continue with next batch rather than terminating
                continue
            
        # Final stats
        total_time = time.time() - start_time
        logger.info(f"Finished processing {vfile}: detected {face_count} faces in {len(frames)} frames")
        logger.info(f"Saved {saved_count} face crops to {fulldir} ({saved_count/max(1,len(frames))*100:.1f}%)")
        logger.info(f"Processing time: {total_time:.2f}s ({len(frames)/max(0.1,total_time):.2f} fps)")
        
        if missed_frames:
            logger.warning(f"Missed faces in {len(missed_frames)} frames: {missed_frames[:5]}...")
            
        return saved_count, len(frames)
        
    except Exception as e:
        # Top level exception handler
        logger.error(f"Critical error processing video {vfile}: {str(e)}")
        traceback.print_exc()
        
        # Ensure GPU memory is released even on error
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        return 0, 0

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
        if torch.cuda.is_available():
            # Check if GPU exists
            if gpu_id >= torch.cuda.device_count():
                logger.error(f"GPU {gpu_id} not available (only {torch.cuda.device_count()} GPUs found)")
                return vfile, (0, 0)
                
            # Set CUDA device for this thread
            torch.cuda.set_device(gpu_id)
            logger.debug(f"Set CUDA device to GPU {gpu_id}")
            
            # Monitor starting memory
            free_mem = torch.cuda.get_device_properties(gpu_id).total_memory - torch.cuda.memory_allocated(gpu_id)
            logger.debug(f"GPU {gpu_id} starting with {free_mem/(1024*1024):.2f}MB free memory")
        
        # Process the video with proper error handling
        stats = process_video_file(vfile, args, gpu_id)
        
        # Release GPU memory after processing file
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        return vfile, stats
    except KeyboardInterrupt:
        # Clean up on interrupt
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except:
                pass
        logger.warning(f"Job for {vfile} interrupted")
        return vfile, (0, 0)
    except Exception as e:
        # Clean up on error
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except:
                pass
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
    
    # Configure max processes based on system
    if args.ngpu > torch.cuda.device_count():
        logger.warning(f"Requested {args.ngpu} GPUs but only {torch.cuda.device_count()} available")
        max_workers = torch.cuda.device_count()
    else:
        max_workers = args.ngpu
        
    if max_workers == 0:
        logger.warning("No GPUs available, using CPU only with 1 worker")
        max_workers = 1
    
    # Safer thread pool execution with resource limits
    with ThreadPoolExecutor(max_workers=max_workers) as p:
        futures = []
        
        # Submit jobs with controlled pacing to avoid memory spikes
        for i, vfile in enumerate(filelist):
            gpu_id = i % max(1, max_workers)
            futures.append(p.submit(mp_handler, (vfile, args, gpu_id)))
            # Brief pause between job submissions to avoid overwhelming the system
            time.sleep(0.1)
        
        results = []
        for future in tqdm(as_completed(futures), total=len(futures)):
            try:
                results.append(future.result(timeout=60))  # 60-second timeout
            except concurrent.futures.TimeoutError:
                logger.error("A job timed out. This may indicate a segfault.")
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
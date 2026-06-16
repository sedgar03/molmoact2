import logging
import numbers
import os
import json
import glob
import pickle
import io

import PIL
import numpy as np
from PIL import Image
from os.path import join

from dataclasses import dataclass
import imageio.v3 as iio

from olmo.io import (
    read_file,
    is_url,
    file_exists,
    get_bytes_range,
    glob as olmo_glob,
    list_directory, write_file
)
from tqdm import tqdm

from olmo.util import resource_path
from olmo.data.dataset import DatasetBase, VIDEO_DATA_HOME

log = logging.getLogger(__name__)

from typing import Any, Dict, List, Literal, Optional, Mapping, Tuple
from typing_extensions import TypedDict

class Point(TypedDict):
    point: List[float] # [x, y] coordinates
    occluded: bool

class PointTrajectoryEntry(TypedDict):
    frame: int # frame index
    time: float # time in seconds
    points: Dict[str, Point] # object_id -> {'point': [x, y], 'occluded': bool}

class QueryPoint(TypedDict):
    id: int | str # point id
    point: List[float] # [x, y] coordinates
    time: float # time in seconds
    frame: int # frame index
class VideoPointTrackMessage(TypedDict):
    style: str
    question: str # format: {instruction}\n{normalized start points}
    points: List[PointTrajectoryEntry] # point trajectory per frame from first visible frame including query points
    initial_points: Optional[List[QueryPoint]] = None # list of initial query points
    point_id: Optional[List] = None # list of point ids being tracked
    fps: Optional[int] = None
    width: Optional[int] = None
    height: Optional[int] = None

def create_video_from_frames(output_path, frames_dir, start_frame, end_frame, fps=6,):
    """
    Creates a video file from a sequence of frames in a directory.
    Helper for kubric-style dataset

    Args:
        frames_dir (str): Directory containing the frames
        start_frame (int): Starting frame number
        end_frame (int): Ending frame number
        fps (int): Frames per second for the output video
        pad_frames (bool): Whether to pad frames to make them divisible by 16

    Returns:
        str: Path to the created video file
    """
    # Generate output path
    if file_exists(output_path):
        return output_path

    # Get list of frame files within the range
    frame_files = []
    for i in range(start_frame, end_frame + 1):
        frame_path = join(frames_dir, f"{i:03d}.png")  # Using 3-digit frame numbers
        if file_exists(frame_path):
            frame_files.append(frame_path)

    if not frame_files:
        raise ValueError(f"No frames found in range {start_frame} to {end_frame} in {frames_dir}")
    
    # Read and pad frames
    frames = []
    for f in frame_files:
        frame = iio.imread(f)
        frames.append(frame)

    # Read frames and write video
    # frames = [iio.imread(f) for f in frame_files]
    iio.imwrite(output_path, frames, fps=fps, codec='libx264')

    return output_path

class KubricPointTracking(DatasetBase):
    # NOTE: hard-coded based on CoTracker3-Kubric dataset
    NUM_VIDEOS=5869
    NUM_FRAMES=120

    data_path = join(VIDEO_DATA_HOME, "point_track", "CoTracker3_Kubric") # Define correct data path in subclass
    dataset_name = "cotracker_kubric"

    def __init__(
        self, 
        split: Literal["train"],
        sample_strategy: Literal["random", "random_no_overlap", "grid_5x5"] = "random",
        prompt_type: Literal["point_track_all_frames"] = "point_track_all_frames",
        num_points: int = 25,
        num_samples_per_video: int = 1,
        max_frames: int = 120,
        sliding_window_stride: int = None, # only used if max_frames < total frames
    ):
        self.split = split
        self.sample_strategy = sample_strategy
        self.prompt_type = prompt_type
        self.num_points = num_points
        self.num_samples_per_video = num_samples_per_video
        self.max_frames = max_frames
        self.sliding_window_stride = sliding_window_stride if sliding_window_stride is not None else max_frames

        super().__init__(split)
    
    def process_bounded_video(self, video_info):
        video_path, video_id, start_frame, end_frame, fps = video_info
        frames_dir = join(self.data_path, 'data', video_id, 'frames',)
        try:
            output_video_path = create_video_from_frames(video_path, frames_dir, start_frame, end_frame, fps)
            iio.immeta(output_video_path) # NOTE: Check if video was created successfully. Remove if not needed.
            return output_video_path
        except Exception as e:
            logging.error(f"Error processing video {video_path}: {e}")
            return None
    
    def load(self):

        data_file_name = f"{self.split}/llava/prompt_type-{self.prompt_type}/sample-{self.sample_strategy}-num_points_{self.num_points}-num_samples_per_video-{self.num_samples_per_video}.jsonl"
        data_file = join(self.data_path, data_file_name)

        data = []
        logging.info(f"Loading data from {data_file}...")
        with open(resource_path(data_file), 'r') as f:
            for line in tqdm(f):
                item = json.loads(line)
                data += self.format_data(item)
        
        from concurrent.futures import ProcessPoolExecutor, as_completed
        video_infos = {d['video']: 
            (d['video'],
            d['metadata']['video_id'],
            d['metadata']['batch_start_frame'],
            d['metadata']['batch_end_frame'],
            d['metadata']['fps']) 
            for d in data}

        ###################################################################### 
        # This is only necessary to run only once to cache the processed videos.
        # Process all bounded videos with multiprocessing
        ######################################################################

        # logging.info(f"Processing {len(video_infos)} bounded videos using 64 workers...")
        # failed_videos = set()
        # with ProcessPoolExecutor(max_workers=64) as executor:
        #     future_to_info = {
        #         executor.submit(self.process_bounded_video, info): info
        #         for info in video_infos.values()
        #     }
        #     processed_videos = {}
        #     with tqdm(total=len(future_to_info)) as pbar:
        #         for future in as_completed(future_to_info):
        #             info = future_to_info[future]
        #             result = future.result()
        #             if result is not None:
        #                 processed_videos[info] = result
        #             else:
        #                 failed_videos.add(info[0])
        #             pbar.update(1)
        
        # content = "\n".join(f"{vid}" for vid in failed_videos) + "\n" if failed_videos else ""
        # failed_videos_file = f"{self.split}_max_frames_{self.max_frames}_failed_videos.txt"
        # write_file(self.data_path, failed_videos_file, 
        #            content, save_overwrite=True)
        # logging.info(f"Processed {len(processed_videos)}/{len(video_infos)} videos successfully.")
        # logging.info(f"Saved {len(failed_videos)} failed videos to {self.data_path}/{failed_videos_file}")
        ######################################################################

        # Quickly check that video files exist and filter data with missing videos
        with ProcessPoolExecutor(max_workers=32) as executor:
            future_to_video = {
                executor.submit(file_exists, info[0]): info
                for info in video_infos.values()
            }
            missing_videos = set()
            with tqdm(total=len(future_to_video)) as pbar:
                for future in as_completed(future_to_video):
                    info = future_to_video[future]
                    exists = future.result()
                    if not exists:
                        missing_videos.add(info[0])
                    pbar.update(1)
            logging.info(f"Checked existence of {len(future_to_video)} videos, {len(missing_videos)} missing.")
        data = [d for d in data if d['video'] not in missing_videos]
        logging.info(f"Final dataset size after filtering missing videos: {len(data)}")
            
        return data
    
    def format_data(self, item: Dict[str, Any]) -> List[Dict[str, Any]]:
        video_fps = item['fps']
        w = item['metadata']['w']
        h = item['metadata']['h']

        # [{'point_id': int, 'point_trajectory': [[x, y, frame_idx, occluded], ...]}, ...]
        point_trajectory: list = item['point_trajectory']
        max_frame = max([pt['point_trajectory'][-1][2] for pt in point_trajectory])

        # Create point message for this batch
        # frame_idx -> {'frame': int, 'time': float, 'points': {point_id: {'position': [x, y], 'occluded': bool}, ...}}
        points_per_frame = {}
        for pt in point_trajectory:
            point_id = pt['point_id']
            for pos in pt['point_trajectory']:
                x, y, frame_idx, visible, out_of_view, occluded = pos
                if out_of_view:
                    continue

                if not visible:
                    assert occluded, "If point is not visible and in view, it must be occluded"

                if frame_idx not in points_per_frame:
                    points_per_frame[frame_idx] = {
                        'frame': frame_idx,
                        'points': {}
                    }
                points_per_frame[frame_idx]['points'][point_id] = {
                    'point': [x, y],
                    'occluded': bool(occluded)
                }
        points_message: List[PointTrajectoryEntry] = [{ # not PointMessage class but follows its dict format.
            'frame': frame_idx,
            'points': frame_data['points'], 
            # 'time': frame_data['time'], # will be filled later after batching
        } for frame_idx, frame_data in points_per_frame.items()]
        points_message = sorted(points_message, key=lambda x: x['frame'])

        # Split frames into batches of max_frames (e.g. [0-39], [40-79], [80-119])
        video_point_track_message = []

        for batch_start in range(0, max_frame + 1, self.sliding_window_stride):
            batch_end = batch_start + self.max_frames - 1

            # Filter points that fall within this batch range
            batch_points_message = [pt for pt in points_message 
                                    if batch_start <= pt['frame'] <= batch_end]

            # Only create a batch if there are points in this range
            if batch_points_message:
                # offset frame indices to start from 0 in the batch
                for pt in batch_points_message:
                    pt['frame'] -= batch_start
                    pt['time'] = pt['frame'] / video_fps
                    
                video_path = join(self.data_path, 'data', item['video'], f"video_{batch_start:03d}_{batch_end:03d}_fps_{video_fps}.mp4")
                video_point_track_message.append({
                    'style': 'video_point_track_all_frames_with_occlusion',
                    'video': video_path,
                    # 'question': item['prompt'],
                    'points': batch_points_message,
                    'width': w,
                    'height': h,
                    'metadata': {
                        'w': w,
                        'h': h,
                        'example_id': item['id'],
                        'video_id': item['video'],
                        'fps': video_fps,
                        'point_id': item['point_id'],
                        'batch_start_frame': batch_start,
                        'batch_end_frame': batch_end,
                        'frame_range': f"{batch_start}-{batch_end}"
                    }
                })

        return video_point_track_message
    
    def get(self, idx, rng):
        return self.data[idx]


def _resize_image(image, shape):
    image = np.asarray(image)
    if len(image.shape) != 3 or image.dtype not in [np.uint8, np.float32]:
        raise ValueError(f"Invalid image shape={image.shape} dtype={image.dtype}")
    return np.array(
        PIL.Image.fromarray(image).resize(
            shape[::-1], resample=PIL.Image.Resampling.LANCZOS
        ),
        dtype=image.dtype,
    )


def resize_video(video, shape):
    """Resizes `video` to specified spatial dimensions using a Lanczos filter.

    Args:
      video: Iterable of images.
      shape: 2D spatial dimensions (height, width) of output video.

    Returns:
      A resampled video whose spatial dimensions match `shape`.
    """
    # Follows mediapy.resize_video
    if len(shape) != 2:
        raise ValueError(f'Shape {shape} is not of the form (height, width).')
    if not all(isinstance(i, numbers.Integral) for i in shape):
        raise ValueError(f'Shape {shape} contains non-integers.')
    return np.array([_resize_image(image, shape) for image in video])


class VideoPointTrackingEval(DatasetBase):
    data_path = os.path.join(VIDEO_DATA_HOME)
    dataset_name = None # abstract, must be defined in subclass

    def __init__(
        self, 
        split:str="valid",
        prompt_type: Literal["point_track_all_frames"] = "point_track_all_frames",
        video_fps:int=6,
        num_points:int=5,
        max_frames:int=60,
        sliding_window_stride:int=None, # only used if max_frames < total frames
        resize_to:list=[256, 256],
    ):
        """
        TODO: Add documentation
        ref: https://github.com/facebookresearch/co-tracker/blob/main/cotracker/datasets/tap_vid_datasets.py#L136
        """
        assert prompt_type in ["point_track_all_frames"]
        
        self.split = split
        self.prompt_type = prompt_type        
        self.resize_to = resize_to
        self.queried_first = not "strided" in self.dataset_name
        self.video_fps = video_fps
        self.for_eval = split not in ["train"]
        self.max_frames = max_frames
        self.num_points = num_points

        super().__init__(split)

    def resize_video(self, video: np.ndarray, output_size: Tuple[int, int]) -> np.ndarray:
        """Resize a video to output_size."""
        # If you have a GPU, consider replacing this with a GPU-enabled resize op,
        # such as a jitted jax.image.resize.  It will make things faster.
        resize_video(video, output_size)

    def sample_queries_first(
        self,
        target_occluded: np.ndarray,
        target_points: np.ndarray,
        frames: np.ndarray,
    ) -> Mapping[str, np.ndarray]:
        """Package a set of frames and tracks for use in TAPNet evaluations.
        Given a set of frames and tracks with no query points, use the first
        visible point in each track as the query.
        Args:
            target_occluded: Boolean occlusion flag, of shape [n_tracks, n_frames],
                where True indicates occluded.
            target_points: Position, of shape [n_tracks, n_frames, 2], where each point
                is [x,y] scaled between 0 and 1.
            frames: Video tensor, of shape [n_frames, height, width, 3].  Scaled between
                -1 and 1.
        Returns:
        A dict with the keys:
            video: Video tensor of shape [1, n_frames, height, width, 3]
            query_points: Query points of shape [1, n_queries, 3] where
            each point is [t, y, x] scaled to the range [-1, 1]
            target_points: Target points of shape [1, n_queries, n_frames, 2] where
            each point is [x, y] scaled to the range [-1, 1]
        """
        valid = np.sum(~target_occluded, axis=1) > 0
        target_points = target_points[valid, :]
        target_occluded = target_occluded[valid, :]

        query_points = []
        for i in range(target_points.shape[0]):
            index = np.where(target_occluded[i] == 0)[0][0]
            x, y = target_points[i, index, 0], target_points[i, index, 1]
            query_points.append(np.array([index, y, x]))  # [t, y, x]
        query_points = np.stack(query_points, axis=0)

        return {
            "video": frames,
            "query_points": query_points,
            "target_points": target_points,
            "occluded": target_occluded,
        }

    def sample_queries_strided(
        self,
        target_occluded: np.ndarray,
        target_points: np.ndarray,
        frames: np.ndarray,
        query_stride: int = 5,
    ) -> Mapping[str, np.ndarray]:
        """Package a set of frames and tracks for use in TAPNet evaluations.
        Given a set of frames and tracks with no query points, sample queries
        strided every query_stride frames, ignoring points that are not visible
        at the selected frames.
        Args:
        target_occluded: Boolean occlusion flag, of shape [n_tracks, n_frames],
            where True indicates occluded.
        target_points: Position, of shape [n_tracks, n_frames, 2], where each point
            is [x,y] scaled between 0 and 1.
        frames: Video tensor, of shape [n_frames, height, width, 3].  Scaled between
            -1 and 1.
        query_stride: When sampling query points, search for un-occluded points
            every query_stride frames and convert each one into a query.
        Returns:
        A dict with the keys:
            video: Video tensor of shape [1, n_frames, height, width, 3].  The video
            has floats scaled to the range [-1, 1].
            query_points: Query points of shape [1, n_queries, 3] where
            each point is [t, y, x] scaled to the range [-1, 1].
            target_points: Target points of shape [1, n_queries, n_frames, 2] where
            each point is [x, y] scaled to the range [-1, 1].
            trackgroup: Index of the original track that each query point was
            sampled from.  This is useful for visualization.
        """
        tracks = []
        occs = []
        queries = []
        trackgroups = []
        total = 0
        trackgroup = np.arange(target_occluded.shape[0])
        for i in range(0, target_occluded.shape[1], query_stride):
            mask = target_occluded[:, i] == 0
            query = np.stack(
                [
                    i * np.ones(target_occluded.shape[0:1]),
                    target_points[:, i, 1],
                    target_points[:, i, 0],
                ],
                axis=-1,
            )
            queries.append(query[mask])
            tracks.append(target_points[mask])
            occs.append(target_occluded[mask])
            trackgroups.append(trackgroup[mask])
            total += np.array(np.sum(target_occluded[:, i] == 0))

        return {
            "video": frames,
            "query_points": np.concatenate(queries, axis=0),
            "target_points": np.concatenate(tracks, axis=0),
            "occluded": np.concatenate(occs, axis=0),
            "trackgroup": np.concatenate(trackgroups, axis=0),
        }

    def load(self):

        if self.dataset_name == "tapvid_davis":
            with open(os.path.join(self.data_path, "tapvid_davis.pkl"), "rb") as f:
                self.points_dataset = pickle.load(f)
            self.video_names = list(self.points_dataset.keys())

        elif self.dataset_name == "tapvid_kinetics":
            all_paths = glob.glob(os.path.join(self.data_path, "*_of_0010.pkl"))
            points_dataset = []
            for pickle_path in all_paths:
                with open(pickle_path, "rb") as f:
                    data = pickle.load(f)
                    points_dataset = points_dataset + data
            self.points_dataset = points_dataset

        elif self.dataset_name == "tapvid_robotap":
            all_paths = glob.glob(os.path.join(self.data_path, "robotap_split*.pkl"))
            points_dataset = None
            for pickle_path in all_paths:
                with open(pickle_path, "rb") as f:
                    data = pickle.load(f)
                    if points_dataset is None:
                        points_dataset = dict(data)
                    else:
                        points_dataset.update(data)
            self.points_dataset = points_dataset
            self.video_names = list(self.points_dataset.keys())


        elif self.dataset_name == "tapvid_rgb_stacking":
            with open(os.path.join(self.data_path, "tapvid_rgb_stacking.pkl"), "rb") as f:
                self.points_dataset = pickle.load(f)
            self.video_names = [i for i in range(len(self.points_dataset))]
        
        else:
            raise ValueError(f"Unknown dataset_name: {self.dataset_name}")

        logging.info("found %d unique videos in %s" % (len(self.points_dataset), self.data_path))

        data = []
        for index in tqdm(range(len(self.points_dataset))):
            if self.dataset_name in ["tapvid_davis", "tapvid_robotap"]:
                video_name = self.video_names[index]
            else:
                video_name = index
            video = self.points_dataset[video_name]
            frames = video["video"] # [t, h, w, 3], uint8

            if isinstance(frames[0], bytes):
                # TAP-Vid is stored and JPEG bytes rather than `np.ndarray`s.
                def decode(frame):
                    byteio = io.BytesIO(frame)
                    img = Image.open(byteio)
                    return np.array(img)

                frames = np.array([decode(frame) for frame in frames])

            target_points = self.points_dataset[video_name]["points"]
            if self.resize_to is not None:
                frames = self.resize_video(frames, self.resize_to)
                target_points *= np.array(
                    [self.resize_to[1] - 1, self.resize_to[0] - 1]
                )  # 1 should be mapped to resize_to-1
            else:
                target_points *= np.array([frames.shape[2] - 1, frames.shape[1] - 1])

            target_occ = self.points_dataset[video_name]["occluded"]
            if self.queried_first:
                converted = self.sample_queries_first(target_occ, target_points, frames)
            else:
                converted = self.sample_queries_strided(target_occ, target_points, frames)
            assert converted["target_points"].shape[0] == converted["query_points"].shape[0]

            # Format initial query points
            initial_points = []
            for point in converted['query_points']:
                t, y, x = point
                initial_points.append({
                    'id': len(initial_points),
                    'point': [x, y],
                    'time': t / self.video_fps,
                    'frame': int(t),
                })
            converted['initial_points'] = initial_points

            # FIXME: figure out how to load previous points when max_frames < total frames
            # For now, just cut video by max_frames
            if self.max_frames is not None and frames.shape[0] > self.max_frames:
                converted['video'] = frames[:self.max_frames]
                converted['target_points'] = converted['target_points'][:, :self.max_frames]
                converted['occluded'] = converted['occluded'][:, :self.max_frames]
                logging.warning(f"Video {video_name} has more than {self.max_frames} frames, cutting to first {self.max_frames} frames.")

            # Split into max num_points for the query points for each batch example
            query_point_ids = [pt['id'] for pt in initial_points]
            for i in range(0, len(converted['query_points']), self.num_points):
                query_point_ids_batch = query_point_ids[i:i+self.num_points]
                ex_id = f"{video_name}_points_{i}_{i+len(query_point_ids_batch)}"
                ex = {'id': ex_id, 
                      'video_id': video_name, 
                      'query_point_ids': query_point_ids_batch, 
                      **converted}
                message_list: List[VideoPointTrackMessage] = self._create_message_list(ex)
                data += message_list
        
        logging.info(f"Final dataset size after processing: {len(data)}")
        return data

    def _create_message_list(self, ex) -> List[VideoPointTrackMessage]:
        """
        Create message list for MeVis dataset with points for object tracking.
        
        Args:
            ex (dict): Dataset example containing {
                'video_id': str,
                'video': np.ndarray [t, h, w, 3],
                'query_points': list of query points, [n_queries]
                'occluded': np.ndarray [n_queries, t],
                'target_points': np.ndarray [n_queries, t, 2],
            }
            
        Returns:
            list: List of message dictionaries with style, prompt, points data
        """
        
        query_point_ids = ex['query_point_ids']
        points_per_frame = {} 
        initial_points = {} # Find initial points (first visible point in each track)
        for frame_idx in range(ex['video'].shape[0]):
            time = frame_idx / self.video_fps
            # Format: {frame_idx: {time: time, points: {obj_id: point_data}}}
            if frame_idx not in points_per_frame:
                points_per_frame[frame_idx] = {
                    'frame': frame_idx,
                    'time': time,
                    'points': {}
                }

            # Check for initial points in this frame
            for point_id in query_point_ids:
                if point_id not in initial_points and not ex['occluded'][point_id, frame_idx]:
                    initial_points[point_id] = {
                        'id': point_id,
                        'point': ex['target_points'][point_id, frame_idx].tolist(),
                        'time': time,
                        'frame': frame_idx,
                    }

                # Add points for this frame
                points_per_frame[frame_idx]['points'][point_id] = {
                    'point': ex['target_points'][point_id, frame_idx].tolist(),
                    'occluded': bool(ex['occluded'][point_id, frame_idx]),
                }
                
            # points_per_frame[frame_idx]['points'].update({
            #     point_id: {
            #         'point': ex['target_points'][point_id, frame_idx],
            #         'occluded': bool(ex['occluded'][point_id, frame_idx]),
            #     } for point_id in query_point_ids
            # })
        points_message: List[PointTrajectoryEntry] = [{
            'frame': pt['frame'],
            'time': pt['time'],
            'points': pt['points'],
        } for pt in points_per_frame.values()]
        points_message.sort(key=lambda x: x['frame'])

        initial_points = list(initial_points.values())
        initial_points.sort(key=lambda x: x['frame']*10000 + x['point'][0]*100 + x['point'][1]) # Sort by frame, then x, then y
        query_point_ids = [pt['id'] for pt in initial_points] # Re-order query_point_ids to match initial_points order

        w, h = ex['video'].shape[2], ex['video'].shape[1]

        # Need to save video to disk if not already exists
        video_path = join(self.data_path, f'videos_fps{self.video_fps}_max_frames_{self.max_frames}', f"{ex['video_id']}.mp4")
        if not file_exists(video_path):
            os.makedirs(os.path.dirname(video_path), exist_ok=True)
            iio.imwrite(video_path, ex['video'], fps=self.video_fps, codec='libx264')
            logging.info(f"Saved video to {video_path}")
        
        return [{
            'style': 'video_point_track_all_frames_with_occlusion',
            'video': video_path,
            "points": points_message,
            "initial_points": initial_points,
            "fps": self.video_fps,
            "width": w,
            "height": h,
            'metadata': {
                'w': w,
                'h': h,
                'example_id': ex['id'],
                'video_id': ex['video_id'],
                'video_fps': self.video_fps,
                'point_id': query_point_ids,
                'query_points': ex['query_points'][query_point_ids],
                'occluded': ex['occluded'][query_point_ids],
                'target_points': ex['target_points'][query_point_ids],
                'dataset_name': self.dataset_name,
            }
        }]

    def get(self, idx, rng):
        """Get a dataset item with the appropriate style."""
        # Use the original style from the message list to ensure proper formatting
        return self.data[idx]

class TapDavis(VideoPointTrackingEval):
    data_path = os.path.join(VIDEO_DATA_HOME, "point_track", "tapvid_davis")
    dataset_name = "tapvid_davis"

class TapKinetics(VideoPointTrackingEval):
    data_path = os.path.join(VIDEO_DATA_HOME, "point_track", "tapvid_kinetics")
    dataset_name = "tapvid_kinetics"

class TapRobotap(VideoPointTrackingEval):
    data_path = os.path.join(VIDEO_DATA_HOME, "point_track", "tapvid_robotap")
    dataset_name = "tapvid_robotap"

class TapRGBStacking(VideoPointTrackingEval):
    data_path = os.path.join(VIDEO_DATA_HOME, "point_track", "tapvid_rgb_stacking")
    dataset_name = "tapvid_rgb_stacking"

class DynamicReplica(VideoPointTrackingEval):
    data_path = os.path.join(VIDEO_DATA_HOME, "point_track", "dynamic_replica_data")
    dataset_name = "dynamic_replica_data"
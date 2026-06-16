import logging
import os
import json

import ast
import random
import numpy as np
from os.path import join
from enum import Enum

from datasets import load_from_disk, load_dataset

from tqdm import tqdm

from olmo.io import (
    read_file,
    is_url,
    file_exists,
    get_bytes_range,
    glob as olmo_glob,
    list_directory, write_file
)
from olmo.torch_util import get_global_rank

from olmo.util import resource_path
from olmo.data.dataset import DatasetBase, VIDEO_DATA_HOME


log = logging.getLogger(__name__)

from typing import Any, Dict, List, Literal, Optional
from typing_extensions import TypedDict

MAX_VIDEO_FPS = 8

PROMPT_TYPES = [
    "point_track_per_frame",       # Track points across all frames
    "point_ground_start_end",      # Only first and last appearance of points
    "single_point_track_per_frame", # Given starting point, track object to end
]

TRAINING_VIDEO_OBJECT_TRACKING_DATASETS = [
    # Tracking tasks
    "mevis_point_track_per_frame_fps_6_sample_fps_2_max_objects_5",
    "mevis_single_point_track_per_frame_fps_6_interval_seconds_0.5",
    "burst_point_track_per_frame_fps_6_sample_fps_2_max_objects_3",
    "burst_single_point_track_per_frame_fps_6_interval_seconds_0.5",
    "ref-yt-vos_point_track_per_frame_fps_6_sample_fps_2",
    "ref-davis17_point_track_per_frame_fps_6_sample_fps_2",
    "lv-vis_point_track_per_frame_fps_4_sample_fps_1_max_objects_5",
    "lv-vis_single_point_track_per_frame_fps_4_interval_seconds_1.0",
    "vicas_point_track_per_frame_fps_6_sample_fps_2_max_objects_5",
    "vicas_single_point_track_per_frame_fps_6_interval_seconds_0.5",
    "revos_point_track_per_frame_fps_6_sample_fps_2_max_objects_5",
    "revos_single_point_track_per_frame_fps_6_interval_seconds_0.5",

    "prolific_video_text_queries_filtered_101925_point_track_per_frame_sample_fps_1_max_objects_7",
    "prolific_video_text_queries_filtered_101925_point_track_per_frame_sample_fps_2_max_objects_5",
    "prolific_video_text_queries_filtered_101925_single_point_track_per_frame_sample_fps_1",
    "prolific_video_text_queries_filtered_101925_single_point_track_per_frame_sample_fps_2",

    # Bbox single point tasks
    "bbox-single-point-track_lvos-v1_single_point_track_per_frame_sample_fps_1",
    "bbox-single-point-track_lvos-v1_single_point_track_per_frame_sample_fps_2",
    "bbox-single-point-track_lvos-v2_single_point_track_per_frame_sample_fps_1",
    "bbox-single-point-track_lvos-v2_single_point_track_per_frame_sample_fps_2",
    "bbox-single-point-track_tnl2k_single_point_track_per_frame_sample_fps_1",

    # Grounding tasks
    "mevis_point_ground_start_end_fps_6_sample_fps_2_max_objects_10",
    "burst_point_ground_start_end_fps_6_sample_fps_2_max_objects_10",
    "lv-vis_point_ground_start_end_fps_4_sample_fps_1",
    "vicas_point_ground_start_end_fps_6_sample_fps_2",
    "revos_point_ground_start_end_fps_6_sample_fps_2",

    "prolific_video_text_queries_filtered_101925_point_ground_start_end",

]

EVAL_VIDEO_OBJECT_TRACKING_DATASETS = [

    # Tracking tasks
    'mevis_point_track_per_frame_fps_6_sample_fps_1',
    'mevis_single_point_track_per_frame_fps_6_interval_seconds_1.0',

    'ref-yt-vos_point_track_per_frame_fps_6_sample_fps_1',
    'ref-davis17_point_track_per_frame_fps_6_sample_fps_1',

    'burst_point_track_per_frame_fps_6_sample_fps_1',
    'vpos_point_track_per_frame_fps_6_sample_fps_1',

    # Grounding tasks
    'mevis_point_ground_start_end_fps_6_sample_fps_6',

]

class Point(TypedDict):
    point: List[float] # [x, y] coordinates
    occluded: Optional[bool] = False

class PointTrajectoryEntry(TypedDict):
    frame: int # frame index
    time: float # time in seconds
    points: Dict[str, Point] # object_id -> {'point': [x, y], 'occluded': bool}

class VideoObjectTrackMessage(TypedDict):
    style: str
    question: str
    label: str
    points: Optional[List[PointTrajectoryEntry]] = None # point trajectories for each frame
    sampling_fps: Optional[int] = None # sampling fps for the points
    width: Optional[int] = None
    height: Optional[int] = None

def get_video_object_track_prompt_type(dataset_name: str) -> str:
    """
    Resolve a prompt type by longest substring match.
    HACK: guarantees 'single_point_*' wins over 'point_*' without relying on a list order.
    """

    hits = [m for m in PROMPT_TYPES if m in dataset_name]
    if not hits:
        return None
    return max(hits, key=lambda m: len(m))  # HACK: longest value wins

def get_candidate_sampling_fps(video_fps: int, sampling_fps: int, max_fps=MAX_VIDEO_FPS) -> List[int]:
    """
    Return the subset of `video_fps` factors that remain multiples of `sampling_fps`.

    Examples:
        >>> get_candidate_sampling_fps(video_fps=6, sampling_fps=2)
        [2, 6]
        >>> get_candidate_sampling_fps(video_fps=5, sampling_fps=1)
        [1, 5]
        >>> get_candidate_sampling_fps(video_fps=2, sampling_fps=2)
        [2]
        >>> get_candidate_sampling_fps(video_fps=5, sampling_fps=2)
        Traceback (most recent call last):
            ...
        ValueError: sampling_fps=2 must divide video_fps=5 to produce consistent frame steps.
    """

    video_fps = int(video_fps)
    sampling_fps = int(sampling_fps)

    if sampling_fps is None:
        raise ValueError("sampling_fps must be provided")
    if video_fps <= 0 or sampling_fps <= 0:
        raise ValueError(f"video_fps and sampling_fps must be positive (got {video_fps}, {sampling_fps})")
    if video_fps % sampling_fps != 0:
        raise ValueError(f"sampling_fps={sampling_fps} must divide video_fps={video_fps}.")

    candidates = []
    for candidate in range(sampling_fps, video_fps + 1, sampling_fps):
        if candidate > max_fps:
            break
        if video_fps % candidate == 0:
            candidates.append(candidate)
    return candidates


class ObjectTracking(DatasetBase):
    data_path = os.path.join(VIDEO_DATA_HOME) # Define correct data path in subclass
    dataset_name = None # abstract, must be defined in subclass

    def __init__(self, split, 
                 video_dir=None, 
                 point_type="largest_center", 
                 prompt_type="point_track_per_frame", 
                 video_fps:int=None, 
                 sampling_fps:int=None,
                 interval_seconds:float=None,
                 max_objects:int=None,
                 use_fps_sampling:bool=True,
                 min_sampling_fps:int=1
                 ):
        """
        Base dataset for video pointing task with grounding and tracking objects.
        Can control different point sampling methods and prompt types.

        Args:
            split (str): Dataset split - ["train", "validation", "test"]
            video_dir (str, optional): Custom video directory path; otherwise, creates default path based on split.
            point_type (str, optional): Type of point extraction method. Defaults to "largest_center".
            prompt_type (str, optional): Type of point data to generate.
                Options: 
                    - "point_track_per_frame" (default): Track points across all frames
                    - "point_ground_start_end": Only first and last appearance of points
                    - "single_point_track_per_frame": Given starting point, track object to end
            video_fps (int, optional): FPS of source videos. Defaults to 6.
            sampling_fps (int, optional): Rate to sample points at, in frames per second.
                If provided, points will be sampled at this frame rate.
            interval_seconds (float, optional): If provided, will sample points at this interval in seconds.
                For example, 0.5 means sample every 0.5 seconds.
            max_objects (int, optional): Maximum number of objects to track in each video. Defaults to None (no limit).
            predicted_points_file (str, optional): Path to JSON file containing initial set of predicted points.
                Only used for single_point_track_per_frame prompt type if you want to provide your own starting points.
            use_fps_sampling (bool, optional): Whether to use fps-based sampling overrides during data loading.
            min_sampling_fps (int, optional): Minimum sampling fps to consider when generating candidate sampling fps values.

        Note:
            Either `sampling_fps` OR `interval_seconds` must be provided.
            If both are provided, `sampling_fps` will be used for frame-based sampling,
            - Use sampling_fps for frame-based sampling (e.g., sampling_fps=2 means 2 frames per second)
            - Use interval_seconds for time-based sampling (e.g., interval_seconds=0.5 means every 0.5 seconds)
        """
        assert split in ["train", "validation", "test"], f"Invalid split: {split}. Expected: train|validation|test"
        assert prompt_type in PROMPT_TYPES

        # Dataset parameters
        self.dataset_name = self._get_dataset_name()
        self.split_dir = self._get_split_dir(split)
        self.is_eval = split not in ["train"]

        # Video parameters
        self.video_fps = video_fps
        self.video_dir = video_dir if video_dir else self._get_video_dir(split)

        # Point-Track parameters
        self.prompt_type = prompt_type
        self.point_type = point_type
        self.sampling_fps = sampling_fps
        self.interval_seconds = interval_seconds
        self.max_objects = max_objects
        if prompt_type in ["point_track_per_frame", "single_point_track_per_frame"]:
            assert (self.sampling_fps is not None) or (self.interval_seconds is not None), \
                "Either `sampling_fps` or `interval_seconds` should be provided."
        self.use_fps_sampling = use_fps_sampling
        self.min_sampling_fps = min_sampling_fps
        
        self.task_name = self._build_task_name()
        
        # Load segmentation dataset from HF for evaluation
        if self.is_eval:
            self._segmentation_dataset, self._example_id_to_index = \
                self._load_segmentation_dataset(self.dataset_name, self.point_type)
        self.data_lookup = {} # example_id -> data index
        super().__init__(split)

    def _try_get_fps(self, video_fps):
        if video_fps is None:
            video_fps = self.video_fps
        try:
            self.get_candidate_sampling_fps(video_fps)
            return True
        except ValueError:
            return False

    def _get_dataset_name(self):
        if self.dataset_name is None:
            raise NotImplementedError("Subclasses must define `dataset_name`.")
        return self.dataset_name

    def _get_split_dir(self, split: str) -> str:
        dataset_split = self._get_dataset_split(split)
        return join(self.data_path, dataset_split)

    def _get_dataset_split(self, split: str) -> str:
        """Map generic split names to dataset-specific splits if needed."""
        return split
    
    def _get_video_dir(self, split):
        """Override this method in subclasses to customize video directory logic."""
        if self.video_fps is not None:
            return join(self.split_dir, f"videos_fps{self.video_fps}")
        else:
            return join(self.split_dir, "videos")
    
    def _build_task_name(self) -> str:
        """Generate a concise task name based on configuration."""              
        components = [                                                                          
            self.dataset_name,                   
            self.prompt_type,                                                                                                                        
        ]
        
        if self.video_fps is not None:                                                                
            components.append(f"fps_{self.video_fps}")
                                                                                                
        if self.sampling_fps is not None:        
            components.append(f"sample_fps_{self.sampling_fps}")                                
        elif self.interval_seconds is not None:  
            components.append(f"interval_seconds_{self.interval_seconds}")

        if self.max_objects is not None:                                                        
            components.append(f"max_objects_{self.max_objects}")
                                                                                                
        return "_".join(components)
    
    def _build_annotation_file_path(self) -> str:
        """Build the relative path to annotation file within split directory."""                                                                                     
        point_subdir = f"point-{self.point_type}"              
        task_subdir = f"prompt_type-{self.prompt_type}"              

        if self.sampling_fps is not None:                                                       
            filename = f"sample_fps-{self.sampling_fps}.json"          
        elif self.interval_seconds is not None: 
            filename = f"interval_seconds-{self.interval_seconds}.json"    
        else:
            if self.prompt_type in ["point_track_per_frame", "single_point_track_per_frame"]:
                raise ValueError(f"For {self.prompt_type}, either `sampling_fps` or `interval_seconds` must be provided.")
            else:
                filename = f"sample_fps-None.json"
                                                                                                
        if self.video_fps:
            filename = f"fps-{self.video_fps}-{filename}"

        if self.max_objects is not None:                                                        
            filename = filename.replace(".json", f"-max_objects-{self.max_objects}.json")       

        return join(point_subdir, task_subdir, filename)                                        
                                                                                                
    def _get_full_annotation_file_path(self) -> str:                                            
        """
        Get the complete absolute path to the annotation file.
        Probably not override this in subclasses.
        """                            
        annotation_relative_path = self._build_annotation_file_path()                           
        return join(self.split_dir, "llava-v3", annotation_relative_path)

    def _create_message_list(self, ex):
        """
        Create message list for a single example, formatting point data based on prompt_type.
        
        Args:
            ex (dict): Dataset example with trajectories and metadata
            
        Returns:
            list: List of message dictionaries with style, prompt, points data
        """
        # style: video_per_frame_point or video_start_end_point
        prompt_type = self.prompt_type.replace('per_frame_point', 'per_frame').replace('start_end_point', 'start_end')
        style = f"video_{prompt_type}"

        point_frames = []
        object_id_to_idx = {obj_id: idx for idx, obj_id in enumerate(ex['metadata']['mask_id'])}

        # Extract trajectory data depending on format
        # if self.prompt_type == "point_track_per_frame":
        # Process all frames with their trajectory points
        frame_trajectories = ex['frame_trajectories']
        for frame_data in frame_trajectories:
            frame_idx = frame_data['frame']
            time = frame_data['time']
            points = {k: v for k, v in frame_data['points'].items() if v is not None}  # Object ID -> point location mapping

            # Format: {frame_idx: {time: time, points: {obj_id: point_data}}}
            point_frames.append({
                'frame': frame_idx,
                'time': time,
                'points': {
                    object_id_to_idx[obj_id]: {
                        'point': point_info['point'],
                        'occluded': point_info['occluded']
                    } for obj_id, point_info in points.items()
                }
            })
            
        point_frames.sort(key=lambda x: x['frame'])
        
        if not point_frames:
            point_frames = None

        return [{
            "style": style,
            "question": ex.get('prompt', None), # NOTE: label is used to create tracking prompt instead
            "label": ex['expression'],
            "points": point_frames,
            "sampling_fps": self.sampling_fps,
            "width": ex['metadata']['w'],
            "height": ex['metadata']['h'],
        }]
    
    def load(self):
        data = []
        annotation_file_path = self._get_full_annotation_file_path()
        assert annotation_file_path.endswith(".json")
        hf_path = annotation_file_path.replace(".json", "-hf")

        if file_exists(hf_path):
            log.info(f"Loading {self.dataset_name} HF data from {hf_path}")
            data = load_from_disk(hf_path, keep_in_memory=True)
            n_pre_filter = len(data)
            self.data_lookup = {ex_id: i for i, ex_id in enumerate(data["id"])}
            data = data.filter(self._try_get_fps, input_columns="fps")
        else:
            log.info(f"Loading {self.dataset_name} json data from {annotation_file_path}")
            data = json.loads(read_file(annotation_file_path))
            n_pre_filter = len(data)
            self.data_lookup = {ex["id"]: i for i, ex in enumerate(data)}
            data = [x for x in data if self._try_get_fps(x.get("fps"))]
        if n_pre_filter != len(data) and get_global_rank() == 0:
            log.warning(f"Skipping {n_pre_filter-len(data)} because of FPS sampling mismatchs")
        return data

    def get_candidate_sampling_fps(self, video_fps: int) -> List[int]:
        """Get candidate sampling fps values for current video fps and dataset sampling fps."""
        if self.sampling_fps:
            return get_candidate_sampling_fps(video_fps, self.sampling_fps)
        elif self.interval_seconds:
            sampling_fps = int(round(1.0 / self.interval_seconds))
            return get_candidate_sampling_fps(video_fps, sampling_fps)
        else:
            return get_candidate_sampling_fps(video_fps, self.min_sampling_fps)
            # raise ValueError("Either `sampling_fps` or `interval_seconds` must be provided.")
    
    def _load_segmentation_dataset(self, dataset_name: str, point_type: str):
        """
        Load HuggingFace dataset containing segmentation masks for each frame.
        Used for evaluating if point is on object.
        
        Expected path format: {VIDEO_DATA_PATH}/{dataset_name}/{split}/annotation/{point_type}/

        Returns:
            segmentation_dataset: Loaded HF dataset object
            example_id_to_index (Dict[str, int]): Dict mapping example_id to HF dataset index for quick lookup

        Raises:
            FileNotFoundError: If dataset path does not exist
            ValueError: If dataset structure is invalid
        """
        try:
            from datasets import load_from_disk
            
            # Construct HF dataset path
            # hf_dataset_path = join(self.split_dir, "annotation", point_type+"_all_objects")
            hf_dataset_path = join(self.split_dir, "annotation", point_type)
            
            # Validate path exists
            if not os.path.exists(hf_dataset_path):
                raise FileNotFoundError(
                    f"Segmentation dataset not found at: {hf_dataset_path}. "
                )

            log.info(f"Loading segmentation dataset from: {hf_dataset_path}")
            segmentation_dataset = load_from_disk(hf_dataset_path)

            # Validate dataset structure
            if len(segmentation_dataset) == 0:
                raise ValueError(f"Empty dataset loaded from {hf_dataset_path}")
            
            # Check required fields in first item
            first_item = segmentation_dataset[0]
            required_fields = ['id', 'masks', 'h', 'w']
            missing_fields = [field for field in required_fields if field not in first_item]
            if missing_fields:
                raise ValueError(f"Dataset missing required fields: {missing_fields}. Found fields: {list(first_item.keys())}")
            
            # Create lookup index: example_id -> dataset index
            example_id_to_index: Dict[str, int] = {}
            for idx, item in enumerate(segmentation_dataset):
                example_id = item['id']
                if example_id in example_id_to_index:
                    log.warning(f"Duplicate example_id found: {example_id}")
                example_id_to_index[example_id] = idx
            
            log.info(f"Successfully loaded {len(segmentation_dataset)} examples, created lookup for {len(example_id_to_index)} unique IDs")

            return segmentation_dataset, example_id_to_index
            
        except (ImportError, ModuleNotFoundError) as e:
            raise ImportError(f"Failed to import required modules for dataset loading: {e}")

    def get_segmentation_masks_for_example(self, example_id: str) -> Dict[str, Any]:
        """
        Retrieve segmentation masks and metadata for a specific example.
        
        Args:
            example_id: Unique identifier for the video example
            
        Returns:
            Dictionary containing segmentation data:
            {
                "mask_id": List[str] - Object mask identifiers,
                "masks": Dict[str, List] - Masks for each object across frames,
                "h": int - Video frame height,
                "w": int - Video frame width
            }
            
        Returns:
            None if example not found or data invalid
        """
        if self._segmentation_dataset is None:
            log.error("Segmentation dataset not loaded. Call _load_segmentation_dataset() first.")
            return None
            
        if example_id not in self._example_id_to_index:
            log.warning(f"Example ID '{example_id}' not found in segmentation dataset. Available: {len(self._example_id_to_index)} examples")
            return None
            
        dataset_index = self._example_id_to_index[example_id]
        dataset_item = self._segmentation_dataset[dataset_index]

        # Extract and validate required fields for mask evaluation
        try:
            segmentation_data = {
                "mask_id": dataset_item["mask_id"],
                "masks": dataset_item["masks"],  # Note: 'mask' vs 'masks' field name
                "h": dataset_item["h"],
                "w": dataset_item["w"],
            }
            
            # Validate data structure
            if not isinstance(segmentation_data["mask_id"], list):
                log.warning(f"Invalid mask_id format for {example_id}: expected list, got {type(segmentation_data['mask_id'])}")
                return None
                
            return segmentation_data
            
        except KeyError as e:
            log.warning(f"Missing required field in dataset item for example '{example_id}': {e}")
            return None

    def get(self, idx, rng):
        """Get a dataset item with the appropriate style."""
        ex = self.data[idx]
        video = ex['video']
        example_id = ex['id']
        video_path = join(self.video_dir, video)
        self.data_lookup[example_id] = idx

        # Create message list with structured point data
        message_list = self._create_message_list(ex)
        video_fps = ex.get("fps", self.video_fps)

        # Prepare metadata
        metadata = ex.get('metadata', {})
        metadata.update({
            'example_id': example_id,
            'task': self.task_name,
            'expression': ex['expression'],
            'w': ex['metadata']['w'],
            'h': ex['metadata']['h'],
            'video_fps': self.video_fps,
            'video': video,
        })

        if self.use_fps_sampling:
            candidate_sampling_fps = self.get_candidate_sampling_fps(video_fps)
            metadata.update({
                'sampler_overrides': {
                    'frame_sample_mode': 'fps',
                    'candidate_sampling_fps': candidate_sampling_fps,
                    'min_fps': self.sampling_fps,
                }
            })

        # Include points and segmentation if not training
        if self.is_eval:
            metadata['points'] = message_list[0]['points']
            if self.prompt_type == 'single_point_track_per_frame':
                example_id = example_id.rsplit('_', 1)[0] # strip off last suffix that corresponds to individual object.

            # Load all object segmentation masks for query
            segmentation_data = self.get_segmentation_masks_for_example(example_id)
            metadata.update(segmentation_data)

        # Create dataset item
        return {
            'video': video_path,
            'message_list': message_list,
            'sampling_fps': self.sampling_fps, # in message_list but also here for convenience
            'metadata': metadata
        }

    def get_by_example_id(self, example_id: str):
        """Retrieve a dataset item by its example ID."""
        for item in self.data:
            if item['metadata']['example_id'] == example_id:
                return item
        log.warning(f"Example ID '{example_id}' not found in dataset.")
        return None

class Mevis(ObjectTracking):
    dataset_name = "mevis"
    data_path = os.path.join(VIDEO_DATA_HOME, "mevis", "MeViS_release")

    def _get_dataset_split(self, split):
        return 'train' if split == "train" else 'valid_u'

class MevisValid(ObjectTracking):
    ''' Codalab submission split for MeViS '''
    dataset_name = "mevis-valid"
    data_path = os.path.join(VIDEO_DATA_HOME, "mevis", "MeViS_release")

    def _get_dataset_split(self, split):
        return 'valid'
    
    def _create_message_list(self, ex):
        """
        Create message list for a single example, formatting point data based on prompt_type.
        
        Args:
            ex (dict): Dataset example with trajectories and metadata
            
        Returns:
            list: List of message dictionaries with style, prompt, points data
        """
        # style: video_per_frame_point or video_start_end_point
        prompt_type = self.prompt_type.replace('per_frame_point', 'per_frame').replace('start_end_point', 'start_end')
        style = f"video_{prompt_type}"

        return [{
            "style": style,
            "question": ex.get('prompt', None), # NOTE: label is used to create tracking prompt instead
            "label": ex['expression'],
            "sampling_fps": self.sampling_fps,
            "width": ex['metadata']['w'],
            "height": ex['metadata']['h'],
        }]
    
    def get(self, idx, rng):
        """Get a dataset item with the appropriate style."""
        ex = self.data[idx]
        video = ex['video']
        example_id = ex['id']
        video_path = join(self.video_dir, video)

        # Create message list with structured point data
        message_list = self._create_message_list(ex)
        video_fps = ex.get("fps", self.video_fps)

        # Prepare metadata
        metadata = ex.get('metadata', {})
        metadata.update({
            'example_id': example_id,
            'task': self.task_name,
            'expression': ex['expression'],
            'w': ex['metadata']['w'],
            'h': ex['metadata']['h'],
            'video_fps': self.video_fps,
            'video': video,
        })

        if self.use_fps_sampling:
            candidate_sampling_fps = self.get_candidate_sampling_fps(video_fps)
            metadata.update({
                'sampler_overrides': {
                    'frame_sample_mode': 'fps',
                    'candidate_sampling_fps': candidate_sampling_fps,
                    'min_fps': self.sampling_fps,
                }
            })
            
        # Create dataset item
        return {
            'video': video_path,
            'message_list': message_list,
            'sampling_fps': self.sampling_fps, # in message_list but also here for convenience
            'metadata': metadata
        }
    
    def _load_segmentation_dataset(self, dataset_name: str, point_type: str):
        log.info("MeViS valid split does not have segmentation masks for evaluation.")
        return None, None
    

class Burst(ObjectTracking):
    dataset_name = "burst"
    data_path = os.path.join(VIDEO_DATA_HOME, "TAO-Amodal", "BURST_annotations")

    def _get_dataset_split(self, split):
        return 'train' if split == "train" else 'test'

    def _get_video_dir(self, split):
        if self.video_fps is not None:
            return join(VIDEO_DATA_HOME, "TAO-Amodal", split, f"videos_fps{self.video_fps}")
        else:
            return join(VIDEO_DATA_HOME, "TAO-Amodal", split, f"videos")
    
    def get_segmentation_masks_for_example(self, example_id: str) -> Dict[str, Any]:
        """
        Burst test set only annotates masks at 1 fps instead of all frames, so we need to specify frame index in the masks. 
        Retrieve segmentation masks and metadata for a specific example.
        
        Args:
            example_id: Unique identifier for the video example
            
        Returns:
            Dictionary containing segmentation data:
            {
                "mask_id": List[str] - Object mask identifiers,
                "masks": Dict[str, List] - Masks for each object across frames,
                "h": int - Video frame height,
                "w": int - Video frame width
            }
            
        Returns:
            None if example not found or data invalid
        """

        segmentation_data = super().get_segmentation_masks_for_example(example_id)
        if segmentation_data is None:
            return None

        if self.split != 'train':
            ''' Adjust masks to include frame indices for evaluation '''
            dataset_index = self._example_id_to_index[example_id]
            dataset_item = self._segmentation_dataset[dataset_index]

            masks = segmentation_data["masks"] # masks in sampling fps
            frame_step= dataset_item['fps'] / dataset_item['sampling_fps']
            if not frame_step.is_integer():
                log.warning(f"Non-integer frame step {frame_step} for example {example_id} with fps {dataset_item['fps']} and sampling_fps {dataset_item['sampling_fps']}")

            for mask_id, mask_list in masks.items():
                if mask_list is None:
                    continue

                # Each mask_list corresponds to a list of masks for that object
                mask_list_with_frame_index = []
                for frame_idx, mask in enumerate(mask_list):
                    mask_list_with_frame_index.append({
                        'frame': int(frame_idx * frame_step),
                        'mask': mask
                    })
                masks[mask_id] = mask_list_with_frame_index
        
        return segmentation_data

class RefYoutubeVOS(ObjectTracking):
    dataset_name = "ref-yt-vos"
    data_path = os.path.join(VIDEO_DATA_HOME, "Ref-YT-VOS")

    def _get_dataset_split(self, split):
        return 'train' if split == "train" else 'valid'
    
class RefDavis17(ObjectTracking):
    dataset_name = "ref-davis17"
    data_path = os.path.join(VIDEO_DATA_HOME, "Ref-DAVIS17")

    def _get_dataset_split(self, split):
        return 'train' if split == "train" else 'valid'
    
class LVVIS(ObjectTracking):
    dataset_name = "lv-vis"
    data_path = os.path.join(VIDEO_DATA_HOME, "LV-VIS")

class YTVIS(ObjectTracking):
    dataset_name = "yt-vis"
    data_path = os.path.join(VIDEO_DATA_HOME, "YT-VIS")

class ViCaS(ObjectTracking):
    dataset_name = "vicas"
    data_path = os.path.join(VIDEO_DATA_HOME, "ViCaS")

    def _get_video_dir(self, split):
        return join(VIDEO_DATA_HOME, "ViCaS", f"videos_fps{self.video_fps}")

class ReVOS(ObjectTracking):
    dataset_name = "revos"
    data_path = os.path.join(VIDEO_DATA_HOME, "ReVOS")

class ReasonVOS(ObjectTracking):
    dataset_name = "reasonvos"
    data_path = os.path.join(VIDEO_DATA_HOME, "ReasonVOS")

    def _get_split_dir(self, split):
        return self.data_path
    
    def _get_video_dir(self, split):
        return join(VIDEO_DATA_HOME, "ReasonVOS", f"videos_fps{self.video_fps}")

class MoCA(ObjectTracking):
    dataset_name = "moca"
    data_path = os.path.join(VIDEO_DATA_HOME, "MoCA")

    def _get_split_dir(self, split):
        return self.data_path

    def _get_video_dir(self, split):
        return join(VIDEO_DATA_HOME, "MoCA", f"videos")

class VPoS(ObjectTracking):
    dataset_name = "vpos"
    data_path = os.path.join(VIDEO_DATA_HOME, "VPoS")

    def _get_split_dir(self, split):
        return self.data_path # always use same directory for all splits
    
    def get(self, idx, rng):
        item = super().get(idx, rng)
        # VPoS supports category
        ex = self.data[idx]
        item['metadata']['category'] = ex.get('category', None)
        return item


class Prolific(ObjectTracking):
    dataset_name = "prolific"
    data_path = os.path.join(VIDEO_DATA_HOME, "prolific")

    def __init__(self, split, 
                 subset_name,
                 video_dir=None, 
                 point_type="largest_center", 
                 prompt_type="point_track_per_frame", 
                 video_fps:int=None, 
                 sampling_fps:int=None,
                 interval_seconds:float=None,
                 max_objects:int=None,
                 predicted_points_file:Optional[str]=None,
                 use_fps_sampling:bool=True,
                 min_sampling_fps:int=1
                 ):
        """
        Base dataset for video pointing task with grounding and tracking objects.
        Can control different point sampling methods and prompt types.

        Args:
            split (str): Dataset split - ["train", "validation", "test"]
            subset_name (str): Specific subset of Prolific dataset to use.
            video_dir (str, optional): Custom video directory path; otherwise, use VIDEO_DATA_HOME
            point_type (str, optional): Type of point extraction method. Defaults to "largest_center".
            prompt_type (str, optional): Type of point data to generate.
                Options: 
                    - "point_track_per_frame" (default): Track points across all frames
                    - "point_ground_start_end": Only first and last appearance of points
                    - "single_point_track_per_frame": Given starting point, track object to end
            video_fps (int, optional): FPS of source videos, if provided
            sampling_fps (int, optional): Rate to sample points at, in frames per second.
                If provided, points will be sampled at this frame rate.
            interval_seconds (float, optional): If provided, will sample points at this interval in seconds.
                For example, 0.5 means sample every 0.5 seconds.
                For single_point_track_per_frame, this defines the interval between sampled frames after the starting point.
                     And if sampling_fps is provided, this is usedd to determine the starting point.
            max_objects (int, optional): Maximum number of objects to track in each video. Defaults to None (no limit).
            predicted_points_file (str, optional): Path to JSON file containing initial set of predicted points.
                Only used for single_point_track_per_frame prompt type if you want to provide your own starting points.
            use_fps_sampling (bool, optional): Whether to use fps-based sampling overrides during data loading.
            min_sampling_fps (int, optional): Minimum sampling fps to consider when generating candidate sampling fps values.

        Note:
            Either `sampling_fps` OR `interval_seconds` must be provided.
            If both are provided, `sampling_fps` will be used for frame-based sampling,
            - Use sampling_fps for frame-based sampling (e.g., sampling_fps=2 means 2 frames per second)
            - Use interval_seconds for time-based sampling (e.g., interval_seconds=0.5 means every 0.5 seconds)
        """
        assert split in ["train", "validation", "test"], f"Invalid split: {split}. Expected: train|validation|test"
        assert prompt_type in PROMPT_TYPES

        # Dataset parameters
        self.split_dir = join(self.data_path, subset_name)
        self.is_eval = split not in ["train"]

        # Video parameters
        self.video_fps = video_fps  # only use if fps not in metadata
        self.video_dir = video_dir if video_dir else VIDEO_DATA_HOME

        # Point-Track parameters
        self.prompt_type = prompt_type
        self.point_type = point_type
        self.sampling_fps = sampling_fps
        self.interval_seconds = interval_seconds
        self.max_objects = max_objects
        if prompt_type in ["point_track_per_frame", "single_point_track_per_frame"]:
            assert (self.sampling_fps is not None) or (self.interval_seconds is not None), \
                "Either `sampling_fps` or `interval_seconds` should be provided."
        self.use_fps_sampling = use_fps_sampling
        self.min_sampling_fps = min_sampling_fps
        self.task_name = self._build_task_name()
        
        # Load segmentation dataset from HF for evaluation
        if self.is_eval:
            self._segmentation_dataset, self._example_id_to_index = \
                self._load_segmentation_dataset(self.dataset_name, self.point_type)

        self.predicted_points_file = predicted_points_file
        self.data_lookup = {} # example_id -> data index

        DatasetBase.__init__(self, split)

    def _get_split_dir(self, split):
        return self.data_path # always use same directory for all splits

    def load(self):
        if self.predicted_points_file:
            assert self.prompt_type == "single_point_track_per_frame", \
                "Predicted points can only be used with 'single_point_track_per_frame' prompt type."

            assert self.predicted_points_file.endswith(".json")
            hf_path = self.predicted_points_file.replace(".json", "-hf")
            if file_exists(hf_path):
                data = load_from_disk(hf_path, keep_in_memory=False)
                log.info(f"Loading {self.dataset_name} HF data from {hf_path} for {len(data)} examples")
            else:
                with open(self.predicted_points_file, 'r') as f:
                    predicted_points = json.load(f)
                log.info(f"Loading predicted points from {self.predicted_points_file} for {len(predicted_points)} examples")
            return predicted_points
        else:
            annotation_file_path = self._get_full_annotation_file_path()
            assert annotation_file_path.endswith(".json")
            hf_path = annotation_file_path.replace(".json", "-hf")
            # if not file_exists(hf_path):
            #     log.info(f"Building HF dataset {hf_path} from {annotation_file_path}")
            #     df = load_dataset("json", data_files=dict(train=annotation_file_path), split="train")
            #     df.save_to_disk(hf_path)

            if file_exists(hf_path):
                log.info(f"Loading {self.dataset_name} HF data from {hf_path}")
                data = load_from_disk(hf_path, keep_in_memory=False)
                n_pre_filter = len(data)
                self.data_lookup = {ex_id: i for i, ex_id in enumerate(data["id"])}
                data = data.filter(self._try_get_fps, input_columns="fps")
            else:
                log.info(f"Loading {self.dataset_name} data from {annotation_file_path}")
                data = json.loads(read_file(annotation_file_path))
                n_pre_filter = len(data)
                data = [x for x in data if self._try_get_fps(x.get("fps"))]
            n_skipped = n_pre_filter - len(data)
            if get_global_rank() == 0 and n_skipped:
                log.warning(f"Skipped {n_skipped} examples because of FPS sampling mismatch")
            return data

    def get(self, idx, rng):
        ex = self.data[idx]
        if self.predicted_points_file:
            video = ex['video']
            single_point_example_id = ex['id']
            video_path = join(self.video_dir, video)

            # Create message list with structured point data
            frame_idx = round(ex['start_time'] * self.video_fps)
            ex['frame_trajectories'] = [{
                'frame': frame_idx,
                'time': ex['start_time'],
                'points': {
                    '0': {
                        'point': ex['start_point_pixel'],
                        'occluded': False
                    }
                }
            }]
            message_list = self._create_message_list(ex)

            # Prepare metadata
            metadata = {
                'example_id': single_point_example_id,
                'task': self.task_name,
                'expression': ex['expression'],
                'w': ex['metadata']['w'],
                'h': ex['metadata']['h'],
                'video_fps': self.video_fps,
                'video': video,
            }

            # Include points and segmentation if not training
            if self.is_eval:
                metadata['points'] = message_list[0]['points']

                # strip off object index after last underscore. Not needed for segmentation lookup
                example_id = single_point_example_id.rsplit('_', 1)[0]
                segmentation_data = self.get_segmentation_masks_for_example(example_id)
                metadata.update(segmentation_data)

            # Create dataset item
            return {
                'video': video_path,
                'message_list': message_list,
                'sampling_fps': self.sampling_fps,
                'metadata': metadata
            }
        else:
            video = ex['video']
            example_id = ex['id']
            video_path = join(self.video_dir, ex['video_path'], video)
            self.data_lookup[example_id] = idx

            # Create message list with structured point data
            message_list = self._create_message_list(ex)

            # Prepare metadata
            video_fps = ex.get("fps", self.video_fps)
            metadata = ex.get('metadata', {})
            metadata.update({
                'example_id': example_id,
                'task': self.task_name,
                'expression': ex['expression'],
                'w': ex['metadata']['w'],
                'h': ex['metadata']['h'],
                'video_fps': video_fps,
                'video': video,
            })
            if self.use_fps_sampling:
                candidate_sampling_fps = self.get_candidate_sampling_fps(video_fps)
                metadata.update({
                    'sampler_overrides': {
                        'frame_sample_mode': 'fps',
                        'candidate_sampling_fps': candidate_sampling_fps,
                        'min_fps': self.sampling_fps,
                    }
                })

            # Include points and segmentation if not training
            if self.is_eval:
                metadata['points'] = message_list[0]['points']
                if self.prompt_type == 'single_point_track_per_frame':
                    example_id = example_id.rsplit('_', 1)[0] # strip off last suffix that corresponds to individual object.

                # Load all object segmentation masks for query
                segmentation_data = self.get_segmentation_masks_for_example(example_id)
                metadata.update(segmentation_data)

            # Create dataset item
            return {
                'video': video_path,
                'message_list': message_list,
                'sampling_fps': self.sampling_fps,
                'metadata': metadata
            }


class BboxSinglePointTrack(ObjectTracking):
    dataset_name = "bbox-single-point-track"
    data_paths = {
        # 3741 videos, no expressions
        'mosev1': {
            'annotation': os.path.join(VIDEO_DATA_HOME, "MOSEv1", 'train', 'annotation', ),
            'video_dir': os.path.join(VIDEO_DATA_HOME, "MOSEv1", "videos_fps5"),
        },
        # 7631 videos, no expressions
        'mosev2': {
            'annotation': os.path.join(VIDEO_DATA_HOME, "MOSEv2", 'train', 'annotation', ),
            'video_dir': os.path.join(VIDEO_DATA_HOME, "MOSEv2", "videos_fps5"),
        },
        # 149 videos
        'lvosv1': {
            'annotation': os.path.join(VIDEO_DATA_HOME, "LVOSv1", 'train', 'annotation', ),
            'video_dir': os.path.join(VIDEO_DATA_HOME, "LVOSv1", "videos_fps6"),
        },
        # 601 videos, no expressions
        'lvosv2': {
            'annotation': os.path.join(VIDEO_DATA_HOME, "LVOSv2", 'train', 'annotation', ),
            'video_dir': os.path.join(VIDEO_DATA_HOME, "LVOSv2", "videos_fps6"),
        },
        # 2628 videos (dense); 7311 videos (annotation_full, partial expressions)
        'uvo': {
            'annotation': os.path.join(VIDEO_DATA_HOME, "UVO", "annotation_full", "dense", ),
            'video_dir': os.path.join(VIDEO_DATA_HOME, "UVO", "videos", "uvo_videos_dense_clips"),
        },
        # 1105 videos
        'lasot': {
            'annotation': os.path.join(VIDEO_DATA_HOME, "LaSOT", 'annotation', ),
            'video_dir': os.path.join(VIDEO_DATA_HOME, "LaSOT", "videos"),
        },
        # 212 videos
        'uwcot': {
            'annotation': os.path.join(VIDEO_DATA_HOME, "UW-COT220", 'annotation', ),
            'video_dir': os.path.join(VIDEO_DATA_HOME, "UW-COT220"),
        },
        # 844 videos
        'webuot': {
            'annotation': os.path.join(VIDEO_DATA_HOME, "webuot", 'annotation', ),
            'video_dir': os.path.join(VIDEO_DATA_HOME, "webuot", "Train"),
        },
        # 3166 videos
        'webuav': {
            'annotation': os.path.join(VIDEO_DATA_HOME, "WebUAV", 'annotation', ),
            'video_dir': os.path.join(VIDEO_DATA_HOME, "WebUAV", "videos"),
        },
        # 223 videos
        'latot': {
            'annotation': os.path.join(VIDEO_DATA_HOME, "LaTOT", 'annotation', ),
            'video_dir': os.path.join(VIDEO_DATA_HOME, "LaTOT", "videos"),
        },
        # 9209 videos
        'got10k': {
            'annotation': os.path.join(VIDEO_DATA_HOME, "GOT-10k", 'annotation', ),
            'video_dir': os.path.join(VIDEO_DATA_HOME, "GOT-10k", "videos"),
        },
        # 877 videos
        'tnl2k': {
            'annotation': os.path.join(VIDEO_DATA_HOME, "TNL2K", 'annotation', ),
            'video_dir': os.path.join(VIDEO_DATA_HOME, "TNL2K", "videos"),
        },
        # 46500 videos
        'vasttrack': {
            'annotation': os.path.join(VIDEO_DATA_HOME, "VastTrack", 'annotation', ),
            'video_dir': os.path.join(VIDEO_DATA_HOME, "VastTrack", "videos"),
        },
        # 147 videos
        'tnllt': {
            'annotation': os.path.join(VIDEO_DATA_HOME, "TNLLT", 'annotation', ),
            'video_dir': os.path.join(VIDEO_DATA_HOME, "TNLLT", "videos"),
        },
        # 2445 videos, no expressions
        'trackingnet0': {
            'annotation': os.path.join(VIDEO_DATA_HOME, "TrackingNet", "TRAIN_0", 'annotation', ),
            'video_dir': os.path.join(VIDEO_DATA_HOME, "TrackingNet", "TRAIN_0", "videos"),
        },
        # 2423 videos, no expressions
        'trackingnet1': {
            'annotation': os.path.join(VIDEO_DATA_HOME, "TrackingNet", "TRAIN_1", 'annotation', ),
            'video_dir': os.path.join(VIDEO_DATA_HOME, "TrackingNet", "TRAIN_1", "videos"),
        },
        # 2440 videos, no expressions
        'trackingnet2': {
            'annotation': os.path.join(VIDEO_DATA_HOME, "TrackingNet", "TRAIN_2", 'annotation', ),
            'video_dir': os.path.join(VIDEO_DATA_HOME, "TrackingNet", "TRAIN_2", "videos"),
        },
        # 2441 videos, no expressions
        'trackingnet3': {
            'annotation': os.path.join(VIDEO_DATA_HOME, "TrackingNet", "TRAIN_3", 'annotation', ),
            'video_dir': os.path.join(VIDEO_DATA_HOME, "TrackingNet", "TRAIN_3", "videos"),
        },
        # 2432 videos, no expressions
        'trackingnet4': {
            'annotation': os.path.join(VIDEO_DATA_HOME, "TrackingNet", "TRAIN_4", 'annotation', ),
            'video_dir': os.path.join(VIDEO_DATA_HOME, "TrackingNet", "TRAIN_4", "videos"),
        },
        # 2442 videos, no expressions
        'trackingnet5': {
            'annotation': os.path.join(VIDEO_DATA_HOME, "TrackingNet", "TRAIN_5", 'annotation', ),
            'video_dir': os.path.join(VIDEO_DATA_HOME, "TrackingNet", "TRAIN_5", "videos"),
        },
        # 2442 videos, no expressions
        'trackingnet6': {
            'annotation': os.path.join(VIDEO_DATA_HOME, "TrackingNet", "TRAIN_6", 'annotation', ),
            'video_dir': os.path.join(VIDEO_DATA_HOME, "TrackingNet", "TRAIN_6", "videos"),
        },
        # 2432 videos, no expressions
        'trackingnet7': {
            'annotation': os.path.join(VIDEO_DATA_HOME, "TrackingNet", "TRAIN_7", 'annotation', ),
            'video_dir': os.path.join(VIDEO_DATA_HOME, "TrackingNet", "TRAIN_7", "videos"),
        },
        # 2428 videos, no expressions
        'trackingnet8': {
            'annotation': os.path.join(VIDEO_DATA_HOME, "TrackingNet", "TRAIN_8", 'annotation', ),
            'video_dir': os.path.join(VIDEO_DATA_HOME, "TrackingNet", "TRAIN_8", "videos"),
        },
        # 2436 videos, no expressions
        'trackingnet9': {
            'annotation': os.path.join(VIDEO_DATA_HOME, "TrackingNet", "TRAIN_9", 'annotation', ),
            'video_dir': os.path.join(VIDEO_DATA_HOME, "TrackingNet", "TRAIN_9", "videos"),
        },
        # 2429 videos, no expressions
        'trackingnet10': {
            'annotation': os.path.join(VIDEO_DATA_HOME, "TrackingNet", "TRAIN_10", 'annotation', ),
            'video_dir': os.path.join(VIDEO_DATA_HOME, "TrackingNet", "TRAIN_10", "videos"),
        },
        # 2424 videos, no expressions
        'trackingnet11': {
            'annotation': os.path.join(VIDEO_DATA_HOME, "TrackingNet", "TRAIN_11", 'annotation', ),
            'video_dir': os.path.join(VIDEO_DATA_HOME, "TrackingNet", "TRAIN_11", "videos"),
        },
    }
    
    def __init__(self, split, 
                 subset_name,
                 video_dir=None, 
                 point_type="largest_center", 
                 prompt_type="single_point_track_per_frame", 
                 video_fps:int=None, 
                 sampling_fps:int=None,
                 interval_seconds:float=None,
                 use_fps_sampling:bool=True,
                 min_sampling_fps:int=1
                 ):
        """
        Base dataset for video pointing task with grounding and tracking objects.
        Can control different point sampling methods and prompt types.

        Args:
            split (str): Dataset split - ["train", "validation", "test"]
            subset_name (str): Specific subset of Prolific dataset to use.
            video_dir (str, optional): Custom video directory path; otherwise, use VIDEO_DATA_HOME
            point_type (str, optional): Type of point extraction method. Defaults to "largest_center".
            prompt_type (str, optional): Type of point data to generate.
                Only supported Options: 
                    - "single_point_track_per_frame": Given starting point, track object to end
            video_fps (int, optional): FPS of source videos, if provided
            sampling_fps (int, optional): Rate to sample points at, in frames per second.
                If provided, points will be sampled at this frame rate.
            interval_seconds (float, optional): If provided, will sample points at this interval in seconds.
                For example, 0.5 means sample every 0.5 seconds.
                For single_point_track_per_frame, this defines the interval between sampled frames after the starting point.
                     And if sampling_fps is provided, this is usedd to determine the starting point.
            use_fps_sampling (bool, optional): Whether to use fps-based sampling overrides during data loading.
            min_sampling_fps (int, optional): Minimum sampling fps to consider when generating candidate sampling fps values.

        Note:
            Either `sampling_fps` OR `interval_seconds` must be provided.
            If both are provided, `sampling_fps` will be used for frame-based sampling,
            - Use sampling_fps for frame-based sampling (e.g., sampling_fps=2 means 2 frames per second)
            - Use interval_seconds for time-based sampling (e.g., interval_seconds=0.5 means every 0.5 seconds)
        """
        assert split in ["train", "validation", "test"], f"Invalid split: {split}. Expected: train|validation|test"
        # assert prompt_type in PROMPT_TYPES
        assert prompt_type == "single_point_track_per_frame"

        # Dataset parameters
        data_info = self.data_paths[subset_name]
        self.data_path = data_info['annotation']
        self.split_dir = self._get_split_dir(split)
        self.is_eval = split not in ["train"]

        # Video parameters
        self.video_fps = video_fps  # only use if fps not in metadata
        self.video_dir = video_dir if video_dir else data_info['video_dir']

        # Point-Track parameters
        self.prompt_type = prompt_type
        self.point_type = point_type
        self.sampling_fps = sampling_fps
        self.interval_seconds = interval_seconds
        self.max_objects = None # not used for single point track
        self.use_fps_sampling = use_fps_sampling
        self.min_sampling_fps = min_sampling_fps
        self.task_name = self._build_task_name()
        
        self.data_lookup = {} # example_id -> data index

        DatasetBase.__init__(self, split)

    def _get_split_dir(self, split):
        return self.data_path # already defined in data_paths
    
    def _get_full_annotation_file_path(self) -> str:                                            
        """
        Get the complete absolute path to the annotation file.
        Probably not override this in subclasses.
        """                            
        # 10/31/25 mixture
        return join(self.split_dir, f"{self.point_type}_20251031")

    def preprocess_example(self, ex):
        """
        Preprocess a single example from the dataset to tracking data format.
        
        Returns:
            frame_trajectories: List of frame dictionaries, each containing:
                - 'frame': Frame index (int)
                - 'time': float in seconds
                - 'points': Dictionary mapping object_id -> point data for this frame (if only point is present)
                    Note: only one point is supported, so we always use object_id '0'.
        """

        # create frame_trajectories list
        frame_trajectories = []

        assert '0' in ex['points']
        assert len(ex['frame_times']) == len(ex['points']['0'])
        for idx, point in enumerate(ex['points']['0']):
            if point is not None:
                frame_trajectories.append({
                    'frame': idx,
                    'time': ex['frame_times'][idx],
                    'points': {
                        '0': {
                            'point': point,
                            'occluded': False
                        }
                    }
                })

        metadata_keys = ['w', 'h', 'qid', 'obj_id', 'anno_id', 'n_frames', 'mask_id']
        metadata = {key: ex[key] for key in metadata_keys if key in ex}

        result = {
            "id": ex['id'],
            "video": f"{ex['video']}.mp4",
            "prompt_type": self.prompt_type,
            "frame_trajectories": frame_trajectories,
            "expression": ex['exp'],
            "fps": ex.get("fps", None),
            "sampling_fps": self.sampling_fps,
            "metadata": metadata,
        }

        return result

    def load(self):
        annotation_file_path = self._get_full_annotation_file_path()
        log.info(f"Loading {self.dataset_name} data from {annotation_file_path}")
        dataset = load_from_disk(annotation_file_path, keep_in_memory=False)
        if "fps" in dataset.column_names and self.use_fps_sampling:
            def _try_get_fps(video_fps):
                if video_fps is None:
                    video_fps = self.video_fps
                try:
                    self.get_candidate_sampling_fps(video_fps)
                    return True
                except ValueError:
                    return False
            n = len(dataset)
            dataset = dataset.filter(_try_get_fps, input_columns="fps")
            n_skipped = n - len(dataset)
            if get_global_rank() == 0 and n_skipped > 0:
                log.warning(f"Skipping {n_skipped} because of FPS sampling mismatchs")
        return dataset

    def get(self, idx, rng):
        ex = self.data[idx]
        ex = self.preprocess_example(ex)
        video = ex['video']
        example_id = ex['id']
        message_list = self._create_message_list(ex)
        video_fps = ex.get("fps", self.video_fps)
        metadata = ex.get('metadata', {})
        metadata.update({
            'example_id': example_id,
            'task': self.task_name,
            'expression': ex['expression'],
            'w': ex['metadata']['w'],
            'h': ex['metadata']['h'],
            'video_fps': video_fps,
            'video': video,
        })
        if self.use_fps_sampling:
            candidate_sampling_fps = self.get_candidate_sampling_fps(video_fps)
            metadata.update({
                'sampler_overrides': {
                    'frame_sample_mode': 'fps',
                    'candidate_sampling_fps': candidate_sampling_fps,
                    'min_fps': self.sampling_fps,
                }
            })
        if self.is_eval:
            metadata['points'] = message_list[0]['points']
            if self.prompt_type == 'single_point_track_per_frame':
                example_id = example_id.rsplit('_', 1)[0] # strip off last suffix that corresponds to individual object.

            # Load all object segmentation masks for query
            segmentation_data = self.get_segmentation_masks_for_example(example_id)
            metadata.update(segmentation_data)
        return {
            'video': join(self.video_dir, video),
            'message_list': message_list,
            'sampling_fps': self.sampling_fps,
            'metadata': metadata
        }

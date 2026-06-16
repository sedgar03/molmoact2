"""
Spatial and Embodied Training Datasets for mm_olmo

Includes:
- SAT: Spatial Aptitude Training (Camera motion, counting, etc.)
- SIMS-VSI: Video Spatial Intelligence benchmark
- VSI-590K: Video Spatial Intelligence dataset
- RoboPoint: Robot pointing and localization (converted to Molmo pointing format)
- SenseNova-SI-800K: Egocentric-Exocentric matching
- VST-P: Visual Spatial Tasks (Template-based, non-VLM)
- RefSpatial: Spatial referring expressions (depth maps skipped)

All datasets support balanced sampling for training mixtures.
"""

import ast
import io
import json
import logging
import os
from os.path import join, exists
from typing import List, Optional

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from PIL import Image

from olmo.data.dataset import Dataset, DatasetBase

logger = logging.getLogger(__name__)

# Spatial datasets path
SPATIAL_DATA_HOME = os.environ.get(
    "SPATIAL_DATA_HOME",
    "/weka/oe-training-default/weikaih/molmo3_er/spatial_embodied_training_data"
)


def load_image_from_bytes(image_bytes: bytes) -> np.ndarray:
    """Load image from bytes to numpy array."""
    img = Image.open(io.BytesIO(image_bytes))
    if img.mode != "RGB":
        img = img.convert("RGB")
    return np.array(img)


def shuffle_options(options: List[str], answer_idx: int, rng):
    """Shuffle MCQ options and update answer index accordingly.

    This prevents position bias (e.g., always selecting 'A') during training.

    Args:
        options: List of option strings
        answer_idx: Index of correct answer in original options
        rng: Random state (Python random.Random or numpy RandomState)

    Returns:
        Tuple of (shuffled_options, new_answer_idx)
    """
    # Create shuffled indices - compatible with both Python random and numpy
    indices = list(range(len(options)))
    rng.shuffle(indices)
    shuffled_options = [options[i] for i in indices]
    # Find new position of correct answer
    new_answer_idx = indices.index(answer_idx)
    return shuffled_options, new_answer_idx


def parse_point_tuples(point_str: str) -> np.ndarray:
    """
    Parse point tuples from string format like '[(0.461, 0.527), (0.498, 0.521)]'
    Returns numpy array of shape (N, 2) with values in 0-1 range.
    """
    try:
        points = ast.literal_eval(point_str)
        if isinstance(points, list) and len(points) > 0:
            return np.array(points, dtype=np.float32)
    except (ValueError, SyntaxError):
        pass
    return np.array([], dtype=np.float32).reshape(0, 2)


# ============================================================================
# MCQ Options Parsing Helpers
# ============================================================================

import re

def parse_mc_options_standard(text: str):
    """
    Parse 'Options:\nA. xxx\nB. xxx' or 'A. xxx\nB. xxx' format.
    Returns (question_without_options, list_of_options).
    """
    # Try to find "Options:" section
    options_match = re.search(r'Options:\s*\n?(.*)', text, re.DOTALL | re.IGNORECASE)
    if options_match:
        question_part = text[:options_match.start()].strip()
        options_text = options_match.group(1).strip()
    else:
        # Try to find options directly in text (A. xxx\nB. xxx pattern)
        # Look for first "A." or "A)" pattern
        first_option = re.search(r'\n[A-D][.\)]\s', text)
        if first_option:
            question_part = text[:first_option.start()].strip()
            options_text = text[first_option.start():].strip()
        else:
            # No options found, return original
            return text, []

    # Parse individual options: "A. xxx" or "A) xxx" format
    # Use newline + letter pattern to split, not [^A-D] which fails on words like "Answer"
    options = []
    # Split by newline + option letter pattern
    option_pattern = re.compile(r'([A-D])[.\)]\s*')

    # Find all option starts
    parts = option_pattern.split(options_text)
    # parts = ['', 'A', 'content1\n', 'B', 'content2\n', ...]

    i = 1  # Skip first empty part
    while i < len(parts) - 1:
        letter = parts[i]
        content = parts[i + 1].strip()
        # Remove trailing instruction text (e.g., "Answer with...")
        content = re.split(r'\n(?:Answer|Please|Select)', content, maxsplit=1)[0].strip()
        content = content.rstrip(',').strip()
        if content:
            options.append(content)
        i += 2

    return question_part, options


def parse_mc_options_parentheses(text: str):
    """
    Parse '(A) xxx (B) xxx' format (used by RefSpatial).
    Handles nested parentheses like '(A) (0.407, 0.852) (B) (0.195, 0.681)'.
    Returns (question_without_options, list_of_options).
    """
    # Find positions of (A), (B), (C), (D) markers
    markers = []
    for letter in 'ABCD':
        pattern = f'\\({letter}\\)'
        for m in re.finditer(pattern, text):
            markers.append((m.start(), m.end(), letter))

    if not markers:
        return text, []

    # Sort by position
    markers.sort(key=lambda x: x[0])

    # Extract question (everything before first marker)
    question_part = text[:markers[0][0]].strip()

    # Extract options (content between markers)
    options = []
    for i, (start, end, letter) in enumerate(markers):
        if i + 1 < len(markers):
            # Content is from end of this marker to start of next marker
            content = text[end:markers[i + 1][0]].strip()
        else:
            # Last option - content is from end of marker to end of text
            content = text[end:].strip()

        if content:
            options.append(content)

    return question_part, options


def get_answer_idx_from_letter(answer: str, options: list) -> int:
    """
    Extract answer index from answer string.
    Handles formats: 'A', 'A.', 'A. content', '(A)', '(A) content'
    """
    if not answer or not options:
        return 0

    answer = answer.strip()

    # Try to extract letter from various formats
    letter = None
    if answer.startswith('(') and len(answer) > 1:
        letter = answer[1].upper()
    elif len(answer) >= 1:
        letter = answer[0].upper()

    if letter and letter in 'ABCD':
        idx = ord(letter) - ord('A')
        if 0 <= idx < len(options):
            return idx

    return 0


class SAT(Dataset):
    """
    Spatial Aptitude Training Dataset

    Format: Parquet with embedded images
    - image_bytes: list of bytes (embedded images)
    - question: string
    - answers: list of option strings
    - correct_answer: string
    - question_type: string (camera_motion, counting, direction, other)
    """

    PATH = join(SPATIAL_DATA_HOME, "SAT")

    def __init__(self, split: str = "train", sample: int = None, keep_in_memory: bool = False):
        assert split in ["train", "val", "test", "static"], f"Invalid split: {split}"
        self.split = split
        self.sample = sample
        self.keep_in_memory = keep_in_memory

        self.parquet_path = join(self.PATH, f"SAT_{split}.parquet")
        # Only read metadata to get length, don't store ParquetFile (not picklable)
        with pq.ParquetFile(self.parquet_path) as pf:
            self._length = pf.metadata.num_rows

        if sample is not None:
            self._length = min(self._length, sample)

        self._cache = None
        if keep_in_memory:
            self._load_to_memory()

    def _load_to_memory(self):
        """Load entire dataset to memory."""
        self._cache = []
        with pq.ParquetFile(self.parquet_path) as pf:
            for batch in pf.iter_batches(batch_size=1000):
                for i in range(batch.num_rows):
                    row = {col: batch.column(col)[i].as_py() for col in batch.column_names}
                    self._cache.append(row)
                    if self.sample and len(self._cache) >= self.sample:
                        return

    def __len__(self):
        return self._length

    def _get_row(self, idx: int) -> dict:
        """Get a single row from the parquet file."""
        if self._cache is not None:
            return self._cache[idx]

        batch_size = 1000
        batch_idx = idx // batch_size
        row_in_batch = idx % batch_size

        with pq.ParquetFile(self.parquet_path) as pf:
            for i, batch in enumerate(pf.iter_batches(batch_size=batch_size)):
                if i == batch_idx:
                    row = {col: batch.column(col)[row_in_batch].as_py() for col in batch.column_names}
                    return row

        raise IndexError(f"Index {idx} out of range")

    def get(self, item, rng):
        row = self._get_row(item)

        # Load images from bytes
        image_bytes_list = row["image_bytes"]
        images = [load_image_from_bytes(b) for b in image_bytes_list]

        # Format as multi-image if multiple images
        if len(images) == 1:
            image = images[0]
            style = "a_okvqa_mc"  # Single image MCQ
        else:
            image = images
            style = "multi_image_mc"  # Multi-image MCQ

        # MCQ format: provide options and answer_idx
        question = row["question"]
        options = list(row["answers"])  # List of option strings
        correct_answer = row["correct_answer"]

        try:
            answer_idx = options.index(correct_answer)
        except ValueError:
            answer_idx = 0

        # Shuffle options to prevent position bias
        options, answer_idx = shuffle_options(options, answer_idx, rng)

        return dict(
            image=image,
            question=question,
            options=options,
            answer_idx=answer_idx,
            style=style,
            metadata=dict(
                question_type=row["question_type"],
                split=self.split,
            )
        )


class SIMSVSI(DatasetBase):
    """
    SIMS-VSI: Simulated Video Spatial Intelligence Dataset

    Format: Parquet with video paths
    - video: path to video file
    - conversations: list of {from, value} dicts
    - question_type: 9 types (obj_abs_distance, obj_count, obj_rel_direction_*, etc.)
    - fmt: "oe" (open-ended) or "mc" (multiple choice)
    """

    PATH = join(SPATIAL_DATA_HOME, "SIMS-VSI")

    # Question type categories
    DIRECTION_TYPES = ["obj_rel_direction_easy", "obj_rel_direction_medium", "obj_rel_direction_hard"]
    DISTANCE_TYPES = ["obj_abs_distance", "obj_rel_distance"]
    COUNTING_TYPES = ["obj_count", "obj_size_est", "room_size_est"]
    OTHER_TYPES = ["obj_appearance_order"]

    def __init__(self, split: str = "train", sample: int = None,
                 fmt: str = None, question_category: str = None):
        """
        Args:
            fmt: "oe" for open-ended, "mc" for multiple choice, None for all
            question_category: "direction", "distance", "counting", "other", or None for all
        """
        self.fmt = fmt
        self.question_category = question_category
        self.parquet_path = join(self.PATH, "sims_vsi_200k.parquet")
        super().__init__(split=split, sample=sample)

    def load(self):
        df = pd.read_parquet(self.parquet_path)

        # Filter by format
        if self.fmt:
            df = df[df['fmt'] == self.fmt]

        # Filter by question category
        if self.question_category == "direction":
            df = df[df['question_type'].isin(self.DIRECTION_TYPES)]
        elif self.question_category == "distance":
            df = df[df['question_type'].isin(self.DISTANCE_TYPES)]
        elif self.question_category == "counting":
            df = df[df['question_type'].isin(self.COUNTING_TYPES)]
        elif self.question_category == "other":
            df = df[df['question_type'].isin(self.OTHER_TYPES)]

        return df.to_dict('records')

    def get(self, item, rng, _retry=0):
        row = self.data[item]

        video_path = join(self.PATH, "data", row["video"])

        convs = row["conversations"]
        # Handle None values in conversations
        if convs[0]["value"] is None or convs[1]["value"] is None:
            if _retry < 10:
                return self.get((item + 1) % len(self.data), rng, _retry + 1)
            raise ValueError(f"Too many None values in SIMSVSI dataset around index {item}")
        question_raw = convs[0]["value"].replace("<image>", "").replace("These are frames of a video.", "").strip()
        answer = convs[1]["value"]

        fmt = row.get("fmt", "oe")

        if fmt == "mc":
            # MCQ: parse options from question
            question_part, options = parse_mc_options_standard(question_raw)
            if options:
                answer_idx = get_answer_idx_from_letter(answer, options)
                # Note: SIMSVSI has balanced answer distribution (~25% each), no shuffle needed
                return dict(
                    video=video_path,
                    message_list=[dict(
                        question=question_part,
                        options=options,
                        answer_idx=answer_idx,
                        style="video_multiple_choice"
                    )],
                    metadata=dict(
                        question_type=row.get("question_type", "unknown"),
                        fmt=fmt,
                    )
                )

        # Open-ended or fallback: use user_qa style
        return dict(
            video=video_path,
            message_list=[dict(question=question_raw, answer=answer, style="user_qa")],
            metadata=dict(
                question_type=row.get("question_type", "unknown"),
                fmt=fmt,
            )
        )


class VSI590K(DatasetBase):
    """
    VSI-590K: Video Spatial Intelligence Dataset (590K samples)

    Format: JSONL with mixed video/image data
    - video: path to video file (374K samples)
    - image: path to image file (216K samples)
    - conversations: list of {from, value} dicts
    - question_type: string
    """

    PATH = join(SPATIAL_DATA_HOME, "VSI-590K")

    def __init__(self, split: str = "train", sample: int = None, media_type: str = None):
        """
        Args:
            media_type: "video", "image", or None for all
        """
        self.media_type = media_type
        self.jsonl_path = join(self.PATH, "vsi_590k.jsonl")
        super().__init__(split=split, sample=sample)

    def load(self):
        data = []
        with open(self.jsonl_path, 'r') as f:
            for line in f:
                if line.strip():
                    row = json.loads(line.strip())
                    # Filter by media type if specified
                    if self.media_type == "video" and "video" not in row:
                        continue
                    if self.media_type == "image" and "image" not in row:
                        continue
                    data.append(row)
        return data

    def get(self, item, rng):
        row = self.data[item]

        convs = row["conversations"]
        question_raw = convs[0]["value"].replace("<image>", "").replace("These are frames of a video.", "").strip()
        answer = convs[1]["value"]

        # VSI-590K has MCQ format with "Options:\nA. xxx\nB. xxx"
        question_part, options = parse_mc_options_standard(question_raw)

        # Determine if this is video or image
        is_video = "video" in row

        if is_video:
            media_path = join(self.PATH, row["video"])
            if options:
                answer_idx = get_answer_idx_from_letter(answer, options)
                # Note: VSI590K has mild bias (~36% A/B, ~14% C/D), no shuffle needed
                return dict(
                    video=media_path,
                    message_list=[dict(
                        question=question_part,
                        options=options,
                        answer_idx=answer_idx,
                        style="video_multiple_choice"
                    )],
                    metadata=dict(question_type=row.get("question_type", "unknown"), media_type="video")
                )
            else:
                return dict(
                    video=media_path,
                    message_list=[dict(question=question_raw, answer=answer, style="user_qa")],
                    metadata=dict(question_type=row.get("question_type", "unknown"), media_type="video")
                )
        else:
            # Image sample
            media_path = join(self.PATH, row["image"])
            if options:
                answer_idx = get_answer_idx_from_letter(answer, options)
                # Note: VSI590K has mild bias (~36% A/B, ~14% C/D), no shuffle needed
                return dict(
                    image=media_path,
                    question=question_part,
                    options=options,
                    answer_idx=answer_idx,
                    style="a_okvqa_mc",
                    metadata=dict(question_type=row.get("question_type", "unknown"), media_type="image")
                )
            else:
                return dict(
                    image=media_path,
                    question=question_raw,
                    answer=answer,
                    style="user_qa",
                    metadata=dict(question_type=row.get("question_type", "unknown"), media_type="image")
                )


class RoboPoint(DatasetBase):
    """
    RoboPoint: Robot Pointing and Localization Dataset

    Format: JSON with image paths and mixed answer types
    - image: path to image file
    - conversations: QA with different answer formats

    Data sources:
    - region_ref/object_ref (667K): Points [(x, y), ...] → style="pointing"
    - coco (100K): Boxes [(x, y, w, h), ...] → style="detection"
    - coco/ocr_vqa/gqa/text_vqa (598K): Text answers → style="user_qa"
    """

    PATH = join(SPATIAL_DATA_HOME, "RoboPoint")

    def __init__(self, split: str = "train", sample: int = None, task_type: str = None):
        """
        Args:
            task_type: Filter by task type - "pointing", "detection", "qa", or None for all
        """
        self.task_type = task_type
        self.json_path = join(self.PATH, "robopoint_1432k.json")
        super().__init__(split=split, sample=sample)

    def _is_box_format(self, answer_str):
        """Check if answer is box format [(x,y,w,h), ...]."""
        if not answer_str.startswith("["):
            return False
        try:
            parsed = ast.literal_eval(answer_str)
            if isinstance(parsed, list) and len(parsed) > 0 and isinstance(parsed[0], tuple):
                return len(parsed[0]) == 4
        except:
            pass
        return False

    def _is_point_format(self, answer_str):
        """Check if answer is point format [(x,y), ...]."""
        if not answer_str.startswith("["):
            return False
        try:
            parsed = ast.literal_eval(answer_str)
            if isinstance(parsed, list) and len(parsed) > 0 and isinstance(parsed[0], tuple):
                return len(parsed[0]) == 2
        except:
            pass
        return False

    def load(self):
        with open(self.json_path, 'r') as f:
            data = json.load(f)

        # Filter by task type if specified
        if self.task_type == "pointing":
            # Only 2D points
            data = [x for x in data if self._is_point_format(x["conversations"][1]["value"])]
        elif self.task_type == "detection":
            # Only 4D boxes
            data = [x for x in data if self._is_box_format(x["conversations"][1]["value"])]
        elif self.task_type == "qa":
            # Only text answers
            data = [x for x in data if not x["conversations"][1]["value"].startswith("[")]

        return data

    def get(self, item, rng):
        row = self.data[item]

        image_path = join(self.PATH, "images", row["image"])

        convs = row["conversations"]
        question = convs[0]["value"].replace("<image>", "").strip()
        answer_str = convs[1]["value"]

        # Check if answer is point/box format or text
        if answer_str.startswith("["):
            # Parse point/box tuples from answer
            points = parse_point_tuples(answer_str)

            if len(points) > 0:
                # Extract label from question
                label = "target"
                q_lower = question.lower()
                if "left of" in q_lower:
                    label = "left region"
                elif "right of" in q_lower:
                    label = "right region"
                elif "above" in q_lower:
                    label = "above region"
                elif "below" in q_lower:
                    label = "below region"
                elif "vacant" in q_lower or "empty" in q_lower:
                    label = "vacant space"
                elif "find all" in q_lower:
                    label = "objects"

                # Check if it's boxes (x, y, w, h) or points (x, y)
                if points.shape[1] == 4:
                    # Boxes: convert (x, y, w, h) → (x1, y1, x2, y2) format
                    boxes = points.copy()
                    boxes[:, 2] = boxes[:, 0] + boxes[:, 2]  # x2 = x + w
                    boxes[:, 3] = boxes[:, 1] + boxes[:, 3]  # y2 = y + h
                    boxes_scaled = boxes * 100

                    return dict(
                        image=image_path,
                        boxes=boxes_scaled,
                        box_scale=100,
                        label=label,
                        style="detection",
                        metadata=dict(id=row.get("id", ""), answer_type="boxes")
                    )
                else:
                    # Points: direct conversion
                    points_scaled = points * 100

                    return dict(
                        image=image_path,
                        points=points_scaled,
                        point_scale=100,
                        label=label,
                        style="pointing",
                        metadata=dict(id=row.get("id", ""), answer_type="points")
                    )

        # Text QA format - use user_qa (in DEMO_STYLES, no tag)
        return dict(
            image=image_path,
            question=question,
            answer=answer_str,
            style="user_qa",
            metadata=dict(id=row.get("id", ""), answer_type="text")
        )


class SenseNovaSI(DatasetBase):
    """
    SenseNova-SI-800K: Egocentric-Exocentric Spatial Intelligence

    Format: JSONL with multi-image paths
    - image: list of image paths (egocentric + exocentric views)
    - conversations: QA about spatial matching
    """

    PATH = join(SPATIAL_DATA_HOME, "SenseNova-SI-800K")

    def __init__(self, split: str = "train", sample: int = None):
        self.jsonl_path = join(self.PATH, "SenseNova-SI-800K.jsonl")
        super().__init__(split=split, sample=sample)

    def load(self):
        data = []
        with open(self.jsonl_path, 'r') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line.strip()))
        return data

    def get(self, item, rng):
        row = self.data[item]

        # Data already contains paths like "images/000/xxx.jpg"
        image_paths = [join(self.PATH, p) for p in row["image"]]

        convs = row["conversations"]
        question = convs[0]["value"].replace("<image>", "").strip()
        answer = convs[1]["value"].strip()

        # SenseNova-SI is multi-image MCQ with A, B, C, D representing different views
        # Options are the labeled images (A, B, C, D)
        options = ["A", "B", "C", "D"]
        answer_idx = get_answer_idx_from_letter(answer, options)

        return dict(
            image=image_paths,  # Multi-image
            question=question,
            options=options,
            answer_idx=answer_idx,
            style="multi_image_mc",
            metadata=dict(id=row.get("id", item), num_images=len(image_paths))
        )


class VSTP(Dataset):
    """
    VST-P: Visual Spatial Tasks (Template-based portion only)

    Format: Parquet files with embedded images
    - images: list of image dicts with 'bytes' key
    - conversations: QA about depth/distance
    - data_source: source identifier (filter out vst_r_* for VLM-generated)

    Task types (template-based only):
    - si_*: single-image tasks (depth_comparison, distance, measurement, scene_caption)
    - mi_*: multi-image tasks (camera_motion, correspondence, object_object_relation, scene_caption)

    Uses lazy loading to avoid loading all parquet files into memory at once.
    """

    PATH = join(SPATIAL_DATA_HOME, "VST-500K")

    SI_TASKS = ["si_depth_comparison", "si_distance", "si_measurement", "si_scene_caption"]
    MI_TASKS = ["mi_camera_motion", "mi_correspondence", "mi_object_object_relation", "mi_scene_caption"]
    TASK_DIRS = SI_TASKS + MI_TASKS

    def __init__(self, split: str = "train", sample: int = None,
                 task: str = None, image_type: str = None):
        """
        Args:
            task: specific task name, or None for all
            image_type: "single" for si_* tasks, "multi" for mi_* tasks, None for all
        """
        self.task = task
        self.image_type = image_type
        self.sample = sample

        # Build file index: list of (parquet_path, task_type, num_rows)
        self._file_index = []
        self._cumulative_rows = [0]  # Cumulative row counts for binary search
        self._total_rows = 0

        self._build_file_index()

        if sample is not None:
            self._total_rows = min(self._total_rows, sample)

    def _build_file_index(self):
        """Scan parquet files and build index without loading data."""
        # Determine which task directories to scan
        if self.task:
            task_dirs = [self.task]
        elif self.image_type == "single":
            task_dirs = self.SI_TASKS
        elif self.image_type == "multi":
            task_dirs = self.MI_TASKS
        else:
            task_dirs = self.TASK_DIRS

        for task_dir in task_dirs:
            task_path = join(self.PATH, task_dir)
            if not exists(task_path):
                continue

            parquet_files = sorted([f for f in os.listdir(task_path) if f.endswith('.parquet')])

            for pf in parquet_files:
                file_path = join(task_path, pf)
                try:
                    # Only read metadata to get row count
                    pf_obj = pq.ParquetFile(file_path)
                    num_rows = pf_obj.metadata.num_rows
                    self._file_index.append((file_path, task_dir, num_rows))
                    self._total_rows += num_rows
                    self._cumulative_rows.append(self._total_rows)
                except Exception as e:
                    logger.warning(f"Error scanning {pf}: {e}")

    def __len__(self):
        return self._total_rows

    def _find_file_and_row(self, idx: int):
        """Binary search to find which file contains the given index."""
        import bisect
        file_idx = bisect.bisect_right(self._cumulative_rows, idx) - 1
        if file_idx < 0:
            file_idx = 0
        row_in_file = idx - self._cumulative_rows[file_idx]
        return file_idx, row_in_file

    def _load_row(self, file_idx: int, row_in_file: int) -> dict:
        """Load a specific row from a parquet file."""
        file_path, task_type, _ = self._file_index[file_idx]

        # Read the specific row using row groups for efficiency
        pf = pq.ParquetFile(file_path)

        # Find the right row group
        cumulative = 0
        for rg_idx in range(pf.metadata.num_row_groups):
            rg_rows = pf.metadata.row_group(rg_idx).num_rows
            if cumulative + rg_rows > row_in_file:
                # Read this row group
                table = pf.read_row_group(rg_idx)
                local_idx = row_in_file - cumulative
                row = {col: table.column(col)[local_idx].as_py() for col in table.column_names}
                row["_task_type"] = task_type
                return row
            cumulative += rg_rows

        raise IndexError(f"Row {row_in_file} not found in {file_path}")

    def get(self, item, rng):
        # Lazy load the specific row
        file_idx, row_in_file = self._find_file_and_row(item)
        row = self._load_row(file_idx, row_in_file)

        # Filter out VLM-generated data (vst_r_* sources)
        data_source = row.get("data_source", "")
        if isinstance(data_source, str) and data_source.startswith("vst_r"):
            # Skip this row, try next one (this is a simplification - in practice
            # the index should be pre-filtered, but for now we just return a placeholder)
            # For training, the DataLoader will handle this gracefully
            pass

        # Load images from embedded bytes
        images_data = row["images"]
        images = []
        for img_dict in images_data:
            if isinstance(img_dict, dict) and "bytes" in img_dict:
                images.append(load_image_from_bytes(img_dict["bytes"]))

        convs = row["conversations"]
        question_raw = convs[0]["value"].replace("<image>", "").replace("<|image_pad|>", "").strip()
        answer = convs[1]["value"]

        task_type = row.get("_task_type", row.get("type", ""))
        is_multi_image = task_type.startswith("mi_") or len(images) > 1

        # Try to parse MCQ options
        question_part, options = parse_mc_options_standard(question_raw)

        if is_multi_image:
            if options:
                # Multi-image MCQ
                answer_idx = get_answer_idx_from_letter(answer, options)
                # Note: VSTP has mild bias (~36% A/B, ~14% C/D), no shuffle needed
                return dict(
                    image=images,
                    question=question_part,
                    options=options,
                    answer_idx=answer_idx,
                    style="multi_image_mc",
                    metadata=dict(type=task_type, num_images=len(images))
                )
            else:
                # Multi-image but not MCQ format - use user_qa
                return dict(
                    image=images,
                    question=question_raw,
                    answer=answer,
                    style="user_qa",
                    metadata=dict(type=task_type, num_images=len(images))
                )
        else:
            # Single image - use user_qa (no MCQ for si_* tasks based on data)
            return dict(
                image=images[0] if images else None,
                question=question_raw,
                answer=answer,
                style="user_qa",
                metadata=dict(type=task_type, data_source=row.get("data_source", ""))
            )


def _has_coordinate_answer(answer: str) -> bool:
    """Check if answer contains coordinate like (0.407, 0.852)."""
    return bool(re.search(r'\(0\.\d+,\s*0\.\d+\)', answer))


def _extract_coordinate_from_answer(answer: str) -> tuple:
    """Extract coordinate from answer like '(A) (0.407, 0.852)' -> (0.407, 0.852)."""
    match = re.search(r'\(0\.(\d+),\s*0\.(\d+)\)', answer)
    if match:
        x = float(f"0.{match.group(1)}")
        y = float(f"0.{match.group(2)}")
        return (x, y)
    return None


class RefSpatialVQA(DatasetBase):
    """
    RefSpatial VQA: Non-coordinate spatial questions (Yes/No, Left/Right, etc.)

    Format: JSON with image paths (depth maps NOT used)
    - Filters out questions with coordinate answers
    - Keeps: yes/no, left/right, size comparison, etc.
    """

    PATH = join(SPATIAL_DATA_HOME, "RefSpatial")
    SUBSETS = ["2D", "3D", "Simulator"]

    def __init__(self, split: str = "train", sample: int = None, subset: str = None):
        self.subset = subset
        super().__init__(split=split, sample=sample)

    def load(self):
        data = []
        subsets = [self.subset] if self.subset else self.SUBSETS

        for subset in subsets:
            subset_path = join(self.PATH, subset)
            if not exists(subset_path):
                continue

            json_path = join(subset_path, "choice_qa.json")
            if exists(json_path):
                with open(json_path, 'r') as f:
                    for item in json.load(f):
                        # Filter: only keep non-coordinate questions
                        convs = item.get("conversations", [])
                        if len(convs) >= 2:
                            answer = convs[1].get("value", "")
                            if not _has_coordinate_answer(answer):
                                item["_subset"] = subset
                                item["_image_dir"] = join(subset_path, "image")
                                data.append(item)

        return data

    def get(self, item, rng):
        row = self.data[item]

        image_names = row.get("image", [])
        image_dir = row.get("_image_dir", "")
        image_paths = [join(image_dir, name) for name in image_names]

        convs = row.get("conversations", [])
        if len(convs) >= 2:
            question_raw = convs[0]["value"]
            answer = convs[1]["value"]
        else:
            question_raw, answer = "", ""

        # Parse MCQ options: "(A) xxx (B) xxx" format
        question_part, options = parse_mc_options_parentheses(question_raw)

        if options:
            answer_idx = get_answer_idx_from_letter(answer, options)
            style = "multi_image_mc" if len(image_paths) > 1 else "a_okvqa_mc"

            if len(image_paths) == 1:
                return dict(
                    image=image_paths[0],
                    question=question_part,
                    options=options,
                    answer_idx=answer_idx,
                    style=style,
                    metadata=dict(id=row.get("id", ""), subset=row.get("_subset", ""))
                )
            else:
                return dict(
                    image=image_paths,
                    question=question_part,
                    options=options,
                    answer_idx=answer_idx,
                    style=style,
                    metadata=dict(id=row.get("id", ""), subset=row.get("_subset", ""), num_images=len(image_paths))
                )
        else:
            style = "user_qa"
            if len(image_paths) == 1:
                return dict(
                    image=image_paths[0],
                    question=question_raw,
                    answer=answer,
                    style=style,
                    metadata=dict(id=row.get("id", ""), subset=row.get("_subset", ""))
                )
            else:
                return dict(
                    image=image_paths,
                    question=question_raw,
                    answer=answer,
                    style=style,
                    metadata=dict(id=row.get("id", ""), subset=row.get("_subset", ""), num_images=len(image_paths))
                )


class RefSpatialPoint(DatasetBase):
    """
    RefSpatial Pointing: Coordinate-based spatial questions converted to Molmo2 pointing format.

    Format: JSON with image paths
    - Filters for questions with coordinate answers like "(A) (0.407, 0.852)"
    - Converts to pointing format: style="pointing", points=[[x, y]]
    - Questions like "Which point is closest to the camera?" -> "Point to the closest location."
    """

    PATH = join(SPATIAL_DATA_HOME, "RefSpatial")
    SUBSETS = ["2D", "3D", "Simulator"]

    # Prompt templates for pointing conversion
    POINTING_PROMPTS = [
        "Point to the {description}.",
        "Indicate the {description}.",
        "Show me the {description}.",
        "Where is the {description}?",
        "Locate the {description}.",
    ]

    def __init__(self, split: str = "train", sample: int = None, subset: str = None):
        self.subset = subset
        super().__init__(split=split, sample=sample)

    def load(self):
        data = []
        subsets = [self.subset] if self.subset else self.SUBSETS

        for subset in subsets:
            subset_path = join(self.PATH, subset)
            if not exists(subset_path):
                continue

            json_path = join(subset_path, "choice_qa.json")
            if exists(json_path):
                with open(json_path, 'r') as f:
                    for item in json.load(f):
                        # Filter: only keep coordinate questions
                        convs = item.get("conversations", [])
                        if len(convs) >= 2:
                            answer = convs[1].get("value", "")
                            if _has_coordinate_answer(answer):
                                item["_subset"] = subset
                                item["_image_dir"] = join(subset_path, "image")
                                data.append(item)

        return data

    def _question_to_pointing_label(self, question: str) -> str:
        """Convert question to pointing label/description."""
        q_lower = question.lower()

        if "closest" in q_lower or "nearest" in q_lower:
            if "observer" in q_lower or "camera" in q_lower:
                return "point closest to the observer"
            return "closest point"
        elif "farthest" in q_lower or "furthest" in q_lower:
            return "farthest point"
        elif "closer" in q_lower:
            return "closer point"
        else:
            return "indicated location"

    def get(self, item, rng):
        row = self.data[item]

        image_names = row.get("image", [])
        image_dir = row.get("_image_dir", "")
        image_paths = [join(image_dir, name) for name in image_names]

        convs = row.get("conversations", [])
        if len(convs) >= 2:
            question_raw = convs[0]["value"]
            answer = convs[1]["value"]
        else:
            return None

        # Extract coordinate from answer
        coord = _extract_coordinate_from_answer(answer)
        if coord is None:
            return None

        # Convert to points array (scale 100 for Molmo format)
        x, y = coord
        points = np.array([[x * 100, y * 100]], dtype=np.float32)

        # Generate pointing label
        question_part, _ = parse_mc_options_parentheses(question_raw)
        label = self._question_to_pointing_label(question_part)

        # Use single image (first one if multiple)
        image_path = image_paths[0] if image_paths else ""

        return dict(
            image=image_path,
            points=points,
            point_scale=100,
            label=label,
            style="pointing",
            metadata=dict(
                id=row.get("id", ""),
                subset=row.get("_subset", ""),
                original_question=question_part,
                original_answer=answer,
            )
        )


# Keep original RefSpatial for backward compatibility
class RefSpatial(DatasetBase):
    """
    RefSpatial: Spatial Referring Expressions Dataset (Original - all questions)

    Format: JSON with image paths (depth maps NOT used)
    - image: list of image filenames
    - conversations: multi-turn QA about spatial relations

    Subsets: 2D/, 3D/, Simulator/
    QA types: choice_qa (multiple choice) - default, reasoning_template_qa (reasoning) - excluded

    Note: Consider using RefSpatialVQA (non-coordinate) or RefSpatialPoint (coordinate->pointing) instead.
    """

    PATH = join(SPATIAL_DATA_HOME, "RefSpatial")
    SUBSETS = ["2D", "3D", "Simulator"]

    def __init__(self, split: str = "train", sample: int = None,
                 subset: str = None, qa_type: str = "choice"):
        """
        Args:
            subset: "2D", "3D", "Simulator", or None for all
            qa_type: "choice" for choice_qa (default), "reasoning" for reasoning_template_qa
        """
        self.subset = subset
        self.qa_type = qa_type if qa_type else "choice"  # Default to choice
        super().__init__(split=split, sample=sample)

    def load(self):
        data = []
        subsets = [self.subset] if self.subset else self.SUBSETS

        # Determine which QA files to load (default: only choice_qa)
        if self.qa_type == "choice":
            qa_files = ["choice_qa.json"]
        elif self.qa_type == "reasoning":
            qa_files = ["reasoning_template_qa.json"]
        else:
            qa_files = ["choice_qa.json"]  # Default to choice only

        for subset in subsets:
            subset_path = join(self.PATH, subset)
            if not exists(subset_path):
                continue

            for qa_file in qa_files:
                json_path = join(subset_path, qa_file)
                if exists(json_path):
                    with open(json_path, 'r') as f:
                        for item in json.load(f):
                            item["_subset"] = subset
                            item["_qa_type"] = qa_file.replace(".json", "")
                            item["_image_dir"] = join(subset_path, "image")
                            data.append(item)

        return data

    def get(self, item, rng):
        row = self.data[item]

        image_names = row.get("image", [])
        image_dir = row.get("_image_dir", "")
        image_paths = [join(image_dir, name) for name in image_names]

        convs = row.get("conversations", [])
        if len(convs) >= 2:
            question_raw = convs[0]["value"]
            answer = convs[1]["value"]
        else:
            question_raw, answer = "", ""

        # Parse MCQ options: "(A) xxx (B) xxx" format
        question_part, options = parse_mc_options_parentheses(question_raw)

        if options:
            # MCQ format
            answer_idx = get_answer_idx_from_letter(answer, options)
            # Note: RefSpatial has balanced distribution (A:47%, B:47%), no shuffle needed
            # Also contains coordinate options that shouldn't be shuffled
            style = "multi_image_mc" if len(image_paths) > 1 else "a_okvqa_mc"

            if len(image_paths) == 1:
                return dict(
                    image=image_paths[0],
                    question=question_part,
                    options=options,
                    answer_idx=answer_idx,
                    style=style,
                    metadata=dict(id=row.get("id", ""), subset=row.get("_subset", ""))
                )
            else:
                return dict(
                    image=image_paths,
                    question=question_part,
                    options=options,
                    answer_idx=answer_idx,
                    style=style,
                    metadata=dict(id=row.get("id", ""), subset=row.get("_subset", ""), num_images=len(image_paths))
                )
        else:
            # Fallback to user_qa if no options parsed
            style = "user_qa"
            if len(image_paths) == 1:
                return dict(
                    image=image_paths[0],
                    question=question_raw,
                    answer=answer,
                    style=style,
                    metadata=dict(id=row.get("id", ""), subset=row.get("_subset", ""))
                )
            else:
                return dict(
                    image=image_paths,
                    question=question_raw,
                    answer=answer,
                    style=style,
                    metadata=dict(id=row.get("id", ""), subset=row.get("_subset", ""), num_images=len(image_paths))
                )


# ============================================================================
# Abstract/Commonsense Reasoning Datasets (V3)
# ============================================================================

class CLEVR(DatasetBase):
    """
    CLEVR: Compositional Language and Elementary Visual Reasoning
    https://cs.stanford.edu/people/jcjohns/clevr/

    Format: JSON with image paths
    - 70K train images, 700K train questions
    - Questions about counting, comparison, spatial relations, etc.
    - Programmatically generated (NOT VLM-generated)

    Question types: count, exist, compare_integer, compare_attribute, query_attribute
    """

    PATH = join(SPATIAL_DATA_HOME, "CLEVR/CLEVR_v1.0")

    def __init__(self, split: str = "train", sample: int = None):
        self.json_path = join(self.PATH, f"questions/CLEVR_{split}_questions.json")
        self.images_dir = join(self.PATH, f"images/{split}")
        super().__init__(split=split, sample=sample)

    def load(self):
        with open(self.json_path, 'r') as f:
            data = json.load(f)
        return data['questions']

    def get(self, item, rng):
        row = self.data[item]

        image_path = join(self.images_dir, row['image_filename'])
        question = row['question']
        answer = str(row['answer']).lower()  # Answers can be yes/no/numbers/colors

        return dict(
            image=image_path,
            question=question,
            answer=answer,
            style="user_qa",
            metadata=dict(
                question_family_index=row.get('question_family_index', -1),
                split=self.split,
            )
        )


class Rel3D(DatasetBase):
    """
    Rel3D: Spatial Relation Classification in 3D Scenes
    https://github.com/princeton-vl/Rel3D

    Format: JSON metadata + H5 compressed images
    - 22K train, 5K test samples
    - Binary classification: does relation hold? (True/False)
    - 30 spatial predicates (on, in, behind, etc.)
    - Programmatically generated (NOT VLM-generated)

    Images are stored compressed in H5 files, decoded with cv2.imdecode.
    """

    PATH = join(SPATIAL_DATA_HOME, "Rel3D/data")

    # Question templates for different predicates
    QUESTION_TEMPLATES = [
        "Is the {subject} {predicate} the {object}?",
        "Does the {subject} appear to be {predicate} the {object}?",
        "Looking at this image, is the {subject} {predicate} the {object}?",
    ]

    def __init__(self, split: str = "train", sample: int = None):
        self.json_path = join(self.PATH, "full.json")
        self.h5_handles = {}  # Cache H5 file handles
        self.path_to_h5 = {}  # Map rgb_path -> (h5_file, index)
        super().__init__(split=split, sample=sample)

    def _build_path_mapping(self):
        """Build mapping from image paths to H5 file locations."""
        import h5py
        h5_suffix = "_True.h5" if self.split == "train" else "_True.h5"  # Use True label H5 files

        for fname in os.listdir(self.PATH):
            if fname.endswith('.h5') and self.split in fname:
                h5_path = join(self.PATH, fname)
                try:
                    with h5py.File(h5_path, 'r') as f:
                        mapping_bytes = f['rgb_path_to_id'][()]
                        path_to_id = json.loads(mapping_bytes)
                        for rgb_path, idx in path_to_id.items():
                            self.path_to_h5[rgb_path] = (h5_path, idx)
                except Exception as e:
                    continue

    def load(self):
        with open(self.json_path, 'r') as f:
            data = json.load(f)

        # Build path mapping after loading JSON
        self._build_path_mapping()

        # Use train/test split from the JSON, filter to only samples with H5 images
        samples = data.get(self.split, data.get('train', []))
        valid_samples = [s for s in samples if s['rgb'] in self.path_to_h5]
        return valid_samples

    def _load_image_from_h5(self, rgb_path):
        """Load and decode image from H5 file."""
        import h5py
        import cv2

        if rgb_path not in self.path_to_h5:
            raise FileNotFoundError(f"Image path not found in H5 mapping: {rgb_path}")

        h5_path, idx = self.path_to_h5[rgb_path]

        # Use cached handle or open new one
        if h5_path not in self.h5_handles:
            self.h5_handles[h5_path] = h5py.File(h5_path, 'r')

        h5f = self.h5_handles[h5_path]
        img_bytes = h5f['rgb'][idx]

        # Decode compressed image
        img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Failed to decode image at {rgb_path}")

        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def get(self, item, rng):
        row = self.data[item]

        rgb_path = row['rgb']

        # Load image from H5 (returns numpy array)
        image = self._load_image_from_h5(rgb_path)

        subject = row['subject']['name']
        predicate = row['predicate']
        obj = row['object']['name']
        label = row['label']  # True or False

        # Generate question using template
        template = rng.choice(self.QUESTION_TEMPLATES) if rng else self.QUESTION_TEMPLATES[0]
        question = template.format(subject=subject, predicate=predicate, object=obj)
        answer = "yes" if label else "no"

        return dict(
            image=image,  # numpy array, not path
            question=question,
            answer=answer,
            style="user_qa",
            metadata=dict(
                subject=subject,
                predicate=predicate,
                object=obj,
                label=label,
            )
        )

    def __del__(self):
        """Close H5 file handles."""
        for h5f in self.h5_handles.values():
            try:
                h5f.close()
            except:
                pass


class GRiD3D(DatasetBase):
    """
    GRiD-3D: Grounding Relative Directions in 3D Scenes
    https://github.com/knowledgetechnologyuhh/grid-3d

    Format: JSON with image paths
    - 365K train, 46K val, 44K test samples
    - 6 tasks: existence, orientation, link, relation prediction, counting, triple classification
    - Programmatically generated (NOT VLM-generated)
    """

    PATH = join(SPATIAL_DATA_HOME, "GRiD-3D/grid-3d")

    def __init__(self, split: str = "train", sample: int = None):
        self.questions_path = join(self.PATH, "questions.json")
        self.split = split
        super().__init__(split=split, sample=sample)

    def load(self):
        # Load questions
        with open(self.questions_path, 'r') as f:
            all_questions = json.load(f)

        # Load split indices
        split_map = {'train': 'train_idxs.json', 'val': 'val_idxs.json', 'test': 'test_idxs.json'}
        if self.split not in split_map:
            split_map[self.split] = f'{self.split}_idxs.json'

        split_file = join(self.PATH, split_map.get(self.split, 'train_idxs.json'))
        with open(split_file, 'r') as f:
            indices = json.load(f)

        # Filter questions by split indices
        return [all_questions[i] for i in indices if i < len(all_questions)]

    def _find_image_path(self, img_idx: int) -> str:
        """Find image path across part directories."""
        for part in range(1, 9):
            img_path = join(self.PATH, f"part_0{part}", f"{img_idx}.png")
            if exists(img_path):
                return img_path
        # Fallback - assume part_01
        return join(self.PATH, "part_01", f"{img_idx}.png")

    def get(self, item, rng):
        row = self.data[item]

        img_idx = row['img_idx']
        image_path = self._find_image_path(img_idx)

        # Reconstruct question from tokens
        question_tokens = row['question']
        question = ' '.join(question_tokens).replace(' ?', '?').replace(' .', '.')
        # Clean up underscores in object names
        question = question.replace('_', ' ')

        answer = str(row['answer']).lower()
        task = row.get('task', 'unknown')

        return dict(
            image=image_path,
            question=question,
            answer=answer,
            style="user_qa",
            metadata=dict(
                task=task,
                input_entities=row.get('input', []),
                img_idx=img_idx,
            )
        )


class MindCube(DatasetBase):
    """
    MindCube: Spatial Mental Modeling from Limited Views
    https://huggingface.co/datasets/MLL-Lab/MindCube

    Format: JSONL with multi-image paths
    - 10K train, 21K total questions
    - Tests cognitive mapping, perspective-taking, mental simulation
    - Multi-image MCQ format (4 options A/B/C/D)
    - NOT VLM-generated
    """

    PATH = join(SPATIAL_DATA_HOME, "MindCube/data")

    def __init__(self, split: str = "train", sample: int = None):
        if split == "train":
            self.jsonl_path = join(self.PATH, "raw/MindCube_train.jsonl")
        else:
            self.jsonl_path = join(self.PATH, "raw/MindCube.jsonl")
        super().__init__(split=split, sample=sample)

    def load(self):
        data = []
        with open(self.jsonl_path, 'r') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line.strip()))
        return data

    def get(self, item, rng):
        row = self.data[item]

        # Build image paths
        image_paths = [join(self.PATH, img_path) for img_path in row['images']]

        # Parse question and extract MCQ options
        question_raw = row['question']

        # MindCube format: "... A. xxx B. xxx C. xxx D. xxx"
        # Options are inline without newlines
        options = []
        question_part = question_raw

        # Find first option marker to split question from options
        first_option_match = re.search(r'\s+A\.\s+', question_raw)
        if first_option_match:
            question_part = question_raw[:first_option_match.start()].strip()
            options_text = question_raw[first_option_match.start():].strip()

            # Parse options: "A. xxx B. xxx C. xxx D. xxx"
            # Use lookahead to split at option boundaries
            option_matches = re.split(r'\s+(?=[A-D]\.\s)', options_text)
            for opt in option_matches:
                opt = opt.strip()
                if opt and len(opt) > 2:
                    # Remove "A. ", "B. ", etc. prefix
                    content = re.sub(r'^[A-D]\.\s*', '', opt).strip()
                    if content:
                        options.append(content)

        # Get answer index
        gt_answer = row.get('gt_answer', 'A')
        answer_idx = ord(gt_answer.upper()) - ord('A')
        answer_idx = max(0, min(answer_idx, len(options) - 1)) if options else 0

        if options and len(options) >= 2:
            return dict(
                image=image_paths,
                question=question_part,
                options=options,
                answer_idx=answer_idx,
                style="multi_image_mc",
                metadata=dict(
                    id=row.get('id', ''),
                    category=row.get('category', []),
                    type=row.get('type', ''),
                    num_images=len(image_paths),
                )
            )
        else:
            # Fallback to user_qa
            return dict(
                image=image_paths,
                question=question_raw,
                answer=gt_answer,
                style="user_qa",
                metadata=dict(
                    id=row.get('id', ''),
                    category=row.get('category', []),
                )
            )


class CosmosReason1(DatasetBase):
    """
    Cosmos-Reason1 SFT Dataset from NVIDIA
    https://huggingface.co/datasets/nvidia/Cosmos-Reason1-SFT-Dataset

    Embodied reasoning dataset with video-text pairs for robot/autonomous vehicle understanding.
    Contains 1.7M+ samples across 4 subsets: robovqa, bridgev2, agibot, holoassist

    Each subset has "understanding" and "reasoning" splits.
    - understanding: descriptive QA about objects/actions in video (VLM-generated verbose)
    - reasoning: chain-of-thought reasoning about agent success (VLM-generated with <think> tags)

    Format: video + conversations (system/user/assistant roles)

    Note: The conversations contain VLM-generated verbose answers. The original human
    annotations are in task_metadata with shorter answers. Use use_human_annotations=True
    to use human annotations instead.
    """
    PATH = join(SPATIAL_DATA_HOME, "Cosmos-Reason1")

    def __init__(self, split: str = "train", sample: int = None,
                 subset: str = None, task_type: str = None,
                 use_human_annotations: bool = False):
        """
        Args:
            subset: "robovqa", "bridgev2", "agibot", "holoassist", or None for all
            task_type: "understanding", "reasoning", or None for all
            use_human_annotations: If True, use original human annotations from task_metadata
                                   instead of VLM-generated verbose conversations.
                                   This filters out reasoning (VLM chain-of-thought) and uses
                                   short human answers for understanding tasks.
                                   NOTE: Only robovqa subset has human QA pairs in task_metadata.
                                   Other subsets (bridgev2/holoassist/agibot) only have action descriptions.
        """
        self.subset = subset
        self.task_type = task_type
        self.use_human_annotations = use_human_annotations

        if use_human_annotations:
            # Only robovqa has human QA pairs in task_metadata
            # Other subsets (bridgev2/holoassist/agibot) only have action descriptions, not QA pairs
            if self.subset is None:
                self.subset = "robovqa"
            # Only load understanding (reasoning files are VLM chain-of-thought)
            if task_type is None:
                self.task_type = "understanding"

        super().__init__(split=split, sample=sample)

    def load(self):
        data = []

        # Define which JSON files to load based on subset and task_type
        subsets_to_load = [self.subset] if self.subset else ["robovqa", "bridgev2", "agibot", "holoassist"]

        for subset in subsets_to_load:
            subset_path = join(self.PATH, subset)
            if not exists(subset_path):
                logger.warning(f"Subset path not found: {subset_path}")
                continue

            # Find JSON files for this subset
            json_files = []
            if self.task_type == "understanding" or self.task_type is None:
                understanding_file = join(subset_path, f"{subset}_understanding.json")
                if exists(understanding_file):
                    json_files.append((understanding_file, "understanding"))

            if self.task_type == "reasoning" or self.task_type is None:
                # Reasoning files may be split into parts (robovqa has multiple)
                for i in range(10):  # Check for reasoning_0, reasoning_1, etc.
                    reasoning_file = join(subset_path, f"{subset}_reasoning_{i}.json")
                    if exists(reasoning_file):
                        json_files.append((reasoning_file, "reasoning"))
                # Also check for single reasoning file
                reasoning_file = join(subset_path, f"{subset}_reasoning.json")
                if exists(reasoning_file):
                    json_files.append((reasoning_file, "reasoning"))

            for json_file, task in json_files:
                try:
                    with open(json_file, 'r') as f:
                        file_data = json.load(f)
                    for item in file_data:
                        item['_subset'] = subset
                        item['_task_type'] = task
                    data.extend(file_data)
                    logger.info(f"Loaded {len(file_data)} samples from {json_file}")
                except Exception as e:
                    logger.warning(f"Failed to load {json_file}: {e}")

        logger.info(f"CosmosReason1 total samples: {len(data)}")
        return data

    def get(self, item, rng):
        row = self.data[item]

        # Get video path - convert clips/xxx.mp4 to clips_extracted/xxx.mp4
        video_rel_path = row["video"]
        if video_rel_path.startswith("clips/"):
            video_rel_path = video_rel_path.replace("clips/", "clips_extracted/", 1)

        subset = row.get('_subset', 'robovqa')
        video_path = join(self.PATH, subset, video_rel_path)

        if self.use_human_annotations:
            # Use original human annotations from task_metadata
            metadata = row.get("metadata", {})
            task_metadata_list = metadata.get("task_metadata", [])

            if not task_metadata_list:
                return self.get((item + 1) % len(self.data), rng)

            # Pick a random task from the metadata (each video can have multiple QAs)
            task_meta = rng.choice(task_metadata_list) if rng else task_metadata_list[0]

            # Use processed question/answer if available, else original
            question = task_meta.get("question_processed") or task_meta.get("question", "")
            answer = task_meta.get("answer_processed") or task_meta.get("answer", "")

            if not question or not answer:
                return self.get((item + 1) % len(self.data), rng)

            return dict(
                video=video_path,
                message_list=[dict(
                    question=question,
                    answer=answer,
                    style="user_qa"
                )],
                metadata=dict(
                    subset=subset,
                    task_type=row.get('_task_type', 'unknown'),
                    task=task_meta.get("task", "unknown"),
                    annotation_type="human",
                )
            )

        # Default: use VLM-generated conversations
        conversations = row.get("conversations", [])

        # Extract question and answer from conversations
        question = None
        answer = None
        for conv in conversations:
            role = conv.get("role", "")
            content = conv.get("content", "")
            if role == "user":
                question = content
            elif role == "assistant":
                answer = content

        if question is None or answer is None:
            # Skip invalid samples
            return self.get((item + 1) % len(self.data), rng)

        # Clean up question (remove video placeholder if any)
        question = question.replace("<video>", "").strip()

        return dict(
            video=video_path,
            message_list=[dict(
                question=question,
                answer=answer,
                style="user_qa"
            )],
            metadata=dict(
                subset=subset,
                task_type=row.get('_task_type', 'unknown'),
                annotation_type="vlm_generated",
                width=row.get('metadata', {}).get('width', 0),
                height=row.get('metadata', {}).get('height', 0),
            )
        )

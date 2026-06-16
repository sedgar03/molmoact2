import logging
import os
import json
import re
import tarfile
import time
import ast
import unicodedata
from collections import Counter, defaultdict
from io import BytesIO
from typing import Literal, Optional, List, Union
from pathlib import Path
from os.path import join, exists, dirname, basename, relpath
from typing import Literal, Optional
import subprocess
import random
import math

import datasets
import yaml
import pandas as pd
import numpy as np
import decord
from decord import VideoReader, cpu
import imageio.v3 as iio
from datasets import Dataset as HfDataset, DatasetDict
from moviepy import VideoFileClip
from tqdm import tqdm

from olmo.torch_util import get_global_rank
from olmo import tokenizer

from olmo.io import (
    read_file,
    is_url,
    file_exists,
    get_bytes_range,
    glob as olmo_glob,
    list_directory, write_file, dir_is_empty
)
from olmo.util import flatten_lists, resource_path, split_into_groups, set_example_style

from olmo.data.dataset import DatasetBase, VIDEO_DATA_HOME, PUBLIC_VIDEO_BASE_URL, Dataset
from olmo.data.vixmo_datasets import sample_random_clip

decord.logging.set_level(2)
log = logging.getLogger(__name__)

def create_video_from_frames(frames_dir, start_frame, end_frame, fps=3, pad_frames=True):
    """
    Creates a video file from a sequence of frames in a directory.

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
    output_path = os.path.join(frames_dir, f"video_{start_frame:05d}_{end_frame:05d}.mp4")

    if file_exists(output_path):
        return output_path

    # Get list of frame files within the range
    frame_files = []
    for i in range(start_frame, end_frame + 1):
        frame_path = os.path.join(frames_dir, f"{i:05d}.jpg")  # Using 5-digit frame numbers
        if file_exists(frame_path):
            frame_files.append(frame_path)

    if not frame_files:
        raise ValueError(f"No frames found in range {start_frame} to {end_frame} in {frames_dir}")
    
    # Read and pad frames
    frames = []
    for f in frame_files:
        frame = iio.imread(f)
        h, w = frame.shape[:2]
        if pad_frames:
            # Calculate padded dimensions (divisible by 16)
            new_h = ((h + 15) // 16) * 16
            new_w = ((w + 15) // 16) * 16
            
            if h != new_h or w != new_w:
                # Pad with black pixels
                if len(frame.shape) == 3:
                    padded = np.zeros((new_h, new_w, frame.shape[2]), dtype=frame.dtype)
                else:
                    padded = np.zeros((new_h, new_w), dtype=frame.dtype)
                padded[:h, :w] = frame
                frame = padded 

        frames.append(frame)

    # Read frames and write video
    # frames = [iio.imread(f) for f in frame_files]
    iio.imwrite(output_path, frames, fps=fps, codec='libx264')

    return output_path


def save_bounded_video(video_path, start_time, end_time, task_type="default",
                       output_folder=None, clip_tag="bounded_decimal_2",
                       use_video_file_clip=False):
    """
    Creates a new video file containing only the segment between start_time and end_time.

    Args:
        video_path (str): Path to the original video file or frames directory
        start_time (float): Start time in seconds
        end_time (float): End time in seconds
        task_type (str): Type of task, determines handling of frames vs video
        output_folder (str): Path to the output folder, if None, output folder will be created
        clip_tag (str): Tag for the new video file, if None, no tag will be added to the new video file

    Returns:
        str: Path to the new bounded video file
    """
    if task_type == "Episodic Reasoning":
        # For frame-based videos, convert frame numbers from time
        fps = 3  # As specified in the frames directory name
        start_frame = int(start_time * fps) + 1  # +1 because frames start at 1
        end_frame = int(end_time * fps)
        return create_video_from_frames(video_path, start_frame, end_frame, fps)

    # Original video handling code
    base_path, ext = os.path.splitext(video_path)

    # round start_time and end_time to 2 decimal places
    start_time = round(start_time, 2)
    end_time = round(end_time, 2)

    clip_id = f"{start_time}_{end_time}{ext}"
    if clip_tag:
        clip_id = f"{clip_tag}_{clip_id}"

    if output_folder is None:
        output_path = f"{base_path}_{clip_id}"
    else:
        file_name_no_ext = os.path.splitext(os.path.basename(video_path))[0]
        output_path = os.path.join(output_folder, f"{file_name_no_ext}_{clip_id}")

    if file_exists(output_path):
        return output_path

    metadata = iio.immeta(video_path)
    duration = metadata['duration']

    if use_video_file_clip:
        video = VideoFileClip(video_path)
        clip = video.subclipped(start_time, min(end_time, duration))
        clip.write_videofile(output_path, codec='libx264')
        video.close()
        clip.close()
    else:
        import ffmpeg
        input_file = ffmpeg.input(video_path).trim(start=start_time, end=end_time).setpts('PTS-STARTPTS')
        output_file = ffmpeg.output(input_file, output_path)
        ffmpeg.run(output_file)

    return output_path


def reformat_webm_to_mp4_if_needed(input_path):
    """
    Reformats a WebM file to MP4 format.

    Args:
        input_path (str): Path to the input WebM file.

    Returns:
        str: The path to the reformatted MP4 file.
    """

    if ".webm" not in input_path:
        return input_path

    output_path = input_path.replace(".webm", ".mp4")
    if file_exists(output_path):
        return output_path

    video = VideoFileClip(input_path)
    video.write_videofile(output_path, codec="libx264", audio_codec="aac")
    video.close()
    return output_path


class DatasetSampleDifficulty():
    data_path = os.path.join(VIDEO_DATA_HOME, "DatasetSampleDifficulty")

    def __init__(self, difficulty_file_name):
        f = resource_path(os.path.join(self.data_path, difficulty_file_name))
        with open(f) as f:
            benchmark_results = json.load(f)

        self.example_id_to_difficulty = {}
        for example_id, example_data in benchmark_results.items():
            frame_results = example_data.get("frames", {})
            correct_count = sum(1 for frame_data in frame_results.values() if frame_data.get("correct", False))
            if correct_count == 3:
                self.example_id_to_difficulty[example_id] = "easy"
            elif correct_count == 2:  # 2 out of 3 frames correct -> majority correct
                self.example_id_to_difficulty[example_id] = "medium"
            else:
                self.example_id_to_difficulty[example_id] = "hard"

    def get_difficulty(self, example_id):
        if example_id not in self.example_id_to_difficulty:
            raise ValueError(f"Example ID {example_id} not found in difficulty file.")
        return self.example_id_to_difficulty[example_id]


class PlmFGQAEval(DatasetBase):
    data_path = os.path.join(VIDEO_DATA_HOME, "PLM-FGQA")
    ego_exo_data_path = os.path.join(VIDEO_DATA_HOME, "PLM-FGQA/egoexo4d")
    ego_4d_data_path = os.path.join(VIDEO_DATA_HOME, "Ego4d/ego4d_data/v2/full_scale")
    yt_dataset_path = os.path.join(VIDEO_DATA_HOME, "PLM-FGQA/plm-eval-videos/")

    def __init__(self, split):
        assert split in ["validation"]
        super().__init__(split)

    @staticmethod
    def qa_template(option_candidates, correct_choice, question):
        option_text = "\n".join(f"{chr(ord('A') + idx)}. {opt}" for idx, opt in enumerate(option_candidates))
        answer = f"{chr(ord('A') + correct_choice)}"
        question = "\n".join(
            [
                question,
                option_text,
                "Answer with the option's letter from the given choices directly.",
            ]
        )
        return question, answer

    def get_ego_exo_video_path(self, item, row, egoexo_segment_id_to_camera_id):
        segment_id = row['video'].split(".mp4")[0]
        selected_camera_id = egoexo_segment_id_to_camera_id[segment_id]
        camera, cam_source = selected_camera_id.split("_")
        camera_metadata = item['frame_aligned_videos'][camera]
        return os.path.join(self.ego_exo_data_path, item['root_dir'], camera_metadata[cam_source]["relative_path"])

    def load(self):
        ego_exo_metadata_path = os.path.join(self.ego_exo_data_path, "takes.json")
        # with open(ego_exo_metadata_path) as f:
            # ego_exo_metadata = json.load(f)
        ego_exo_metadata = json.loads(read_file(ego_exo_metadata_path))
        ego_exo_uid_to_metadata = {}
        for item in ego_exo_metadata:
            ego_exo_uid_to_metadata[item['take_uid']] = item

        # egoexo_segment_to_camera_map = pd.read_csv(os.path.join(self.data_path, "fgqa_test_egoexo4d_segment2cam.csv"))
        bytes_io = BytesIO(read_file(os.path.join(self.data_path, "fgqa_test_egoexo4d_segment2cam.csv"), "rb"))
        # egoexo_segment_to_camera_map = pd.read_csv(os.path.join(self.data_path, "fgqa_test_egoexo4d_segment2cam.csv"))
        egoexo_segment_to_camera_map = pd.read_csv(bytes_io)
        egoexo_segment_id_to_camera_id = {}
        for index, row in egoexo_segment_to_camera_map.iterrows():
            egoexo_segment_id_to_camera_id[row['segment_uid']] = row['camera_name']

        data = []
        # get the metadata column and get list of source
        # fgqa_df = pd.read_parquet(os.path.join(self.data_path, "plm_fgqa_test.parquet"))
        bytes_io = BytesIO(read_file(os.path.join(self.data_path, "plm_fgqa_test.parquet"), "rb"))
        # fgqa_df = pd.read_parquet(read_file(os.path.join(self.data_path, "plm_fgqa_test.parquet"), "rb"))
        fgqa_df = pd.read_parquet(bytes_io)
        for index, row in fgqa_df.iterrows():
            metadata = row['metadata']
            source_id = metadata['source_video_id']
            source = metadata['source_dataset']
            question_group_id = row['qa_uid']
            if source == "egoexo4d":
                video_path = self.get_ego_exo_video_path(ego_exo_uid_to_metadata[source_id], row, egoexo_segment_id_to_camera_id)
            elif source == "ego4d":
                video_path = os.path.join(self.ego_4d_data_path, source_id + ".mp4")
            else:
                if not file_exists(os.path.join(self.yt_dataset_path, source_id)):
                    continue  # Some videos in the PLM eval set and no longer found on the internet

                file_list = os.listdir(os.path.join(self.yt_dataset_path, source_id))
                assert len(file_list) >= 2, f"Number of files in {source_id} is {len(file_list)}, expected >= 2. {file_list}"
                video_path = None
                for file_name in file_list:
                    if ".json" not in file_name and "_bounded_" not in file_name:
                        video_path = os.path.join(self.yt_dataset_path, source_id, file_name)
                        break
                assert video_path is not None, f"No video file found for {source_id}"

                _, video_ext = os.path.splitext(video_path)
                if "webm" in video_ext:
                    video_path = reformat_webm_to_mp4_if_needed(video_path)

            start_time = metadata['source_start_time']
            end_time = metadata['source_end_time']
            video_path = save_bounded_video(video_path, start_time, end_time, task_type="default")

            option_tuple_list = [(option[0], option[1]) for option in row['options'].items()]
            sorted_options = sorted(option_tuple_list, key=lambda x: int(x[0].lstrip("option_")))
            option_list = [option[1] for option in sorted_options]

            question, answer = self.qa_template(option_list, int(row['answer_index']), row['question'])
            example = {
                "question": question,
                "answer": answer,
                "video": video_path,
                "metadata": dict(
                    question_id=row["uid"],
                    question_group_id=question_group_id,
                    video_segment_id=row['video'],
                    video_source_id=source_id,
                    video_source_type=source,
                    options=option_list,
                    video_url=PUBLIC_VIDEO_BASE_URL + video_path[len(VIDEO_DATA_HOME):],
                )
            }
            data.append(example)
        return data

    def get(self, idx, rng):
        return dict(**self.data[idx], style="video_eval_multiple_choice")


class VideoHallucer(DatasetBase):
    data_path = os.path.join(VIDEO_DATA_HOME, "VideoHallucer")
    categories = ["temporal", "semantic_detail", "object_relation", "external_nonfactual", "external_factual"]

    def __init__(self, split):
        assert split in ["validation"]
        super().__init__(split)

    def load(self):
        data = []
        for category in self.categories:
            category_path = os.path.join(self.data_path, category)
            category_json = os.path.join(category_path, f"{category}.json")
            for pair_idx, paired_qa in enumerate(json.load(open(category_json, "r"))):
                qa_group_id = f"{category}-{pair_idx}-basic_vid-{paired_qa['basic']['video']}"
                for qa in [paired_qa["basic"], paired_qa["hallucination"]]:
                    question = "\n".join([qa["question"], "Please answer yes or no:"])
                    answer = qa["answer"]
                    video_path = os.path.join(category_path, "videos", qa["video"])
                    example = {
                        "question": question,
                        "answer": answer,
                        "video": video_path,
                        "metadata": dict(
                            question_group_id=qa_group_id,
                            type=paired_qa['type'],
                        )
                    }
                    data.append(example)
        return data

    def get(self, idx, rng):
        return dict(**self.data[idx], style="video_eval_short_answer")


class PlmFGQATrain(DatasetBase):
    data_path = os.path.join(VIDEO_DATA_HOME, "PLM-FGQA")
    coin_video_path = os.path.join(VIDEO_DATA_HOME, "coin/videos")
    coin_video_segments_path = os.path.join(VIDEO_DATA_HOME, "coin/video_segments")
    youcook2_path = os.path.join(VIDEO_DATA_HOME, "YouCook2/all_data")
    youcook2_video_path = os.path.join(VIDEO_DATA_HOME, "YouCook2/all_data/raw_videos")
    youcook2_video_wo_ext_to_video_path = os.path.join(VIDEO_DATA_HOME, "YouCook2/all_data/video_wo_ext_to_video_path.json")
    crosstask_path = os.path.join(VIDEO_DATA_HOME, "crosstask")
    ego_4d_data_path = os.path.join(VIDEO_DATA_HOME, "Ego4d/ego4d_data/v2/full_scale")
    # ht100m_data_path = os.path.join(VIDEO_DATA_HOME, "ht100m")
    ht100m_data_path = os.path.join(VIDEO_DATA_HOME, "ht100m_mp4")
    # ht100m_non_h264_path = os.path.join(VIDEO_DATA_HOME, "ht100m/non_h264_ht100m_may_6_2025_57k_filtered.txt")

    corrupt_files = set(
        ['PjMSsT8Yy_M.mp4', 'X0MNgGS4PSY.mp4', 'fHvuWaClMz0.mp4', 'qT-OohQciG0.mp4', 
         'N49jLLpfyco.mp4', 'cGXfvgXnk7c.mp4', 'eXAbTeQEBOg.mp4', 'T-ShtSvQaSA.mp4', 
         'dX5hY0pVNb0.mp4']
    )
    

    def __init__(self, split):
        assert split in ["train"]
        super().__init__(split)

    @staticmethod
    def probe_for_h264(video_path):
        result = subprocess.run(
            ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
             '-show_entries', 'stream=codec_name', '-of', 'default=nw=1:nk=1', video_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        return result.stdout.strip() == 'h264'

    def load(self):
        data = []
        parquet_path = os.path.join(self.data_path, "plm_fgqa_train_w_src_file_exists.parquet")
        fgqa_df = pd.read_parquet(resource_path(dirname(parquet_path), basename(parquet_path)))
        fgqa_df = fgqa_df[fgqa_df["segment_id"].notna()]
        source_groups = fgqa_df.groupby('source_dataset')

        segment_id_to_message_list = {}
        missing = defaultdict(int) 

        n_rows = len(fgqa_df)
        processed_groups = pd.DataFrame()

        with tqdm(total=n_rows, desc="Processing FGQA train data") as pbar:
            for source, group_df in source_groups:
                log.info(f"Processing source: {source}, number of rows: {len(group_df)}")
                if DEBUG := False:
                    log.info("Sampling 1000 rows for debugging")
                    group_df = group_df.sample(n=min(1000, len(group_df)), random_state=42)
                if source == "coin":
                    group_df['video_name'] = group_df.apply(
                        lambda row: f"{row['source_video_id']}_{round(row['source_start_time'], 2)}_{round(row['source_end_time'], 2)}.mp4", axis=1
                    )
                    group_df['video_path'] = group_df.apply(
                        lambda row: os.path.join(self.coin_video_segments_path, row["video_name"]), axis=1
                    )
                    group_df = group_df[group_df['file_exists'] == True]
                    missing[source] = len(group_df) - group_df['file_exists'].sum()
                    processed_groups = pd.concat([processed_groups, group_df], ignore_index=True)
                elif source == "youcook2":
                    group_df['video_name'] = group_df['weka_path'].apply(
                        lambda x: os.path.basename(x) if isinstance(x, str) else None
                    )
                    group_df['video_path'] = group_df['weka_path'].apply(
                        lambda weka_path: os.path.join(
                            self.youcook2_video_path,
                            *weka_path.split(os.sep)[-3:]
                        )
                    )
                    group_df = group_df[group_df['file_exists'] == True].copy()
                    missing[source] = len(group_df) - group_df['file_exists'].sum()
                elif source == "crosstask":
                    group_df['video_path'] = group_df.apply(
                        lambda row: os.path.join(
                            self.crosstask_path, row['source_video_id'], f"{row['source_video_id']}.mp4"
                        ), axis=1
                    )
                    missing[source] = len(group_df) - group_df['file_exists'].sum()
                    group_df = group_df[group_df['file_exists'] == True].copy()
                elif source == "ego4d":
                    missing[source] += len(group_df) - group_df['file_exists'].sum()
                    group_df['video_path'] = group_df.apply(
                        lambda row: os.path.join(self.ego_4d_data_path, row['source_video_id'] + ".mp4"), axis=1
                    )
                    group_df = group_df[group_df['file_exists'] == True]
                elif source == "ht100m":
                    missing[source] = len(group_df) - group_df['file_exists'].sum()
                    group_df['video_path'] = group_df.apply(
                        lambda row: os.path.join(self.ht100m_data_path, row['source_video_id'], f"{row['source_video_id']}.mp4"), axis=1
                    )
                    group_df = group_df[group_df['file_exists'] == True]
                else:
                    continue

                for _, row in group_df.iterrows():
                    if row['video_path'].split("/")[-1] in self.corrupt_files:
                        log.warning(f"Skipping corrupt file: {row['video_path']}")
                        continue
                    if row['segment_id'] not in segment_id_to_message_list:
                        segment_id_to_message_list[row['segment_id']] = {
                            "video": row['video_path'],
                            "metadata": dict(
                                video_segment_id=row['segment_id'],
                                video_source_id=row['source_video_id'],
                                video_source_dataset=source,
                            ),
                            "message_list": [dict(
                                question=row['question'],
                                answer=row['answer'],
                                style="llava_video_da",
                            )]
                        }
                        if source in ["ht100m", "ego4d"]:
                            segment_id_to_message_list[row['segment_id']]['metadata']['clip_start_time'] = row['source_start_time']
                            segment_id_to_message_list[row['segment_id']]['metadata']['clip_end_time'] = row['source_end_time']
                    else:
                        segment_id_to_message_list[row['segment_id']]['message_list'].append(
                            dict(
                                question=row['question'],
                                answer=row['answer'],
                                style="llava_video_da",
                            )
                        )

                pbar.update(len(group_df))

        for segment_id, example in segment_id_to_message_list.items():
            data.append(example)

        return data

    def get(self, idx, rng):
        return dict(**self.data[idx], )


class NeXTQA(DatasetBase):
    data_path = os.path.join(VIDEO_DATA_HOME, "NeXTQA")

    def __init__(self, split, task, flat: bool = False, max_per_video: Optional[int] = None, difficulty="all"):
        # implement MC task for now
        if task == "multiple-choice":
            assert split in ["train", "val", "test"]
        else:
            raise NotImplementedError(f"Task {task} not implemented")
        assert difficulty in ["easy", "medium", "hard", "all"]
        self.difficulty = difficulty
        self.task = task
        self.split = split
        self.flat = flat
        self.max_per_video = max_per_video
        super().__init__(split)

    def mc_qoa_template(self, data):
        options = [data[f'a{idx}'].strip() for idx in range(5)]
        option_text = "\n".join(
            f"{chr(ord('A') + idx)}. {options[idx]}" for idx in range(5)
        )
        answer = f"{chr(ord('A') + int(data['answer']))}"
        question = "\n".join(
            [
                data["question"].strip(),
                option_text,
                "Answer with the option's letter from the given choices directly.",
            ]
        )
        return question, options, answer

    def load(self):
        difficulty_lookup = DatasetSampleDifficulty("next_qa_mc_three_frame_gemini_2_0_filter_results.json")

        task = self.task
        data_list = []
        if task == "multiple-choice" and self.split == "test":
            # df = pd.read_parquet(os.path.join(self.data_path, "MC", "test-00000-of-00001.parquet"))
            # bytes_io = BytesIO(read_file(os.path.join(self.data_path, "MC", "test-00000-of-00001.parquet"), "rb"))
            # df = pd.read_parquet(bytes_io)
            df_path = join(self.data_path, "MC", "test-00000-of-00001.parquet")
            df = pd.read_parquet(resource_path(dirname(df_path), basename(df_path)))

            for idx, row in df.iterrows():
                video_path = os.path.join(self.data_path, "NExTVideo", f"{row['video']}.mp4")
                question, options, answer = self.mc_qoa_template(row)

                example_id = f"{row['video']}_{idx}_type_{row['type']}"
                sample_difficulty = difficulty_lookup.get_difficulty(example_id)
                if self.difficulty != "all" and sample_difficulty != self.difficulty:
                    continue

                example = {
                    "question": question,
                    "answer": answer,
                    "video": video_path,
                    "style": "video_eval_multiple_choice",
                    "metadata": dict(
                        example_id=example_id,
                        question_id=str(idx),
                        question_type=row["type"],
                        video_id=row["video"],
                        options=options,

                        video_url=PUBLIC_VIDEO_BASE_URL + video_path[len(VIDEO_DATA_HOME):],
                        sample_difficulty=sample_difficulty
                    )
                }
                data_list.append(example)

        elif task == "multiple-choice" and self.split in {"train", "val"}:
            df_path = join(self.data_path, f"{self.split}.csv")
            id_map_path = join(self.data_path, "map_vid_vidorID.json")

            df = pd.read_csv(resource_path(dirname(df_path), basename(df_path)))
            id_map = json.load(open(resource_path(dirname(id_map_path), basename(id_map_path))))
            df["video_path"] = df["video"].apply(lambda x: join(self.data_path, "NExTVideo-all-videos", id_map[str(x)] + ".mp4"))

            video2msgs = {}
            for row in df.itertuples(False):
                video_path = row.video_path
                video2msgs[video_path] = video2msgs.get(video_path, [])
                msg = dict(
                    question=row.question,
                    options=[row.a0, row.a1, row.a2, row.a3, row.a4],
                    answer_idx=row.answer,
                    style="video_multiple_choice",
                )
                video2msgs[video_path].append(msg)

                if self.flat:
                    formatted_ex = {
                        "video": video_path,
                        "message_list": [msg]
                    }
                    data_list.append(formatted_ex)
            
            if not self.flat:
                for video, msgs in video2msgs.items():
                    if len(msgs) == 0: continue
                    if self.max_per_video:
                        for msg in split_into_groups(msgs, self.max_per_video):
                            formatted_ex = {
                                "video": video,
                                "message_list": msg
                            }
                            data_list.append(formatted_ex)
                    else:
                        formatted_ex = {
                            "video": video,
                            "message_list": msgs,
                        }
                        data_list.append(formatted_ex)
        else:
            raise NotImplementedError(f"Task {task} not implemented")

        return data_list

    def get(self, idx, rng):
        return self.data[idx]


class MMEVideoOCR(DatasetBase):
    data_path = os.path.join(VIDEO_DATA_HOME, "MME-VideoOCR")

    def __init__(self, split, subset='mc'):
        assert split in ["test"]
        assert subset in ["mc", "all"]
        self.subset = subset
        super().__init__(split)

    def qa_template(self, qa_data):
        option_text = "\n".join(f"{chr(ord('A') + idx)}. {opt}" for idx, opt in enumerate(qa_data['option']))
        answer = qa_data['answer']
        question = "\n".join(
            [
                qa_data["question"],
                option_text,
                "Answer with the option's letter from the given choices directly.",
            ]
        )
        return question, answer

    def load(self):

        json_data = pd.read_json(resource_path(join(self.data_path, "dataset.json")))
        data = []
        for row_idx, example in json_data.iterrows():
            if self.subset == 'mc':
                if example['eval_method'] != 'multiple_choice':
                    continue

            example_id = f"{example['video_index']}_{example['index']}"
            video_path = os.path.join(self.data_path, "Video", f"{example['video_index']}.mp4")

            if example['eval_method'] == 'multiple_choice':
                question, answer = self.qa_template(example)
                example = {
                    "question": question,
                    "answer": answer,
                    "video": video_path,
                    "metadata": dict(
                        example_id=example_id,
                        question_id=example["index"],
                        video_id=example["video_index"],
                        options=example["option"],
                        question_category=example["task_type"],
                        task=example["task"],
                        eval_method='multiple_choice',
                    ),
                    "style": "video_eval_multiple_choice", #, style="video_short_answer"
                }
            else:
                question, answer = self.qa_template(example)
                example = {
                    "question": question,
                    "answer"  : answer,
                    "video"   : video_path,
                    "metadata": dict(
                        example_id=example_id,
                        question_id=example["index"],
                        video_id=example["video_index"],
                        question_category=example["task_type"],
                        task=example["task"],
                        eval_method='containment_match',
                    ),
                    "style"   : "video_eval_short_answer",  # , style="video_short_answer"
                }

            data.append(example)
        return data

    def get(self, idx, rng):
        return dict(**self.data[idx])


def time_to_seconds(time_str: str) -> float:
    h, m, s = time_str.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


class LVBench(DatasetBase):

    def __init__(self):
        self.data_path = os.path.join(VIDEO_DATA_HOME, "LVBench")
        super().__init__("test")

    @staticmethod
    def parse_mcq_lines(text: str):
        """
        Assumes:
          - line 1 = question
          - lines 2..n = options (one per line)
        Accepts options like "(A) foo", "A) foo", "A. foo", or just "foo".
        Returns: (question, [(label, option_text), ...])
        """
        lines = [ln.strip() for ln in text.replace("\r\n", "\n").split("\n") if ln.strip()]
        if not lines:
            return "", []

        question = lines[0]
        raw_options = lines[1:]

        options = []
        for i, opt in enumerate(raw_options):
            # Try to pull a leading label; otherwise fall back to 1-based index
            # Patterns: (A) foo | A) foo | A. foo
            label = None
            if len(opt) >= 3:
                if opt[0] == "(" and ")" in opt[:4]:
                    label = opt[1:opt.index(")")]
                    opt = opt[opt.index(")") + 1:].strip()
                elif opt[1:3] in (") ", ").") and opt[0].isalnum():
                    label = opt[0]
                    opt = opt[3:].strip()
                elif len(opt) > 2 and opt[1] == "." and opt[0].isalnum():
                    label = opt[0]
                    opt = opt[2:].strip()

            if label is None:
                label = str(i + 1)  # fallback label

            options.append((label, opt))

        return question, options

    def qa_template(self, data):
        question, options = self.parse_mcq_lines(data['question'])
        options = "\n".join(f"{idx}. {c}" for idx, c in options)
        answer = f"{data['answer']}."
        question = "\n".join(
            [
                question,
                options,
                "Please respond with only the letter of the correct answer.",
            ]
        )
        return question, answer

    def load(self):
        data = []

        with open(join(self.data_path, "video_info.meta.jsonl"), "r") as f:
            video_info = [json.loads(line) for line in f.readlines()]

        for info in video_info:
            for qa in info["qa"]:
                question, answer = self.qa_template(qa)
                data.append(dict(
                    video=join(self.data_path, info["key"] + ".mp4"),
                    question=question,
                    metadata=dict(
                        answer=answer,
                        type=info["type"],
                        uid=qa["uid"],
                        qtype=qa["question_type"],
                        time_reference=qa["time_reference"],
                        video_url=PUBLIC_VIDEO_BASE_URL + join("LVBench", info["key"] + ".mp4")
                    ),
                    style="video_eval_multiple_choice",
                ))

        return data

    def get(self, item, rng):
        return self.data[item]


class LongVideoBench(DatasetBase):
    data_path = os.path.join(VIDEO_DATA_HOME, "LongVideoBench")
    duration_groups = ["15", "60", "600", "3600"]

    def __init__(self, split, allow_subtitle=True, difficulty="all", duration_group="all", with_subtitle=False):
        assert split in ["validation", "test"]
        assert difficulty in ["easy", "medium", "hard", "all"]
        assert duration_group in (self.duration_groups + ["all"])

        self.difficulty = difficulty
        self.duration_group = duration_group
        self.allow_subtitle = allow_subtitle
        self.with_subtitle = with_subtitle
        if with_subtitle:
            assert self.allow_subtitle is True, "with_subtitle at True, want to include subtitle questions"
            self.style = "video_eval_multiple_choice_w_subtitle"
        else:
            self.style = "video_eval_multiple_choice"
        super().__init__(split)

    def qa_template(self, qa_data):
        option_text = "\n".join(f"{chr(ord('A') + idx)}. {opt}" for idx, opt in enumerate(qa_data['candidates']))
        answer = f"{chr(ord('A') + qa_data['correct_choice'])}" if "correct_choice" in qa_data else None
        question = "\n".join(
            [
                qa_data["question"],
                option_text,
                "Answer with the option's letter from the given choices directly.",
            ]
        )
        return question, answer

    def load(self):
        difficulty_lookup = DatasetSampleDifficulty("long_video_bench_three_frame_gemini_2_0_filter_results.json")

        if self.split == "validation":
            json_data = json.loads(read_file(os.path.join(self.data_path, "lvb_val.json")))
        else:
            json_data = json.loads(read_file(os.path.join(self.data_path, "lvb_test_wo_gt.json")))

        data = []
        for row_idx, qa_data in enumerate(json_data):
            question, answer = self.qa_template(qa_data)
            if not self.allow_subtitle and "subtitle" in question:
                continue
            if self.duration_group != "all" and qa_data["duration_group"] != self.duration_group:
                continue

            video_path = os.path.join(self.data_path, "videos", qa_data["video_path"])
            example = {
                "question": question,
                "answer": answer,
                "video": video_path,
                "metadata": dict(
                    question_id=qa_data["id"],
                    video_id=qa_data["video_id"],
                    level=qa_data["level"],
                    options=qa_data["candidates"],
                    question_category=qa_data["question_category"],
                    duration_group=qa_data["duration_group"],
                    video_url=PUBLIC_VIDEO_BASE_URL + video_path[len(VIDEO_DATA_HOME):],
                )
            }
            if self.split == "validation":
                example_id = f"{qa_data['video_id']}_{qa_data['id']}_{row_idx}"
                sample_difficulty = difficulty_lookup.get_difficulty(example_id)
                if self.difficulty != "all" and sample_difficulty != self.difficulty:
                    continue
                example['metadata']['example_id'] = example_id
                example['metadata']['sample_difficulty'] = sample_difficulty

            if self.with_subtitle:
                subtitle = json.loads(read_file(os.path.join(self.data_path, "subtitles", qa_data["subtitle_path"])))
                subtitle_dict = {}
                starting_timestamp_for_subtitles = qa_data['starting_timestamp_for_subtitles']
                ending_timestamp_for_subtitles = starting_timestamp_for_subtitles + qa_data['duration']
                for entry in subtitle:
                    if "timestamp" in entry:
                        sub_start, sub_end = entry["timestamp"]
                        if not isinstance(sub_end, float):
                            sub_end = qa_data['duration']
                        text = entry["text"]
                    else:
                        sub_start, sub_end = float(time_to_seconds(entry['start'])), float(time_to_seconds(entry['end']))
                        text = entry["line"]

                    if sub_end < starting_timestamp_for_subtitles or sub_start > ending_timestamp_for_subtitles:
                        continue
                    sub_start -= starting_timestamp_for_subtitles
                    sub_end -= starting_timestamp_for_subtitles
                    if sub_start < 0:
                        sub_start = 0.0
                    subtitle_dict[(sub_start, sub_end)] = text
                example['subtitle'] = subtitle_dict
            data.append(example)
        return data

    def get(self, idx, rng):
        return dict(**self.data[idx], style=self.style)


class LongVideoBenchCaption(DatasetBase):
    data_path = os.path.join(VIDEO_DATA_HOME, "LongVideoBench")

    def __init__(self, split):
        assert split in ["test"]
        super().__init__(split)

    def load(self):
        json_data = json.loads(read_file(os.path.join(self.data_path, "lvb_test_caption.json"), "r"))
        data = []
        for k, qa_data in json_data.items():
            video_path = os.path.join(self.data_path, "videos", qa_data["video_path"])
            caption = qa_data["caption"]
            example = {
                "video": video_path,
                "message_list": [dict(text=caption, style="video_long_caption")],
                "metadata": dict(
                    question_id=qa_data["id"],
                    video_id=qa_data["video_id"],
                    caption=caption,
                    statements= qa_data["statements"],
                    question_category=qa_data["topic_category"],
                    duration_group=qa_data["duration_group"],
                    video_url=PUBLIC_VIDEO_BASE_URL + video_path[len(VIDEO_DATA_HOME):]
                )
            }
            data.append(example)
        return data

    def get(self, idx, rng):
        return self.data[idx]


class MLVU(DatasetBase):
    data_path = os.path.join(VIDEO_DATA_HOME, "MVLU", "MLVU")
    test_data_path = os.path.join(VIDEO_DATA_HOME, "MLVU_Test")
    val_mc_tasks = [
        "plotQA",
        "needle",
        "ego",
        "count",
        "order",
        "anomaly_reco",
        "topic_reasoning"
    ]
    vak_gen_tasks = ["sub_scene", "summary"]

    def __init__(self, split, task, use_resize=False):
        assert split in ["validation", "test"]
        assert task in ["multiple-choice", "generation"]
        self.task = task
        self.use_resize = use_resize
        super().__init__(split)

    def mc_qa_template(self, data):
        """lmms-eval uses the MVBench's template, but llava-video uses the different one, so just follow the PerceptionTest's template"""
        option_text = "\n".join(f"{chr(ord('A') + idx)}. {opt}" for idx, opt in enumerate(data['candidates']))
        answer_idx = data['candidates'].index(data['answer'])
        answer = f"{chr(ord('A') + answer_idx)}"
        question = "\n".join(
            [
                data["question"],
                option_text,
                "Answer with the option's letter from the given choices directly.",
            ]
        )
        return question, answer
        """
        option_text = "\n".join(f"({chr(ord('A') + idx)}) {opt}" for idx, opt in enumerate(data['candidates']))
        answer_idx = data['candidates'].index(data['answer'])
        answer = f"{chr(ord('A') + answer_idx)}"
        question = "\n".join([data["question"], option_text, "Only give the best option.", "Best option: ("])
        return question, answer
        """

    def load(self):
        task = self.task
        data = []
        if self.split == "test":
            assert self.task in "multiple-choice"
            gt_video_path = os.path.join(self.test_data_path, "video")
            gt_path = os.path.join(self.test_data_path, "test-ground-truth", "test_mcq_gt.json")
            gt_data = json.load(open(gt_path, "r"))

            data = []
            for qa_data in gt_data:
                video_path = os.path.join(gt_video_path, qa_data['video'])
                question, answer = self.mc_qa_template(qa_data)
                example = {
                    "question": question,
                    "answer": answer,
                    "video": video_path,
                    "style": "video_eval_multiple_choice",
                    "metadata": dict(
                        question_id=str(qa_data['question_id']),
                        video_id=qa_data['video'],
                        video_url=PUBLIC_VIDEO_BASE_URL + video_path[len(VIDEO_DATA_HOME):],
                        duration=qa_data['duration'],
                    )
                }
                data.append(example)

        elif task == "multiple-choice":
            question_id = 0
            for idx, task_type in enumerate(self.val_mc_tasks, 1):
                name = f"{idx}_{task_type}"
                json_data = json.loads(read_file(os.path.join(self.data_path, "json", f"{name}.json")))
                for qa_data in json_data:
                    small_video_path = os.path.join(self.data_path, "video_resized", name, f"{qa_data['video']}")
                    if self.use_resize and os.path.exists(small_video_path):
                        video_path = small_video_path
                    else:
                        video_path = os.path.join(self.data_path, "video", name, f"{qa_data['video']}")
                    question, answer = self.mc_qa_template(qa_data)
                    example = {
                        "question": question,
                        "answer": answer,
                        "video": video_path,
                        "style": "video_eval_multiple_choice",
                        "metadata": dict(
                            question_id=str(question_id),
                            video_id=qa_data['video'],
                            task_type=task_type,
                            video_url=PUBLIC_VIDEO_BASE_URL + video_path[len(VIDEO_DATA_HOME):],
                            duration=qa_data['duration'],
                        )
                    }
                    data.append(example)
                    question_id += 1
        else:
            question_id = 0
            for idx, task_type in enumerate(self.vak_gen_tasks, 1):
                name = f"{len(self.val_mc_tasks) + idx}_{task_type}"
                json_data = json.loads(read_file(os.path.join(self.data_path, "json", f"{name}.json")))
                for qa_data in json_data:
                    video_path = os.path.join(self.data_path, "video", name, f"{qa_data['video']}")
                    example = {
                        "question": qa_data['question'],
                        "answer": qa_data['answer'],
                        "video": video_path,
                        "style": "demo",
                        "metadata": dict(
                            question_id=str(question_id),
                            question=qa_data['question'],
                            answer=qa_data['answer'],
                            video_id=qa_data['video'],
                            task_type=task_type,
                            video_url=PUBLIC_VIDEO_BASE_URL + video_path[len(VIDEO_DATA_HOME):],
                            duration=qa_data['duration'],
                        )
                    }
                    if "scoring_points" in qa_data:
                        example["metadata"]["scoring_points"] = qa_data["scoring_points"]
                    data.append(example)
                    question_id += 1
        data.sort(key=lambda x: x["metadata"]["duration"])
        # # Manually filter outlier that is >32K seconds
        # data = [x for x in data if x["metadata"]["duration"] < 8500]
        return data

    def get(self, idx, rng):
        return self.data[idx]


class PerceptionTest(DatasetBase):
    """PerceptionTest Multiple-Choice Video QA task"""
    data_path = os.path.join(VIDEO_DATA_HOME, "PerceptionTest_Val")
    # data_path = os.path.join(VIDEO_DATA_HOME, "PerceptionTest")

    def __init__(
            self, 
            split, 
            flat=False,
            max_per_video=None,
        ):
        assert split in ["train", "validation", "test"]
        self.split = split
        self.flat = flat
        self.max_per_video = max_per_video
        super().__init__(split)

    def qa_template(self, question, options, answer_id):
        prefixes = "abcdefg".upper()
        option_text = "\n".join(
            f"{prefix}. {opt}" for prefix, opt in zip(prefixes, options)
        )
        question = "\n".join(
            [
                question,
                option_text,
                "Answer with the option's letter from the given choices directly.",
            ]
        )
        if answer_id is not None:
            answer = prefixes[answer_id]
        else:
            answer = None
        return question, answer

    def load(self):
        data_list = []
        if self.split == "validation":
            parquet_path = os.path.join(self.data_path, "mc_question_val", "validation-00000-of-00001.parquet")
            df = pd.read_parquet(resource_path(parquet_path))

            for idx, row in df.iterrows():
                video_path = os.path.join(self.data_path, "videos", row["video_name"] + ".mp4")
                question, answer = self.qa_template(row["question"], row["options"], int(row["answer_id"]))
                example = {
                    "question": question,
                    "answer": answer,
                    "video": video_path,
                    "metadata": dict(
                        question_id=row["question_id"],
                        video_id=row["video_name"],
                        answer_idx=int(row["answer_id"]),
                        area=row["area"],
                        reasoning=row["reasoning"],
                        video_url=PUBLIC_VIDEO_BASE_URL + video_path[len(VIDEO_DATA_HOME):],
                    )
                }
                data_list.append(example)

        elif self.split == "test":
            parquet_path = os.path.join(self.data_path, "Test/mc_question", "test-00000-of-00001.parquet")
            df = pd.read_parquet(resource_path(parquet_path))
            for idx, row in df.iterrows():
                video_path = os.path.join(self.data_path, "Test/videos", row["video_name"] + ".mp4")
                question, _ = self.qa_template(row["question"], row["options"], None)
                example = {
                    "question": question,
                    "video": video_path,
                    "metadata": dict(
                        question_id=row["question_id"],
                        video_id=row["video_name"],
                    )
                }
                data_list.append(example)

        elif self.split == "train":
            json_path = os.path.join(self.data_path, "all_train.json")
            train_anns = json.load(open(resource_path(dirname(json_path), basename(json_path))))

            video2msgs = {}
            for video_id, ann in train_anns.items():
                video_path = os.path.join(self.data_path, "train_videos", video_id + ".mp4")
                for qa_data in ann["mc_question"]:
                    if video_path not in video2msgs:
                        video2msgs[video_path] = []
                    msg = dict(
                        question=qa_data['question'],
                        options=qa_data["options"],
                        answer_idx=qa_data["answer_id"],
                        style="video_multiple_choice",
                    )
                    if self.flat:
                        formatted_ex = {
                            "video": video_path,
                            "message_list": [msg]
                        }
                        data_list.append(formatted_ex)
                    video2msgs[video_path].append(msg)
            
            if not self.flat:
                for video, msgs in video2msgs.items():
                    if len(msgs) == 0: continue
                    if self.max_per_video:
                        for msg in split_into_groups(msgs, self.max_per_video):
                            formatted_ex = {
                                "video": video,
                                "message_list": msg
                            }
                            data_list.append(formatted_ex)
                    else:
                        formatted_ex = {
                            "video": video,
                            "message_list": msgs,
                        }
                        data_list.append(formatted_ex)

        return data_list
    
    def get(self, idx, rng):
        return dict(**self.data[idx], style="video_eval_multiple_choice")


class EgoSchema(DatasetBase):
    data_path = os.path.join(VIDEO_DATA_HOME, "egoschema")

    def __init__(self, split):
        assert split in ["validation", "test"]
        super().__init__(split)

    def question_template(self, question, options):
        question = "\n".join(
            [
                question,
                "\n".join(options),
                "Answer with the option's letter from the given choices directly.",
            ]
        )
        return question

    def load(self):
        subset_tag = "Subset" if self.split == "validation" else "MC"
        parquet_path = os.path.join(self.data_path, subset_tag, "test-00000-of-00001.parquet")
        df = pd.read_parquet(resource_path(dirname(parquet_path), basename(parquet_path)))
        data = []
        for idx, row in df.iterrows():
            video_path = os.path.join(self.data_path, "videos", row["video_idx"] + ".mp4")
            question = self.question_template(row["question"], row["option"])

            if row["answer"] is not None:
                answer = "abcdefg".upper()[int(row["answer"])]
            else:
                answer = None
            example_id = f"{row['question_idx']}_{row['video_idx']}_{idx}"

            example = {
                "question": question,
                "answer": answer,
                "video": video_path,
                "metadata": dict(
                    example_id=example_id,
                    question_id=row["question_idx"],
                    video_id=row["video_idx"],
                    options=list(row["option"]),
                    answer_idx=row["answer"],
                    video_url=PUBLIC_VIDEO_BASE_URL + video_path[len(VIDEO_DATA_HOME):]
                )
            }
            data.append(example)
        return data

    def get(self, idx, rng):
        return dict(**self.data[idx], style="video_eval_multiple_choice")


class VideoMME(DatasetBase):
    data_path = os.path.join(VIDEO_DATA_HOME, "Video-MME")
    duration = ["short", "medium", "long"]

    def __init__(self, split, duration="all", difficulty="all", with_subtitle=False):
        assert split in ["validation"]
        assert duration in (self.duration + ["all"])
        assert difficulty in ["easy", "medium", "hard", "all"]
        self.difficulty = difficulty
        self.target_duration = duration
        self.with_subtitle = with_subtitle
        super().__init__(split)

    def question_template(self, question, options):
        prompt = "Select the best answer to the following multiple-choice question based on the video."
        prompt += " Respond with only the letter (A, B, C, or D) of the correct option."
        question = "\n".join(
            [
                prompt,
                question,
                "\n".join(options),
                "The best answer is:"
            ]
        )
        return question

    def load(self):
        difficulty_lookup = DatasetSampleDifficulty("video_mme_three_frame_gemini_2_0_filter_results.json")
        parquet_path = os.path.join(self.data_path, "videomme", "test-00000-of-00001.parquet")
        df = pd.read_parquet(resource_path(dirname(parquet_path), basename(parquet_path)))
        if self.target_duration != "all":
            df = df[df["duartion"] == self.target_duration]
        data = []
        video_dir = os.path.join(self.data_path, "data")
        subtitles = json.loads(read_file(resource_path(os.path.join(self.data_path, "subtitles.json"))))
        for idx, row in df.iterrows():
            question = self.question_template(row["question"], row["options"])
            video_path = os.path.join(video_dir, row["videoID"] + ".mp4")
            example_id = f"{row['question_id']}_{row['video_id']}_{idx}"
            sample_difficulty = difficulty_lookup.get_difficulty(example_id)
            if self.difficulty != "all" and sample_difficulty != self.difficulty:
                continue

            example = {
                "question": question,
                "answer": row["answer"],
                "video": video_path,
                "style": "video_eval_multiple_choice",
                "metadata": dict(
                    example_id=example_id,
                    video_id=row["video_id"],
                    question_id=row["question_id"],
                    duration=row["duration"],
                    domain=row["domain"],
                    sub_category=row["sub_category"],
                    task_type=row["task_type"],
                    video_url=PUBLIC_VIDEO_BASE_URL + video_path[len(VIDEO_DATA_HOME):],
                    sample_difficulty=sample_difficulty
                )
            }
            if self.with_subtitle and row["videoID"] in subtitles:
                subtitle = {}
                for i, entry in subtitles[row["videoID"]].items():
                    subtitle[(entry["start"], entry["end"])] = entry["text"]
                example["subtitle"] = subtitle
                example["style"] = "video_eval_multiple_choice_w_subtitle"
            data.append(example)

        return data

    def get(self, idx, rng):
        return dict(**self.data[idx])


class TempCompass(DatasetBase):
    data_path = os.path.join(VIDEO_DATA_HOME, "TempCompass")
    tasks = ["multi-choice", "yes_no", "caption_matching", "captioning"]
    answer_prompt = {
        "multi-choice": "Please directly give the best option:",
        "yes_no": "Please answer yes or no:",
        "caption_matching": "Please directly give the best option:",
        "captioning": "" # The answer "Generated Caption:" is already contained in the question
    }

    def question_template(self, question, task):
        question = "\n".join([question, self.answer_prompt[task]])
        return question

    def __init__(self, split, task="all", difficulty="all"):
        assert split in ["validation"]
        assert task in (self.tasks + ["all", "internal"])
        self.target_task = task
        self.difficulty = difficulty
        assert self.difficulty in ["easy", "medium", "hard", "all"]
        super().__init__(split)

    def load(self):
        if self.target_task == "all":
            target_tasks = self.tasks
        elif self.target_task == "internal":
            target_tasks = ["multi-choice", "yes_no","caption_matching"]
        else:
            target_tasks = [self.target_task]

        difficulty_lookup = DatasetSampleDifficulty("temp_compass_three_frame_gemini_2_0_filter_results.json")

        meta_infos = json.loads(read_file(os.path.join(self.data_path, "meta_info.json")))
        data = []
        for task in target_tasks:
            parquet_file = resource_path(join(self.data_path, task), "test-00000-of-00001.parquet")
            df = pd.read_parquet(parquet_file)
            if task == "captioning":
                style = "demo"
            elif task in ["multi-choice", "caption_matching"]:
                style = "video_eval_multiple_choice"
            else:
                style = "video_eval_short_answer"
            for sample_idx, row in df.iterrows():
                vid = row["video_id"]
                question = self.question_template(row["question"], task)
                video_path = os.path.join(self.data_path, "videos", f"{vid}.mp4")
                temp_asp = row["dim"]

                vid_identifier = vid.replace('.jpg', '').replace('.mp4', '')  # Follow the original evaluation script
                fine_grained_temp_asp = meta_infos[vid_identifier]["eval_dim"][temp_asp]["type"] if temp_asp != "order" else "order"

                example_id = f"{vid}_{task}_{sample_idx}"
                if task == "captioning":
                    sample_difficulty = "hard"
                else:
                    sample_difficulty = difficulty_lookup.get_difficulty(example_id)
                if self.difficulty != "all" and sample_difficulty != self.difficulty:
                    continue

                answer = row["answer"]
                if task == "caption_matching":
                    question = row["question"].split("\n")[0]
                    # question\n...\nOption 2: The woman is dancing and singing.
                    options = row["question"].split("\n")[1:]
                    if answer in options:
                        answer_idx = options.index(answer)
                    else:
                        logging.warning("Skipping example with no GT answer in the options")
                        continue
                    # remove prefix and leave options formatting to data formatter
                    no_prefix_options = [option.split(":")[1].strip() for i, option in enumerate(options)]
                    example = {
                        "question": question,
                        "answer_idx": answer_idx,
                        "answer": answer,
                        "options": no_prefix_options,
                        "video": video_path,
                        "metadata": dict(
                            video_id=vid,
                            task=task,
                            question=row["question"],
                            temporal_aspect=temp_asp,
                            fine_grained_temporal_aspect=fine_grained_temp_asp,
                            example_id=example_id,
                            video_url=PUBLIC_VIDEO_BASE_URL + video_path[len(VIDEO_DATA_HOME):],
                            sample_difficulty=sample_difficulty
                        ),
                        "style": style,
                    }
                else:
                    example = {
                        "question": question,
                        "answer": answer,
                        "video": video_path,
                        "metadata": dict(
                            video_id=vid,
                            task=task,
                            question=row["question"],
                            temporal_aspect=temp_asp,
                            fine_grained_temporal_aspect=fine_grained_temp_asp,
                            example_id=example_id,
                            sample_difficulty=sample_difficulty,
                            video_url=PUBLIC_VIDEO_BASE_URL + video_path[len(VIDEO_DATA_HOME):],
                        ),
                        "style": style,
                    }
                if "mc_question" in row:
                    example["metadata"]["mc_question"] = row["mc_question"]
                    example["metadata"]["mc_answer"] = row["mc_answer"]
                data.append(example)

        return data

    def get(self, idx, rng):
        return self.data[idx]


class MVBench(DatasetBase):
    data_path = os.path.join(VIDEO_DATA_HOME, "MVBench")
    data_list = {
        "Action Sequence": ("action_sequence.json", f"{data_path}/video/star/Charades_v1_480/", "video", True), # has start & end
        "Action Prediction": ("action_prediction.json", f"{data_path}/video/star/Charades_v1_480/", "video", True), # has start & end
        "Action Antonym": ("action_antonym.json", f"{data_path}/video/ssv2_video/", "video", False),
        "Fine-grained Action": ("fine_grained_action.json", f"{data_path}/video/Moments_in_Time_Raw/videos/", "video", False),
        "Unexpected Action": ("unexpected_action.json", f"{data_path}/video/FunQA_test/test/", "video", False),
        "Object Existence": ("object_existence.json", f"{data_path}/video/clevrer/video_validation/", "video", False),
        "Object Interaction": ("object_interaction.json", f"{data_path}/video/star/Charades_v1_480/", "video", True), # has start & end
        "Object Shuffle": ("object_shuffle.json", f"{data_path}/video/perception/videos/", "video", False),
        "Moving Direction": ("moving_direction.json", f"{data_path}/video/clevrer/video_validation/", "video", False),
        "Action Localization": ("action_localization.json", f"{data_path}/video/sta/sta_video/", "video", True),  # has start & end
        "Scene Transition": ("scene_transition.json", f"{data_path}/video/scene_qa/video/", "video", False),
        "Action Count": ("action_count.json", f"{data_path}/video/perception/videos/", "video", False),
        "Moving Count": ("moving_count.json", f"{data_path}/video/clevrer/video_validation/", "video", False),
        "Moving Attribute": ("moving_attribute.json", f"{data_path}/video/clevrer/video_validation/", "video", False),
        "State Change": ("state_change.json", f"{data_path}/video/perception/videos/", "video", False),
        "Fine-grained Pose": ("fine_grained_pose.json", f"{data_path}/video/nturgbd_convert/", "video", False),
        "Character Order": ("character_order.json", f"{data_path}/video/perception/videos/", "video", False),
        "Egocentric Navigation": ("egocentric_navigation.json", f"{data_path}/video/vlnqa/", "video", False),
        "Episodic Reasoning": ("episodic_reasoning.json", f"{data_path}/video/tvqa/frames_fps3_hq/", "frame", True),  # has start & end, read frame
        "Counterfactual Inference": ("counterfactual_inference.json", f"{data_path}/video/clevrer/video_validation/", "video", False),
    }
    data_types_with_bound = {"Action Sequence", "Action Prediction", "Object Interaction", "Action Localization", "Episodic Reasoning"}

    def __init__(self, split, difficulty="all", sample=None):
        assert split in ["validation", "val"]
        if split == "validation":
            split = "val"
        self.difficulty = difficulty
        assert self.difficulty in ["easy", "medium", "hard", "all"]
        super().__init__(split, sample)

    def qa_template(self, data):
        # question = f"Question: {data['question']}\n"
        # question += "Options:\n"
        question = data['question']
        answer = data['answer']
        options = "\n".join(f"{chr(ord('A') + idx)}. {c}" for idx, c in enumerate(data['candidates']))
        answer_idx = data['candidates'].index(answer)
        answer = f"{chr(ord('A') + answer_idx)}."
        question = "\n".join(
            [
                question,
                options,
                "Please respond with only the letter of the correct answer.",
            ]
        )
        return question, answer

    def load(self):
        data = []
        example_id_to_difficulty = DatasetSampleDifficulty("mvbench_three_frame_gemini_2_0_filter_results.json")
        for k, v in self.data_list.items():
            json_data = json.loads(read_file(os.path.join(self.data_path, "json", v[0])))

            for qa_idx, qa_data in enumerate(json_data):
                example_id = f"{k}_{qa_idx}"
                sample_difficulty = example_id_to_difficulty.get_difficulty(example_id)
                if self.difficulty != "all" and sample_difficulty != self.difficulty:
                    continue

                question, answer = self.qa_template(qa_data)
                if k == "Fine-grained Pose":
                    video_name = qa_data['video']
                    converted_video_name = video_name.replace(".avi", ".mp4")
                    video_path = os.path.join(v[1], converted_video_name)
                else:
                    video_path = os.path.join(v[1], qa_data['video'])

                if k in self.data_types_with_bound:
                    if is_url(video_path):
                        # Assume the bounded video has already been saved since even calling
                        # `file_exists` on each example can be slow if they are URLs
                        base_path, ext = os.path.splitext(video_path)
                        start_time, end_time = qa_data['start'], qa_data['end']
                        if k == "Episodic Reasoning":
                            fps = 3
                            start_frame = int(start_time * fps) + 1  # +1 because frames start at 1
                            end_frame = int(end_time * fps)
                            video_path = os.path.join(video_path, f"video_{start_frame:05d}_{end_frame:05d}.mp4")
                        else:
                            start_time = round(start_time, 2)
                            end_time = round(end_time, 2)
                            video_path = f"{base_path}_bounded_decimal_2_{start_time}_{end_time}{ext}"
                    else:
                        video_path = save_bounded_video(video_path, qa_data['start'], qa_data['end'], k)

                data.append({
                    'question': question,
                    'answer': answer,
                    'video': video_path,
                    "metadata": dict(
                        example_id=example_id,
                        sample_difficulty=sample_difficulty,
                        video_path=video_path,
                        task_type=k,
                        prefix=v[1],
                        data_type=v[2],
                        start_time=qa_data['start'] if 'start' in qa_data else None,
                        end_time=qa_data['end'] if 'end' in qa_data else None,
                        video_url= PUBLIC_VIDEO_BASE_URL + video_path[len(VIDEO_DATA_HOME):]
                    )
                })
        if self.sample:
            np.random.RandomState(6817).shuffle(data)
            data = data[:self.sample]
        return data

    def __len__(self):
        return len(self.data)

    def get(self, idx, rng):
        return dict(**self.data[idx], style="video_eval_multiple_choice")


class MotionBench(DatasetBase):
    data_path = os.path.join(VIDEO_DATA_HOME, "MotionBench/MotionBench")
    metadata_path = os.path.join(data_path, "video_info.meta.jsonl")
    self_collected_path = os.path.join(data_path, "self-collected")
    public_dataset_path = os.path.join(data_path, "public-dataset")

    def __init__(self, split):
        if split == "val":
            split = "validation"
        assert split in ["validation", "test"]
        self.split = split
        super().__init__(split)

    def find_video_path(self, video_filename, all_files):
        # Check if the file exists in self-collected directory
        self_collected_video_path = os.path.join(self.self_collected_path, video_filename)
        if self_collected_video_path in all_files:
            return self_collected_video_path

        # Check if the file exists in public-dataset directory
        public_dataset_video_path = os.path.join(self.public_dataset_path, video_filename)
        if public_dataset_video_path in all_files:
            return public_dataset_video_path

        # If not found in either location, return the default path
        return None

    def load(self):
        data = []
        # Build a list of all video files since it is faster than checking for file existence
        # one-by-one, especially when dealing with remote files
        all_files = set(list_directory(self.self_collected_path, recurse=True, include_files=True, include_dirs=False))
        all_files.update(list_directory(self.public_dataset_path, recurse=True, include_files=True, include_dirs=False))
        with open(resource_path(self.metadata_path)) as f:
            data_entries = f.readlines()
        for line in data_entries:
            entry = json.loads(line.strip().strip("\n"))
            video_path = self.find_video_path(entry["video_path"], all_files)
            if video_path is None:
                # Eval sets should never have missing videos
                raise ValueError(f"Missing video {video_path}")
            for qa_instance in entry["qa"]:
                answer = qa_instance["answer"]
                if self.split == "validation" and answer == "NA":  # Samples in the test set don't have answers - https://github.com/zai-org/MotionBench/issues/9
                    continue

                data.append(
                    dict(
                        question=qa_instance["question"],
                        answer=answer,
                        video=video_path,
                        metadata=dict(
                            video=video_path,
                            example_id=f"{qa_instance['uid']}_{entry['key']}",
                            qa_uid=qa_instance["uid"],
                            video_info=entry.get("video_info", {}),
                            task_type=entry.get("question_type", ""),
                            video_type=entry.get("video_type", "")
                        ),
                    ))
                # For whatever reason these videos have a chance of hanging decord, and the hanges
                # seems to occur for different videos depending on the video loading settings,
                # such as the number of frames. The videos are short anyway so just never
                # use decord for any of them.
                data[-1]["metadata"]["decode_method"] = "av_noseek"
        return data

    def __len__(self):
        return len(self.data)

    def get(self, idx, rng):
        return dict(**self.data[idx], style="video_eval_multiple_choice")


class QVHighlights(DatasetBase):
    data_path = os.path.join(VIDEO_DATA_HOME, "QVHighlights")
    def __init__(self, split, minimum=0.0):
        if split == "val":
            split = "validation"
        assert split in ["validation", "train"]
        self.max_detected_windows = 5
        self.minimum = minimum
        assert 0 <= self.minimum < 1.0
        super().__init__(split)

    @staticmethod
    def qa_template(data):
        question = f"Here is a query: {data['query'].rstrip('.')}. Where are all the segments for the query? Report the segments in seconds."

        answer_parts = []
        for start, end in data['relevant_windows']:
            answer_parts.append(f"[{start}-{end}]")

        return question, f"Segments: {', '.join(answer_parts)}", data['query'].rstrip('.')

    def load(self):
        data = []
        if self.split == "validation":
            metadata_file_path = resource_path(join(self.data_path, "highlight_val_release.jsonl"))
        else:
            metadata_file_path = resource_path(join(self.data_path, "highlight_train_release.jsonl"))

        with open(metadata_file_path, "r") as f:
            lines = f.readlines()

        skipped = 0
        for line in lines:
            data_info = json.loads(line.strip())
            if len(data_info['relevant_windows']) > self.max_detected_windows:
                # Skipping example with more than self.max_detected_windows relevant windows
                continue

            video_path = join(self.data_path, "videos", f"{data_info['vid']}.mp4")
            if not file_exists(video_path):
                skipped += 1
                continue

            example_id = f"{data_info['qid']}_{data_info['vid']}"

            question, answer, frame_sel_input = self.qa_template(data_info)

            # scale from 0 to 4 to 0 to (1 - self.minimum)
            scaled_scored_id_to_score = {}
            for index, clip_id in enumerate(data_info['relevant_clip_ids']):
                scaled_scored_id_to_score[clip_id] = (sum(data_info['saliency_scores'][index]) / 3.0) / 4.0 * (1 - self.minimum)

            # Outputs scores between self.minimum and 1.0
            scaled_avg_scores = [self.minimum for _ in range(data_info['duration'] // 2)]
            for clip_id in range(len(scaled_avg_scores)):
                if clip_id in scaled_scored_id_to_score:
                    scaled_avg_scores[clip_id] += scaled_scored_id_to_score[clip_id]

            data.append({
                'question': question,
                'answer': answer,
                'video': video_path,
                "metadata": dict(
                    example_id=example_id,
                    video_path=video_path,
                    query=data_info['query'],
                    duration=data_info['duration'],
                    relevant_windows=data_info['relevant_windows'],
                    relevant_clip_ids=data_info['relevant_clip_ids'],
                    saliency_scores=data_info['saliency_scores'],
                    scaled_avg_scores=scaled_avg_scores,
                    frame_sel_input=frame_sel_input
                )
            })

        if skipped > 0:
            if get_global_rank() == 0:
                log.warning(f"Skipped {skipped} examples with missing videos.")

        return data

    def __len__(self):
        return len(self.data)

    def get(self, idx, rng):
        return dict(**self.data[idx], style="video_short_answer")


class LLaVAVideo178K(DatasetBase):
    data_path = os.path.join(VIDEO_DATA_HOME, "LLaVA-Video-178K")
    file_path = os.path.join(VIDEO_DATA_HOME, "LLaVA-Video-178K", "data_subset_config.yaml")
    shuffled_video_names_path = os.path.join(VIDEO_DATA_HOME, "LLaVA-Video-178K", "shuffled_llava_video_names.json")

    files_not_found = set(["ytb_GqeRnxSuLFI.mp4", "ytb_y6ReUXtm_VE.mp4"])
    corrupt_files = set([
        "ytb_3ujEaKQBqqE.mp4",
        "ytb_93RkWNK3BZc.mp4",
        "ytb_FwoZBsssEXg.mp4",
        "v_iB20nDf5yJs.mp4",
        "ytb_-CTxMb7fsWE.mp4",
        "ytb_F0IdifHpXRc.mp4",
        "ytb_bRwdpNx6bdM.mp4",
        "v_ZTHsS5lQyvQ.mp4",
        "ytb_pvf5ykfo5Ko.mp4",
        "ytb_nJ11r1kVt14.mp4",
        "ytb_pWRqmt6EEqw.mp4",
        "ytb_ZIGajSaQQLM.mp4",
        "ytb_4s2QqSla2CA.mp4",
        "ytb_UKLnTkIzsxs.mp4",
        "ytb_KWmrJ_jxozc.mp4"
    ])

    def __init__(self, split, answer_type="multi_choice", flat=False, max_per_video=None,
                 id_source=None, cap_source="lv", cap_kw="merged_caption", subset="all"):
        if split == "val":
            split = "validation"    
        assert split in ["train", "validation"]
        assert answer_type in ["multi_choice", "open_ended", "caption", "all", "caption_no_prompt"]
        assert cap_source in ["lv", "human"]
        # assert cap_kw in ["merged_caption", "concat_caption", "concat_transcript"]
        if answer_type == "caption_no_prompt":
            self.answer_type = "caption"
            self.no_question = True
        else:
            self.answer_type = answer_type
            self.no_question = False
        self.flat = flat
        self.max_per_video = max_per_video
        self.id_source = id_source
        self.cap_source = cap_source
        self.cap_kw = cap_kw
        self.subset = subset
        assert subset in ["all", "academic", "gpt"]
        if self.subset == "academic":
            self.allowed_suffixes = ["nextqa", "activitynetqa", "perceptiontest"]
        elif self.subset == "gpt":
            self.allowed_suffixes = ["youtube_v0_1", "academic_v0_1"]
        super().__init__(split)

    def load(self):
        config = yaml.safe_load(read_file(join(self.data_path, "data_subset_config.yaml")))

        shuffled_video_names = json.loads(read_file(self.shuffled_video_names_path))
        if self.split == "train":
            subset_video_names = set(shuffled_video_names[:int(len(shuffled_video_names) * 0.95)])
        elif self.split == "validation":
            subset_video_names = set(shuffled_video_names[int(len(shuffled_video_names) * 0.95):])
        else:
            raise NotImplementedError(self.split)

        # Add human caption data as the captions in training
        allowed_subset_to_caption = {}
        if self.id_source:
            data_frame = pd.read_parquet(resource_path(dirname(self.id_source), basename(self.id_source)))
            for _, row in data_frame.iterrows():
                allowed_subset_to_caption[row['video_path']] = row[self.cap_kw]

        data = {}
        data_list_format = []
        self.video_paths = []
        for config_item in config.get('configs', []):
            for data_file in config_item['data_files']:
                question_type = data_file['split']
                data_path = data_file['path']
                if self.subset != "all":
                    dir = os.path.dirname(data_path)
                    dir_suffix = "_".join(dir.split("_")[3:])
                    if dir_suffix not in self.allowed_suffixes:
                        continue

                if self.answer_type != "all" and question_type != self.answer_type:
                    continue

                if question_type == "caption":
                    style = "video_long_caption"
                else:
                    style = "video_" + ("short_answer" if question_type == "open_ended" else "multiple_choice")

                config_path = os.path.join(self.data_path, data_file['path'])
                first_file_data = None
                for file in olmo_glob(config_path):
                    first_file_data = json.loads(read_file(file))
                    break

                for qa_data in first_file_data:
                    relative_video_path = os.path.join(qa_data['data_source'], qa_data['video'])
                    video_path = os.path.join(self.data_path, qa_data['data_source'], qa_data['video'])
                    if self.id_source and (relative_video_path not in allowed_subset_to_caption) and (video_path not in allowed_subset_to_caption):
                        continue

                    video_name = os.path.basename(video_path)
                    if video_name in self.files_not_found or video_name in self.corrupt_files:
                        continue
                    video_id = os.path.join(qa_data['data_source'], qa_data['video'])
                    if video_id not in subset_video_names:
                        continue

                    self.video_paths.append(video_path)
                    example_id = f"{qa_data['id']}_{qa_data['data_source']}_{qa_data['video']}_{question_type}"

                    conversations = qa_data['conversations']
                    if example_id not in data:
                        messages = []
                    else:
                        messages = data[example_id]['message_list']

                    for conv_idx in range(0, len(conversations), 2):
                        question = conversations[conv_idx]['value']
                        if tokenizer.IMAGE_PROMPT in question:
                            raise ValueError()
                        if question.startswith("<image>\n"):
                            question = question[len("<image>\n"):]
                        answer = conversations[conv_idx + 1]['value']
                        answer = answer.lstrip().strip()

                        if self.id_source and self.cap_source == "human":
                            answer = allowed_subset_to_caption[relative_video_path]

                        msg = dict(answer=answer, style=style)
                        if not self.no_question:
                            msg["question"] = question
                        messages.append(msg)

                    data[example_id] = {
                        'video': video_path,
                        'prefix': data_file['path'],
                        'message_list': messages
                    }

        for example_id, example in data.items():
            data_list_format.append({
                "video": example["video"],
                "metadata": dict(
                    example_id=example_id,
                    prefix=example["prefix"],
                    video_url=PUBLIC_VIDEO_BASE_URL + example["video"][len(VIDEO_DATA_HOME):],
                ),
                "message_list": example["message_list"],
            })

        if self.flat:
            data_list_format = flatten_lists(
                [dict(ex, message_list=[message]) for message in ex["message_list"]]
                for ex in data_list_format
            )
        elif self.max_per_video:
            flat = []
            for ex in tqdm(data_list_format):
                for msg in split_into_groups(ex["message_list"], self.max_per_video):
                    flat.append(dict(ex, message_list=msg))
            logging.info(f"Split {len(data_list_format)} in {len(flat)} examples")
            data_list_format = flat

        return data_list_format

    def __len__(self):
        return len(self.data)

    def get_shuffle_subset(self, set_video_paths):
        """
        Code to create a shuffled video names file that can be used to get train/val split
        """
        video_id_list = []
        for video_path in set_video_paths:
            video_id_list.append(video_path.split("LLaVA-Video-178K/")[1])
        print(f"Unique video names: {len(video_id_list)}")

        # save all names to a dictionary
        sorted_video_names = sorted(video_id_list)
        random.Random(42).shuffle(sorted_video_names)
        json.dump(sorted_video_names, open(self.shuffled_video_names_path, 'w'))

    def get(self, idx, rng):
        return self.data[idx]


class InternVid(DatasetBase):
    SPLITS = ["train", "validation"]
    INTERN_VID = join(VIDEO_DATA_HOME, "intern_vid")

    def __init__(self, split, n_val=20):
        assert split in self.SPLITS
        self.n_val = n_val
        super().__init__(split)

    def load(self):
        metadata_source = join(self.INTERN_VID, "metadata")
        files = [f for f in os.listdir(metadata_source) if f.startswith('internvid_10m_flt-seed42-')]
        files = sorted([f for f in files if not f.endswith("_filtered.csv")])

        if self.split == "train":
            files = files[:-self.n_val]
        else:
            files = files[-self.n_val:]

        data = []
        for file in files:
            video_folder_id = file.split("flt-")[1].split(".csv")[0]
            video_folder_path = join(self.INTERN_VID, "videos", video_folder_id)
            video_files = [f for f in os.listdir(video_folder_path) if f.endswith(".mp4") or f.endswith(".mkv")]

            df = pd.read_csv(resource_path(metadata_source, file))
            for video_file in video_files:
                row_id = video_file.split("_")[0]
                row = df.loc[int(row_id)]
                identifier = row['YoutubeID']
                caption = row['Caption']
                start_time = pd.to_timedelta(row['Start_timestamp'])
                end_time = pd.to_timedelta(row['End_timestamp'])

                video_path = join(video_folder_path, video_file)

                data.append(
                    dict(
                        video=video_path,
                        messages=dict(text=caption, style="video_short_caption"),
                        metadata=dict(
                            example_id=f"{row_id}_{identifier}",
                            start_time=start_time,
                            end_time=end_time
                        )
                    )
                )

        return data

    def get(self, item, rng):
        return self.data[item]


class Koala(DatasetBase):
    SPLITS = ["train", "validation"]
    KOALA_SRC = join(VIDEO_DATA_HOME, "koala_36m")

    def __init__(self, split, n_val=4):
        assert split in self.SPLITS
        self.n_val = n_val
        super().__init__(split)

    def load(self):
        metadata_source = join(self.KOALA_SRC, "metadata")
        files = sorted([f for f in os.listdir(metadata_source) if f.startswith('koala_36m-seed42-')])

        if self.split == "train":
            files = files[:-self.n_val]
        else:
            files = files[-self.n_val:]

        data = []
        for file in files:
            video_folder_id = file.split("koala_36m-")[1].split(".csv")[0]
            video_folder_path = join(self.KOALA_SRC, "videos", video_folder_id)
            video_files = sorted([f for f in os.listdir(video_folder_path) if f.endswith(".mp4") or f.endswith(".mkv")])

            df = pd.read_csv(resource_path(metadata_source, file))
            for video_file in video_files:
                video_id = video_file.split(".")[0]
                row = df.loc[video_id]
                caption = row["caption"]
                st, et = ast.literal_eval(row['timestamp'])
                start_time = pd.to_timedelta(st)
                end_time = pd.to_timedelta(et)

                video_path = join(video_folder_path, video_file)

                data.append(
                    dict(
                        video=video_path,
                        messages=dict(text=caption, style="video_long_caption"),
                        metadata=dict(
                            example_id=video_id,
                            start_time=start_time,
                            end_time=end_time
                        )
                    )
                )

        return data

    def get(self, item, rng):
        return self.data[item]


def load_all_frames_decord_or_pyav(video_path: str) -> np.ndarray:
    try:
        vr = VideoReader(video_path, num_threads=1, ctx=cpu(0))
        total_frames = len(vr)  # Total frame count
        frame_indices = np.arange(0, total_frames)
        return vr.get_batch(frame_indices).asnumpy()

    except Exception as e:
        frames = []
        for frame in iio.imiter(video_path, plugin="pyav"):
            frames.append(frame)
        return np.stack(frames)


class PeVideo(DatasetBase):
    data_path = os.path.join(VIDEO_DATA_HOME, "PE-Video")

    @classmethod
    def build_index(cls):
        for split in ["train", "test"]:
            split_path = join(cls.data_path, split)
            index_path = join(cls.data_path, f"{split_path}_index.json")
            if file_exists(index_path):
                continue
            indices = {}
            logging.info(f"Building index for {split}")
            video_ids = Counter()
            for file in tqdm(os.listdir(split_path)):
                index = []
                example_to_files = {}
                with tarfile.open(join(split_path, file), 'r') as db:
                    it = iter(db)
                    while True:
                        try:
                            json_tarinfo = next(it)
                        except StopIteration:
                            break
                        mp4_tarinfo = next(it)
                        json_data = json.load(db.extractfile(json_tarinfo))
                        video_id = json_data["video_id"]
                        video_ids[video_id] += 1
                        example_id = int(json_tarinfo.name.split('.')[0])
                        assert example_id == int(mp4_tarinfo.name.split('.')[0])
                        assert video_id == example_id
                        index.append((
                            example_id,
                            json_tarinfo.offset_data, json_tarinfo.size,
                            mp4_tarinfo.offset_data, mp4_tarinfo.size,
                        ))
                        db.members = []
                indices[file] = index
            with open(index_path, "w") as f:
                json.dump(indices, f)

    def load(self):
        if self.split == "test":
            data = json.loads(read_file(join(self.data_path, "test_index.json")))
            data = flatten_lists(((file,) + val for val in vals) for file, vals in data.items())
        else:
            data = json.loads(read_file(join(self.data_path, "train_index.json")))
            data = flatten_lists(([file] + val for val in vals) for file, vals in data.items())
            np.random.RandomState(1321).shuffle(data)
            if self.split == "validation":
                data = data[:4096]
            else:
                data = data[4096:]
        return data

    def get(self, item, rng):
        file, example_id, json_off, json_sz, mp_off, mp_size = self.data[item]
        if self.split == "test":
            file = join(self.data_path, "test", file)
        else:
            file = join(self.data_path, "train", file)
        data = json.loads(get_bytes_range(file, json_off, json_sz).decode("utf-8"))
        video = get_bytes_range(file, mp_off, mp_size)
        return dict(
            text=data["model_caption"],
            video=video,
            style="long_caption",
            metadata=dict(example_id=data["video_id"]),
        )


class VideoEvalProMC(DatasetBase):
    data_path = os.path.join(VIDEO_DATA_HOME, "VideoEvalPro", "videos_filtered")
    resized_data_path = os.path.join(VIDEO_DATA_HOME, "VideoEvalPro", "videos_filtered_resized")
    parquet_path = os.path.join(VIDEO_DATA_HOME, "VideoEvalPro", "test-00000-of-00001.parquet")
    resized_video_list = os.path.join(VIDEO_DATA_HOME, "VideoEvalPro", "resized_videos_list.json")

    def __init__(self, split, difficulty="all", use_resize=False):
        assert split in ["test"]
        assert difficulty in ["easy", "medium", "hard", "all"]
        self.difficulty = difficulty
        self.use_resize = use_resize
        super().__init__(split)

    def load(self):
        difficulty_lookup = DatasetSampleDifficulty("video_eval_pro_three_frame_gemini_2_0_filter_results.json")
        with open(resource_path(self.resized_video_list), "r") as f:
            resized_videos = set(json.load(f))

        data = []
        data_table = pd.read_parquet(resource_path(dirname(self.parquet_path), basename(self.parquet_path)))
        for row_index, row in data_table.iterrows():
            # Check if resized version exists and use_resize is True
            resized_video_path = os.path.join(self.resized_data_path, row['video'])
            if self.use_resize and row['video'] in resized_videos:
                video_path = resized_video_path
            else:
                video_path = os.path.join(self.data_path, row['video'])
            question = row['question']
            options = list(row['options'])
            answer = row["answer"]

            # metadata = json.loads(row['meta'])
            metadata = {}
            metadata['answer_text'] = row["answer_text"]
            metadata['source_set'] = row['source']
            metadata["original_qa_subtype"] = row['qa_subtype']
            metadata["original_qa_type"] = row['qa_type']
            example_id = f"{row_index}_{row['video']}"
            metadata['example_id'] = example_id
            metadata["display_in_eval"] = True
            metadata['video_url'] = PUBLIC_VIDEO_BASE_URL + video_path[len(VIDEO_DATA_HOME):]


            sample_difficulty = difficulty_lookup.get_difficulty(example_id)
            if self.difficulty != "all" and sample_difficulty != self.difficulty:
                continue
            metadata["sample_difficulty"] = sample_difficulty

            # For MC mode, we use the options
            formatted_question = "\n".join([question, "\n".join(options), "Please respond with only the letter of the correct answer."])

            data.append({
                'question': formatted_question,
                'answer': answer,
                'video': video_path,
                "metadata": metadata
            })
        return data

    def __len__(self):
        return len(self.data)

    def get(self, idx, rng):
        return dict(**self.data[idx], style="video_eval_multiple_choice")


class Vinoground(DatasetBase):
    def __init__(self):
        self.HOME = join(VIDEO_DATA_HOME, "Vinoground")
        super().__init__("test")

    def load(self):
        data = []

        for qtype in ["textscore", "videoscore"]:
            questions = json.loads(read_file(join(self.HOME, f"vinoground_{qtype}.json"), "r"))
            for q in questions:
                data.append(dict(
                    video=join(self.HOME, q["video_name"]),
                    prompt=q["question"],
                    metadata=dict(
                        qtype=qtype,
                        answer=q["GT"],
                        id=q["idx"],
                    ),
                    style="video_eval_multiple_choice"
                ))

        return data

    def get(self, item, rng):
        return self.data[item]


class TemporalBenchQa(DatasetBase):
    data_path = os.path.join(VIDEO_DATA_HOME, "TemporalBench")

    def __init__(self, split, format="original"):
        assert split in ["test"]
        self.format = format
        super().__init__(split)

    def load(self):
        examples = []
        for src in ["long_qa", "short_qa"]:
            file = resource_path(join(self.data_path, f"temporalbench_{src}.json"))
            with open(file, "r") as f:
                data = json.load(f)
            for example in data:
                video = join(self.data_path, example["video_name"])
                metadata = dict(example_id=example["idx"], type=src, video=example["video_name"])
                if self.format == "original":
                    examples.append(dict(
                        video=video,
                        question=example['question'],
                        style="video_short_answer",
                        answer=example["GT"],
                        metadata=metadata
                    ))
                elif self.format == "mc":
                    parts = [x.strip() for x in example['question'].split("\n") if x.strip()]
                    question = parts[0]
                    instructions = parts[-1]
                    assert instructions.startswith("Answer with the option'")
                    options = []
                    answer_idx = None
                    for option_ix, option_part in enumerate(parts[1:-1]):
                        group = re.match("([A-Z]).(.*)", option_part)
                        options.append(group.group(2).strip())
                        if group.group(1) == example["GT"]:
                            assert answer_idx is None, "Multiple option matched the ground truth"
                            answer_idx = option_ix
                    assert answer_idx is not None, "No option matched the ground truth"
                    assert len(options) > 1, "<=1 options"
                    examples.append(dict(
                        video=video,
                        question=question,
                        options=options,
                        answer_idx=answer_idx,
                        style="video_multiple_choice",
                        metadata=metadata
                    ))
                else:
                    raise NotImplementedError(self.format)
        return examples

    def __len__(self):
        return len(self.data)

    def get(self, idx, rng):
        return self.data[idx]


class Tomato(DatasetBase):
    data_path = os.path.join(VIDEO_DATA_HOME, "TOMATO")
    reasoning_type_choices = [
        "count",
        "direction",
        "rotation",
        "shape&trend",
        "velocity&frequency",
        "visual_cues"
    ]
    demonstration_type_choices = [
        "human",
        "object",
        "simulated"
    ]

    @staticmethod
    def validate_choices(input_value, all_choices, input_name):
        if input_value == 'ALL':
            return all_choices
        else:
            selected_values = [item.strip() for item in input_value.split(",")]
            invalid_values = [item for item in selected_values if item not in all_choices]
            if invalid_values:
                raise ValueError(f"Invalid {input_name} type(s): {', '.join(invalid_values)}. "
                                 f"Valid choices are: {', '.join(all_choices + ['ALL'])}")
            return selected_values

    def __init__(self, split, reasoning_type="ALL", demonstration_type="ALL"):
        assert split in ["test"]
        self.reasoning_type = reasoning_type
        self.demonstration_type = demonstration_type
        super().__init__(split)

    def load(self):
        queries = defaultdict(list)
        existing_paths = list()
        reasoning_type = self.validate_choices(self.reasoning_type, self.reasoning_type_choices, "reasoning")
        demonstration_type = self.validate_choices(self.demonstration_type, self.demonstration_type_choices, "demonstration")
        queries = []
        for rt in reasoning_type:
            dataset_path = resource_path(join(self.data_path, f"data/{rt}.json"))
            with open(dataset_path, "r") as f:
                qas = json.load(f)
            for id_, qa in qas.items():
                if qa['demonstration_type'] in demonstration_type:
                    if (qa["demonstration_type"], qa["key"]) == ("object", "0390-01"):
                        # This is a super special snowflake video the causes decord to hang
                        decode_method = "av_noseek"
                    else:
                        decode_method = None
                    queries.append(dict(
                        video=join(self.data_path, "videos", qa["demonstration_type"], qa["key"] + ".mp4"),
                        question=qa['question'],
                        options=qa["options"],
                        answer_idx=qa["answer"],
                        metadata=dict(
                            decode_method=decode_method,
                            example_id=id_,
                            key=qa["key"],
                            reasoning_type=rt,
                            demonstration_type=qa["demonstration_type"],
                            motion_type=qa["motion_type"],
                        ),
                        style="video_multiple_choice",
                    ))
        return queries

    def __len__(self):
        return len(self.data)

    def get(self, idx, rng):
        return self.data[idx]


class CLEVRER(DatasetBase):
    """CLEVRER Video QA dataset"""
    data_path = join(VIDEO_DATA_HOME, "CLEVRER")

    def __init__(self, 
        split, 
        answer_type: Literal["open_ended", "multi_choice", "all"] = "all",
        include_multiple_correct: bool = False,
        flat: bool = False,
        max_per_video: Optional[int] = None,
    ):
        assert split in ["train", "validation", "test"], f"Invalid split: {split}"
        assert answer_type in ["open_ended", "multi_choice", "all"], f"Invalid answer type: {answer_type}"
        self.answer_type = answer_type
        # There are ~20K / 150K questions with multiple correct answers in CLEVRER
        self.include_multiple_correct = include_multiple_correct
        self.flat = flat
        self.max_per_video = max_per_video
        super().__init__(split)

    @staticmethod
    def format_options_and_answer(choices):
        choice_texts = []
        correct_choices = []
        answer_idx = None
        for i, choice in enumerate(choices):
            choice_texts.append(choice['choice'])
            if choice['answer'] == 'correct':
                answer_idx = i
                correct_choices.append(f"{chr(ord('A') + choice['choice_id'])}")
        multiple_correct = len(correct_choices) > 1
        answer = "\n".join(correct_choices)
        return choice_texts, answer_idx, answer, multiple_correct

    def load(self):
        # load data from json path
        json_path = join(self.data_path, f"{self.split}.json")
        with open(resource_path(dirname(json_path), basename(json_path))) as f:
            data = json.load(f)

        data_list = []
        num_no_gt_ans = 0
        total_num = 0
        for ex in data:
            questions = ex['questions']
            video = ex['video_filename']
            video_id = int(video.split(".")[0].replace("video_", ""))
            k = video_id // 1000
            video_folder = f"video_{k * 1000:05d}-{(k + 1) * 1000:05d}"
            video_path = join(self.data_path, video_folder, video)
            msgs = []
            for q in questions:
                if self.answer_type == "open_ended" and "answer" not in q:
                    continue
                if self.answer_type == "multi_choice" and "choices" not in q:
                    continue

                if "answer" in q:
                    msg = dict(
                        question=q['question'],
                        answer=q['answer'],
                        style="video_short_answer",
                    )
                    msgs.append(msg)
                elif "choices" in q:
                    options, answer_idx, answer, multiple_correct = self.format_options_and_answer(q['choices'])
                    if multiple_correct:
                        if self.include_multiple_correct:
                            msg = dict(
                                    question=q['question'],
                                    options=options,
                                    answer=answer,
                                    style="video_multiple_choice_multiple_correct",
                                )
                            msgs.append(msg)

                    else:
                        if answer_idx is not None:
                            msg = dict(
                                question=q['question'],
                                options=options,
                                answer_idx=answer_idx,
                                style="video_multiple_choice",
                            )
                            msgs.append(msg)
                        else:
                            num_no_gt_ans += 1
                        total_num += 1
                else:
                    raise ValueError("Question must have either 'answer' or 'choices' field.")

                if self.flat:
                    formatted_ex = {
                        "video": video_path,
                        "metadata": {
                            "question_id": q['question_id'],
                            "question_type": q['question_type'],
                            "program": q['program']
                        },
                        "message_list": [msg],
                        }
                    data_list.append(formatted_ex)
            if not self.flat:
                if len(msgs) == 0: continue
                if self.max_per_video:
                    for msg in split_into_groups(msgs, self.max_per_video):
                        formatted_ex = {
                            "video": video_path,
                            "message_list": msg,
                            }
                        data_list.append(formatted_ex)
                else:
                    formatted_ex = {
                        "video": video_path,
                        "message_list": msgs,
                        }
                    data_list.append(formatted_ex)
        if get_global_rank() == 0:
            log.warning(f"Skipped {num_no_gt_ans} / {total_num} examples that don't have GT answer.")
        return data_list

    def get(self, item, rng):
        return self.data[item]


class STAR(DatasetBase):
    """STAR Video QA dataset"""
    data_path = join(VIDEO_DATA_HOME, "STAR")
    video_path = join(VIDEO_DATA_HOME, "Charades/Charades_v1")

    @classmethod
    def download(cls, num_procs=None):
        # STAR jsons are huge and can take a while to load, so we save a slimmed-down version
        # the only contains the fields we need here for faster startup time
        for split in ["train", "val", "test"]:
            json_path = join(cls.data_path, f"STAR_{split}.json")
            compressed_path = join(cls.data_path, f"STAR_{split}_qa.json")
            if file_exists(compressed_path):
                continue
            with open(resource_path(dirname(json_path), basename(json_path)), 'r') as f:
                data = json.load(f)
            data = [
                {k: x[k] for k in
                 ["video_id", "start", "end", "question", "answer", "choices"] if k in x}
                for x in data
            ]
            write_file(cls.data_path, f"STAR_{split}_compressed.json", json.dumps(data), True)

    def __init__(
        self, 
        split, 
        answer_type: Literal["open_ended", "multi_choice", "all"] = "all", 
        flat: bool = False,
        max_per_video: Optional[int] = None
    ):
        assert split in ["train", "validation", "test"], f"Invalid split: {split}"
        assert answer_type in ["open_ended", "multi_choice", "all"], f"Invalid answer type: {answer_type}"
        self.answer_type = answer_type
        self.flat = flat
        self.max_per_video = max_per_video
        super().__init__(split)

    @staticmethod
    def format_options_and_answer(answer, choices):
        """ Format the options and answer for multiple choice questions."""
        choice_texts = []
        answer_idx = None
        for i, choice in enumerate(choices):
            choice_texts.append(choice['choice'])
            if choice['choice'] == answer:
                answer_idx = i
        return choice_texts, answer_idx

    def load(self):
        json_path = join(self.data_path, f"STAR_{self.split}.json")
        compressed_path = join(self.data_path, f"STAR_{self.split}_compressed.json")
        if file_exists(compressed_path):
            with open(resource_path(compressed_path), 'r') as f:
                data = json.load(f)
        else:
            with open(resource_path(json_path), 'r') as f:
                data = json.load(f)

        data_list = []
        video2msgs = {}
        video2meta = {}
        num_no_gt_ans = 0
        total_num = 0
        for ex in data:
            video_id = ex['video_id']
            start = float(ex['start'])
            end = float(ex['end'])
            abs_video_path = join(self.video_path, f"{video_id}.mp4")

            clip_id = f"{video_id}_{start}_{end}"
            video2msgs[clip_id] = video2msgs.get(clip_id, [])
            if clip_id not in video2meta:
                video2meta[clip_id] = [start, end, abs_video_path]

            msgs = []
            if self.answer_type == "open_ended" or self.answer_type == "all":
                msg = dict(
                    question=ex['question'],
                    answer=ex['answer'],
                    style="video_short_answer"
                )
                msgs.append(msg)

            if self.answer_type == "multi_choice" or self.answer_type == "all":
                choices = ex['choices']
                options, answer_idx = self.format_options_and_answer(ex['answer'], choices)
                if answer_idx is not None:
                    msg = dict(
                        question=ex['question'],
                        options=options,
                        answer_idx=answer_idx,
                        style="video_multiple_choice"
                    )
                    msgs.append(msg)
                else:
                    num_no_gt_ans += 1
                total_num += 1

            video2msgs[clip_id] += msgs

            if self.flat:
                for msg in msgs:
                    formatted_ex = {
                        "video": abs_video_path,
                        "metadata": {
                            "clip_start_time": start,
                            "clip_end_time": end,
                        },
                        "message_list": [msg]
                    }
                    data_list.append(formatted_ex)

        if get_global_rank() == 0:
            log.warning(f"Skipped {num_no_gt_ans} / {total_num} examples without GT answer.")

        if not self.flat:
            for clip_id, msgs in video2msgs.items():
                if len(msgs) == 0: continue
                start, end, video = video2meta[clip_id]
                if self.max_per_video:
                    for msg in split_into_groups(msgs, self.max_per_video):
                        formatted_ex = {
                            "video": video,
                            "message_list": msg,
                            "metadata": {
                                "clip_start_time": float(start),
                                "clip_end_time": float(end),
                            }
                        }
                        data_list.append(formatted_ex)
                else:
                    formatted_ex = {
                        "video": video,
                        "message_list": msgs,
                        "metadata": {
                            "clip_start_time": float(start),
                            "clip_end_time": float(end),
                        }
                    }
                    data_list.append(formatted_ex)
        return data_list

    def get(self, item, rng):
        return self.data[item]


class FunQA(DatasetBase):
    """FunQA Video QA dataset"""
    data_path = join(VIDEO_DATA_HOME, "FunQA")

    def __init__(
        self, 
        split, 
        answer_type: Literal["open_ended", "multi_choice", "all"] = "all", 
        flat: bool = False,
        max_per_video: Optional[int] = None
    ):
        assert split in ["train", "val", "test"], f"Invalid split: {split}"
        self.answer_type = answer_type
        self.flat = flat
        self.max_per_video = max_per_video
        super().__init__(split)

    @staticmethod
    def format_options_and_answer(answer_idx, options):
        modified_options = []
        answer = None
        for i, opt in enumerate(options):
            choice_letter = chr(ord('A') + i)
            if i == answer_idx - 1:  
                answer = choice_letter
            opt_content = opt.replace(f"Options {i+1}: ", "").strip()
            modified_options.append(opt_content)
        return modified_options, answer

    def load(self):
        # H1, C1, M1
        video_path = join(VIDEO_DATA_HOME, f"FunQA/{self.split}")
        oe_json_path = resource_path(join(self.data_path, f"FunQA_{self.split}.json"))
        mc_json_path = resource_path(join(self.data_path, f"Funqa_mcqa_v1.json"))
        if self.answer_type == "open_ended":
            oe_data = json.load(open(resource_path(dirname(oe_json_path), basename(oe_json_path)), 'r'))
            mc_data = []
        elif self.answer_type == "multi_choice":
            mc_data = json.load(open(resource_path(dirname(mc_json_path), basename(mc_json_path)), 'r'))
            oe_data = []
        elif self.answer_type == "all":
            oe_data = json.load(open(resource_path(dirname(oe_json_path), basename(oe_json_path)), 'r'))
            mc_data = json.load(open(resource_path(dirname(mc_json_path), basename(mc_json_path)), 'r'))
        else:
            raise ValueError(f"Invalid answer type: {self.answer_type}")

        # Faster on weka and GCP to fetch all the filepaths and then check against that
        # instead of using `file_exists` repeatedly
        all_files = set(list_directory(self.data_path, recurse=True, include_dirs=False))
        data_list = []
        video2msgs = {}
        for ex in oe_data:
            if ex['task'] in ['H1', 'C1', 'M1', 'C4', 'C5']:
                continue  # Skip these tasks with timestamps as outputs
            if ex['task'].startswith("H"):
                video_dir = f"{self.split}_humor"
            elif ex['task'].startswith("C"):
                video_dir = f"{self.split}_creative"
            elif ex['task'].startswith("M"):
                video_dir = f"{self.split}_magic"
            else:
                raise ValueError(f"Unknown task {ex['task']}")

            key = join(video_dir, ex['visual_input'])
            video = join(video_path, video_dir, ex['visual_input'])
            if video not in all_files:
                continue

            question = ex['instruction']
            answer = ex['output']
            msg = dict(
                question=question,
                answer=answer,
                style="video_short_answer"
            )
            if self.flat:
                formatted_ex = {
                    "video": video,
                    "metadata": {
                        "task": ex['task'],
                    },
                    "message_list": [msg]
                }
                data_list.append(formatted_ex)
            video2msgs[video] = video2msgs.get(video, [])
            video2msgs[video].append(msg)

        gt_oob_cnt = 0
        for ex in mc_data:
            if ex['visual_input'].startswith("H"):
                video_dir = f"{self.split}_humor"
            elif ex['visual_input'].startswith("C"):
                video_dir = f"{self.split}_creative"
            elif ex['visual_input'].startswith("M"):
                video_dir = f"{self.split}_magic"

            key = ex['visual_input']
            video = join(video_path, video_dir, ex['visual_input'])
            if video not in all_files:
                continue
            sentences = ex['instruction'].split("\n")
            question = sentences[-2].replace("The Question is:", "").strip()
            options = sentences[-1]

            # instruction: ...\n The Options are: ['Options 1: Black', 'Options 2: White'...]
            options_list = ast.literal_eval(options.replace(" The Options are: ", ""))
            try:
                # gt is 1-indexed
                gt_idx = int(ex['gt'])
                if gt_idx > len(options_list):
                    gt_oob_cnt += 1
                    raise IndexError(f"{gt_idx} out of bounds for options {options_list}")
            except Exception as e:
                continue

            options, answer = self.format_options_and_answer(gt_idx, options_list)
            msg = dict(
                question=question,
                options=options,
                answer_idx=gt_idx-1,
                style="video_multiple_choice"
            )
            if self.flat:
                formatted_ex = {
                    "video": video,
                    "metadata": {
                        "id": ex['id'],
                    },
                    "message_list": [msg]
                }
                data_list.append(formatted_ex)
            video2msgs[video] = video2msgs.get(video, [])
            video2msgs[video].append(msg)
        if get_global_rank() == 0:
            log.warning(f"Number of questions with out-of-bound GT index: {gt_oob_cnt}")

        if not self.flat:
            for video, msgs in video2msgs.items():
                if len(msgs) == 0: continue
                if self.max_per_video:
                    for msg in split_into_groups(msgs, self.max_per_video):
                        formatted_ex = {
                            "video": video,
                            "message_list": msg,
                        }
                        data_list.append(formatted_ex)
                else:
                    formatted_ex = {
                        "video": video,
                        "message_list": msgs,
                    }
                    data_list.append(formatted_ex)
        return data_list

    def get(self, item, rng):
        return self.data[item]


class TGIF(DatasetBase):
    """TGIF Video QA dataset"""
    data_path = join(VIDEO_DATA_HOME, "TGIF")
    video_path = join(VIDEO_DATA_HOME, "TGIF/videos")

    def __init__(
        self, 
        split, 
        answer_type: Literal["open_ended", "multi_choice", "all"] = "all", 
        subset: Literal["all", "action", "count", "transition", "frameqa"] = "all",
        flat: bool = False
    ):
        assert split in ["train", "test"], f"Invalid split: {split}"
        self.answer_type = answer_type
        self.subset = subset
        self.flat = flat
        super().__init__(split)

    @staticmethod
    def format_options_and_answer(answer_idx, choices):
        choice_texts = []
        correct_choice = None
        for i, choice in enumerate(choices):
            choice_texts.append(choice)
            if i == answer_idx:
                correct_choice = f"{chr(ord('A') + i)}"
        return choice_texts, correct_choice

    def load(self):
        subset2data = {}
        if self.subset != "all":
            csv_path = join(self.data_path, f"{self.split.capitalize()}_{self.subset}_question.csv")
            data = pd.read_csv(resource_path(dirname(csv_path), basename(csv_path)), sep="\t")
            subset2data[self.subset] = data
        else:
            all_dfs = []
            for subset in ["action", "count", "transition", "frameqa"]:
                csv_path = join(self.data_path, f"{self.split.capitalize()}_{subset}_question.csv")
                df = pd.read_csv(resource_path(dirname(csv_path), basename(csv_path)), sep="\t")
                subset2data[subset] = df
                all_dfs.append(df)

        data_list = []
        video2msgs = {}
        for subset, df in subset2data.items():
            for row in df.itertuples(False):
                msgs = []
                question = row.question
                video = f"{row.gif_name}.mp4"
                abs_video_path = join(self.video_path, video)

                # these subsets have choices 
                if subset in ["action", "transition"]:
                    options = [getattr(row, f'a{i}') for i in range(1, 6)]
                    answer_idx = row.answer
                    if answer_idx >= len(options):
                        raise IndexError(f"Answer index {answer_idx} out of bounds for options {options}")
                    answer = options[answer_idx]  # TGIF uses 0-indexed answers

                    options, mc_answer = self.format_options_and_answer(answer_idx, options)
                    if self.answer_type == "multi_choice" or self.answer_type == "all":
                        mc_msg = dict(
                            question=question,
                            options=options,
                            answer_idx=answer_idx,
                            style="video_multiple_choice",
                        )
                        msgs.append(mc_msg)

                        if self.flat:
                            formatted_ex = {
                                "video": abs_video_path,
                                "metadata": {
                                    "vid_id": row.vid_id,
                                    "subset": subset,
                                },
                                "message_list": [mc_msg]
                            }
                            data_list.append(formatted_ex)
                    if self.answer_type == "open_ended" or self.answer_type == "all":
                        oe_msg = dict(
                            question=question,
                            answer=str(answer),
                            style="video_short_answer",
                        )

                        msgs.append(oe_msg)

                        if self.flat:
                            formatted_ex = {
                                "video": abs_video_path,
                                "metadata": {
                                    "vid_id": row.vid_id,
                                    "subset": subset,
                                },
                                "message_list": [oe_msg]
                            }
                            data_list.append(formatted_ex)
                # these subsets don't have choices
                elif subset in ["count", "frameqa"]:
                    if self.answer_type == "open_ended" or self.answer_type == "all":
                        answer = row.answer
                        msg = dict(
                            question=question,
                            answer=str(answer),
                            style="video_short_answer",
                        )
                        msgs.append(msg)

                        if self.flat:
                            formatted_ex = {
                                "video": abs_video_path,
                                "metadata": {
                                    "vid_id": row.vid_id,
                                    "subset": subset,
                                },
                                "message_list": [msg]
                            }
                            data_list.append(formatted_ex)
                else:
                    raise ValueError(f"Unknown subset: {subset}")

                video2msgs[abs_video_path] = video2msgs.get(abs_video_path, [])
                video2msgs[abs_video_path].extend(msgs)


        if not self.flat:
            for video, msgs in video2msgs.items():
                if len(msgs) == 0: continue
                formatted_ex = {
                    "video": video,
                    "message_list": msgs,
                }
                data_list.append(formatted_ex)
        return data_list

    def get(self, item, rng):
        return self.data[item]


class IntentQA(DatasetBase):
    """IntentQA Video QA dataset"""
    video_path = join(VIDEO_DATA_HOME, "IntentQA", "videos")

    def __init__(self, split, answer_type: Literal["open_ended", "multi_choice", "all"] = "all", flat: bool = False):
        assert split in ["train", "val", "test"], f"Invalid split: {split}"
        assert answer_type in ["open_ended", "multi_choice", "all"], f"Invalid answer type: {answer_type}"
        self.answer_type = answer_type
        self.flat = flat
        super().__init__(split)

    @staticmethod
    def format_options_and_answer(answer_idx, choices):
        choice_texts = []
        correct_choice = None
        for i, choice in enumerate(choices):
            choice_texts.append(choice)
            if i == answer_idx:
                correct_choice = f"{chr(ord('A') + i)}"
        return choice_texts, correct_choice

    def load(self):
        csv_path = resource_path(join(VIDEO_DATA_HOME, "IntentQA", f"{self.split}.csv"))
        df = pd.read_csv(csv_path)
        data_list = []
        video2msgs = {}
        for row in df.itertuples(False):
            abs_video_path = join(self.video_path, f"{row.video_id}.mp4")
            video2msgs[abs_video_path] = video2msgs.get(abs_video_path, [])
            question = row.question

            options = [getattr(row, f'a{i}') for i in range(5)]
            answer_idx = row.answer
            # answer_idx is 0-indexed
            if answer_idx >= len(options):
                raise IndexError(f"Answer index {answer_idx} out of bounds for options {options}")
            answer = options[answer_idx]  # IntentQA uses 0-indexed answers
            if self.answer_type == "open_ended" or self.answer_type == "all":
                msg = dict(
                    question=question,
                    answer=answer,
                    style="video_short_answer"
                )
                video2msgs[abs_video_path].append(msg)
                if self.flat:
                    formatted_ex = {
                        "video": abs_video_path,
                        "metadata": {
                            "video_id": row.video_id,
                            "type": row.type,
                        },
                        "message_list": [msg]
                    }
                    data_list.append(formatted_ex)

            if self.answer_type == "multi_choice" or self.answer_type == "all":
                options, mc_answer = self.format_options_and_answer(answer_idx, options)
                msg = dict(
                    question=question,
                    options=options,
                    answer_idx=answer_idx,
                    style="video_multiple_choice"
                )
                video2msgs[abs_video_path].append(msg)

                if self.flat:
                    formatted_ex = {
                        "video": abs_video_path,
                        "metadata": {
                            "video_id": row.video_id,
                            "type": row.type,
                        },
                        "message_list": [msg]
                    }
                    data_list.append(formatted_ex)

        if not self.flat:
            for video, msgs in video2msgs.items():
                if len(msgs) == 0: continue
                formatted_ex = {
                    "video": video,
                    "message_list": msgs,
                }
                data_list.append(formatted_ex)

        return data_list

    def get(self, item, rng):
        return self.data[item]


class Paxion(DatasetBase):
    """Paxion dataset"""
    ssv2_video_path = join(VIDEO_DATA_HOME, "sth-sth-v2", "videos")
    ego4d_video_path = join(VIDEO_DATA_HOME, "Ego4d", "ego4d_data", "v2", "full_scale")
    ego4d_clips_path = join(VIDEO_DATA_HOME, "paxion-ego4d-clips")
    corrupt_files = ["703d550a-0a84-4bcf-9b45-e25c864ade70"]
    corrupt_clips = ["1348c9f9-fc8b-40c7-b1ab-5b6281e5d390_978.617_979.098.mp4"]

    def __init__(
        self,
        split,
        flat: bool = False,
        max_per_video: Optional[int] = None,
        use_extracted_clips: bool = True
    ):
        assert split in ["train", "val", "test"], f"Invalid split: {split}"
        self.flat = flat
        self.question_templates = [  # MVBench question templates
            "What activity does the video depict?",
            "What is the action performed by the person in the video?",
            "Which one of these descriptions correctly matches the actions in the video?"
        ]
        self.max_per_video = max_per_video
        self.use_extracted_clips = use_extracted_clips
        super().__init__(split)
    
    def load(self):
        ssv2_json_path = join(VIDEO_DATA_HOME, "paxion", "ssv2", "antonyms", f"{self.split}_with_rel_path.json")
        ego4d_json_path = join(VIDEO_DATA_HOME, "paxion", "ego4d", "egoclip_subset_action_antonyms_train_val_test_split", f"{self.split}.jsonl")

        ssv2_df = pd.read_json(resource_path(ssv2_json_path))
        ego4d_df = pd.read_json(resource_path(ego4d_json_path), lines=True)

        data_list = []
        video2msgs = defaultdict(list)
        rng = random.Random(42)

        for row in ssv2_df.itertuples(False):
            abs_video_path = join(self.ssv2_video_path, row.rel_vid_path)
            question = rng.choice(self.question_templates)

            options = [row.label, row.label_action_antonym_clip_text, 'Not sure']
            # shuffle options
            options = rng.sample(options, len(options))
            answer_idx = options.index(row.label)

            msg = dict(
                question=question,
                options=options,
                answer_idx=answer_idx,
                style="video_multiple_choice"
            )

            video2msgs[(abs_video_path, None, None)].append(msg)

            if self.flat:
                formatted_ex = {
                    "video": abs_video_path,
                    "message_list": [msg]
                }
                data_list.append(formatted_ex)

        for row in ego4d_df.itertuples(False):
            if row.clip_start >= row.clip_end:
                continue
            if row.video_uid in self.corrupt_files:
                continue

            # Determine video path based on whether to use extracted clips
            if self.use_extracted_clips:
                start_time = float(row.clip_start)
                end_time = float(row.clip_end)
                clip_filename = f"{row.video_uid}_{start_time:.3f}_{end_time:.3f}.mp4"
                abs_video_path = join(self.ego4d_clips_path, clip_filename)
                if clip_filename in self.corrupt_clips:
                    continue
                video_key = (abs_video_path, None, None)
                clip_metadata = None  # No need for metadata when using extracted clips
            else:
                # Use original full video with clip timing
                abs_video_path = join(self.ego4d_video_path, f"{row.video_uid}.mp4")
                clip_metadata = {
                    "clip_start_time": row.clip_start,
                    "clip_end_time": row.clip_end,
                }
                video_key = (abs_video_path, row.clip_start, row.clip_end)

            question = rng.choice(self.question_templates)

            label = row.clip_text
            antonym = row.action_antonym_clip_text

            if isinstance(label, str) and label.startswith("#") and len(label) > 1 and not label[1].isspace():
                label = label[2:].lstrip()
            if isinstance(antonym, str) and antonym.startswith("#") and len(antonym) > 1 and not antonym[1].isspace():
                antonym = antonym[2:].lstrip()
 
            # Capitalize the first letter of label if not already
            if isinstance(label, str) and label and not label[0].isupper():
                label = label[0].upper() + label[1:] 
            if isinstance(antonym, str) and antonym and not antonym[0].isupper():
                antonym = antonym[0].upper() + antonym[1:]

            options = [label, antonym, 'Not sure']
            # shuffle options
            options = rng.sample(options, len(options))
            answer_idx = options.index(label)
            msg = dict(
                question=question,
                options=options,
                answer_idx=answer_idx,
                style="video_multiple_choice"
            )

            video2msgs[video_key].append(msg)

            if self.flat:
                formatted_ex = {
                    "video": abs_video_path,
                    "message_list": [msg]
                }
                # Only add metadata if using original video (not extracted clips)
                if clip_metadata is not None:
                    formatted_ex["metadata"] = clip_metadata
                data_list.append(formatted_ex) 
        
        if not self.flat:
            for video_start_end, msgs in video2msgs.items():
                video, start, end = video_start_end
                meta = None

                # Only include metadata if using original video (not extracted clips)
                # Check if this is an extracted clip by looking at the path
                if not video.startswith(self.ego4d_clips_path) and start is not None and end is not None:
                    meta = {
                        "clip_start_time": start,
                        "clip_end_time": end,
                    }

                if len(msgs) == 0:
                    continue
                if self.max_per_video:
                    for msg in split_into_groups(msgs, self.max_per_video):
                        formatted_ex = {
                            "video": video,
                            "message_list": msg,
                        }
                        formatted_ex.update({"metadata": meta} if meta is not None else {})
                        data_list.append(formatted_ex)
                else:
                    formatted_ex = {
                        "video": video,
                        "message_list": msgs,
                    }
                    formatted_ex.update({"metadata": meta} if meta is not None else {})
                    data_list.append(formatted_ex)

        return data_list

    def get(self, item, rng):
        return self.data[item]


class Ego4d(DatasetBase):
    """Ego4d NLQ/MQ subsets
    **Please use Ego4dCachedClips for faster training as it uses pre-extracted clips.**

    NLQ: the query is expressed in text (e.g., “What did I put in the drawer?”),
    and the output response is the temporal window where the answer is visible
    or deducible. Annotators wrote these queries based on a set of 13 template
    questions.

    MQ: in which the query is the name of a high-level activity or “moment”,
    and the response consists of all temporal windows where the activity occurs
    (e.g., “When did I read to my children?”). They established a taxonomy of 
    110 activities in a data-driven, semi-automatic manner by mining the narration
    summaries. Moments capture high-level activities in the camera wearer’s
    day, e.g., setting the table is a moment. For MQ, we provide the taxonomy of 
    labels and ask annotators to label clips with each and every temporal segment 
    containing a moment instance.
    """
    video_path = join(VIDEO_DATA_HOME, "Ego4d", "ego4d_data", "v2", "full_scale")
    clips_path = join(VIDEO_DATA_HOME, "ego4d-clips")

    def __init__(
            self,
            split,
            task: Literal["mq_label_clip", "mq_label_start_end",
                          "mq_temporal_grounding", "nlq_temporal_grounding",
                          "all"] = "all",
            max_per_video: Optional[int] = None,
            video_segment_length: Optional[int] = 180,
            use_extracted_clips: bool = True
        ):
        """
        Args:
            split (str): Dataset split to use. Must be one of ["train", "val", "test"].
            task (Literal, optional): Task type to include in the dataset. Options:
                - "mq_label_clip": given a short clip, cropped from start to end,
                  label what's shown in the video.
                - "mq_label_start_end": Given a few minute long video and a start and 
                  end timestamp in the prompt, output a label for that segment.
                - "nlq_temporal_grounding": Given a natural language query, localize the
                  part in the video which shows the answer to the question.
                - "all": Include all task types
                Defaults to "all".
            flat (bool, optional): Whether to flatten the dataset so each example contains
                only a single message instead of grouping multiple messages per video.
                Defaults to False.
            max_per_video (Optional[int], optional): Maximum number of messages to group
                per video example. If None, all messages for a video are grouped together.
                Only used when flat=False. Defaults to None.
            video_segment_length: as ego4d videos are long, for some of the tasks such as
                mq_label_start_end and temporal_grounding, if this arg is provided, we split
                the video into segments with some max length and only load that segment
                during training.
            use_extracted_clips: whether to use pre-extracted clips or extract on-the-fly
                from full videos. When True, looks for clips in clips_path first.
        """
        assert split in ["train", "val", "test"], f"Invalid split: {split}"
        assert task in ["mq_label_clip", "mq_label_start_end",
                        "mq_temporal_grounding", "nlq_temporal_grounding",
                        "all"], f"Invalid task: {task}"
        if self.video_path.startswith("gs://"):
            raise ValueError("Ego4d dataset not supported on GCP. Please use Ego4dCachedClips if you're training on GCP.")
        self.task = task
        self.max_per_video = max_per_video
        self.video_segment_length = video_segment_length
        self.use_extracted_clips = use_extracted_clips
        if self.use_extracted_clips:
            json_path = resource_path(join(VIDEO_DATA_HOME, "ego4d-clips", f"existing_clips.json"))
            with open(json_path, 'r') as f:
                self.existing_clips = set(json.load(f))

        super().__init__(split)
    
    def get_video_path(self, video_uid, start_time, end_time):
        """Get the appropriate video path, preferring extracted clips when available."""
        if self.use_extracted_clips:
            # Try to find pre-extracted clip
            clip_filename = f"{video_uid}_{start_time:.3f}_{end_time:.3f}_custom_clip.mp4"
            clip_path = join(self.clips_path, clip_filename)

            if clip_filename in self.existing_clips:
                self.existing_clips.add(clip_filename)
                return clip_path, None  # No metadata needed for extracted clips
            else:
                return None, None
        # Fall back to original video with metadata
        original_path = join(self.video_path, f"{video_uid}.mp4")
        metadata = {
            "clip_start_time": start_time,
            "clip_end_time": end_time,
        }
        return original_path, metadata
    
    def extract_segment(self, moment_start, moment_end, video_dur):
        """
        Extract a segment from the video that contains the moment.
        Args:
            moment_start (float): Start time of the moment in seconds.
            moment_end (float): End time of the moment in seconds.
            video_dur (float): Duration of the full ego4d video in seconds.
        """
        segment_start = random.uniform(max(0, moment_end - self.video_segment_length), moment_start)
        segment_end = min(segment_start + self.video_segment_length, video_dur)
        moment_start_within_segment = moment_start - segment_start
        moment_end_within_segment = moment_end - segment_start

        return segment_start, segment_end, moment_start_within_segment, moment_end_within_segment

    def load(self):
        ego4d_meta_json_path = join(VIDEO_DATA_HOME, "Ego4d", "ego4d_data", "ego4d.json")
        ego4d_meta = json.load(open(resource_path(dirname(ego4d_meta_json_path), basename(ego4d_meta_json_path))))
        video_uid_to_duration = {el['video_uid']: el['duration_sec'] for el in ego4d_meta['videos']}

        nlq_json_path = join(VIDEO_DATA_HOME, "Ego4d", "ego4d_data", "v2", "annotations", f"nlq_{self.split}.json")
        nlq_df = pd.read_json(resource_path(dirname(nlq_json_path), basename(nlq_json_path)))
        nlq_df['video_uid'] = nlq_df['videos'].apply(lambda x: x['video_uid'])

        mq_json_path = join(VIDEO_DATA_HOME, "Ego4d", "ego4d_data", "v2", "annotations", f"moments_{self.split}.json")
        mq_df = pd.read_json(resource_path(dirname(mq_json_path), basename(mq_json_path)))
        mq_df['video_uid'] = mq_df['videos'].apply(lambda x: x['video_uid'])

        data_list = []
        video2msgs = defaultdict(list)
        skipped = 0

        np.random.seed(42)
        random.seed(42)

        # Helper function to check overlap and merge moments
        def merge_overlapping_moments(moments, new_start, new_end, threshold=0.75):
            for i, (existing_start, existing_end) in enumerate(moments):
                intersection_start = max(new_start, existing_start)
                intersection_end = min(new_end, existing_end)
                intersection = max(0, intersection_end - intersection_start)
                
                new_len = new_end - new_start
                existing_len = existing_end - existing_start
                
                if new_len > 0 and existing_len > 0:
                    new_overlap_ratio = intersection / new_len
                    existing_overlap_ratio = intersection / existing_len
                    
                    if new_overlap_ratio > threshold or existing_overlap_ratio > threshold:
                        moments.pop(i)
                        if intersection_end > intersection_start:
                            moments.append((intersection_start, intersection_end))
                        return True
            return False

        # Process MQ tasks
        if self.task in ["mq_temporal_grounding", "mq_label_clip", "mq_label_start_end", "all"]:
            for _, row in mq_df.iterrows():
                video_uid = row['video_uid']
                video_dur = video_uid_to_duration.get(video_uid)
                
                # Collect all labels for temporal grounding
                if self.task in ["mq_temporal_grounding", "all"]:
                    label_to_moments = {}
                
                for clip in row['videos']['clips']:
                    for ann in clip['annotations']:
                        for label in ann['labels']:
                            if 'label' not in label:
                                continue

                            start, end = label['video_start_time'], label['video_end_time']
                            if end <= start:
                                skipped += 1
                                continue

                            label_name = label['label']
                            
                            # Handle temporal grounding
                            if self.task in ["mq_temporal_grounding", "all"]:
                                if label_name not in label_to_moments:
                                    label_to_moments[label_name] = []

                                if not merge_overlapping_moments(label_to_moments[label_name], start, end):
                                    label_to_moments[label_name].append((start, end))

                            # Handle clip labeling
                            if self.task in ["mq_label_clip", "all"]:
                                # Get the appropriate video path (extracted clip or original video)
                                video_path, clip_metadata = self.get_video_path(video_uid, start, end)
                                if video_path is None:
                                    continue

                                msg = dict(
                                    answer=label_name,
                                    question="Label the clip.",
                                    style="ego4d_mq_label_clip"
                                )

                                # Use extracted clip path if available, otherwise use original path with metadata
                                if clip_metadata is None:
                                    # Using extracted clip - no start/end needed in key
                                    video_key = (video_path, None, None)
                                else:
                                    # Using original video with clip metadata
                                    video_key = (video_path, start, end)

                                if msg not in video2msgs[video_key]:
                                    video2msgs[video_key].append(msg)

                            # Handle start/end labeling
                            if self.task in ["mq_label_start_end", "all"]:
                                if self.video_segment_length is not None:
                                    if end - start > self.video_segment_length:
                                        skipped += 1
                                        continue
                                        
                                    segment_start, segment_end, moment_start_within_segment, moment_end_within_segment = \
                                        self.extract_segment(start, end, video_dur)
                                    
                                    # Get the appropriate video path (extracted segment clip or original video)
                                    video_path, clip_metadata = self.get_video_path(video_uid, segment_start, segment_end) 
                                    if video_path is None:
                                        continue
                                    
                                    msg = dict(
                                        answer=label_name,
                                        style="ego4d_mq_label_start_end",
                                        question=f"Label the segment from {moment_start_within_segment:.2f} to {moment_end_within_segment:.2f}.",
                                    )
 
                                    # Use extracted clip path if available, otherwise use original path with metadata
                                    if clip_metadata is None:
                                        # Using extracted clip - no start/end needed in key
                                        video_key = (video_path, None, None)
                                    else:
                                        # Using original video with clip metadata
                                        video_key = (video_path, segment_start, segment_end)
                                    
                                    video2msgs[video_key].append(msg)
                                else:
                                    # Get the appropriate video path (extracted clip or original video)
                                    video_path, clip_metadata = self.get_video_path(video_uid, start, end)
                                    
                                    if video_path is None:
                                        continue

                                    msg = dict(
                                        answer=label_name,
                                        style="ego4d_mq_label_start_end",
                                        question=f"Label the segment from {start:.2f} to {end:.2f}.",
                                    )
                                    
                                    # Use extracted clip path if available, otherwise use original path
                                    if clip_metadata is None:
                                        # Using extracted clip - no start/end needed in key
                                        video_key = (video_path, None, None)
                                    else:
                                        # Using original video with clip metadata
                                        video_key = (video_path, None, None)
                                    
                                    video2msgs[video_key].append(msg)
                
                # Process temporal grounding messages
                if self.task in ["mq_temporal_grounding", "all"]:
                    for label_name, moments in label_to_moments.items():
                        for start, end in moments:
                            if self.video_segment_length and end - start > self.video_segment_length:
                                continue
                            
                            # Collect all moments for this label (including the current one)
                            localized_moments = []
                            
                            if self.video_segment_length is not None:
                                segment_start, segment_end, moment_start_within_segment, moment_end_within_segment = \
                                    self.extract_segment(start, end, video_dur)
                                
                                # Get the appropriate video path (extracted segment clip or original video)
                                video_path, clip_metadata = self.get_video_path(video_uid, segment_start, segment_end)
                                
                                if video_path is None:
                                    continue
                                
                                # Use extracted clip path if available, otherwise use original path with metadata
                                if clip_metadata is None:
                                    # Using extracted clip - no start/end needed in key
                                    video_key = (video_path, None, None)
                                else:
                                    # Using original video with clip metadata
                                    video_key = (video_path, segment_start, segment_end)
                                
                                # Add current moment relative to segment
                                localized_moments.append((moment_start_within_segment, moment_end_within_segment))
                                
                                # Add other overlapping moments within this segment
                                for other_start, other_end in moments:
                                    if other_start == start and other_end == end:
                                        continue
                                    if other_start < segment_end and other_end > segment_start:
                                        clipped_start = max(other_start, segment_start) - segment_start
                                        clipped_end = min(other_end, segment_end) - segment_start
                                        localized_moments.append((clipped_start, clipped_end))
                            else:
                                # Get the appropriate video path (extracted clip or original video)
                                video_path, clip_metadata = self.get_video_path(video_uid, start, end)
                                
                                if video_path is None:
                                    continue
                                
                                # Use extracted clip path if available, otherwise use original path
                                if clip_metadata is None:
                                    # Using extracted clip - no start/end needed in key
                                    video_key = (video_path, None, None)
                                else:
                                    # Using original video with clip metadata
                                    video_key = (video_path, None, None)
                                
                                # Add all moments in absolute time
                                for moment_start, moment_end in moments:
                                    localized_moments.append((moment_start, moment_end))
                            
                            answer = "\n".join([f"{s:.2f} - {e:.2f}" for s, e in localized_moments])
                            
                            msg = dict(
                                question=f"Localize the event {label_name} in the video.",
                                style="ego4d_mq_temporal_grounding",
                                answer=answer
                            )
                            video2msgs[video_key].append(msg)

        # Process NLQ tasks
        if self.task in ["nlq_temporal_grounding", "all"]:
            for _, row in nlq_df.iterrows():
                video_uid = row['video_uid']
                video_dur = video_uid_to_duration.get(video_uid)

                for clip in row['videos']['clips']:
                    for ann in clip['annotations']:
                        for query in ann['language_queries']:
                            if (question := query.get('query')) is None:
                                continue
                            start, end = query['clip_start_sec'], query['clip_end_sec']
                            if end <= start:
                                skipped += 1
                                continue

                            if self.video_segment_length is not None:
                                if end - start > self.video_segment_length:
                                    skipped += 1
                                    continue
                                
                                segment_start, segment_end, moment_start_within_segment, moment_end_within_segment = \
                                    self.extract_segment(start, end, video_dur)
                                
                                # Get the appropriate video path (extracted segment clip or original video)
                                video_path, clip_metadata = self.get_video_path(video_uid, segment_start, segment_end) 
                                if video_path is None:
                                    continue
                                
                                answer = f"Start: {moment_start_within_segment:.2f}, end: {moment_end_within_segment:.2f}"
                                
                                # Use extracted clip path if available, otherwise use original path with metadata
                                if clip_metadata is None:
                                    # Using extracted clip - no start/end needed in key
                                    video_key = (video_path, None, None)
                                else:
                                    # Using original video with clip metadata
                                    video_key = (video_path, segment_start, segment_end)
                            else:
                                # Get the appropriate video path (extracted clip or original video)
                                video_path, clip_metadata = self.get_video_path(video_uid, start, end) 
                                if video_path is None:
                                    continue
                                
                                answer = f"Start: {start:.2f}, end: {end:.2f}"
                                
                                # Use extracted clip path if available, otherwise use original path
                                if clip_metadata is None:
                                    # Using extracted clip - no start/end needed in key
                                    video_key = (video_path, None, None)
                                else:
                                    # Using original video with clip metadata
                                    video_key = (video_path, None, None)

                            msg = dict(
                                question=question,
                                answer=answer,
                                style="ego4d_nlq_temporal_grounding"
                            )
                            video2msgs[video_key].append(msg)

        if skipped > 0:
            log.warning(f"Skipped {skipped} clips due to invalid start and end times.")

        for video_start_end, msgs in video2msgs.items():
            video, start, end = video_start_end
            meta = None
            if start is not None and end is not None:
                meta = {
                    "clip_start_time": start,
                    "clip_end_time": end,
                } 
            if len(msgs) == 0:
                continue 
            if self.max_per_video:
                for msg_group in split_into_groups(msgs, self.max_per_video):
                    formatted_ex = {"video": video, "message_list": msg_group}
                    if meta is not None:
                        formatted_ex["metadata"] = meta
                    data_list.append(formatted_ex)
            else:
                formatted_ex = {"video": video, "message_list": msgs}
                if meta is not None:
                    formatted_ex["metadata"] = meta
                data_list.append(formatted_ex)
        
        return data_list

    def get(self, item, rng):
        return self.data[item]


class Ego4dCachedClips(DatasetBase):
    """Ego4d with cached clips extracted offline."""
    clips_path = join(VIDEO_DATA_HOME, "ego4d-clips")
    corrupt_clips_file = join(VIDEO_DATA_HOME, "Ego4d", "corrupt_9_15_25.parquet")

    def __init__(
            self,
            split,
            task: Literal["all"] = "all",
            max_per_video: Optional[int] = None,
        ):
        assert split in ["train", "val", "test"], f"Invalid split: {split}"
        assert task in ["all"], f"Invalid task: {task}"
        self.task = task
        self.max_per_video = max_per_video
        corrupt_clips_df = pd.read_parquet(resource_path(self.corrupt_clips_file))
        self.corrupt_clips = set(corrupt_clips_df['video'].apply(lambda x: x.split('/')[-1]).tolist())

        super().__init__(split)

    def load(self):
        ego4d_meta_json_path = join(VIDEO_DATA_HOME, "Ego4d", "data_list_with_clips.jsonl")
        ego4d_meta = pd.read_json(resource_path(dirname(ego4d_meta_json_path), basename(ego4d_meta_json_path)), lines=True)

        ego4d_meta = ego4d_meta[~ego4d_meta['clip'].isin(self.corrupt_clips)]
        ego4d_meta['video'] = ego4d_meta['clip'].apply(lambda x: join(self.clips_path, x))
        ego4d_meta = ego4d_meta.drop(columns=['clip'])
        data_list = ego4d_meta.to_dict(orient="records")

        return data_list
    
    def get(self, item, rng):
        return self.data[item]


class TVQA(DatasetBase):
    """TVQA dataset"""
    video_path = join(VIDEO_DATA_HOME, "TVQA", "video-frames", "frames_hq")
    corrupt_files = [
        "grey_s03e15_seg02_clip_05/video_00001_00051.mp4",
        "castle_s08e08_seg02_clip_14/video_00001_00019.mp4",
    ]

    def __init__(
            self,
            split,
            flat: bool = False,
            max_per_video: Optional[int] = None,
            with_subtitle: bool = False
        ):
        assert split in ["train", "val", "test"], f"Invalid split: {split}"
        if split == "test":
            split = "test_public"
        self.flat = flat
        self.split = split
        self.max_per_video = max_per_video
        self.with_subtitle = with_subtitle
        super().__init__(split)

    def process_bounded_video(self, video_info):
        """Helper function to process a single bounded video"""
        abs_video_path, start, end = video_info
        try:
            return save_bounded_video(
                video_path=abs_video_path,
                start_time=start, 
                end_time=end,
                task_type="Episodic Reasoning"
            )
        except Exception as e:
            return None

    def get_clip_subtitles(self, subtitle_df, vid_name, start, end):
        """Extract subtitles that overlap with the given time range."""
        if not self.with_subtitle:
            return []
        
        sub = subtitle_df.get(vid_name, [])
        if not isinstance(sub, list):
            return []
        
        clip_sub = {}
        for el in sub:
            try:
                # Validate subtitle structure
                if not all(key in el for key in ['start', 'end', 'text']):
                    continue
                    
                sub_start, sub_end = float(el['start']), float(el['end'])
                
                # Check for overlap: two intervals overlap if neither ends before the other starts
                if not (sub_end <= start or sub_start >= end):
                    clip_sub[(sub_start-start, sub_end-start)] = el['text']

            except (ValueError, TypeError, KeyError):
                continue
        
        return clip_sub

    def load(self):
        show_name_to_frame_dir_map = {
            "Grey's Anatomy": "grey_frames",
            "How I Met You Mother": "met_frames",
            "The Big Bang Theory": "bbt_frames",
            "House M.D.": "house_frames",
            "Castle": "castle_frames",
            "Friends": "friends_frames",
        }
        json_path = join(VIDEO_DATA_HOME, "TVQA", f"tvqa_{self.split}.jsonl")
        df = pd.read_json(resource_path(dirname(json_path), basename(json_path)), lines=True)
        if self.with_subtitle:
            subtitle_json_path = join(VIDEO_DATA_HOME, "TVQA", "tvqa_preprocessed_subtitles.jsonl")
            subtitle_df = pd.read_json(resource_path(dirname(subtitle_json_path), basename(subtitle_json_path)), lines=True)
            subtitle_df = subtitle_df.set_index("vid_name")
            subtitle_df = subtitle_df['sub'].to_dict()

        # FIXME it would be better to do this with an offline scripts then caching it in `load`
        video_infos = []
        valid_rows = []
        missing_clip_f = join(VIDEO_DATA_HOME, "TVQA", f"missing_clips.txt")
        generate_clips = not file_exists(missing_clip_f)
        if generate_clips:
            missing_clips = set()
            log.info("TVQA Clips will be re-generated")
        else:
            log.info("TVQA Clips are pre-built")
            with open(resource_path(missing_clip_f), 'r') as f:
                missing_clips = set(x.strip() for x in f.read().split("\n") if x.strip())

        for row in df.itertuples(False):
            abs_video_path = join(self.video_path, show_name_to_frame_dir_map[row.show_name], f"{row.vid_name}")
            start, end = [float(t) for t in row.ts.split("-")]
            if pd.isna(start) or pd.isna(end):
                continue
            
            video_infos.append((abs_video_path, start, end))
            valid_rows.append(row)

        ###################################################################### 
        # This is only necessary to run only once to cache the processed videos.
        # Process all bounded videos using ThreadPoolExecutor
        ######################################################################
        # import concurrent.futures
        # from concurrent.futures import ThreadPoolExecutor
        # logging.info(f"Processing {len(video_infos)} bounded videos using 64 threads...")
        # with ThreadPoolExecutor(max_workers=64) as executor:
        #     # Submit all jobs
        #     future_to_info = {
        #         executor.submit(self.process_bounded_video, info): info 
        #         for info in video_infos
        #     }
        #     processed_videos = {}
        #     with tqdm(total=len(future_to_info), desc="Processing bounded videos") as pbar:
        #         for future in concurrent.futures.as_completed(future_to_info):
        #             info = future_to_info[future]
        #             result = future.result()
        #             if result is not None:
        #                 processed_videos[info] = result
        #             pbar.update(1)
        ######################################################################

        data_list = []
        video2msgs = {}
        video2meta = {}
        corrupt = 0
        for i, row in enumerate(valid_rows):
            abs_video_path = join(self.video_path, show_name_to_frame_dir_map[row.show_name], f"{row.vid_name}")

            start, end = [float(t) for t in row.ts.split("-")]
            video_info = (abs_video_path, start, end)
            path = Path(abs_video_path)
            clip_key = f"{path.parent.name}/{path.name}:{row.ts}"
            if generate_clips:
                try:
                    abs_video_path = save_bounded_video(
                        video_path=abs_video_path,
                        start_time=start,
                        end_time=end,
                        task_type="Episodic Reasoning"
                    )
                except Exception as e:
                    abs_video_path = None
                    missing_clips.add(clip_key)
            else:
                # Assume the video has already been generated
                if clip_key in missing_clips:
                    abs_video_path = None
                else:
                    fps = 3  # As specified in the frames directory name
                    start_frame = int(start * fps) + 1  # +1 because frames start at 1
                    end_frame = int(end * fps)
                    abs_video_path = os.path.join(abs_video_path, f"video_{start_frame:05d}_{end_frame:05d}.mp4")

            if abs_video_path is None:
                corrupt += 1
                continue

            path = Path(abs_video_path)
            video_key = f"{path.parent.name}/{path.name}"
            if video_key in self.corrupt_files:
                corrupt += 1
                continue

            ######################################################################
            # This is only necessary to run once to find the corrupt videos.
            # try:
            #     _ = iio.immeta(abs_video_path)
            # except Exception as e:
            #     logging.error(f"Error reading video metadata for {abs_video_path}: {e}")
            #     with open("/weka/oe-training-default/reza/mm_olmo/bad_videos.txt", "a") as f:
            #         f.write(f"{abs_video_path}\n")
            #     continue 
            ######################################################################

            video2msgs[abs_video_path] = video2msgs.get(abs_video_path, [])
            video2meta[abs_video_path] = video2meta.get(abs_video_path, {})

            question = row.q

            options = [getattr(row, f'a{i}') for i in range(5)]
            answer_idx = row.answer_idx
            if answer_idx >= len(options):
                raise IndexError(f"Answer index {answer_idx} out of bounds for options {options}")

            style = "video_multiple_choice"

            if self.with_subtitle:
                clip_sub = self.get_clip_subtitles(subtitle_df, row.vid_name, start, end)
                if clip_sub:
                    style = "video_multiple_choice_w_subtitle"

            msg = dict(
                question=question,
                answer_idx=answer_idx,
                options=options,
                style=style
            )

            if self.flat:
                formatted_ex = {
                    "video": abs_video_path,
                    "metadata": video2meta[abs_video_path],
                    "message_list": [msg]
                }
                if self.with_subtitle and clip_sub:
                    formatted_ex["subtitle"] = clip_sub
                data_list.append(formatted_ex)
            else:
                video2msgs[abs_video_path].append(msg)
                video2meta[abs_video_path].update(
                    {
                        "show_name": row.show_name,
                        "ts"       : row.ts,
                    }
                )
                if self.with_subtitle and clip_sub:
                    video2meta[abs_video_path]["subtitle"] = clip_sub

        if get_global_rank() == 0:
            log.warning(f"Skipped {corrupt} corrupt TVQA videos.")

        if not self.flat:
            for video, msgs in video2msgs.items():
                if len(msgs) == 0:
                    continue
                subtitle = video2meta[video].get("subtitle", None)
                if self.max_per_video:
                    for msg in split_into_groups(msgs, self.max_per_video):
                        formatted_ex = {
                            "video": video,
                            "metadata": video2meta[video],
                            "message_list": msg,
                        }
                        if subtitle:
                            formatted_ex["subtitle"] = subtitle
                        data_list.append(formatted_ex)
                else:
                    formatted_ex = {
                        "video": video,
                        "metadata": video2meta[video],
                        "message_list": msgs,
                    }
                    if subtitle:
                        formatted_ex["subtitle"] = subtitle
                    data_list.append(formatted_ex)

        if generate_clips:
            if get_global_rank() == 0:
                log.info("Caching missing clips data")
                write_file(dirname(missing_clip_f), basename(missing_clip_f),
                           "\n".join(missing_clips), True)
        return data_list
    
    def get(self, item, rng):
        return self.data[item]


class VideoLocalizedNarratives(DatasetBase):
    """Video Localized Narratives dataset"""

    # video_path = join(VIDEO_DATA_HOME, "oops/oops_video")
    video_path = join(VIDEO_DATA_HOME, "oops/oops_video_transformed")

    def __init__(self, split, flat: bool = False, answer_type: Literal["open_ended"] = "open_ended"):
        assert split in ["train", "val"], f"Invalid split: {split}"
        self.flat = flat
        self.answer_type = answer_type
        super().__init__(split)

    def load(self):
        data_list = []
        json_path =  join(VIDEO_DATA_HOME, f"video-localized-narratives/videoqa/text_output/oops_{self.split}/qa_text_output_transformed.json")
        data = pd.read_json(resource_path(dirname(json_path), basename(json_path)))
        for ex in data['annotations']:
            video_path = join(self.video_path, ex['transformed_video_name'])
            qa_pairs = ex['qa_pairs']
            msgs = []
            for q in qa_pairs:
                question = q['raw_question']
                answer = q['raw_answer']
                msg = dict(
                    question=question,
                    answer=answer,
                    style="video_short_answer"
                )
                msgs.append(msg)
                if self.flat:
                    formatted_ex = {
                        "video": video_path,
                        "metadata": {
                            "video_name": ex['video_name'],
                            "question_id": q['question_id'],
                        },
                        "message_list": [msg]
                    }
                    data_list.append(formatted_ex)
            if not self.flat:
                if len(msgs) == 0: continue
                formatted_ex = {
                    "video": video_path,
                    "message_list": msgs,
                    "metadata": {
                        "video_name": ex['video_name'],
                    }
                }
                data_list.append(formatted_ex)
        return data_list

    def get(self, item, rng):
        return self.data[item]


class VideoLocalizedNarrativesCaptionHf(Dataset):
    """Video Localized Narratives dataset"""

    oops_video_path = "oops/oops_video"
    kinetics_video_path = "kinetics/kinetics700"
    uvo_sparse_video_path = "UVO/uvo_videos_sparse"
    uvo_dense_video_path = "UVO/uvo_videos_dense"
    ovis_video_path = "OVIS/train_mp4_5fps"

    kinetics_corrupt_files = set([
        'ixQrfusr6k8_000001_000011'
    ])
    @staticmethod
    def normalize_label(s: str) -> str:
        # Normalize Unicode, convert weird spaces to regular spaces, collapse runs
        s = unicodedata.normalize("NFKC", s)
        s = s.replace("\u00A0", " ").replace("\u2009", " ").replace("\u202F", " ")
        s = re.sub(r"\s+", " ", s.strip())
        return s

    @classmethod
    def download(cls, num_proc=None):
        # Build a HF dataset since the raw json files are large and take a long time to load
        hf_dir = join(VIDEO_DATA_HOME, "video-localized-narratives", "hf_dataset")
        if file_exists(hf_dir):
            return
        split = "train"

        data_list = []
        for video_dir, src in [
            (join(cls.oops_video_path, split), f"oops"),
            (join(cls.kinetics_video_path, split), f"mapped_kinetics"),
            (cls.uvo_sparse_video_path, f"UVO_sparse"),
            (cls.uvo_dense_video_path, f"UVO_dense"),
            (cls.ovis_video_path, f"OVIS")
        ]:
            json_path = join(VIDEO_DATA_HOME, f"video-localized-narratives/{src}_{split}.jsonl")
            data = pd.read_json(resource_path(dirname(json_path), basename(json_path)), lines=True, orient='records')

            for i, row in data.iterrows():
                if src == "mapped_kinetics":
                    if row['video_name'] in cls.kinetics_corrupt_files:
                        continue
                    video_path = join(cls.kinetics_video_path, split, cls.normalize_label(row['label']), f"{row['video_name']}.mp4")
                else:
                    video_path = join(video_dir, f"{row['video_id']}.mp4")
                anno = row['actor_narratives']
                if len(anno) > 0:
                    data_list.append(dict(
                        video=video_path,
                        id=row['vidln_id'],
                        captions=[x['caption'] for x in anno],
                        actor_names=[x['actor_name'] for x in anno],
                    ))
        dataset = HfDataset.from_list(data_list)
        dataset.save_to_disk(hf_dir)

    def __init__(self, split):
        assert split in ["train"], f"Invalid split: {split}"
        self.data = HfDataset.load_from_disk(
            join(VIDEO_DATA_HOME, "video-localized-narratives", "hf_dataset"))

    def __len__(self):
        return len(self.data)

    def get(self, item, rng):
        ex = self.data[item]
        messages = [dict(text=c, object=a.lower(), style="video_object_caption")
                    for c, a in zip(ex['captions'], ex["actor_names"])]
        return dict(
            video=join(VIDEO_DATA_HOME, ex["video"]),
            message_list=messages,
            metadata=dict(id=ex["id"])
        )


class VideoLocalizedNarrativesCaption(DatasetBase):
    """Video Localized Narratives dataset"""

    oops_video_path = join(VIDEO_DATA_HOME, "oops/oops_video")
    kinetics_video_path = join(VIDEO_DATA_HOME, "kinetics/kinetics700")
    uvo_sparse_video_path = join(VIDEO_DATA_HOME, "UVO/uvo_videos_sparse")
    uvo_dense_video_path = join(VIDEO_DATA_HOME, "UVO/uvo_videos_dense")
    ovis_video_path = join(VIDEO_DATA_HOME, "OVIS/train_mp4_5fps")

    kinetics_corrupt_files = set([
        'ixQrfusr6k8_000001_000011'
    ])

    def __init__(self, split, flat: bool = False):
        assert split in ["train"], f"Invalid split: {split}"
        self.flat = flat
        super().__init__(split)

    def load(self):
        import re
        import unicodedata

        def normalize_label(s: str) -> str:
            # Normalize Unicode, convert weird spaces to regular spaces, collapse runs
            s = unicodedata.normalize("NFKC", s)
            s = s.replace("\u00A0", " ").replace("\u2009", " ").replace("\u202F", " ")
            s = re.sub(r"\s+", " ", s.strip())
            return s

        data_list = []

        json_path = join(VIDEO_DATA_HOME, f"video-localized-narratives/oops_{self.split}.jsonl")
        data = pd.read_json(resource_path(dirname(json_path), basename(json_path)), lines=True, orient='records')
        for i, row in data.iterrows():
            video_path = join(self.oops_video_path, self.split, f"{row['video_id']}.mp4")

            msgs = []
            for ex in row['actor_narratives']:
                msg = dict(
                    text=ex['caption'],
                    object=ex['actor_name'].lower(),
                    style="video_object_caption"
                )
                msgs.append(msg)
                if self.flat:
                    formatted_ex = {
                        "video"       : video_path,
                        "message_list": [msg]
                    }
                    data_list.append(formatted_ex)

            if not self.flat:
                if len(msgs) == 0: continue
                formatted_ex = {
                    "video"       : video_path,
                    "message_list": msgs,
                    "metadata"    : {
                        "id": row['vidln_id'],
                    }
                }
                data_list.append(formatted_ex)

        json_path = join(VIDEO_DATA_HOME, f"video-localized-narratives/mapped_kinetics_{self.split}.jsonl")
        data = pd.read_json(resource_path(dirname(json_path), basename(json_path)), lines=True, orient='records')
        for i, row in data.iterrows():
            if row['video_name'] in self.kinetics_corrupt_files:
                continue
            video_path = join(self.kinetics_video_path, self.split, normalize_label(row['label']), f"{row['video_name']}.mp4")

            msgs = []
            for ex in row['actor_narratives']:
                msg = dict(
                    text=ex['caption'],
                    object=ex['actor_name'].lower(),
                    style="video_object_caption"
                )
                msgs.append(msg)
                if self.flat:
                    formatted_ex = {
                        "video"       : video_path,
                        "message_list": [msg]
                    }
                    data_list.append(formatted_ex)

            if not self.flat:
                if len(msgs) == 0: continue
                formatted_ex = {
                    "video"       : video_path,
                    "message_list": msgs,
                    "metadata"    : {
                        "id": row['vidln_id'],
                    }
                }
                data_list.append(formatted_ex)

        json_path = join(VIDEO_DATA_HOME, f"video-localized-narratives/UVO_sparse_{self.split}.jsonl")
        data = pd.read_json(resource_path(dirname(json_path), basename(json_path)), lines=True, orient='records')
        for i, row in data.iterrows():
            video_path = join(self.uvo_sparse_video_path, f"{row['video_id']}.mp4")

            msgs = []
            for ex in row['actor_narratives']:
                msg = dict(
                    text=ex['caption'],
                    object=ex['actor_name'].lower(),
                    style="video_object_caption"
                )
                msgs.append(msg)
                if self.flat:
                    formatted_ex = {
                        "video"       : video_path,
                        "message_list": [msg]
                    }
                    data_list.append(formatted_ex)

            if not self.flat:
                if len(msgs) == 0: continue
                formatted_ex = {
                    "video"       : video_path,
                    "message_list": msgs,
                    "metadata"    : {
                        "id": row['vidln_id'],
                    }
                }
                data_list.append(formatted_ex)

        json_path = join(VIDEO_DATA_HOME, f"video-localized-narratives/UVO_dense_{self.split}.jsonl")
        data = pd.read_json(resource_path(dirname(json_path), basename(json_path)), lines=True, orient='records')
        for i, row in data.iterrows():
            video_path = join(self.uvo_dense_video_path, f"{row['video_id']}.mp4")

            msgs = []
            for ex in row['actor_narratives']:
                msg = dict(
                    text=ex['caption'],
                    object=ex['actor_name'].lower(),
                    style="video_object_caption"
                )
                msgs.append(msg)
                if self.flat:
                    formatted_ex = {
                        "video"       : video_path,
                        "message_list": [msg]
                    }
                    data_list.append(formatted_ex)

            if not self.flat:
                if len(msgs) == 0: continue
                formatted_ex = {
                    "video"       : video_path,
                    "message_list": msgs,
                    "metadata"    : {
                        "id": row['vidln_id'],
                    }
                }
                data_list.append(formatted_ex)

        json_path = join(VIDEO_DATA_HOME, f"video-localized-narratives/OVIS_{self.split}.jsonl")
        data = pd.read_json(resource_path(dirname(json_path), basename(json_path)), lines=True, orient='records')
        for i, row in data.iterrows():
            video_path = join(self.ovis_video_path, f"{row['video_id']}.mp4")

            msgs = []
            for ex in row['actor_narratives']:
                msg = dict(
                    text=ex['caption'],
                    object=ex['actor_name'].lower(),
                    style="video_object_caption"
                )
                msgs.append(msg)
                if self.flat:
                    formatted_ex = {
                        "video"       : video_path,
                        "message_list": [msg]
                    }
                    data_list.append(formatted_ex)

            if not self.flat:
                if len(msgs) == 0: continue
                formatted_ex = {
                    "video"       : video_path,
                    "message_list": msgs,
                    "metadata"    : {
                        "id": row['vidln_id'],
                    }
                }
                data_list.append(formatted_ex)

        return data_list

    def get(self, item, rng):
        return self.data[item]


class Countix(DatasetBase):
    """Countix dataset"""

    kinetics_video_path = join(VIDEO_DATA_HOME, "kinetics/kinetics700")
    kinetics_corrupt_files = set([
        'ixQrfusr6k8_000001_000011'
    ])

    def __init__(self, split, answer_format: Literal["mc", "oe"]):
        assert split in ["train"], f"Invalid split: {split}"
        self.answer_format = answer_format
        super().__init__(split)

    def load(self):
        import re
        import unicodedata

        def normalize_label(s: str) -> str:
            # Normalize Unicode, convert weird spaces to regular spaces, collapse runs
            s = unicodedata.normalize("NFKC", s)
            s = s.replace("\u00A0", " ").replace("\u2009", " ").replace("\u202F", " ")
            s = re.sub(r"\s+", " ", s.strip())
            return s

        def generate_consecutive_options(correct, min_val=1):
            # pick an offset from -3 to 0, but clamp so that start >= min_val
            possible_offsets = [o for o in range(-3, 1) if correct + o >= min_val]
            start_offset = random.choice(possible_offsets)
            start = correct + start_offset

            options = [start + i for i in range(4)]
            return options

        data_list = []

        csv_path = join(self.kinetics_video_path, f"countix_train_mapped.csv")
        data = pd.read_csv(resource_path(dirname(csv_path), basename(csv_path)))
        question_template = json.loads(read_file(
            join(self.kinetics_video_path, "class_to_question.json"), "r"
        ))
        for i, row in data.iterrows():
            video_path = join(self.kinetics_video_path, row['video_path'])
            action = row['class']
            question = random.choice(question_template[action])
            count = row['count']

            if self.answer_format == "oe":
                msg = dict(
                    question=question,
                    answer=str(count),
                    style="video_short_answer"
                )
            else:
                options = generate_consecutive_options(count, min_val=1)
                answer_idx = options.index(count)
                msg = dict(
                    question=question,
                    options=options,
                    answer_idx=answer_idx,
                    style="video_multiple_choice"
                )
            formatted_ex = {
                "video"       : video_path,
                "message_list": [msg]
            }
            data_list.append(formatted_ex)


        return data_list

    def get(self, item, rng):
        return self.data[item]



class RoadTextVQA(DatasetBase):
    """RoadText Video QA dataset"""
    video_path = join(VIDEO_DATA_HOME, "RoadTextVQA", "videos")

    def __init__(self, split, flat: bool = False, answer_type: Literal["open_ended"] = "open_ended"):
        assert split in ["train", "val", "test"], f"Invalid split: {split}"
        self.flat = flat
        self.answer_type = answer_type
        super().__init__(split)

    def load(self):
        data_list = []
        json_path = join(VIDEO_DATA_HOME, "RoadTextVQA", f"{self.split}.json")
        data = json.loads(read_file(json_path, "r"))
        video2msgs = {}
        for ex in data['data']:
            video_path = join(self.video_path, ex['video'])
            video2msgs[video_path] = video2msgs.get(video_path, [])
            question = ex['question']

            if len(ex['answer']) == 1:
                answer = ex['answer'][0]
            else:
                answer = random.choice(ex['answer'])

            msg = dict(
                question=question,
                answer=answer,
                style="video_short_answer"
            )
            video2msgs[video_path].append(msg)
            if self.flat:
                formatted_ex = {
                    "video": video_path,
                    "metadata": {
                        "questionId": ex['questionId'],
                        "answer": ex['answer']
                    },
                    "message_list": [msg]
                }
                data_list.append(formatted_ex)
        if not self.flat:
            for video, msgs in video2msgs.items():
                if len(msgs) == 0: continue
                formatted_ex = {
                    "video": video,
                    "message_list": msgs,
                }
                data_list.append(formatted_ex)
        return data_list

    def get(self, item, rng):
        return self.data[item]


class CinepileHf(Dataset):
    """Cinepile Video QA dataset saved in HF form for faster loading times"""
    video_path = join(VIDEO_DATA_HOME, "cinepile", "videos")

    @classmethod
    def download(cls, num_proc=None):
        dataset_dict = {}
        hf_dir = join(VIDEO_DATA_HOME, "cinepile", f"hf_dataset")
        if not dir_is_empty(hf_dir):
            return
        for split in ["train", "test"]:
            json_path = join(VIDEO_DATA_HOME, "cinepile", f"{split}.json")
            log.info("Reading json...")
            df = pd.read_json(resource_path(json_path))
            n_missing = 0
            table = []

            log.info("Building list...")
            video_to_examples = defaultdict(list)
            for _, row in df.iterrows():
                video_to_examples[row['video_id']].append(row.to_dict())

            for video_id, examples in tqdm(video_to_examples.items()):
                abs_video_path = join(cls.video_path, video_id, f"{video_id}.mp4")
                if not file_exists(abs_video_path):
                    n_missing += 1
                    continue
                row = dict(video_id=video_id)
                # Collapse some of the redundant fields
                for k in ["movie_name", "subtitles", "genre", "year", "yt_clip_link", "yt_clip_title", "videoID"]:
                    val = examples[0][k]
                    assert all(val == ex.pop(k) for ex in examples)
                    row[k] = val
                for ex in examples:
                    del ex["movie_scene"]
                row["examples"] = examples
                table.append(row)
            log.warning(f"Missing {n_missing}/{len(df)} videos")
            dataset_dict[split] = HfDataset.from_list(table)
        dataset = DatasetDict(**dataset_dict)
        dataset.save_to_disk(hf_dir)

    def __init__(
        self,
        split,
        with_subtitle: bool = False
    ):
        assert split in ["train", "test"], f"Invalid split: {split}"
        self.with_subtitle = with_subtitle
        self.data = datasets.load_from_disk(
            join(VIDEO_DATA_HOME, "cinepile", f"hf_dataset"),
            keep_in_memory=False
        )[split]

    def __len__(self):
        return len(self.data)

    def get(self, item, rng):
        ex = self.data[item]
        abs_video_path = join(self.video_path, ex['video_id'], f"{ex['video_id']}.mp4")

        out = dict(
            video=abs_video_path,
            metadata=dict(video_id=ex['video_id'], movie_name=ex["movie_name"]),
        )
        style = "video_multiple_choice"
        if self.with_subtitle:
            subtitle = ex.get('subtitles', '')
            if subtitle:
                out["subtitle"] = subtitle.replace('<subtitle> ', '').strip()
                style = "video_multiple_choice_w_subtitle"

        messages = []
        for row in ex["examples"]:
            question = row['question']
            answer = row['answer_key']
            options = row['choices']
            answer_idx = row['answer_key_position']
            if answer_idx >= len(options):
                raise IndexError(f"Answer index {answer_idx} out of bounds for options {options}")
            assert answer == options[answer_idx]
            messages.append(dict(
                question=question,
                answer_idx=answer_idx,
                options=options,
                style=style
            ))
        out["message_list"] = messages
        return out


class Cinepile(DatasetBase):
    """Cinepile Video QA dataset"""
    video_path = join(VIDEO_DATA_HOME, "cinepile", "videos")


    def __init__(
            self,
            split,
            flat: bool = False,
            max_per_video: Optional[int] = None,
            with_subtitle: bool = False
        ):
        assert split in ["train", "test"], f"Invalid split: {split}"
        self.flat = flat
        self.max_per_video = max_per_video
        self.with_subtitle = with_subtitle
        super().__init__(split)

    def load(self):
        json_path = join(VIDEO_DATA_HOME, "cinepile", f"{self.split}.json")
        df = pd.read_json(resource_path(json_path))
        data_list = []
        video2msgs = {}
        video2meta = {}

        # We cache what videos are missing since checking for file existence for every video
        # can be slow, especially on remote file systems
        missing_videos_f = join(VIDEO_DATA_HOME, "cinepile", f"missing_videos.json")
        missing_videos_precomputed = file_exists(missing_videos_f)
        if missing_videos_precomputed:
            with open(resource_path(missing_videos_f), 'r') as f:
                missing_videos = set(json.load(f))
        else:
            missing_videos = set()

        corrupt = 0
        for _, row in df.iterrows():
            abs_video_path = join(self.video_path, row['video_id'], f"{row['video_id']}.mp4")
            if not missing_videos_precomputed and not file_exists(abs_video_path):
                missing_videos.add(row["video_id"])
            if row["video_id"] in missing_videos:
                corrupt += 1
                continue
            video2msgs[abs_video_path] = video2msgs.get(abs_video_path, [])
            video2meta[abs_video_path] = video2meta.get(abs_video_path, {})
            question = row['question']
            answer = row['answer_key']

            style = "video_multiple_choice"
            if self.with_subtitle:
                subtitle = row.get('subtitles', '')
                if subtitle:
                    subtitle = subtitle.replace('<subtitle> ', '').strip()
                    style = "video_multiple_choice_w_subtitle"

            options = row['choices']
            answer_idx = row['answer_key_position']
            if answer_idx >= len(options):
                raise IndexError(f"Answer index {answer_idx} out of bounds for options {options}")

            assert answer == options[answer_idx]
            msg = dict(
                question=question,
                answer_idx=answer_idx,
                options=options,
                style=style
            )

            if self.flat:
                formatted_ex = {
                    "video": abs_video_path,
                    "metadata": video2meta[abs_video_path],
                    "message_list": [msg]
                }
                if self.with_subtitle and subtitle:
                    formatted_ex["subtitle"] = subtitle
                data_list.append(formatted_ex)
            else:
                video2msgs[abs_video_path].append(msg)
                video2meta[abs_video_path].update({
                    "video_id"  : row['video_id'],
                    "movie_name": row['movie_name'],
                })
                if self.with_subtitle and subtitle:
                    video2meta[abs_video_path]["subtitle"] = subtitle


        if get_global_rank() == 0:
            log.warning(f"Total corrupt Cinepile videos: {corrupt}")

        if not self.flat:
            for video, msgs in video2msgs.items():
                if len(msgs) == 0:
                    continue
                subtitle = video2meta[video].get("subtitle", None)
                if self.max_per_video:
                    for msg in split_into_groups(msgs, self.max_per_video):
                        formatted_ex = {
                            "video": video,
                            "metadata": video2meta[video],
                            "message_list": msg,
                        }
                        if subtitle:
                            formatted_ex["subtitle"] = subtitle
                        data_list.append(formatted_ex)
                else:
                    formatted_ex = {
                        "video": video,
                        "metadata": video2meta[video],
                        "message_list": msgs,
                    }
                    if subtitle:
                        formatted_ex["subtitle"] = subtitle
                    data_list.append(formatted_ex)

        if not missing_videos_precomputed:
            if get_global_rank() == 0:
                log.info("Caching missing videos to missing_videos.json")
                write_file(join(VIDEO_DATA_HOME, "cinepile"), "missing_videos.json",
                           json.dumps(list(missing_videos)), True)
        return data_list

    def get(self, item, rng):
        return self.data[item]


def generate_action_localization_qa(segments):
    return None, None, []
    action_locate_templates = [
        "In the given video, when does the action '{action}' take place?",
        "During which part of the video does the action '{action}' occur?",
        "Can you identify when the action '{action}' happens in the video?",
        "At what moment in the video does the action '{action}' occur?",
        "When in the video sequence do we observe the action '{action}'?"
    ]
    ALL = "Throughout the entire video."
    START = "At the beginning of the video."
    END = "At the end of the video."
    MIDDLE = "In the middle of the video."

    message_list = []
    N = len(segments)
    if N > 0:
        clip_start = min([s[0] for s in segments])
        clip_end = max([s[1] for s in segments])

        if N == 1:
            action = segments[0][2]
            correct = ALL
            options = [ALL, START, END, MIDDLE]
            random.shuffle(options)
            answer_idx = options.index(correct)
            message_list.append(dict(
                question=random.choice(action_locate_templates).format(action=action),
                options=options,
                answer_idx=answer_idx,
                style="video_multiple_choice"
            ))
        else:
            action_to_time = {}
            for i, (s, e, a) in enumerate(segments):
                if a not in action_to_time:
                    action_to_time[a] = []
                action_to_time[a].append((i, s, e))
            for action, times in action_to_time.items():
                idxs = [i for i, _, _ in times]
                # thirds by index
                t1 = N / 3.0
                t2 = 2.0 * N / 3.0

                start_cnt = sum(i < t1 for i in idxs)
                middle_cnt = sum(t1 <= i < t2 for i in idxs)
                end_cnt = sum(i >= t2 for i in idxs)

                if start_cnt > 0 and middle_cnt == 0 and end_cnt == 0:
                    correct = START
                elif start_cnt == 0 and middle_cnt == 0 and end_cnt > 0:
                    correct = END
                elif start_cnt == 0 and middle_cnt > 0 and end_cnt == 0:
                    correct = MIDDLE
                elif start_cnt > 0 and middle_cnt > 0 and end_cnt > 0:
                    correct = ALL
                else:
                    # skip this action if it doesn't meet simple rules
                    continue

                options = [ALL, START, END, MIDDLE]
                random.shuffle(options)
                answer_idx = options.index(correct)
                message_list.append(dict(
                    question=random.choice(action_locate_templates).format(action=action),
                    options=options,
                    answer_idx=answer_idx,
                    style="video_multiple_choice"
                ))

        return clip_start, clip_end, message_list
    else:
        return None, None, []


class Youcook2(DatasetBase):
    """YouCook2 dataset"""
    video_path = join(VIDEO_DATA_HOME, "youcook2", "videos")

    templates = [
        "What action is being performed?",
        "What action is occurring?",
        "What activity does the video depict?"
    ]

    def __init__(
            self,
            split,
            flat: bool = False,
            task: Literal["action_localization_mc", "caption_start_end", "caption_clip", "all"] = "caption_clip",
            max_per_video: Optional[int] = None,
            qa_format: bool = False,
        ):
        """
        Args:
            split (str): Dataset split to use. Must be one of ["train", "validation", "test"].
            flat (bool, optional): Whether to flatten the dataset so each example contains
                only a single message instead of grouping multiple messages per video.
                Defaults to False.
            task (Literal, optional): Task type to include in the dataset. Options:
                - "caption_start_end": Given a start and end timestamp, output a caption for that segment.
                - "caption_clip": Given a short clip, output a caption for that clip.
                Defaults to "caption_clip".
            max_per_video (Optional[int], optional): Maximum number of messages to group
                per video example. If None, all messages for a video are grouped together.
                Only used when flat=False. Defaults to None.
        """
        assert split in ["train", "validation", "test"], f"Invalid split: {split}"
        self.flat = flat
        self.task = task
        self.max_per_video = max_per_video
        self.qa_format = qa_format
        super().__init__(split)

    def load(self):
        json_path = join(VIDEO_DATA_HOME, "youcook2", f"youcookii_annotations_trainval.json")
        data = pd.read_json(resource_path(json_path))
        corrupt_path = join(VIDEO_DATA_HOME, "youcook2", "corrupt_08_04_25.parquet")
        corrupt = pd.read_parquet(resource_path(corrupt_path))
        corrupt = corrupt.groupby("video_id")
        corrupt_video_ids = corrupt.groups.keys()

        all_files_f = join(VIDEO_DATA_HOME, "youcook2", "all_videos.json")
        if file_exists(all_files_f):
            log.info(f"Using cached filelist {all_files_f}")
            with open(resource_path(all_files_f)) as f:
                all_videos = set(json.load(f))
        else:
            all_videos = list(list_directory(self.video_path, recurse=True, include_files=True, include_dirs=False))
            all_videos = [relpath(x, self.video_path) for x in all_videos]
            if get_global_rank() == 0:
                log.info(f"Saving filelist to {all_files_f}")
                write_file(all_files_f, None, json.dumps(all_videos), True)

        data_list = []
        video2msgs = defaultdict(list)

        skipped = 0
        for video_id, v in data['database'].items():
            if self.split not in v['subset']:
                continue
            for ext in [".mkv", ".mp4", ".webm"]:
                if f"{video_id}{ext}" in all_videos:
                    break
            else:
                skipped += 1
                continue

            abs_video_path = join(self.video_path, f"{video_id}{ext}")
            segments = []
            complete = video_id not in corrupt_video_ids
            for ann in v['annotations']:
                segment = ann.get('segment', None)
                if segment is None:
                    continue
                if segment[1] <= segment[0]:
                    complete = False
                    continue
                start, end = segment
                sentence = ann['sentence']+"."
                segments.append((start, end, sentence))

                if self.task in ["caption_clip", "all"]:
                    if self.qa_format:
                        question = random.choice(self.templates)
                        msg = dict(
                            question=question,
                            answer=sentence,
                            style="youcook2_label"
                        )
                    else:
                        msg = dict(
                            text=sentence,
                            style="video_short_caption"
                        )
                    meta = {
                        "clip_start_time": start,
                        "clip_end_time": end,
                    }
                    is_corrupt = False
                    if video_id in corrupt_video_ids:
                        corrupt_group = corrupt.get_group(video_id)
                        if meta in corrupt_group['metadata'].values:
                            is_corrupt = True

                    if not is_corrupt:
                        video2msgs[(abs_video_path, start, end)].append(msg)
                        if self.flat:
                            formatted_ex = {
                                "video": abs_video_path,
                                "message_list": [msg]
                            }
                            formatted_ex.update({"metadata": meta})
                            data_list.append(formatted_ex)

                if self.task in ["caption_start_end", "all"]:
                    msg = dict(
                        start_time=start,
                        end_time=end,
                        text=sentence,
                        style="video_clip_short_caption_start_end"
                    )
                    video2msgs[(abs_video_path, None, None)].append(msg)
                    if self.flat:
                        formatted_ex = {
                            "video": abs_video_path,
                            "message_list": [msg]
                        }
                        data_list.append(formatted_ex)

            # if complete and self.task in ["action_localization_mc", "all"]:
            #     clip_start, clip_end, message_list = generate_action_localization_qa(segments)
            #     if len(message_list) > 0:
            #         video2msgs[(abs_video_path, clip_start, clip_end)] += message_list
            #         if self.flat:
            #             for msg in message_list:
            #                 formatted_ex = {
            #                     "video"       : abs_video_path,
            #                     "meta"        : {
            #                         "clip_start_time": clip_start,
            #                         "clip_end_time"  : clip_end,
            #                     },
            #                     "message_list": [msg]
            #                 }
            #                 data_list.append(formatted_ex)


        if get_global_rank() == 0:
            log.warning(f"Skipped {skipped}/{skipped+len(data['database'])} missing Youcook2 videos.")

        if not self.flat:
            for video_start_end, msgs in video2msgs.items():
                video, start, end = video_start_end
                meta = None
                if start is not None and end is not None:
                    meta = {
                        "clip_start_time": start,
                        "clip_end_time": end,
                    }
                if len(msgs) == 0:
                    continue
                if self.max_per_video:
                    for msg in split_into_groups(msgs, self.max_per_video):
                        formatted_ex = {
                            "video": video,
                            "message_list": msg,
                        }
                        formatted_ex.update({"metadata": meta} if meta is not None else {})
                        data_list.append(formatted_ex)
                else:
                    formatted_ex = {
                        "video": video,
                        "message_list": msgs,
                    }
                    formatted_ex.update({"metadata": meta} if meta is not None else {})
                    data_list.append(formatted_ex)

        return data_list

    def get(self, item, rng):
        return self.data[item]


class COIN(DatasetBase):
    """COIN dataset"""
    video_path = join(VIDEO_DATA_HOME, "coin", "videos")

    action_templates = [
        "What action is being performed?",
        "What action is occurring?",
        "What activity does the video depict?"
    ]


    def __init__(
            self, 
            split, 
            flat: bool = False,
            task: Literal["action_localization_mc", "caption_clip", "all"] = "caption_clip",
            max_per_video: Optional[int] = None,
            qa_format: bool = False,
        ):
        assert split in ["train", "test"], f"Invalid split: {split}"
        self.flat = flat
        self.task = task
        self.max_per_video = max_per_video
        self.qa_format = qa_format
        super().__init__(split)

    def load(self):
        json_path = join(VIDEO_DATA_HOME, "coin", f"COIN.json")
        data = json.load(open(resource_path(json_path)))
        corrupt_path = join(VIDEO_DATA_HOME, "coin", "coin_corrupt_08_04_25.parquet")
        corrupt = pd.read_parquet(resource_path(corrupt_path))
        corrupt = corrupt.groupby("video_id")
        corrupt_video_ids = corrupt.groups.keys()

        data_list = []
        video2msgs = defaultdict(list)
        extensions_file = join(VIDEO_DATA_HOME, "coin", "file_extensions.json")
        if file_exists(extensions_file):
            precomputed_extensions = True
            with open(resource_path(extensions_file), "r") as f:
                extensions = json.load(f)
        else:
            precomputed_extensions = False
            extensions = {}
        skipped = 0

        for video_id, v in data['database'].items():
            if self.split not in v['subset']:
                continue
            if precomputed_extensions:
                ex = extensions[video_id]
                if ex is None:
                    skipped += 1
                    continue
                else:
                    abs_video_path = join(self.video_path, f"{video_id}{ex}")
            else:
                for ext in [".mkv", ".mp4", ".webm"]:
                    abs_video_path = join(self.video_path, f"{video_id}{ext}")
                    if file_exists(abs_video_path):
                        extensions[video_id] = ext
                        break
                else:
                    extensions[video_id] = None
                    skipped += 1
                    continue

            segments = []
            complete = video_id not in corrupt_video_ids
            for ann in v['annotation']:
                segment = ann.get('segment', None)
                if segment is None:
                    continue
                if segment[1] <= segment[0]:
                    complete = False
                    continue
                start, end = segment
                segments.append((start, end, ann['label']))

                if self.task in ["caption_clip", "all"]:
                    if self.qa_format:
                        question = random.choice(self.action_templates)
                        msg = dict(
                            question=question,
                            answer=ann['label'],
                            style="coin_label"
                        )
                    else:
                        msg = dict(
                            text=ann['label'],
                            style="coin_label"
                        )

                    meta = {
                        "clip_start_time": segment[0],
                        "clip_end_time": segment[1],
                    }
                    if video_id in corrupt_video_ids:
                        corrupt_group = corrupt.get_group(video_id)
                        if meta in corrupt_group['metadata'].values:
                            continue

                    if self.flat:
                        formatted_ex = {
                            "video": abs_video_path,
                            "meta": {
                                "clip_start_time": start,
                                "clip_end_time": end,
                            },
                            "message_list": [msg]
                        }
                        data_list.append(formatted_ex)
                    else:
                        video2msgs[(abs_video_path, start, end)].append(msg)

            # if complete and self.task in ["action_localization_mc", "all"]:
            #     clip_start, clip_end, message_list = generate_action_localization_qa(segments)
            #     if len(message_list) > 0:
            #         video2msgs[(abs_video_path, clip_start, clip_end)] += message_list
            #         if self.flat:
            #             for msg in message_list:
            #                 formatted_ex = {
            #                     "video"       : abs_video_path,
            #                     "meta"        : {
            #                         "clip_start_time": clip_start,
            #                         "clip_end_time"  : clip_end,
            #                     },
            #                     "message_list": [msg]
            #                 }
            #                 data_list.append(formatted_ex)

        if get_global_rank() == 0:
            log.warning(f"Skipped {skipped}/{skipped + len(data['database'])} missing COIN videos.")

        if not self.flat:
            for video_start_end, msgs in video2msgs.items():
                video, start, end = video_start_end
                if len(msgs) == 0:
                    continue
                if self.max_per_video:
                    for msg in split_into_groups(msgs, self.max_per_video):
                        formatted_ex = {
                            "video": video,
                            "metadata": {
                                "clip_start_time": start,
                                "clip_end_time": end,
                            },
                            "message_list": msg,
                        }
                        data_list.append(formatted_ex)
                else:
                    formatted_ex = {
                        "video": video,
                        "metadata": {
                            "clip_start_time": start,
                            "clip_end_time": end,
                        },
                        "message_list": msgs,
                    }
                    data_list.append(formatted_ex)

        if get_global_rank() == 0 and not precomputed_extensions:
            log.info("Caching file extensions")
            write_file(join(VIDEO_DATA_HOME, "coin"), "file_extensions.json", json.dumps(extensions), save_overwrite=False)
        return data_list
    
    def get(self, item, rng):
        return self.data[item]


class SportsQA(DatasetBase):
    """SportsQA dataset"""
    video_path = join(VIDEO_DATA_HOME, "SportsQA")

    def __init__(
            self,
            split,
            max_per_video: Optional[int] = None,
            flat: bool = False,
        ):
        assert split in ["train", "val", "test"], f"Invalid split: {split}"
        self.max_per_video = max_per_video
        self.flat = flat
        super().__init__(split)

    def load(self):
        json_path = join(VIDEO_DATA_HOME, "SportsQA", "meta-data", f"{self.split}.json")
        df = pd.read_json(resource_path(dirname(json_path), basename(json_path)))
        data_list = []
        video2msgs = {}

        all_files_f = join(VIDEO_DATA_HOME, "SportsQA", "all_videos.json")
        if file_exists(all_files_f):
            log.info(f"Using cached filelist {all_files_f}")
            with open(resource_path(all_files_f)) as f:
                all_files = json.load(f)
        else:
            video_files = list_directory(self.video_path, recurse=True, include_files=True, include_dirs=False)
            all_files = [relpath(el, self.video_path) for el in video_files]
            if get_global_rank() == 0:
                log.info(f"Saving filelist to {all_files_f}")
                write_file(all_files_f, None, json.dumps(all_files), True)
        all_files = set(all_files)

        skipped = 0
        for _, row in df.iterrows():
            video_id = row['video']
            for ext in [".avi", ".mp4", ".webm"]:
                if f"{video_id}{ext}" in all_files:
                    abs_video_path = join(self.video_path, f"{video_id}{ext}")
                    break
            else:
                skipped += 1
                continue

            video2msgs[abs_video_path] = video2msgs.get(abs_video_path, [])
            question = row['question']
            answer = row['answer']

            msg = dict(
                question=question,
                answer=answer,
                style="video_short_answer"
            )
            video2msgs[abs_video_path].append(msg)
                
            if self.flat:
                formatted_ex = {
                    "video": abs_video_path,
                    "message_list": [msg]
                }
                data_list.append(formatted_ex)

        if not self.flat:
            for video, msgs in video2msgs.items():
                if self.max_per_video:
                    for msg in split_into_groups(msgs, self.max_per_video):
                        formatted_ex = {
                            "video": video,
                            "message_list": msg,
                        }
                        data_list.append(formatted_ex)
                else:
                    formatted_ex = {
                        "video": video,
                        "message_list": msgs,
                    }
                    data_list.append(formatted_ex)
        return data_list

    def get(self, item, rng):
        return self.data[item]


class SSV2(DatasetBase):
    """Something-something v2 dataset"""
    video_path = join(VIDEO_DATA_HOME, "sth-sth-v2", "videos")

    templates = [
        "What action is being performed?",
        "What is the person doing?",
        "What action is the person taking?"
        "What activity does the video depict?"
    ]

    def __init__(
            self,
            split,
            flat: bool = False,
            max_per_video: Optional[int] = None,
            qa_format: bool = False
        ):
        assert split in ["train", "val", "test"], f"Invalid split: {split}"
        self.flat = flat
        self.max_per_video = max_per_video
        self.qa_format = qa_format
        super().__init__(split)

    def load(self):
        ssv2_json_path = join(VIDEO_DATA_HOME, "sth-sth-v2", "labels", f"v2-{self.split}.json")
        ssv2_df = pd.read_json(resource_path(dirname(ssv2_json_path), basename(ssv2_json_path)))

        all_files_f = join(VIDEO_DATA_HOME, "sth-sth-v2", "all_videos.json")
        if file_exists(all_files_f):
            log.info(f"Using cached filelist {all_files_f}")
            with open(resource_path(all_files_f)) as f:
                id2vidpath = json.load(f)
        else:
            ssv2_videos = list_directory(self.video_path, recurse=True, include_files=True, include_dirs=False)
            ssv2_videos = [el for el in ssv2_videos if el.endswith(".webm")]
            id2vidpath = {el.split("/")[-1].split(".")[0]: relpath(el, self.video_path) for el in ssv2_videos}
            assert len(id2vidpath) == len(ssv2_videos)
            if get_global_rank() == 0:
                log.info(f"Saving filelist to {all_files_f}")
                write_file(all_files_f, None, json.dumps(id2vidpath), True)

        data_list = []
        video2msgs = {}

        # for row in ssv2_df.itertuples(False):
        for id, label in zip(ssv2_df.id[:], ssv2_df.label[:]):
            abs_video_path = join(self.video_path, id2vidpath[str(id)])
            if self.qa_format:
                msg = dict(
                    question=random.choice(self.templates),
                    answer=label,
                    style="ssv2"
                )
            else:
                msg = dict(
                    text=label,
                    style="ssv2"
                )

            if self.flat:
                formatted_ex = {
                    "video": abs_video_path,
                    "message_list": [msg]
                }
                data_list.append(formatted_ex)
            else:
                video2msgs[abs_video_path] = video2msgs.get(abs_video_path, [])
                video2msgs[abs_video_path].append(msg)

        if not self.flat:
            if self.max_per_video:
                for video, msgs in video2msgs.items():
                    for msg in split_into_groups(msgs, self.max_per_video):
                        formatted_ex = {
                            "video": video,
                            "message_list": msg,
                        }
                        data_list.append(formatted_ex)
            else:
                for video, msgs in video2msgs.items():
                    formatted_ex = {
                        "video": video,
                        "message_list": msgs,
                    }
                    data_list.append(formatted_ex)

        return data_list
    
    def get(self, item, rng):
        return self.data[item]


class Kinetics710(DatasetBase):
    """Kinetics710 dataset"""
    root_path = join(VIDEO_DATA_HOME, "kinetics")

    templates = [
        "What action is being performed?",
        "What is the person doing?",
        "What action is the person taking?"
        "What activity does the video depict?"
    ]

    def __init__(
            self,
            split,
            flat: bool = False,
            max_per_video: Optional[int] = None,
            qa_format: bool = False
        ):
        assert split in ["train", "val"], f"Invalid split: {split}"
        self.split = split
        self.flat = flat
        self.max_per_video = max_per_video
        self.qa_format = qa_format
        super().__init__(split)

    def _load_mappings(self):
        """Load all label mappings in one place"""
        # Load k710 labels
        k710_labels = Path(resource_path(join(self.root_path, "kinetics710", "k710_label_map.txt"))).read_text().splitlines()

        # Load mapping files
        k700_map = json.load(open(resource_path(join(self.root_path, "kinetics710", "map_k700.json"))))
        k600_map = json.load(open(resource_path(join(self.root_path, "kinetics710", "map_k600.json"))))

        return {
            'k710_labels': {i: line for i, line in enumerate(k710_labels)},
            'k600_to_k710': {i: int(v) for i, v in enumerate(k600_map)},
            'k700_to_k710': {i: int(v) for i, v in enumerate(k700_map)},
        }

    def _load_video_list(self, dataset_name):
        """Load video list for a specific dataset (k600 or k700)"""
        file_path = join(self.root_path, "kinetics710", f"{dataset_name}_{self.split}_list_videos.txt")
        lines = Path(resource_path(file_path)).read_text().splitlines()

        return {
            line.strip().split(" ")[0]: int(line.strip().split(" ")[1])
            for line in lines
        }

    def load(self):
        # Load all mappings in a more organized way
        mappings = self._load_mappings()

        # Load video lists and create mappings to k710 labels
        k600_video_to_k710_label = {}
        k700_video_to_k710_label = {}

        # Process k600 videos
        k600_list = self._load_video_list("k600")
        for video_path, label_id in k600_list.items():
            video_id = video_path.split("/")[-1]
            k600_video_to_k710_label[video_id] = mappings['k710_labels'][label_id]

        # Process k700 videos
        k700_list = self._load_video_list("k700")
        for video_path, label_id in k700_list.items():
            video_id = video_path.split("/")[-1]
            k700_video_to_k710_label[video_id] = mappings['k710_labels'][label_id]

        corrupt = pd.read_parquet(resource_path(join(self.root_path, "corrupt_10_3_25.parquet")))
        corrupt_videos = set(corrupt['video'].apply(lambda x: join(self.root_path, x)).values)

        with open(resource_path(join(self.root_path, "existing_videos.json")), "r") as f:
            existing_videos = json.load(f)
        existing_videos = set([join(self.root_path, el) for el in existing_videos])

        # Process videos and create dataset
        data_list = []
        video2msgs = {}
        skipped = 0

        for video_id, label in list(k700_video_to_k710_label.items()):
            # Get the K710 label text
            abs_video_path = join(self.root_path, "kinetics700", self.split, label, video_id)
            if abs_video_path not in existing_videos:
                skipped += 1
                continue
            if abs_video_path in corrupt_videos:
                skipped += 1
                continue

            if self.qa_format:
                msg = dict(
                    question=random.choice(self.templates),
                    answer=label,
                    style="kinetics_label"
                )
            else:
                msg = dict(
                    text=label,
                    style="kinetics_label"
                )

            if self.flat:
                formatted_ex = {
                    "video": abs_video_path,
                    "message_list": [msg]
                }
                data_list.append(formatted_ex)
            else:
                video2msgs[abs_video_path] = video2msgs.get(abs_video_path, [])
                video2msgs[abs_video_path].append(msg)

        for video_id, label in list(k600_video_to_k710_label.items()):
            abs_video_path = join(self.root_path, "kinetics600", self.split, label.replace(" ", "_"), video_id)
            if abs_video_path in corrupt_videos:
                skipped += 1
                continue
            if abs_video_path not in existing_videos:
                skipped += 1
                continue

            if self.qa_format:
                msg = dict(
                    question=random.choice(self.templates),
                    answer=label,
                    style="kinetics_label"
                )
            else:
                msg = dict(
                    text=label,
                    style="kinetics_label"
                )

            if self.flat:
                formatted_ex = {
                    "video": abs_video_path,
                    "message_list": [msg]
                }
                data_list.append(formatted_ex)
            else:
                video2msgs[abs_video_path] = video2msgs.get(abs_video_path, [])
                video2msgs[abs_video_path].append(msg)

        if get_global_rank() == 0:
            log.warning(f"Skipped {skipped}/{len(k700_video_to_k710_label) + len(k600_video_to_k710_label)} missing Kinetics710 videos.")

        if not self.flat:
            for video, msgs in video2msgs.items():
                if len(msgs) == 0:
                    continue
                if self.max_per_video:
                    for msg_group in split_into_groups(msgs, self.max_per_video):
                        formatted_ex = {
                            "video": video,
                            "message_list": msg_group,
                        }
                        data_list.append(formatted_ex)
                else:
                    formatted_ex = {
                        "video": video,
                        "message_list": msgs,
                    }
                    data_list.append(formatted_ex)

        return data_list

    def get(self, item, rng):
        return self.data[item]


class CharadesSTA(DatasetBase):
    """CharadesSTA Video dataset"""
    video_path = join(VIDEO_DATA_HOME, "Charades")

    # Videos with incorrect segment annotations
    INCORRECT_SEGMENTS = {"LEOL6", "AKKWU"}

    templates = [
        "What action is being performed?",
        "What is the person doing?",
        "What action is the person taking?",
        "What activity does the video depict?"
    ]

    def __init__(
            self,
            split,
            flat: bool = False,
            task: Literal["action_localization_mc", "caption_clip", "all"] = "caption_clip",
            qa_format: bool = False
    ):
        assert split in ["train"], f"Invalid split: {split}"
        self.split = split
        self.flat = flat
        self.task = task
        self.qa_format = qa_format
        super().__init__(split)

    def load(self):
        data = Path(resource_path(join(VIDEO_DATA_HOME, "Charades", "charades_sta_train.txt"))).read_text().splitlines()
        data_list = []
        video2msgs = defaultdict(list)
        skipped = 0
        with open(resource_path(join(VIDEO_DATA_HOME, "Charades", "existing_videos.json")), "r") as f:
            existing_videos = json.load(f)

        video2segments = defaultdict(list)
        path_to_id = {}
        for line in data:
            rest, caption = line.strip().split("##")
            video_id, start, end = rest.split(" ")
            abs_video_path = join(self.video_path, "Charades_v1", f"{video_id}.mp4")
            if video_id not in existing_videos:
                skipped += 1
                continue
            path_to_id[abs_video_path] = video_id

            if self.qa_format:
                msg = dict(
                    question=random.choice(self.templates),
                    answer=caption,
                    style="charades_sta"
                )
            else:
                msg = dict(
                    text=caption,
                    style="charades_sta"
                )

            start, end = float(start), float(end)

            if end <= start:
                skipped += 1
                continue

            video2segments[abs_video_path].append((start, end, caption))

            if self.task in ["caption_clip", "all"]:
                if self.flat:
                    formatted_ex = {
                        "video": abs_video_path,
                        "metadata": {
                            "clip_start_time": start,
                            "clip_end_time": end,
                        },
                        "message_list": [msg]
                    }
                    data_list.append(formatted_ex)
                else:
                    video2msgs[(abs_video_path, start, end)].append(msg)

#         if self.task in ["action_localization_mc", "all"]:
#             for video, segments in video2segments.items():
#                 if path_to_id[video] in self.INCORRECT_SEGMENTS:
#                     continue
#                 clip_start, clip_end, message_list = generate_action_localization_qa(segments)
#                 if len(message_list) > 0:
#                     video2msgs[(video, clip_start, clip_end)] += message_list
#                     if self.flat:
#                         for msg in message_list:
#                             formatted_ex = {
#                                 "video"       : video,
#                                 "meta"        : {
#                                     "clip_start_time": clip_start,
#                                     "clip_end_time"  : clip_end,
#                                 },
#                                 "message_list": [msg]
#                             }
#                             data_list.append(formatted_ex)

        if not self.flat:
            for video_start_end, msgs in video2msgs.items():
                video, start, end = video_start_end
                if len(msgs) == 0:
                    continue
                formatted_ex = {
                    "video": video,
                    "metadata": {
                        "clip_start_time": start,
                        "clip_end_time": end,
                    },
                    "message_list": msgs,
                }
                data_list.append(formatted_ex)

        if get_global_rank() == 0:
            log.warning(f"Skipped {skipped} missing or corrupt CharadesSTA annotations.")

        return data_list

    def get(self, item, rng):
        return self.data[item]


class ActivityNet(DatasetBase):
    """ActivityNet Video dataset (Captioning OG / ActivityNetQA)"""
    video_path = join(VIDEO_DATA_HOME, "ActivityNet", "all-videos")

    templates = [
        "What activity is being performed?",
        "What activity is occurring?",
        "What activity does the video depict?"
    ]

    def __init__(
            self,
            split,
            flat: bool = False,
            max_per_video: Optional[int] = None,
            task: Literal["captioning", "qa", "action_localization_mc", "all"] = "all",
            qa_format: bool = False
        ):
        assert split in ["train", "validation"], f"Invalid split: {split}"
        self.flat = flat
        if split == "validation":
            split = "val"
        self.split = split
        self.max_per_video = max_per_video
        self.task = task
        self.qa_format = qa_format
        super().__init__(split)

    def load(self):
        caption_json_path = join(VIDEO_DATA_HOME, "ActivityNet", f"activity_net.v1-3.min.json")
        q_json_path = join(VIDEO_DATA_HOME, "ActivityNet", f"{self.split}_q.json")
        a_json_path = join(VIDEO_DATA_HOME, "ActivityNet", f"{self.split}_a.json")

        missing_videos_f = join(VIDEO_DATA_HOME, "ActivityNet", f"missing_videos.json")
        missing_videos_precomputed = file_exists(missing_videos_f)
        if missing_videos_precomputed:
            with open(resource_path(missing_videos_f), "r") as f:
                missing_videos = set(json.load(f))
        else:
            missing_videos = set()

        data_list = []
        video2msgs = {}

        if self.task in ["captioning", "all"]:
            existing = 0
            missing = 0
            bad_annotation = 0

            caption_data = json.load(open(resource_path(caption_json_path)))
            for vid, anns in caption_data['database'].items():
                abs_video_path = join(self.video_path, vid, f"{vid}.mp4")
                if not missing_videos_precomputed and not file_exists(abs_video_path):
                    missing_videos.add(vid)
                if vid in missing_videos:
                    missing += 1
                    continue
                existing += 1

                segments = []
                complete = True
                for ann in anns['annotations']:
                    start, end = ann['segment']
                    if pd.isna(start) or pd.isna(end):
                        bad_annotation += 1
                        continue
                    if end <= start:
                        bad_annotation += 1
                        complete = False
                        continue
                    segments.append((start, end, ann['label']))

                    if self.qa_format:
                        question = random.choice(self.templates)
                        msg = dict(
                            question=question,
                            answer=ann['label'],
                            style="activitynet_label"
                        )
                    else:
                        msg = dict(
                            text=ann['label'],
                            style="activitynet_short_caption"
                        )

                    if self.flat:
                        formatted_ex = {
                            "video": abs_video_path,
                            "meta": {
                                "clip_start_time": start,
                                "clip_end_time": end,
                            },
                            "message_list": [msg]
                        }
                        data_list.append(formatted_ex)
                    else:
                        video2msgs[(abs_video_path, start, end)] = video2msgs.get((abs_video_path, start, end), [])
                        # sometimes there are identical annotations for the same segment
                        if msg not in video2msgs[(abs_video_path, start, end)]:
                            video2msgs[(abs_video_path, start, end)].append(msg)

                # if complete and self.task in ["action_localization_mc", "all"]:
                #     clip_start, clip_end, message_list = generate_action_localization_qa(segments)
                #     if len(message_list) > 0:
                #         video2msgs[(abs_video_path, clip_start, clip_end)] = video2msgs.get((abs_video_path, clip_start, clip_end), [])
                #         video2msgs[(abs_video_path, clip_start, clip_end)] += message_list
                #         if self.flat:
                #             for msg in message_list:
                #                 formatted_ex = {
                #                     "video"       : abs_video_path,
                #                     "meta"        : {
                #                         "clip_start_time": clip_start,
                #                         "clip_end_time"  : clip_end,
                #                     },
                #                     "message_list": [msg]
                #                 }
                #                 data_list.append(formatted_ex)

            if get_global_rank() == 0:
                log.warning(f"Skipped {missing}/{missing+existing} missing Activitynet captioning videos.")
                log.warning(f"Skipped {bad_annotation} corrupt Activitynet captioning annotations.")

        if self.task in ["qa", "all"]:
            existing = 0
            missing = 0
            q_df = pd.read_json(resource_path(dirname(q_json_path), basename(q_json_path)))
            a_df = pd.read_json(resource_path(dirname(a_json_path), basename(a_json_path)))
            qa_df = pd.merge(q_df, a_df, on="question_id", how="inner")

            for video_name, question, answer in zip(qa_df["video_name"], qa_df["question"], qa_df["answer"]):
                abs_video_path = join(self.video_path, video_name, f"{video_name}.mp4")
                if not missing_videos_precomputed and not file_exists(abs_video_path):
                    missing_videos.add(video_name)
                if video_name in missing_videos:
                    missing += 1
                    continue
                existing += 1
                video2msgs[(abs_video_path, None, None)] = video2msgs.get((abs_video_path, None, None), [])
                if not question.endswith("?"):
                    question = question.strip() + "?"

                msg = dict(
                    question=question,
                    answer=answer,
                    style="video_short_answer"
                )
                video2msgs[(abs_video_path, None, None)].append(msg)

                if self.flat:
                    formatted_ex = {
                        "video": abs_video_path,
                        "message_list": [msg]
                    }
                    data_list.append(formatted_ex)

            if get_global_rank() == 0:
                log.warning(f"Skipped {missing}/{missing+existing} Activitynet QA samples with missing videos.")

        if not self.flat:
            for video_start_end, msgs in video2msgs.items():
                video, start, end = video_start_end
                meta = None
                if start is not None and end is not None:
                    meta = {
                        "clip_start_time": start,
                        "clip_end_time": end,
                    }
                if len(msgs) == 0:
                    continue
                if self.max_per_video:
                    for msg in split_into_groups(msgs, self.max_per_video):
                        formatted_ex = {
                            "video": video,
                            "message_list": msg,
                        }
                        formatted_ex.update({"metadata": meta} if meta is not None else {})
                        data_list.append(formatted_ex)
                else:
                    formatted_ex = {
                        "video": video,
                        "message_list": msgs,
                    }
                    formatted_ex.update({"metadata": meta} if meta is not None else {})
                    data_list.append(formatted_ex)

        if not missing_videos_precomputed:
            if get_global_rank() == 0:
                log.info("Caching missing videos to missing_videos.json")
                write_file(join(VIDEO_DATA_HOME, "ActivityNet"), f"missing_videos.json", json.dumps(list(missing_videos)), True)
        return data_list

    def get(self, item, rng):
        return self.data[item]


class MomentsInTime(DatasetBase):
    """Moments in Time dataset"""

    templates = [
        "What action is being performed?",
        "What action is occurring?",
        "What activity does the video depict?"
    ]

    def __init__(
            self,
            split,
            flat: bool = False,
            max_per_video: Optional[int] = None,
            qa_format: bool = False
        ):
        assert split in ["train", "validation"], f"Invalid split: {split}"
        self.video_path = join(VIDEO_DATA_HOME, "moments_in_time", "Moments_in_Time_Raw", f"{split}-videos-renamed")
        self.flat = flat
        self.max_per_video = max_per_video
        self.qa_format = qa_format
        super().__init__(split)

    def load(self):
        csv_path = join(VIDEO_DATA_HOME, "moments_in_time", "Moments_in_Time_Raw", f"{self.split}_with_missing_renamed.csv")
        data = pd.read_csv(resource_path(dirname(csv_path), basename(csv_path)))
        data_list = []
        video2msgs = {}
        skipped = 0

        # Use `itertuples` since this dataset is huge and itertuples is much faster then `iterrows`
        for row in data.itertuples(index=False):
            label = row.label
            abs_video_path = join(self.video_path, row.transformed_video_path)
            if row.missing:
                skipped += 1
                continue

            if '+' in label:
                label = label.replace('+', ' ')
                if label.startswith("child ") or label.startswith("adult "):
                    continue

            if self.qa_format:
                msg = dict(
                    question=random.choice(self.templates),
                    answer=label,
                    style="moments_in_time_label"
                )
            else:
                msg = dict(
                    text=label,
                    style="moments_in_time_label"
                )

            if self.flat:
                formatted_ex = {
                    "video": abs_video_path,
                    "message_list": [msg]
                }
                data_list.append(formatted_ex)
            else:
                video2msgs[abs_video_path] = video2msgs.get(abs_video_path, [])
                video2msgs[abs_video_path].append(msg)

        if get_global_rank() == 0:
            log.warning(f"Skipped {skipped}/{len(data)} missing Moments in Time videos.")
        if not self.flat:
            for video, msgs in video2msgs.items():
                formatted_ex = {
                    "video": video,
                    "message_list": msgs,
                }
                data_list.append(formatted_ex)
        return data_list

    def get(self, item, rng):
        return self.data[item]


class How2QA(DatasetBase):
    data_path = join(VIDEO_DATA_HOME, "how2QA")
    clip_dir = join(data_path, "video-clips")
    corrupt_videos = {"Z1NEMNKgR9U"}  # Unable to download from how2 100m
    corrupt_clips = {"onGTfd-EKrs_89.15_89.44.mp4", "bkl7eK8B6ig_147.73_149.83.mp4"}  # output of bounds clip

    def __init__(self, split, flatten=False):
        if split == "validation":
            split = "val"
        assert split in ["train", "val"]
        self.flatten = flatten
        super().__init__(split)

    def load(self):
        data = pd.read_csv(resource_path(join(self.data_path, f'how2QA_{self.split}_release.csv')), header=None)
        extension_file_path = resource_path(join(self.data_path, "extensions.json"))
        with open(extension_file_path, encoding="utf-8") as json_file:
            extensions = json.load(json_file)

        data_list = []
        errors = Counter()
        clip_to_question = defaultdict(list)
        for row in data.itertuples(index=False):
            start, end = eval(row[1].replace(':', ','))
            if start >= end:
                errors["invalid_clip"] += 1
                continue

            vid = row[0]
            if vid not in extensions:
                errors["missing_in_extensions"] += 1
                continue
            if vid in self.corrupt_videos:
                errors["corrupt"] += 1
                continue

            ext = extensions[vid]
            start = round(start, 2)
            end = round(end, 2)
            clip_id = f"{vid}_{start}_{end}.{ext}"
            if clip_id in self.corrupt_clips:
                errors["corrupt_clip"] += 1
                continue
            clip_path = join(self.clip_dir, clip_id)
            answer = row[6]
            neg_options = list(row[2:5])
            question = row[5]
            answer_idx = random.randint(0, len(neg_options))
            neg_options.insert(answer_idx, answer)
            assert answer_idx < len(neg_options), f"Answer index {answer_idx} out of bounds for options {neg_options}"
            msg = dict(
                question=question,
                answer_idx=answer_idx,
                options=neg_options,
                style="video_multiple_choice",
            )
            clip_to_question[clip_path].append(msg)

        for clip_path, msgs in clip_to_question.items():
            if self.flatten:
                for msg in msgs:
                    data_list.append(dict(video=clip_path, message_list=[msg]))
            else:
                data_list.append(dict(video=clip_path, message_list=msgs))
        return data_list

    def get(self, idx, rng):
        return self.data[idx]


class CameraBenchTrain(DatasetBase):
    data_path = join(VIDEO_DATA_HOME, "CameraBench")
    video_dir = join(data_path, "train")

    def __init__(self, split):
        super().__init__(split)

    def load(self):
        data = pd.read_parquet(resource_path(join(self.data_path, f'camerabench_qa.parquet')))

        data_list = []
        for ex_id, row in data.iterrows():
            video_path = os.path.join(self.video_dir, row["video_path"])
            qa_list = row["mc_qa_list"]
            msgs = []
            for q_id, ex in enumerate(qa_list):
                question = ex["Question"]
                answer = ex["Answer"]
                neg_options = list(ex["NegativeAnswers"])
                answer_idx = random.randint(0, len(neg_options))
                neg_options.insert(answer_idx, answer)
                assert answer_idx < len(neg_options), f"Answer index {answer_idx} out of bounds for options {neg_options}"
                msg = dict(
                    question=question,
                    answer_idx=answer_idx,
                    options=neg_options,
                    style="video_multiple_choice",
                )
                msgs.append(msg)
            if len(msgs) == 0: continue

            formatted_example = {
                "video"       : video_path,
                "message_list": msgs
            }
            data_list.append(formatted_example)

        return data_list

    def get(self, idx, rng):
        return self.data[idx]


class NewsVideoQA(DatasetBase):
    data_path = join(VIDEO_DATA_HOME, "NewsVideoQA")
    video_path = join(VIDEO_DATA_HOME, "NewsVideoQA", "final_data_feb_16", "videos")
    corrupt_files = set(['389x18', '76x16', '254x17', '254x21', '389x15', '254x20', '389x17'])

    def __init__(self, split, flat: bool = False, filter_empty_answers=True):
        assert split in ["train", "val"]
        self.filter_empty_answers = filter_empty_answers
        self.flat = flat
        super().__init__(split)

    def load(self):
        data = json.load(open(resource_path(join(self.data_path, "final_data_feb_16", "json_files", "raw_files", f'new_{self.split}.json')), 'r'))

        if self.flat:
            data_list = []
            for item in data:
                if item['uni_clipped_id'] in self.corrupt_files:
                    continue
                video_path = join(self.video_path, self.split, f"{item['uni_clipped_id']}.mp4")
                answer = item['answer']
                if self.filter_empty_answers and len(answer.strip()) == 0:
                    continue
                question = item['question']
                msg = dict(
                    question=question,
                    answer=answer,
                    style="video_short_answer"
                )
                msgs = [msg]
                formatted_example = {
                    "video"       : video_path,
                    "message_list": msgs,
                    "metadata": dict(
                        decode_method = "av_noseek",
                    )
                }
                data_list.append(formatted_example)
        else:
            data_list = []
            video2msgs = {}
            for item in data:
                if item['uni_clipped_id'] in self.corrupt_files:
                    continue
                video_path = join(self.video_path, self.split, f"{item['uni_clipped_id']}.mp4")
                video2msgs[video_path] = video2msgs.get(video_path, [])
                answer = item['answer']
                if self.filter_empty_answers and len(answer.strip()) == 0:
                    continue
                question = item['question']
                msg = dict(
                    question=question,
                    answer=answer,
                    style="video_short_answer"
                )
                video2msgs[video_path].append(msg)

            for video, msgs in video2msgs.items():

                formatted_example = {
                    "video"       : video,
                    "message_list": msgs,
                    "metadata": dict(
                        decode_method="av_noseek",
                    )
                }
                data_list.append(formatted_example)

        return data_list

    def get(self, idx, rng):
        example = self.data[idx]
        return example


class SUTDTrafficQA(DatasetBase):
    data_path = join(VIDEO_DATA_HOME, "SUTD-TrafficQA")
    video_path = join(VIDEO_DATA_HOME, "SUTD-TrafficQA", "compressed_videos")

    def __init__(self, split, flat: bool = False):
        assert split in ["train", "test"]
        self.flat = flat
        super().__init__(split)

    def load(self):
        tmp = pd.read_json(resource_path(join(self.data_path, f'R2_{self.split}.jsonl')), orient="values", lines=True)

        cols = tmp.iloc[0].tolist()
        data = tmp.iloc[1:].copy()
        data.columns = cols

        data_list = []
        if self.flat:
            for ex_id, row in data.iterrows():
                video_path = join(self.video_path, row['vid_filename'])
                options = [row['option0'], row['option1'], row['option2'], row['option3']]
                question = row['q_body']
                answer = options[row['answer']]
                options = [o for o in options if o]
                answer_idx = options.index(answer)

                assert answer_idx < len(options), f"Answer index {answer_idx} out of bounds for options {options}"
                msg = dict(
                    question=question,
                    answer_idx=answer_idx,
                    options=options,
                    style="video_multiple_choice",
                )
                msgs = [msg]
                formatted_example = {
                    "video"       : video_path,
                    "message_list": msgs
                }
                data_list.append(formatted_example)

        else:
            video2msgs = {}
            for ex_id, row in data.iterrows():
                video_path = join(self.video_path, row['vid_filename'])

                video2msgs[video_path] = video2msgs.get(video_path, [])
                options = [row['option0'], row['option1'], row['option2'], row['option3']]
                question = row['q_body']
                answer = options[row['answer']]
                options = [o for o in options if o]
                answer_idx = options.index(answer)

                assert answer_idx < len(options), f"Answer index {answer_idx} out of bounds for options {options}"
                msg = dict(
                    question=question,
                    answer_idx=answer_idx,
                    options=options,
                    style="video_multiple_choice",
                )
                video2msgs[video_path].append(msg)

            for video, msgs in video2msgs.items():
                formatted_example = {
                    "video"       : video,
                    "message_list": msgs,
                }
                data_list.append(formatted_example)

        return data_list

    def get(self, idx, rng):
        example = self.data[idx]
        return example


class SocialIQ2(DatasetBase):
    data_path = join(VIDEO_DATA_HOME, "social-iq2", "qa")
    video_path = join(VIDEO_DATA_HOME, "social-iq2", "video")

    def __init__(self, split, flat: bool = False):
        assert split in ["train", "val", "test"]
        self.flat = flat
        super().__init__(split)

    def load(self):
        data = pd.read_json(resource_path(join(self.data_path, f'qa_{self.split}.json')), orient="records", lines=True)
        all_videos = set(list_directory(self.video_path))

        data_list = []
        skip_videos = set()
        if self.flat:
            for ex_id, row in data.iterrows():
                video_path = join(self.video_path, f"{row['vid_name']}.mp4")
                if video_path not in all_videos:
                    skip_videos.add(video_path)
                    continue
                options = [row['a0'], row['a1'], row['a2'], row['a3']]
                question = row['q']
                answer_idx = row['answer_idx']
                assert answer_idx < len(options), f"Answer index {answer_idx} out of bounds for options {options}"
                msg = dict(
                    question=question,
                    answer_idx=answer_idx,
                    options=options,
                    style="video_multiple_choice",
                )
                msgs = [msg]
                formatted_example = {
                    "video"       : video_path,
                    "message_list": msgs
                }
                data_list.append(formatted_example)

        else:
            video2msgs = {}
            for ex_id, row in data.iterrows():
                video_path = join(self.video_path, f"{row['vid_name']}.mp4")
                if video_path not in all_videos:
                    skip_videos.add(video_path)
                    continue
                video2msgs[video_path] = video2msgs.get(video_path, [])

                options = [row['a0'], row['a1'], row['a2'], row['a3']]
                question = row['q']
                answer_idx = row['answer_idx']
                assert answer_idx < len(options), f"Answer index {answer_idx} out of bounds for options {options}"
                msg = dict(
                    question=question,
                    answer_idx=answer_idx,
                    options=options,
                    style="video_multiple_choice",
                )
                video2msgs[video_path].append(msg)

            for video, msgs in video2msgs.items():
                formatted_example = {
                    "video"       : video,
                    "message_list": msgs,
                }
                data_list.append(formatted_example)

        if get_global_rank() == 0:
            log.warning(f"Skipped {len(skip_videos)} SocialIQ2 videos that were not found.")
        return data_list

    def get(self, idx, rng):
        example = self.data[idx]
        return example


class EpicKitchens(DatasetBase):
    """Epic Kitchens 100 dataset for short video clip captioning.
    
    Epic Kitchens is a large-scale dataset of egocentric videos in kitchen environments
    with temporal action annotations. Each clip contains narrations describing kitchen
    activities like "open door", "take cup", etc.

    When use_extracted_clips=True (default), it looks for pre-extracted clips in clips_path.
    When use_extracted_clips=False, it uses the original full videos with clip timing metadata.
    """
    data_path = join(VIDEO_DATA_HOME, "epic-kitchens")
    clips_path = join(VIDEO_DATA_HOME, "epic-kitchens-clips")
    corrupt_clips = [
        "P01_102_229.910_234.300.mp4", "P01_102_90.240_93.490.mp4",
        "P01_104_65.170_92.450.mp4", "P02_118_722.700_723.550.mp4",
    ]

    templates = [
        "What action is being performed?",
        "What action is occurring?",
        "What activity does the video depict?"
    ]

    def __init__(
            self,
            split,
            flat: bool = False,
            max_per_video: Optional[int] = None,
            use_extracted_clips: bool = True,
            qa_format: bool = False
        ):
        """
        Args:
            split: Dataset split ('train', 'validation')
            flat: Whether to flatten the dataset (default: False)
            max_per_video: Maximum number of messages per video (default: None)
            use_extracted_clips: Whether to use pre-extracted clips or extract on-the-fly
                from full videos. When True, looks for clips in clips_path first.
        """
        assert split in ["train", "validation"], f"Invalid split: {split}"
        self.flat = flat
        self.max_per_video = max_per_video
        self.use_extracted_clips = use_extracted_clips
        self.qa_format = qa_format
        super().__init__(split)

    def get_video_path(self, participant_id, video_id, start_time, stop_time):
        """Get the appropriate video path, preferring extracted clips when available."""
        if self.use_extracted_clips:
            # Try to find pre-extracted clip
            video_fname = f"{video_id}_{start_time:.3f}_{stop_time:.3f}.mp4"
            if video_fname not in self.corrupt_clips:
                clip_path = os.path.join(
                    self.clips_path,
                    participant_id,
                    "videos",
                    video_fname
                )
                return clip_path, None  # No metadata needed for extracted clips
            else:
                return None, None  # Skip corrupt clips

        # Fall back to original video with metadata
        original_path = os.path.join(
            self.data_path,
            participant_id,
            "videos",
            f"{video_id}.mp4"
        )
        metadata = {
            "clip_start_time": start_time,
            "clip_end_time": stop_time,
        }
        return original_path, metadata

    def load(self):
        annotation_file = os.path.join(
            self.data_path,
            "epic-kitchens-100-annotations",
            f"EPIC_100_{self.split}_with_missing.csv"
        )
        annotation_file = resource_path(dirname(annotation_file), basename(annotation_file))

        if not file_exists(annotation_file):
            raise FileNotFoundError(f"Annotation file not found: {annotation_file}")

        # Read CSV file
        df = pd.read_csv(annotation_file)
        
        data_list = []
        video2msgs = defaultdict(list)
        skipped_clips = 0
        
        for row in df.itertuples(index=False):
            # Parse timestamps
            if row.missing:
                skipped_clips += 1
                continue

            start_time = self._parse_timestamp(row.start_timestamp)
            stop_time = self._parse_timestamp(row.stop_timestamp)
            
            # Get video path and metadata using new method
            participant_id = row.participant_id
            video_id = row.video_id
            video_path, clip_metadata = self.get_video_path(
                participant_id, video_id, start_time, stop_time
            )

            if video_path is None:
                skipped_clips += 1
                continue
            
            # Get the narration (caption)
            narration = row.narration

            if self.qa_format:
                msg = dict(
                    question=random.choice(self.templates),
                    answer=narration,
                    style="epic_kitchens_label"
                )
            else:
                msg = dict(
                    text=narration,
                    style="epic_kitchens_short_caption"
                )

            if self.flat:
                formatted_ex = {
                    "video": video_path,
                    "message_list": [msg]
                }
                # Only add metadata if using original video (not extracted clips)
                if clip_metadata is not None:
                    formatted_ex["metadata"] = clip_metadata
                data_list.append(formatted_ex)
            else:
                # Use a unique key that includes metadata info for grouping
                video_key = (video_path, start_time if clip_metadata else None, stop_time if clip_metadata else None)
                video2msgs[video_key].append(msg)

        log.info(f"Loaded Epic Kitchens clips from {self.split} split")
        if skipped_clips > 0 and get_global_rank() == 0:
            log.warning(f"Skipped {skipped_clips}/{skipped_clips+len(video2msgs)} epic-kitchens clips due to missing video files.")
        
        if not self.flat:
            for video_key, msgs in video2msgs.items():
                video_path, start_time, stop_time = video_key

                if len(msgs) == 0:
                    continue

                # Determine if we need metadata (when using original video)
                clip_metadata = None
                if start_time is not None and stop_time is not None:
                    clip_metadata = {
                        "clip_start_time": start_time,
                        "clip_end_time": stop_time,
                    }

                if self.max_per_video:
                    for msg_group in split_into_groups(msgs, self.max_per_video):
                        formatted_ex = {
                            "video": video_path,
                            "message_list": msg_group,
                        }
                        if clip_metadata is not None:
                            formatted_ex["metadata"] = clip_metadata
                        data_list.append(formatted_ex)
                else:
                    formatted_ex = {
                        "video": video_path,
                        "message_list": msgs,
                    }
                    if clip_metadata is not None:
                        formatted_ex["metadata"] = clip_metadata
                    data_list.append(formatted_ex)
        
        return data_list

    def _parse_timestamp(self, timestamp_str):
        """Parse timestamp string like '00:00:01.089' to seconds."""
        try:
            parts = timestamp_str.split(':')
            hours = int(parts[0])
            minutes = int(parts[1])
            seconds = float(parts[2])
            return hours * 3600 + minutes * 60 + seconds
        except (ValueError, IndexError) as e:
            raise ValueError(f"Invalid timestamp format: {timestamp_str}") from e

    def get(self, item, rng):
        return self.data[item]


class Dream1K(DatasetBase):
    data_path = join(VIDEO_DATA_HOME, "DREAM-1K")

    def __init__(self, split):
        assert split in ["test"]
        super().__init__(split)

    def load(self):
        json_data = pd.read_json(resource_path(join(self.data_path, 'metadata.json')))
        data = []
        for k, ex in json_data.iterrows():
            video_path = join(self.data_path, ex["video_file"])
            example = {
                "video": video_path,
                "message_list": [dict(text="", style="video_long_caption")], # qa_data["aggregated_annotations"]
                "metadata": ex
            }
            data.append(example)
        return data

    def get(self, idx, rng):
        return self.data[idx]


class AcademicTrackingPoints(DatasetBase):
    data_path = join(VIDEO_DATA_HOME, "video_points")
    
    duration_path = join(data_path, "all_subsets/video2duration_academic_tracking_1129.json")
    
    def __init__(self, 
        split: Literal["train", "val"] = "train", 
        subset: Literal["all", "lvvis", "ovis", "burst", "refdavis17", "mevis", "refyoutube"] = "all",
        mode: Union[Literal["point", "count", "point_count", "count_point"], List[str]] = "point",
        flat: bool = False,
        max_points: int = -1,
        point_sort_by: Literal["xy", "yx", None] = None,
        max_seconds: int = -1,
        fake_timestamp_fps: int = None,
        load_clip_times_from_metadata: bool = False,
        fake_fps_candidates: List[float] = None,
        max_raw_duration: int = -1,
    ):
        assert split in ["train", "val"], f"Invalid split: {split}"
        self.split = split
        self.subset2root = {
            "lvvis": join(VIDEO_DATA_HOME, f"LV-VIS/{split}/videos_2fps"),
            "ovis": join(VIDEO_DATA_HOME, "OVIS/videos_2fps"),
            "burst": join(VIDEO_DATA_HOME, f"TAO-Amodal/{split}/videos_2fps"),
            "refdavis17": join(VIDEO_DATA_HOME, f"Ref-DAVIS17/{split}/videos-2fps"),
            "mevis": join(VIDEO_DATA_HOME, f"mevis/MeViS_release/{split}/videos-2fps"),
            "refyoutube": join(VIDEO_DATA_HOME, f"Ref-YT-VOS/{split}/videos-2fps"),
        }
        self.subset = subset
        self.mode = mode
        self.flat = flat
        self.max_points = max_points
        self.point_sort_by = point_sort_by
        self.max_seconds = max_seconds
        # fps for the timestamps associated with the points and used in frame sampling during training
        self.fake_timestamp_fps = fake_timestamp_fps
        # set timestamp step based on the fake fps, default to 0.5s (2fps) if not specified
        self.timestamp_step = 1 / fake_timestamp_fps if fake_timestamp_fps is not None else 0.5
        self.load_clip_times_from_metadata = load_clip_times_from_metadata
        if load_clip_times_from_metadata:
            metadata_path = join(self.data_path, f"all_subsets/academic_points_meta_max_{self.max_seconds}s_1129.json")
            self.clip_metadata = json.load(open(metadata_path))
        self.fake_fps_candidates = fake_fps_candidates
        self.max_raw_duration = max_raw_duration
        # check if only one of fake_timestamp_fps and fake_fps_candidates is provided
        assert self.fake_timestamp_fps is None or self.fake_fps_candidates is None, \
            "Only one of fake_timestamp_fps and fake_fps_candidates should be provided."
        super().__init__(split)
        
    def load(self):
        root_path = join(self.data_path, "all_subsets")
        vid2duration = {}
        if self.max_seconds > 0:
            vid2duration = json.load(open(self.duration_path))
        if self.subset == "all":
            all_dfs = []
            if self.split == "train":
                all_subsets = ["lvvis", "ovis", "burst", "refdavis17", "mevis", "refyoutube"]
            else:
                all_subsets = ["lvvis", "burst"]
            for subset in all_subsets:
                path = f"{subset}_points_{self.split}.parquet"
                data_path = join(root_path, path)
                sub_df = pd.read_parquet(data_path)
                sub_df['subset'] = subset
                all_dfs.append(sub_df)
            df = pd.concat(all_dfs)
        else:
            data_path = join(root_path, f"{self.subset}_points_{self.split}.parquet")
            df = pd.read_parquet(data_path)
            df['subset'] = self.subset

        video2msgs = {}
        formatted_data = []
        total_skipped_examples = 0
        for row in df.itertuples(False):
            subset = row.subset
            root = self.subset2root[subset]
            if subset == "burst":
                if row.dataset not in ["ArgoVerse", "BDD", "Charades", "LaSOT", "YFCC100M"]:
                    continue
                video_path = join(root, row.dataset, f"{row.video_name}.mp4")
            elif subset in ["refdavis17", "mevis", "refyoutube"]:
                video_path = join(root, f"{row.video_name}")
            else:
                video_path = join(root, f"{row.video_name}.mp4")
            if not file_exists(video_path):
                continue
            total_points = sum([len(points) for points in row.points])
            if self.max_points > 0 and total_points > self.max_points:
                continue
            if self.fake_fps_candidates is not None:
                # randomly assign a fake fps from the candidates for this example
                fake_fps = random.choice(self.fake_fps_candidates)
                self.fake_timestamp_fps = fake_fps
                self.timestamp_step = 1 / fake_fps
            ann_start = min(row.timestamps)
            ann_end = max(row.timestamps)
            ann_duration = ann_end - ann_start
            if self.max_seconds > 0 and ann_duration > self.max_seconds:
                continue
            msg = {
                "label": row.category.lower(),
                "answer": str(total_points),
                "count": int(total_points),
                "example_id": f"{video_path.replace(VIDEO_DATA_HOME, '')[1:]}_{row.category.lower()}",
            }
            if self.split != "train":
                # use the same "how many" question template for val/test
                msg["question"] = f"How many \"{msg['label']}\" are there in the video?"
            video_duration = vid2duration.get(video_path, ann_end)
            if self.max_raw_duration > 0 and video_duration > self.max_raw_duration:
                continue
            if self.max_seconds > 0:
                if video_duration <= self.max_seconds:
                    msg["clip_start_time"] = 0.0
                    msg["clip_end_time"] = video_duration
                elif self.load_clip_times_from_metadata:
                    if msg["example_id"] not in self.clip_metadata:
                        logging.warning(f"Example ID {msg['example_id']} not found in clip metadata, skipping.")
                        continue
                    clip_info = self.clip_metadata[msg["example_id"]]
                    msg["clip_start_time"] = clip_info["clip_start_time"]
                    msg["clip_end_time"] = clip_info["clip_end_time"]
                else:
                    rand_start, rand_end = sample_random_clip(
                        video_duration=video_duration,
                        start_time=ann_start,
                        end_time=ann_end,
                        min_seconds=0.5,  # at least one timestamp step
                        max_seconds=self.max_seconds,
                        timestamp_step=0.5,
                        seed=42,
                    )
                    msg["clip_start_time"] = rand_start
                    msg["clip_end_time"] = rand_end
            
                assert msg['clip_start_time'] >= 0, msg['clip_start_time']
                assert msg['clip_end_time'] <= video_duration, (msg['clip_end_time'], video_duration)
                assert msg['clip_start_time'] < msg['clip_end_time'], (msg['clip_start_time'], msg['clip_end_time'])
                assert msg['clip_end_time'] - msg['clip_start_time'] <= self.max_seconds, (msg['clip_end_time'], msg['clip_start_time'], self.max_seconds)
                assert msg['clip_start_time'] <= ann_start, (msg['clip_start_time'], ann_start)
                assert msg['clip_end_time'] >= ann_end, (msg['clip_end_time'], ann_end)
            
            sorted_timestamps = []
            sorted_points = []
            for i, ts in sorted(enumerate(row.timestamps), key=lambda x: x[1]):
                # assuming we use 2fps annotations by default
                assert ts / 0.5 == int(
                    ts / 0.5
                ), f"Original timestamp {ts} not aligned to 0.5s intervals"

                if "clip_start_time" in msg:
                    ts = ts - msg["clip_start_time"]
                # align timestamps to the specified fps intervals
                ts = math.floor(ts / 0.5) * self.timestamp_step           
                sorted_timestamps.append(ts)

                points = list(row.points[i])
                # sort points by x
                if self.point_sort_by == "xy":
                    # sort by x first and then by y
                    points = sorted(points, key=lambda p: (p["x"], p["y"]))
                elif self.point_sort_by == "yx":
                    points = sorted(points, key=lambda p: (p["y"], p["x"]))
                sorted_points.append(points)

            msg["points"] = sorted_points
            msg["timestamps"] = sorted_timestamps

            video2msgs[video_path] = video2msgs.get(video_path, []) + [msg]
            if self.flat or self.max_seconds > 0:
                metadata = {
                    "points": sorted_points,
                    "timestamps": sorted_timestamps,
                    "count": msg["count"],
                    "subset": "object"
                }
                if self.fake_timestamp_fps is not None:
                    metadata["fake_timestamp_fps"] = self.fake_timestamp_fps
                if self.max_seconds > 0:
                    metadata.update({
                        "clip_start_time": msg["clip_start_time"],
                        "clip_end_time": msg["clip_end_time"],
                    })
                msg.update({
                    "video": video_path,
                    "metadata": metadata
                })
                formatted_data.append(msg)
        if not self.flat and self.max_seconds < 0:
            for video_path, msgs in video2msgs.items():
                example = {
                    "video": video_path,
                    "message_list": msgs,
                }
                formatted_data.append(example)
        return formatted_data

    def __len__(self):
        return len(self.data)

    def get(self, idx, rng):
        if isinstance(self.mode, str):
            style = self.mode
        else:
            style = rng.choice(self.mode)
        return set_example_style(self.data[idx], f"video_{style}")


class MotionBenchCaption(DatasetBase):
    """Video Localized Narratives dataset"""

    video_path = join(VIDEO_DATA_HOME, "MotionBench/MotionBenchCaption-train/videos")
    caption_path = join(VIDEO_DATA_HOME, "MotionBench/MotionBenchCaption-train/train.jsonl")
    qa_path = join(VIDEO_DATA_HOME, "MotionBench/MotionBenchCaption-train/motionbench_qa.parquet")

    def __init__(self, split, flat: bool = False):
        assert split in ["train"], f"Invalid split: {split}"
        self.flat = flat
        super().__init__(split)

    def load(self):
        video2qa = {}
        # data = pd.read_parquet(resource_path(self.qa_path))
        # for ex_id, row in data.iterrows():
        #     video_path = join(self.video_path, row['video_path'])
        #     qa_list = row["mc_qa_list"]
        #     msgs = []
        #     for q_id, ex in enumerate(qa_list):
        #         question = ex["Question"]
        #         answer = ex["Answer"]
        #
        #         neg_options = list(ex["NegativeAnswers"])
        #         answer_idx = random.randint(0, len(neg_options))
        #         neg_options.insert(answer_idx, answer)
        #         assert answer_idx < len(neg_options), f"Answer index {answer_idx} out of bounds for options {neg_options}"
        #         msg = dict(
        #             question=question,
        #             answer_idx=answer_idx,
        #             options=neg_options,
        #             style="video_multiple_choice",
        #             category=ex["Category"] if "Category" in ex else None,
        #         )
        #         msgs.append(msg)
        #     video2qa[video_path] = video2qa.get(video_path, []) + msgs


        json_path = self.caption_path
        data = pd.read_json(resource_path(dirname(json_path), basename(json_path)), lines=True, orient='records')
        data_list = []
        for i, row in data.iterrows():
            video_path = join(self.video_path, row['video_path'])
            qa_msgs = video2qa.get(video_path, [])

            msg = dict(
                text=row['motion_caption'],
                style="video_motion_caption"
            )
            formatted_ex = {
                "video"       : video_path,
                "message_list": [msg]+qa_msgs
            }
            data_list.append(formatted_ex)

        return data_list

    def get(self, item, rng):
        return self.data[item]


class VSIBench(Dataset):
    PATH = "nyu-visionx/VSI-Bench"
    MCA_QUESTION_TYPES = [
        "object_rel_direction_easy",
        "object_rel_direction_medium",
        "object_rel_direction_hard",
        "object_rel_distance",
        "route_planning",
        "obj_appearance_order",
    ]
    NA_QUESTION_TYPES = [
        "object_abs_distance",
        "object_counting",
        "object_size_estimation",
        "room_size_estimation",
    ]

    @classmethod
    def download(cls, n_procs=1):
        from huggingface_hub import snapshot_download
        import zipfile
        from pathlib import Path

        logging.info(f"Downloading VSI-Bench...")
        snapshot_download(
            repo_id=cls.PATH,
            repo_type="dataset",
            revision="main",
            local_dir=join(VIDEO_DATA_HOME, "VSI-Bench"),
            local_dir_use_symlinks=False,
        )

        for subset in ["arkitscenes", "scannet", "scannetpp"]:
            zip_path = join(VIDEO_DATA_HOME, "VSI-Bench", f"{subset}.zip")
            extract_dir = join(VIDEO_DATA_HOME, "VSI-Bench")
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(extract_dir)
            Path(zip_path).unlink()
        
        datasets.load_dataset_builder(cls.PATH).download_and_prepare()
    
    def __init__(self, split: str, multi_image: bool = False, max_frames: int = 32, keep_in_memory=False):
        assert split == "test"
        self.multi_image = multi_image
        self.max_frames = max_frames
        self.split = split
        self.dataset = datasets.load_dataset(
            self.PATH, split=split, keep_in_memory=keep_in_memory
        )
    
    def __len__(self):
        return len(self.dataset)
    
    def sample_frames(self, video_path: str):
        from torchcodec.decoders import VideoDecoder
        video_dec = VideoDecoder(video_path, seek_mode="exact", num_ffmpeg_threads=1)
        n_frames = video_dec.metadata.num_frames
        indices = np.linspace(
            0,
            n_frames-1,
            num=min(self.max_frames, n_frames),
            endpoint=True,
            dtype=np.int32
        )
        frames = video_dec.get_frames_at(indices).data.numpy().transpose(0, 2, 3, 1)
        frames = [frame for frame in frames]
        return frames
    
    def qo_template(self, question: str, question_type: str, options: Optional[List[str]] = None):
        if question_type in self.NA_QUESTION_TYPES:
            prompt = "\n".join([question, "Do not response anything other than a single number!"])
        else:
            prompt = "\n".join(
                [
                    question,
                    "Options:",
                    "\n".join(options),
                    "Answer with the option's letter from the given choices directly.",
                ]
            )
            options = [opt[2:].lstrip() for opt in options]
        return prompt, options
    
    def get(self, item, rng):
        ex = self.dataset[item]
        video_path = join(
            VIDEO_DATA_HOME,
            "VSI-Bench",
            ex["dataset"],
            ex["scene_name"] + ".mp4",
        )

        prompt, options = self.qo_template(ex["question"], ex["question_type"], ex["options"])
        out = dict(
            question=prompt,
            answer=ex["ground_truth"],
            metadata=dict(
                example_id=ex["id"],
                question_type=ex["question_type"],
            )
        )

        if self.multi_image:
            out["image"] = self.sample_frames(video_path)
            style = "eval_multi_image_mc" if options is not None else "eval_multi_image_da"
        else:
            out["video"] = video_path
            style = "video_eval_multiple_choice" if options is not None else "video_eval_short_answer"
        
        out["style"] = style
        if options is not None:
            out["metadata"]["options"] = options
        
        return out


class LongText(DatasetBase):
    data_path = join(VIDEO_DATA_HOME, "LongText/longalign_longreward_en_toklen.parquet")

    def __init__(self, split, seq_len: int = 16000):
        assert split in ["train"], f"Invalid split: {split}"
        self.seq_len = seq_len
        super().__init__(split)

    def load(self):
        data = pd.read_parquet(resource_path(self.data_path))
        data_list = []
        for ex_id, row in data.iterrows():
            if row['length'] > self.seq_len:
                continue

            messages = [
                row['prompt'],
                row['response']
            ]

            data_list.append(
                dict(
                    message_list=[dict(messages=messages, style="text_sft")],
                    metadata=dict(
                        example_id=ex_id,
                        source=row["source"],
                    )
                )
            )

        return data_list

    def __len__(self):
        return len(self.data)

    def get(self, item, rng):
        return self.data[item]


if __name__ == "__main__":
    # dataset = CLEVRER(split)
    # print(f"Total samples: {len(dataset)}")
    # dataset = STAR(split)
    # print(f"Total samples: {len(dataset)}")
    # dataset = FunQA(split)
    # print(f"Total samples: {len(dataset)}")
    # dataset = TGIF(split)
    # print(f"Total samples: {len(dataset)}")
    # ds = TempCompass(split="validation", task="captioning")
    # splits = ["train"] # , "val"]
    # for split in splits:
    #     for dataset_cls in [FunQA,]: #TGIF, CLEVRER ,STAR,IntentQA,  VideoLocalizedNarratives, RoadTextVQA
    #         # 10k*14 -> 20k, 3*30 -> 10k, 4*50 -> 22k, 60*3.5, 3*8, 7.5*4, 2*4
    #         for flat in [False]: # , True
    #             # for answer_type in ["open_ended", "multi_choice", "all"]:
    #             print(f"Loading {dataset_cls.__name__} with split {split} and flat={flat}")
    #             ds = dataset_cls(split, flat=flat, answer_type="multi_choice") #   max_per_video=10
    #             print(f"Total samples: {len(ds)}")
    #             # ds = dataset_cls(split, flat=flat, include_multiple_correct=True) #  answer_type="multi_choice" max_per_video=10
    #             # print(f"Total samples: {len(ds)}")
    # ds = AcademicTrackingPoints(split="train", max_points=60, mode="point_count", max_seconds=63, load_clip_times_from_metadata=True)
    ds = AcademicTrackingPoints(split="val", max_points=60, subset="all", mode="point_count", flat=True, max_seconds=63, load_clip_times_from_metadata=True)
    print(len(ds))
    breakpoint()
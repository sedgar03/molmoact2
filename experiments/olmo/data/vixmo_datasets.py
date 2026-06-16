import os
import json
import pickle
import random
import math
import pandas as pd
from os.path import join, exists
from olmo.data.dataset import DatasetBase, VIDEO_DATA_HOME, Dataset
from olmo.util import split_into_groups, resource_path, set_example_style
from olmo.torch_util import get_global_rank
from typing import Literal, Optional, Tuple, List
import numpy as np
import logging
from collections import defaultdict

log = logging.getLogger(__name__)


def seconds_to_timestamp(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = seconds % 60  # Keep decimal
    # keep two decimal places for seconds by default
    formatted = f"{hours:02}:{minutes:02}:{seconds:05.2f}"
    return formatted


class VixMoClippedCaption(DatasetBase):
    """Caption clipped videos"""

    corrupt_videos = set([
        "intern/batch_1/-6xu3wJTRJk.mkv",
        "intern/batch_1/34cuCnucXhE.mkv",
        "intern/batch_1/iQ9Rww7IyAM.mkv",
        "intern/batch_1/kpYkcHQ1yxg.mkv",
        "intern/batch_1/qrgnGsVYwLM.mkv",
        "intern/batch_1/r_qmz7g0Bwc.mkv",
        "intern/batch_1/hHsqxtMdsLA.mkv",
        "intern/batch_1/m7FDcgPjd34.mkv",
        "intern/batch_1/BQD5d6XFG7M.mkv",
        "intern/batch_1/fe51NdK6ASI.mkv",
        "LLaVA-Video-178K/1_2_m_academic_v0_1/academic_source/activitynet/v1-3/train_val/v_fwwo0GsYB7c.mp4",
        "LLaVA-Video-178K/0_30_s_youtube_v0_1/liwei_youtube_videos/videos/youtube_video_2024/ytb_2vJKcdVyjLk.mp4",
    ])
    data_path = os.path.join(VIDEO_DATA_HOME, "video_captions")

    def __init__(
        self,
        split: Literal["train", "val"],
        subset: Literal["all", "llava_10k", "koala_20k", "intern_9k"] = "all",
        min_clip_length: int = 1,
        max_clip_length: int = 1,
        parts: Literal["both", "caption", "transcript"] = "both",
        min_score: int = 0,
        version: Literal["v1", "v2", "v3", "v2_filtered"] = "v2",
        skip_complete_caption=False
    ):
        assert split in ["train", "validation"], f"Invalid split: {split}"
        assert subset in ["all", "llava_10k", "koala_20k", "intern_9k"], f"Invalid subset: {subset}"
        self.subset = subset
        self.min_clip_length = min_clip_length
        self.max_clip_length = max_clip_length
        self.skip_complete_caption = skip_complete_caption
        self.parts = parts
        self.version = version
        self.min_score = min_score
        super().__init__(split)

    def load(self):
        if self.subset == "all":
            if self.version == "v1":
                all_data = pd.read_parquet(resource_path(self.data_path, "vixmo_captions.parquet"))
            elif self.version in ["v2", "v2_filtered"]:
                all_data = pd.read_parquet(resource_path(self.data_path, "vixmo_captions_v2.parquet"))
            elif self.version == "v3":
                all_data = pd.read_parquet(resource_path(self.data_path, "vixmo_captions_v3_w_ann_score.parquet"))
            else:
                raise NotImplementedError(self.version)
        else:
            all_data = pd.read_parquet(resource_path(os.path.join(self.data_path, "all_subsets"), f"video_captions_{self.subset}_split.parquet"))
        if self.version == "v2_filtered":
            tmp = pd.read_parquet(resource_path(self.data_path, "vixmo_captions_v3_w_ann_score.parquet"), columns=["video_path"])
            keep = set(tmp["video_path"])
            all_data = all_data[all_data["video_path"].isin(keep)]

        split = self.split
        if split == "validation":
            split = "val"
        data = all_data[all_data["split"] == split]

        data_list_format = []
        for example in data.itertuples():
            clip_timestamps = example.clip_timestamps
            if example.video_path in self.corrupt_videos:
                continue
            if "annotation_score" in example and example["annotation_score"] < self.min_score:
                continue
            if len(clip_timestamps) < self.min_clip_length:
                continue
            if self.skip_complete_caption and (len(clip_timestamps) == self.min_clip_length):
                continue
            clip_captions = example.clip_captions
            clip_transcripts = example.clip_transcripts
            data_list_format.append((example.video_path, clip_timestamps, clip_captions, clip_transcripts))
        return data_list_format

    def get(self, idx, rng):
        video, timestamps, caps, transcripts = self.data[idx]
        start = rng.randint(0, len(timestamps) - self.min_clip_length + 1)
        max_end = start+self.max_clip_length
        if self.skip_complete_caption and start == 0:
            max_end = min(max_end, len(transcripts))
        else:
            max_end = min(max_end, len(transcripts)+1)
        end = rng.randint(start+1, max_end)  # exclusive
        messages = []
        if "both" in self.parts or "caption" in self.parts:
            messages.append(dict(
                text="\n\n".join(caps[start:end]),
                style="clip_caption"
            ))
        if "both" in self.parts or "transcript" in self.parts:
            messages.append(dict(
                text="\n\n".join(transcripts[start:end]),
                style="clip_transcript"
            ))
        return dict(
            video=join(VIDEO_DATA_HOME, video),
            metadata=dict(
                clip_start_time=timestamps[start][0],
                clip_end_time=timestamps[end-1][1],
            ),
            message_list=messages
        )


class VixMoCaptions(DatasetBase):
    version2style = {
        "merged_caption": "video_merged_caption",
        "video_caption": "video_short_caption",
        "video_transcript": "video_transcript",
        "video_image_merged_caption": "video_long_caption"
    }

    data_path = os.path.join(VIDEO_DATA_HOME, "video_captions")
    corrupt_videos = set([
        "intern/batch_1/-6xu3wJTRJk.mkv",
        "intern/batch_1/34cuCnucXhE.mkv",
        "intern/batch_1/iQ9Rww7IyAM.mkv",
        "intern/batch_1/kpYkcHQ1yxg.mkv",
        "intern/batch_1/qrgnGsVYwLM.mkv",
        "intern/batch_1/r_qmz7g0Bwc.mkv",
        "intern/batch_1/hHsqxtMdsLA.mkv", 
        "intern/batch_1/m7FDcgPjd34.mkv", 
        "intern/batch_1/BQD5d6XFG7M.mkv", 
        "intern/batch_1/fe51NdK6ASI.mkv",
        "youtube_temporal/batch_1/0OfUqrbRa2Y.mp4",
        "youtube_temporal/batch_1/mTXBQzptWAg.mp4",
        "youtube_temporal/batch_1/NyuA4FatDQk.mp4",
        "LLaVA-Video-178K/1_2_m_academic_v0_1/academic_source/activitynet/v1-3/train_val/v_fwwo0GsYB7c.mp4",
        "LLaVA-Video-178K/0_30_s_youtube_v0_1/liwei_youtube_videos/videos/youtube_video_2024/ytb_2vJKcdVyjLk.mp4",
    ])

    def __init__(
            self, 
            split: Literal["train", "val"],
            subset: Literal["all", "llava_10k", "koala_20k", "intern_9k"] = "all",
            include_video_transcript: bool = False,
            include_video_caption: bool = False,
            include_merged_caption: bool = False,
            include_video_image_merged_caption: bool = False,
            n_clip_captions: int = 0,
            n_clip_transcripts: int = 0,
            n_frame_captions: int = 0,
            expanded_clip_captions: int = 0,
            included_expanded_clip_transcripts: bool = False,
            min_score: int = 0,
            clip_video: bool = False,
            max_caption_per_video: int = 4,
            weight: float = None,
            version: Literal["v1", "v2", "v3", "v2_filtered"] = "v2",
        ):
        assert split in ["train", "validation", "val"], f"Invalid split: {split}"
        """
        Each video will be annotated with:
        
        include_video_transcript: Include human transcript after seeing the full video
        include_video_caption: Include human re-written caption after seeing the full video
        include_merged_caption: Include caption build from all clip-captions
        include_video_image_merged_caption: Caption built from clip and frame captions
        n_clip_captions: Include n randomly selected clip captions
        n_clip_transcripts: Include n randomly selected clip transcripts
        n_frame_captions: Include n randomly selected frame captions
        expanded_clip_captions: Include concatenated captions for sequences of clips of length
                                `expanded_clip_captions`
        included_expanded_clip_transcripts: Include concatenated clips transcripts for sequences of 
                                            clips of length `expanded_clip_captions`                 
        """
        # subset of video sources to include, or include all video sources
        assert subset in ["all", "llava_10k", "koala_20k", "intern_9k"], f"Invalid subset: {subset}"
        self.subset = subset
        # the version of video captions to use
        self.expanded_clip_captions = expanded_clip_captions
        self.clip_video = clip_video
        self.include_merged_caption = include_merged_caption
        self.include_video_caption = include_video_caption
        self.include_video_image_merged_caption = include_video_image_merged_caption
        self.n_clip_captions = n_clip_captions
        self.n_clip_transcripts = n_clip_transcripts
        self.n_frame_captions = n_frame_captions
        self.include_video_transcript = include_video_transcript
        self.max_caption_per_video = max_caption_per_video
        self.included_expanded_clip_transcripts = included_expanded_clip_transcripts
        self.weight = weight
        self.version = version
        self.min_score = min_score

        if not include_video_image_merged_caption:
            self.version2style['merged_caption'] = "video_long_caption"

        assert any([
            include_merged_caption,
            include_video_caption,
            include_video_image_merged_caption,
            n_clip_captions,
            n_frame_captions,
            expanded_clip_captions,
            include_video_transcript,
        ]), "No inputs were specified"

        super().__init__(split)

    def load(self):
        if self.subset == "all":
            if self.version == "v1":
                all_data = pd.read_parquet(resource_path(self.data_path, "vixmo_captions.parquet"))
            elif self.version in ["v2", "v2_filtered"]:
                all_data = pd.read_parquet(resource_path(self.data_path, "vixmo_captions_v2.parquet"))
            elif self.version == "v3":
                all_data = pd.read_parquet(resource_path(self.data_path, "vixmo_captions_v3_w_ann_score.parquet"))
            else:
                raise NotImplementedError(self.version)
        else:
            all_data = pd.read_parquet(resource_path(os.path.join(self.data_path, "all_subsets"), f"video_captions_{self.subset}_split.parquet"))
        if self.version == "v2_filtered":
            tmp = pd.read_parquet(resource_path(self.data_path, "vixmo_captions_v3_w_ann_score.parquet"), columns=["video_path"])
            keep = set(tmp["video_path"])
            all_data = all_data[all_data["video_path"].isin(keep)]

        split = self.split
        if split == "validation":
            split = "val"
        data = all_data[all_data["split"] == split]

        cap_versions = []
        if self.include_merged_caption:
            cap_versions.append("merged_caption")
        if self.include_video_caption:
            cap_versions.append("video_caption")
        if self.include_video_transcript:
            cap_versions.append("video_transcript")
        if self.include_video_image_merged_caption:
            cap_versions.append("video_image_merged_caption")

        data_list_format = []
        for example in data.itertuples():
            example = example._asdict()
            if example["video_path"] in self.corrupt_videos:
                continue
            if "annotation_score" in example and example["annotation_score"] < self.min_score:
                continue
            messages = []
            for version in cap_versions:
                caption = example[version]
                if caption is None:
                    continue
                message = dict(
                    text=caption,
                    style=self.version2style[version],
                )
                messages.append(message)

            if self.expanded_clip_captions:
                # include clip-level captions with clip timestamps and captions
                for c in range(1, self.expanded_clip_captions+1):
                    last_start = len(example["clip_timestamps"]) - c + 1
                    # if there are not enough clips to expand to c clips, skip
                    if last_start <= 0:
                        continue

                    for i in range(0, last_start, 1):
                        clip_start_time = example["clip_timestamps"][i][0]
                        # start of next clip (e.g. i+c) == end of previous clip (i+c-1)
                        clip_end_time = example["clip_timestamps"][i+c-1][1]
                        if self.included_expanded_clip_transcripts:
                            clip_cap_versions = ["clip_captions", "clip_transcripts"]
                        else:
                            clip_cap_versions = ["clip_captions"]

                        for clip_cap_version in clip_cap_versions:
                            caption = "\n".join(example[clip_cap_version][i:i+c])
                            message = dict(
                                text=caption,
                                style="video_clip_caption_start_end" if clip_cap_version != "clip_transcripts" else "video_clip_transcript_start_end",
                                start_time=clip_start_time,
                                end_time=clip_end_time,
                            )
                            messages.append(message)

            # placeholder to indicate we should randomly sample some of these elements in `get`
            # We sample so we get different elements each epoch
            # Note we always merge these samples so we can ensure we always sample different
            # captions each epoch if sampling multiple captions
            if self.n_frame_captions:
                messages.append(dict(style="frame_captions"))
            if self.n_clip_captions:
                messages.append(dict(style="clip_captions"))
            if self.n_clip_transcripts:
                messages.append(dict(style="clip_transcripts"))

            if len(messages) == 0:
                # Might occur since some videos are missing `video_caption`/`video_transcript`
                continue

            if self.max_caption_per_video:
                for msg in split_into_groups(messages, self.max_caption_per_video):
                    data_list_format.append((msg, example))
            else:
                data_list_format.append((messages, example))

        return data_list_format

    def get(self, idx, rng: np.random.RandomState):
        messages, example = self.data[idx]
        frame_captions_ix = None
        post_processed_messages = []
        for message in messages:
            if message["style"] == "frame_captions":
                n_captions = len(example["frame_captions"])
                for ix in rng.choice(n_captions, min(self.n_frame_captions, n_captions), replace=False):
                    post_processed_messages.append(dict(
                        text=example["frame_captions"][ix],
                        timestamp=example["frame_timestamps"][ix],
                        style="video_frame_caption_timestamp",
                    ))
            elif message["style"] in ["clip_captions", "clip_transcripts"]:
                n_captions = len(example["clip_timestamps"])
                if message["style"] == "clip_captions":
                    n = self.n_clip_captions
                    captions = example["clip_captions"]
                    style = "caption"
                else:
                    n = self.n_clip_transcripts
                    captions = example["clip_transcripts"]
                    style = "transcript"
                for ix in rng.choice(len(captions), min(n, len(captions)), replace=False):
                    post_processed_messages.append(dict(
                        text=captions[ix],
                        start_time=example["clip_timestamps"][ix][0],
                        end_time=example["clip_timestamps"][ix][1],
                        style=f"video_clip_{style}_start_end",
                    ))
            else:
                post_processed_messages.append(message)
        video_path = os.path.join(VIDEO_DATA_HOME, example["video_path"])
        return {
            "video": video_path,
            "metadata": dict(
                video_path=video_path,
                example_id=idx,
                category=example["category"] if "category" in example else None,
            ),
            "message_list": post_processed_messages,
            "weight": self.weight
        }



class VixMoCaptionsEval(DatasetBase):
    data_path = join(VIDEO_DATA_HOME, "video_captions")

    def __init__(self, split, version='v1'):
        assert split in ["test"]
        self.version = version
        super().__init__(split)

    def load(self):
        if self.version == 'v1':
            json_data_wg = json.load(open(resource_path(join(self.data_path, "vixmo_captions_test_0905_wg.json")), "r"))
            json_data_ng = json.load(open(resource_path(join(self.data_path, "vixmo_captions_test_0905_ng.json")), "r"))
            data = []
            for k, qa_data in json_data_wg.items():
                video_path = join(VIDEO_DATA_HOME, qa_data["video"])
                example = {
                    "video": video_path,
                    "style": "video_long_caption",
                    "metadata": {
                        "data_with_gemini": qa_data,
                        "data_without_gemini": json_data_ng[k],
                        "video_path": video_path,
                    }
                }
                data.append(example)
        else:

            json_data = json.load(open(resource_path(join(self.data_path, "vixmo_captions_test_1013_gpt5.json")), "r"))
            data = []
            for k, qa_data in json_data.items():
                video_path = join(VIDEO_DATA_HOME, qa_data["video"])
                example = {
                    "video"       : video_path,
                    "style": "video_long_caption",
                    "metadata"    : {
                        "data"   : qa_data,
                        "video_path"         : video_path,
                    }
                }
                data.append(example)

        return data

    def get(self, idx, rng):
        return self.data[idx]


class VixMoSynClipCaptions(DatasetBase):
    data_path = os.path.join(VIDEO_DATA_HOME, "video_captions", "syn_clip_caption")
    # filtered here mean gpt filtered captions
    filtered = ['exist_203k', 'kw_765k', 'temporal_1392k']  # 2.3M
    unfiltered = ['exist', 'kw', 'mammalnet', 'temporal_0', 'temporal_1']  # 3278892
    corrupted = set([
        'youtube-cc/youtube-cc-temporal/52OCh5Oy9OY/52OCh5Oy9OY.mp4',
        'youtube-cc/youtube-cc-kw/tSTGHrd83O4/tSTGHrd83O4.mp4'
    ])

    def __init__(
            self,
            split: str = "train",
            subset: str = "filtered",
    ):
        assert split in ["train"], f"Invalid split: {split}"
        """
        """
        # subset of video sources to include, or include all video sources
        assert subset in ["filtered", "unfiltered"], f"Invalid subset: {subset}"
        self.subset = subset

        super().__init__(split)

    def load(self):
        files = []
        if self.subset == "filtered":
            for name in self.filtered:
                files.append(resource_path(self.data_path, f"{name}_gpt_filter_clip_caption.parquet"))
        else:
            for name in self.unfiltered:
                files.append(resource_path(self.data_path, f"{name}_syn_clip_caption.parquet"))

        data_list = []
        for f in files:
            data = pd.read_parquet(f)

            for example in data.itertuples(False):
                example = example._asdict()
                if example["video_path"] in self.corrupted:
                    continue
                video_path = os.path.join(VIDEO_DATA_HOME, example["video_path"])

                for clip_timestamp, clip_caption in zip(example['clip_timestamps'], example['clip_captions']):
                    if isinstance(clip_timestamp, str):
                        # convert string to tuple of floats
                        s, e = clip_timestamp.split(':')
                        start, end = float(s), float(e)
                    else:
                        start, end = clip_timestamp
                    msgs = [
                        dict(
                            text=clip_caption,
                            style='video_long_caption',
                        )
                    ]
                    formatted_example = {
                        "video"       : video_path,
                        "metadata"    : dict(
                            example_id=f"vixmo_syn_clip_cap_{example['video_path']}",
                            clip_start_time=start,
                            clip_end_time=end,
                        ),
                        "message_list": msgs
                    }
                    data_list.append(formatted_example)
        return data_list

    def get(self, idx, rng):
        example = self.data[idx]
        return example

    def __len__(self):
        return len(self.data)


class VixMoSynVideoCaptions(DatasetBase):
    data_path = os.path.join(VIDEO_DATA_HOME, "video_captions", "syn_video_caption")
    subsets = ['exist', 'kw', 'temporal']
    corrupted = set([
        'youtube-cc/youtube-cc-temporal/52OCh5Oy9OY/52OCh5Oy9OY.mp4',
        'youtube-cc/youtube-cc-kw/tSTGHrd83O4/tSTGHrd83O4.mp4'
    ])

    def __init__(
            self,
            split: str = "train",
            subset: str = "all",
            version: str = "v1",
    ):
        assert split in ["train"], f"Invalid split: {split}"
        self.subset = subset
        assert version in ["v1", "v2", "v3"], f"Invalid version: {version}"
        self.version = version

        super().__init__(split)

    def load(self):
        files = []
        if self.version == 'v3':
            files = [resource_path(self.data_path, f"syn_video_cap_v3.parquet")]
        else:
            if self.subset == "all":
                for name in self.subsets:
                    if self.version == "v1":
                        files.append(resource_path(self.data_path, f"{name}_gpt_merged_video_caption.parquet"))
                    else:
                        files.append(resource_path(self.data_path, f"{name}_gpt_merged_video_caption_v2.parquet"))
            else:
                if self.version == "v1":
                    files.append(resource_path(self.data_path, f"{self.subset}_gpt_merged_video_caption.parquet"))
                else:
                    files.append(resource_path(self.data_path, f"{self.subset}_gpt_merged_video_caption_v2.parquet"))

        data_list = []
        for f in files:
            data = pd.read_parquet(f)

            for example in data.itertuples(False):
                if example.video_path in self.corrupted:
                    continue
                video_path = os.path.join(VIDEO_DATA_HOME, example.video_path)

                msgs = [
                    dict(
                        text=example.video_caption,
                        style='video_long_caption',
                    )
                ]
                formatted_example = {
                    "video"       : video_path,
                    "message_list": msgs
                }
                data_list.append(formatted_example)
        return data_list

    def get(self, idx, rng):
        example = self.data[idx]
        return example

    def __len__(self):
        return len(self.data)


class VixMoCaptionsQA(DatasetBase):
    captions_path = join(VIDEO_DATA_HOME, "video_captions")
    data_path = join(VIDEO_DATA_HOME, "LLM-QA")

    def __init__(
        self, 
        split: Literal["train", "val"],
        subset: Literal["llava_10k", "all"] = "all",
        flatten: bool = False, 
        answer_type: Literal["multi_choice", "open_ended", "all"] = "all", 
        format: Literal["all", "answer_last", "answer_first", "short_answer", "long_answer", "random"] = "random"
    ):
        assert split in ["train", "val"], f"Invalid split: {split}"
        assert subset in ["llava_10k", "all"], f"Invalid subset: {subset}"
        # each video has 3 qas for perception_mc, perception_oe, reasoning_mc, and reasoning_oe
        # mc has 3 answer formats (answer_first, answer_last, short_answer) 
        # oe has 2 answer formats (long_answer, short_answer)
        assert answer_type in ["multi_choice", "open_ended", "all"], f"Invalid answer type: {answer_type}"
        assert format in ["all", "answer_last", "answer_first", "short_answer", "long_answer", "random"], f"Invalid format: {format}"
        self.subset = subset
        self.flatten = flatten
        self.answer_type = answer_type
        self.format = format
        super().__init__(split)

    def load(self):
        # get the videos in a particular split
        if self.subset == "all":
            cap_file = resource_path(join(self.captions_path, "vixmo_captions_v2.parquet"))
            qa_path = resource_path(join(self.data_path, "human_38k", f"all_qas.json"))
        else:
            cap_file = resource_path(join(self.captions_path, "all_subsets", f"video_captions_{self.subset}_split.parquet"))
            qa_path = resource_path(join(self.data_path, self.subset, f"all_qas.json"))
        captions = pd.read_parquet(cap_file)
        
        all_qas = json.load(open(qa_path, "r"))
        videos = captions[captions["split"] == self.split]['video_path'].tolist()
        data_list = []
        for video, qas in all_qas.items():
            full_video_path = video
            # if video.startswith("intern") or video.startswith("koala"):
            #     full_video_path = video
            # else:
            #     # add the llava dir for llava videos
            #     full_video_path = join("LLaVA-Video-178K", video)
            if full_video_path not in videos:
                continue
            
            messages = []
            for q_id, qa in enumerate(qas):
                if self.answer_type == "multi_choice" and not qa["type"].endswith("mc"):
                    continue
                if self.answer_type == "open_ended" and not qa["type"].endswith("oe"):
                    continue
                
                # select only questions of a particular answer format
                if self.format not in ['all', 'random'] and qa['format'] != self.format:
                    continue
                
                if self.format == "random":
                    # sample a random type from 3 types for mc and 2 types for oe
                    prob = 0.33 if qa["type"].endswith("mc") else 0.5
                    if random.random() > prob:
                        continue

                question = qa["question"]
                answer = qa["answer"]
                message = dict(
                    question=question,
                    answer=answer,
                    # style="video_multiple_choice" if qa["type"] == "mc" else "video_short_answer"
                    style="plain"
                )
                example = {
                    "video": join(VIDEO_DATA_HOME, full_video_path),
                    "metadata": dict(
                        example_id=f"{video}-{q_id}",
                        category=qa["category"],
                        format=qa["format"],
                        question_type=qa["type"],
                    ),
                    "message_list": [
                        message
                    ]
                }
                messages.append(message)
                if self.flatten:
                    data_list.append(example)

            if not self.flatten:
                if len(messages) > 0:
                    example = {
                        "video": join(VIDEO_DATA_HOME, full_video_path),
                        "metadata": dict(
                            example_id=video,
                        ),
                        "message_list": messages,
                    }
                    data_list.append(example)
        return data_list
    
    def get(self, idx, rng):
        example = self.data[idx]
        return example


class VixMoCaptionsQA2(DatasetBase):
    data_path = join(VIDEO_DATA_HOME, "LLM-QA")
    SKIP_VIDEO = {
        "youtube_temporal/batch_1/ImRn2eqmU58.mp4",
        "youtube_temporal/batch_1/Lvg6fb0DXSY.mp4"
    }
    # These videos have an audio stream > video stream and annotations for times where
    # the video stream has already ended. This break the video loaders, and its not clear what
    # the right thing to do that case is anyway, so we just skip them

    def __init__(
            self,
            split,
            answer_type: Literal["mc", "oe"] = "mc",
            subset: Literal["count", "motion", "order", "all"] = "all",
            exclude_counting: bool = False,
    ):
        assert answer_type in ["mc", "oe"], f"Invalid answer type: {answer_type}"
        self.answer_type = answer_type
        self.subset = subset
        self.exclude_counting = exclude_counting
        super().__init__(split)

    def load(self):

        subsets = []
        if self.subset in ['count','all'] and not self.exclude_counting:
            subsets += ['human_video_cap_gpt_count.parquet']
        if self.subset in ['motion','all']:
            subsets += ['human_video_cap_gpt_motion.parquet']
        if self.subset in ['order','all']:
            subsets += ['human_video_cap_gpt_order.parquet']
        files = [join(self.data_path, "human_video_cap_qa", name) for name in subsets]

        video2msg = defaultdict(list)
        for f in files:
            data = pd.read_parquet(resource_path(f))
            for row in data.itertuples(False):
                if row.video_path in self.SKIP_VIDEO:
                    continue
                video_path = os.path.join(VIDEO_DATA_HOME, row.video_path)
                qa_list = row.qa_list
                clip_timestamps = row.clip_timestamps
                for q_id, ex in enumerate(qa_list):
                    if len(clip_timestamps) == 1:
                        clip_start_time, clip_end_time = None, None
                    else:
                        clip_ids = ex['ClipIDs']
                        s, e = min(clip_ids), max(clip_ids)
                        if s < 0 or e >= len(clip_timestamps):
                            continue
                        clip_start_time = clip_timestamps[s][0]
                        clip_end_time = clip_timestamps[e][1]

                    question = ex["Question"]
                    answer = ex["Answer"]

                    if self.answer_type == 'mc':
                        neg_options = list(ex["NegativeAnswers"])
                        answer_idx = random.randint(0, len(neg_options))
                        neg_options.insert(answer_idx, answer)
                        assert answer_idx < len(neg_options), f"Answer index {answer_idx} out of bounds for options {neg_options}"
                        msg = dict(
                            question=question,
                            answer_idx=answer_idx,
                            options=neg_options,
                            style="video_multiple_choice",
                            category=ex["Category"] if "Category" in ex else None,
                        )
                    else:
                        msg = dict(
                            question=question,
                            answer=str(answer),
                            style="video_short_answer",
                            category=ex["Category"] if "Category" in ex else None,
                        )
                    video2msg[(video_path, clip_start_time, clip_end_time)].append(msg)

        data_list = []
        for (video_path, clip_start_time, clip_end_time), msgs in video2msg.items():
            if len(msgs) == 0: continue
            if clip_start_time is None and clip_end_time is None:
                formatted_example = {
                    "video"       : video_path,
                    "metadata"    : dict(
                        example_id=f"vixmo_cap_qa_{os.path.basename(video_path)}_{clip_start_time}_{clip_end_time}",
                    ),
                    "message_list": msgs
                }
            else:
                formatted_example = {
                    "video"       : video_path,
                    "metadata"    : dict(
                        example_id=f"vixmo_cap_qa_{os.path.basename(video_path)}_{clip_start_time}_{clip_end_time}",
                        clip_start_time=clip_start_time,
                        clip_end_time=clip_end_time,
                    ),
                    "message_list": msgs
                }
            data_list.append(formatted_example)

        return data_list

    def get(self, idx, rng):
        example = self.data[idx]
        return example

    def __len__(self):
        return len(self.data)



class VixMoHumanQA(DatasetBase):
    data_path = join(VIDEO_DATA_HOME, "video_humanqa")

    def __init__(self, split):
        super().__init__(split)

    def load(self):

        data = pd.read_parquet(resource_path(join(self.data_path, 'human_qa_filtered_1101.parquet')))
        data_list = []
        for row in data.itertuples():
            video_path = os.path.join(VIDEO_DATA_HOME, row.video_path)
            qa_list = row.qa_list
            messages = []
            for q_id, ex in enumerate(qa_list):
                question = ex["question"]
                answer = ex["answer"]
                if answer.strip():
                    messages.append(dict(question=question.strip(), answer=answer.strip(), style="user_qa"))

            if messages:
                formatted_example = {
                    "video"       : video_path,
                    "message_list": messages
                }
                data_list.append(formatted_example)

        return data_list

    def get(self, idx, rng):
        example = self.data[idx]
        return example

    def __len__(self):
        return len(self.data)


class Molmo2HumanEval(Dataset):
    SRC = "video-pref-eval/sample_for_human_eval_agg_v2.json"

    def __init__(self, split, cap_prompt=None, task=None):
        assert split in ["test"]
        self.cap_prompt = cap_prompt
        self.split = split
        with open(resource_path(join(VIDEO_DATA_HOME, self.SRC))) as f:
            self.data = json.load(f)
        if task:
            self.data = [ex for ex in self.data if ex["task"] == task]

    def __len__(self):
        return len(self.data)

    def get(self, item, rng):
        ex = self.data[item]
        task = ex["task"]
        question = ex["question"]
        if task == "caption":
            if self.cap_prompt:
                question = self.cap_prompt
        return dict(
            style="user_qa",
            video=join(VIDEO_DATA_HOME, ex["video_path"]),
            question=question,
            metadata=dict(id=ex["id"], task=ex["task"])
        )


class VixMoSynCaptionsQA(DatasetBase):
    data_path = join(VIDEO_DATA_HOME, "LLM-QA")

    def __init__(
            self,
            split,
            version: str = "v1",
            exclude_counting: bool = False,
    ):
        self.version = version
        self.exclude_counting = exclude_counting
        assert version in ["v1", "v2", "v3", "long", "long2", "long3"], f"Invalid version: {version}"
        super().__init__(split)

    def load(self):

        data_list = []
        subsets = ['exist']
        if self.version == "v1":
            subsets += ['kw', 'temporal']
        else:
            subsets += ['kw2', 'temporal2']
        files = [join(self.data_path, "syn_video_cap_qa", f"{name}_syn_video_cap_gpt_qa.parquet") for name in subsets]

        if self.version == "v3":
            files += [
                join(self.data_path, "syn_video_cap_qa", "kw_syn_video_cap_gpt_qa_long_option.parquet"),
                join(self.data_path, "syn_video_cap_qa", "temporal_syn_video_cap_gpt_qa_long_option.parquet"),

            ]

        if self.version == "long":
            files = [
                join(self.data_path, "syn_video_cap_qa", "long_temporal_syn_video_cap_gpt_qa.parquet"),
            ]
        if self.version == "long2":
            files = [
                join(self.data_path, "syn_video_cap_qa", "long_temporal2_syn_video_cap_gpt_qa.parquet"),
            ]
        if self.version == "long3":
            files = [
                join(self.data_path, "syn_video_cap_qa", "long_temporal3_syn_video_cap_gpt_qa.parquet"),
            ]

        video2msgs = defaultdict(list)
        for f in files:
            data = pd.read_parquet(resource_path(f))
            for row in data.itertuples(False):
                video_path = os.path.join(VIDEO_DATA_HOME, row.video_path)
                qa_list = row.qa_list
                msgs = []
                for q_id, ex in enumerate(qa_list):
                    if 'count' in ex.get('Category', '').lower().split():
                        if self.exclude_counting:
                            continue
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
                        category=ex["Category"] if "Category" in ex else None,
                    )
                    msgs.append(msg)
                video2msgs[video_path].extend(msgs)

        for video_path, msgs in video2msgs.items():
            if len(msgs) == 0: continue
            formatted_example = {
                "video"       : video_path,
                "metadata"    : dict(
                    example_id=f"vixmo_syn_qa_{row.video_path}",
                ),
                "message_list": msgs
            }
            data_list.append(formatted_example)

        return data_list

    def get(self, idx, rng):
        example = self.data[idx]
        return example

    def __len__(self):
        return len(self.data)


class VixMoSynCaptionsSubtitleQA(DatasetBase):
    data_path = join(VIDEO_DATA_HOME, "LLM-QA")

    def __init__(
            self,
            split,
            answer_type: Literal["multi_choice", "open_ended", "all"] = "all",
            version: str = "v1"
    ):
        assert answer_type in ["multi_choice", "open_ended", "all"], f"Invalid answer type: {answer_type}"
        self.answer_type = answer_type
        self.version = version
        assert version in ["v1", "v2"], f"Invalid version: {version}"
        super().__init__(split)

    def load(self):

        data_list = []
        subsets = ['exist']
        if self.version == "v1":
            subsets += ['temporal']
        else:
            subsets += ['temporal2']
        files = [join(self.data_path, "syn_video_cap_qa", f"{name}_syn_video_cap_gpt_qa_subtitle.parquet") for name in subsets]

        for f in files:

            data = pd.read_parquet(resource_path(f))
            for row in data.itertuples():
                video_path = os.path.join(VIDEO_DATA_HOME, row.video_path)
                qa_list = row.qa_list
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
                        category=ex["Category"] if "Category" in ex else None,
                    )
                    msgs.append(msg)
                if len(msgs) == 0: continue

                subtitle = {}
                for sub in row.subtitle:
                    subtitle[(sub['start'], sub['end'])] = sub['text']

                formatted_example = {
                    "video"       : video_path,
                    "metadata"    : dict(
                        example_id=f"vixmo_syn_qa_{row.video_path}",
                    ),
                    "subtitle": subtitle,
                    "message_list": msgs
                }
                data_list.append(formatted_example)

        return data_list

    def get(self, idx, rng):
        example = self.data[idx]
        return example

    def __len__(self):
        return len(self.data)


def generate_consecutive_options(correct, min_val=0):
    # pick an offset from -3 to 0, but clamp so that start >= min_val
    possible_offsets = [o for o in range(-3, 1) if correct + o >= min_val]
    start_offset = random.choice(possible_offsets)
    start = correct + start_offset
    return [start + i for i in range(4)]


class VixMoPointCountQA(DatasetBase):
    data_path = join(VIDEO_DATA_HOME, "video_points")

    def __init__(
            self,
            split,
            mode: Literal["only_count", "mc"] = "only_count",
            clip_aug: bool = False,
            clip_aug_min: float = 10,
            clip_aug_max: float = 60,
            clip_aug_ratio: float = 0.5,
            use_clip: bool = False,
    ):
        assert mode in ["only_count", "mc"], f"Invalid mode: {mode}"
        self.mode = mode
        self.use_clip = use_clip
        if use_clip:
            assert not clip_aug, "clip_aug can only be used when use_clip is False"
        self.clip_aug = clip_aug
        self.clip_aug_min = clip_aug_min
        self.clip_aug_max = clip_aug_max
        self.clip_aug_ratio = clip_aug_ratio
        super().__init__(split)

    @staticmethod
    def expand_intervals(start: float, end: float, min_len=10.0, max_len=60.0):
            """
            Randomly expand [start, end] so the final interval:
              - contains the original
              - has length L chosen uniformly in [min_len, max_len]
              - distributes the extra padding randomly on both sides
            If the original length >= max_len, returns the original (can't expand to <= max_len).
            If the original length is between [min_len, max_len], optionally expands up to max_len.

            Args:
                start, end: positive floats with end >= start
                min_len, max_len: desired length range (defaults 10..60)
                seed: optional int for reproducibility

            Returns:
                (new_start, new_end)
            """
            if start > end:
                raise ValueError("start must be <= end")

            orig_len = end - start
            if orig_len >= max_len:
                # Already too big to fit within the cap; return as-is (or you could choose to raise).
                return None, None

            # Choose a target length:
            lo = max(min_len, orig_len)  # must be at least the original length
            hi = max_len
            target_len = random.uniform(lo, hi)

            # How much padding we need to add total:
            pad_total = target_len - orig_len

            # Split padding randomly left/right
            pad_left = random.uniform(0.0, pad_total)
            pad_right = pad_total - pad_left

            new_start = max(0.0, start - pad_left)  # keep non-negative
            # If we clipped the left side at 0, push the leftover to the right to preserve target length
            clipped_left = (start - pad_left) - new_start
            if clipped_left < 0:  # means we clipped (since new_start == 0 and start - pad_left < 0)
                pad_right += -clipped_left  # compensate on the right

            new_end = end + pad_right

            return new_start, new_end

    def load(self):
        data_list = []
        data_path = join(self.data_path, "vixmo_points_all_0925.parquet")
        video2duration = pickle.load(open(join(self.data_path, "video2duration_0925.pkl"), "rb"))

        data = pd.read_parquet(resource_path(data_path))
        video_path2data = {}
        for row in data.itertuples(False):
            row = row._asdict()
            if row["unsure"] or row['unanswerable'] or row["split"] != self.split:
                continue
            if row["count"] > 0:
                if row["2fps_video_path"].startswith('youtube-cc'):
                    video_path = join(VIDEO_DATA_HOME, 'youtube-cc', row["2fps_video_path"])
                else:
                    video_path = join(VIDEO_DATA_HOME, row["2fps_video_path"])


                count = int(row["count"])
                s, e = min(row['timestamps']), max(row['timestamps'])
                if e - s > 200:
                    # skip if the segment is too long
                    continue
                start = max(0, s - 1)
                end = e + 1

                if self.mode == "only_count":
                    msg = dict(
                        question=row["question"],
                        answer=str(count),
                        style="video_short_answer"
                    )

                else:
                    question = row["question"]
                    options = generate_consecutive_options(count, min_val=1)
                    answer_idx = options.index(count)

                    msg = dict(
                        question=question,
                        options=options,
                        answer_idx=answer_idx,
                        style="video_multiple_choice"
                    )

                if self.clip_aug:
                    video_duration = video2duration[row["2fps_video_path"]]
                    if video_duration > self.clip_aug_min:
                        clip_length = end - start
                        if clip_length < video_duration * self.clip_aug_ratio and clip_length < self.clip_aug_max:
                            start, end = self.expand_intervals(s, e, min_len=self.clip_aug_min, max_len=self.clip_aug_max)
                            formatted_ex = {
                                "video"       : video_path,
                                "meta"        : {
                                    "clip_start_time": start,
                                    "clip_end_time"  : end,
                                },
                                "message_list": [msg]
                            }
                            data_list.append(formatted_ex)

                if self.use_clip:
                    formatted_ex = {
                        "video"       : video_path,
                        "meta": {
                            "clip_start_time": start,
                            "clip_end_time"  : end,
                        },
                        "message_list": [msg]
                    }
                    data_list.append(formatted_ex)
                else:
                    if video_path not in video_path2data:
                        video_path2data[video_path] = []
                    video_path2data[video_path].append(msg)

        for video_path, msgs in video_path2data.items():
            formatted_ex = {
                "video"       : video_path,
                "message_list": msgs
            }
            data_list.append(formatted_ex)

        return data_list

    def get(self, idx, rng):
        example = self.data[idx]
        return example

    def __len__(self):
        return len(self.data)




class VixMoClipCaptionsQA(DatasetBase):
    data_path = join(VIDEO_DATA_HOME, "LLM-QA")
    captions_path = join(VIDEO_DATA_HOME, "video_captions", "vixmo_captions.parquet")

    def __init__(self, split, answer_type: Literal["multi_choice", "open_ended", "all"] = "all"):
        assert split in ["train", "val"]
        assert answer_type in ["multi_choice", "open_ended", "all"], f"Invalid answer type: {answer_type}"
        self.answer_type = answer_type
        super().__init__(split)

    def load(self):
        captions = pd.read_parquet(self.captions_path)
        videos = captions[captions["split"] == self.split]['video_path'].tolist()

        data_list = []

        if self.answer_type == "multi_choice" or self.answer_type == "all":

            data = pd.read_parquet(join(self.data_path, f"clipqa_mc_v2.parquet"))
            for row in data.itertuples(False):
                row = row._asdict()
                if row["video"] not in videos:
                    continue
                video_path = os.path.join(VIDEO_DATA_HOME, row["video"])
                start = row["timestamp_start"]
                end = row["timestamp_end"]
                qa_list = row["qa_list"]
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
                        category=ex["Category"] if "Category" in ex else None,
                    )
                    msgs.append(msg)
                if len(msgs) == 0: continue
                formatted_example = {
                    "video"       : video_path,
                    "metadata"    : dict(
                        example_id=f"vixmo_clip_qa_{row['video']}",
                        clip_start_time=start,
                        clip_end_time=end,
                    ),
                    "message_list": msgs
                }
                data_list.append(formatted_example)

        if self.answer_type == "open_ended" or self.answer_type == "all":

            data = pd.read_parquet(os.path.join(self.data_path, f"clipqa_oe_v2.parquet"))
            for row in data.itertuples():
                if row.video not in videos:
                    continue
                video_path = os.path.join(VIDEO_DATA_HOME, row.video)
                start = row.timestamp_start
                end = row.timestamp_end
                qa_list = row.qa_list
                msgs = []
                for q_id, ex in enumerate(qa_list):
                    question = ex["Question"]
                    answer = ex["Answer"]
                    msg = dict(
                        question=question,
                        answer=answer,
                        style="video_short_answer",
                        category=ex["Category"] if "Category" in ex else None,
                    )
                    msgs.append(msg)
                if len(msgs) == 0: continue
                formatted_example = {
                    "video"       : video_path,
                    "metadata"    : dict(
                        example_id=f"vixmo_clip_qa_{row.video}",
                        clip_start_time=start,
                        clip_end_time=end,
                    ),
                    "message_list": msgs
                }
                data_list.append(formatted_example)

        return data_list

    def get(self, idx, rng):
        example = self.data[idx]
        return example

    def __len__(self):
        return len(self.data)

def sample_random_clip(
    video_duration: float,
    start_time: float,
    end_time: float,
    min_seconds: float,
    max_seconds: float,
    timestamp_step: float = 0.5,
    seed: Optional[int] = None
) -> Tuple[float, float]:
    """
    Randomly choose a clip [clip_start, clip_end] such that:
      - 0 <= clip_start <= start_time <= end_time <= clip_end <= video_duration
      - min_seconds <= (clip_end - clip_start) <= max_seconds
      - clip_start is a multiple of timestamp_step (e.g., 0.5, 1/30, etc.)
      - Uniform randomness by default:
          * start is uniform over all feasible aligned starts
          * duration is uniform within the feasible range for that start

    Args:
      timestamp_step: grid step for clip_start alignment (must be > 0).
      seed: optional RNG seed for reproducibility.

    Raises:
      ValueError if inputs are invalid or constraints are impossible.
    """
    EPS = 1e-9

    # Validation
    if video_duration <= 0:
        raise ValueError("video_duration must be positive.")
    if min_seconds <= 0 or max_seconds <= 0:
        raise ValueError("min_seconds and max_seconds must be positive.")
    if timestamp_step <= 0:
        raise ValueError("timestamp_step must be positive.")
    if min_seconds - max_seconds > EPS:
        raise ValueError("min_seconds cannot exceed max_seconds.")
    if not (0 <= start_time <= end_time <= video_duration):
        raise ValueError(f"Require 0 <= start_time <= end_time <= video_duration but got {start_time, end_time, video_duration}.")

    seg_len = end_time - start_time
    if seg_len - max_seconds > EPS:
        raise ValueError(f"Required segment is longer than max_seconds. Got {start_time, end_time, max_seconds}")

    # Global feasible duration range
    W_min = max(seg_len, min_seconds)
    W_max = min(max_seconds, video_duration)
    if W_min - W_max > EPS:
        raise ValueError("No feasible clip length given the constraints and video length.")

    step = float(timestamp_step)

    # k range from lower bound (ensuring end_time can be included with max W)
    start_lower = max(0.0, end_time - W_max)
    # and upper bound (must not exceed start_time, and leave room for at least W_min)
    start_upper = min(start_time, video_duration - W_min)

    k_min = math.ceil((start_lower - EPS) / step)
    k_max = math.floor((start_upper + EPS) / step)

    if k_min > k_max:
        raise ValueError("No grid-aligned start can satisfy the constraints.")

    # Collect feasible k where the local W interval is non-empty
    feasible_k = []
    for k in range(k_min, k_max + 1):
        clip_start = k * step
        w_low = max(W_min, end_time - clip_start)
        w_high = min(W_max, video_duration - clip_start)
        if w_low <= w_high + EPS:
            feasible_k.append((k, w_low, w_high))

    if not feasible_k:
        raise ValueError("No grid-aligned start yields a feasible duration window.")

    rng = random.Random(seed) if seed is not None else random

    # Sample start index uniformly among feasible grid points
    k, w_low, w_high = rng.choice(feasible_k)
    clip_start = k * step

    # Sample duration uniformly within feasible window for this start
    W = w_low if abs(w_high - w_low) <= EPS else rng.uniform(w_low, w_high)
    clip_end = clip_start + W

    # Final safety checks (tolerant to float noise)
    assert -EPS <= clip_start <= start_time + EPS
    assert end_time - EPS <= clip_end <= video_duration + EPS
    assert min_seconds - EPS <= (clip_end - clip_start) <= max_seconds + EPS
    assert abs((clip_start / step) - round(clip_start / step)) <= 1e-6  # grid aligned

    return float(clip_start), float(clip_end)


class VixMoPoints(DatasetBase):
    corrupt_videos = set(
        [
            'youtube-cc-exist-2fps/5fvMsY0fdlA_2fps.mp4',
            'youtube-cc-exist-2fps/S4QRCBJHjcM_2fps.mp4',
            'youtube-cc-exist-2fps/bj9kuemDLy4_2fps.mp4',
            'youtube-cc-exist-2fps/53jDAb9lM3c_2fps.mp4',
            'youtube-cc-exist-2fps/fcCwR-EzZwU_2fps.mp4',
            'youtube-cc-exist-2fps/H_Jht3DKbsQ_2fps.mp4',
            'youtube-cc-exist-2fps/lkG7m3yjj2k_2fps.mp4',
            'youtube-cc-exist-2fps/pqcim7sQjyI_2fps.mp4',
            'youtube-cc-exist-2fps/5GLgBKdkbR0_2fps.mp4',
            'youtube-cc-exist-2fps/uTo5c47bZzk_2fps.mp4',
            'youtube-cc-exist-2fps/0LkJ_zcSDqs_2fps.mp4',
            'youtube-cc-exist-2fps/JwSfuxRm8WA_2fps.mp4',
            'youtube-cc-exist-2fps/8Sh7XpIW55o_2fps.mp4',
            'youtube-cc-exist-2fps/DpYxzINmX8w_2fps.mp4',
            'youtube-cc-exist-2fps/Ag1fW6GX0M8_2fps.mp4',
            'youtube-cc-exist-2fps/I2tZQUAnAPA_2fps.mp4',
            'youtube-cc-exist-2fps/8Cvz8k9MzW8_2fps.mp4',
            'youtube-cc-exist-2fps/FHUxcO6jH2Q_2fps.mp4',
            'youtube-cc-exist-2fps/bvRB-E9v6K8_2fps.mp4',
            'youtube-cc-exist-2fps/WzDnorAzWVU_2fps.mp4',
            'youtube-cc-exist-2fps/KlQGPlbtOE8_2fps.mp4',
            'youtube-cc-exist-2fps/KTrDXKnQRTM_2fps.mp4',
            'youtube-cc-exist-2fps/lfapRt3RId0_2fps.mp4',
            'youtube-cc-exist-2fps/XhOp91r3Ed4_2fps.mp4',
            'youtube-cc-exist-2fps/T0WudxReNwE_2fps.mp4',
            'youtube-cc-exist-2fps/UbBe3Dt7QaI_2fps.mp4',
            'youtube-cc-exist-2fps/mJKFIGUzA6A_2fps.mp4',
            'youtube-cc-exist-2fps/yQ9J10QMiDk_2fps.mp4',
            'youtube-cc-exist-2fps/wzAl3L83LmI_2fps.mp4',
            'youtube-cc-exist-2fps/YcAqdzsqESs_2fps.mp4'
        ]
    )
    data_path = join(VIDEO_DATA_HOME, "video_points")
    video_root = join(VIDEO_DATA_HOME, "youtube-cc")
    gen_video_root = join(VIDEO_DATA_HOME, "defection_video")

    def __init__(
        self,
        split: Literal["train", "val"] = "train",
        subset: Literal[
            "all", 
            "object", 
            "animal", 
            "action/event",
            "referring expression",
            "indirect reference",
            "anomaly",
            "spatial reference",
            "comparative"
        ] = "all",
        capability: Literal["all", "point", "count"] = "all",
        mode: Literal["point", "count", "point_count", "count_point"] = "point",
        include_unsure: bool = False,
        include_zero: bool = True,
        include_nonzero_unanswerable: bool = False,
        flat: bool = False,
        min_points: int = -1,
        max_points: int = -1,
        point_sort_by: Literal["xy", "yx", None] = "xy",
        use_2fps_video: bool = True, # assuming we use 2fps videos and annotations by default
        max_seconds: int = -1,
        max_raw_duration: int = -1,
        fake_timestamp_fps: int = None,
        load_clip_times_from_metadata: bool = False,
        fake_fps_candidates: List[float] = None,
        multi_message_short_clips=False
    ):
        assert split in ["train", "val"], f"Invalid split: {split}"
        self.split = split
        self.subset = subset
        self.capability = capability
        self.include_unsure = include_unsure
        self.include_zero = include_zero
        self.include_nonzero_unanswerable = include_nonzero_unanswerable
        self.mode = mode
        self.flat = flat
        self.min_points = min_points
        self.max_points = max_points
        self.point_sort_by = point_sort_by
        self.use_2fps_video = use_2fps_video
        self.max_seconds = max_seconds
        self.max_raw_duration = max_raw_duration  # skip videos longer than this
        # fps for the timestamps associated with the points and used in frame sampling during training
        self.fake_timestamp_fps = fake_timestamp_fps
        # set timestamp step based on the fake fps, default to 0.5s (2fps) if not specified
        self.timestamp_step = 1 / fake_timestamp_fps if fake_timestamp_fps is not None else 0.5
        self.load_clip_times_from_metadata = load_clip_times_from_metadata
        if self.load_clip_times_from_metadata:
            # load the preprocessed metadata where the clips have been sanitized
            metadata_path = join(self.data_path, f"vixmo_points_meta_max_{self.max_seconds}s_1206.json")
            if not exists(metadata_path):
                raise FileNotFoundError(f"Preprocessed metadata file not found: {metadata_path}")
            self.clip_metadata = json.load(open(metadata_path, "r"))
        self.fake_fps_candidates = fake_fps_candidates
        # check if only one of fake_timestamp_fps and fake_fps_candidates is provided
        assert self.fake_timestamp_fps is None or self.fake_fps_candidates is None, \
            "Only one of fake_timestamp_fps and fake_fps_candidates should be provided."
        self.multi_message_short_clips = multi_message_short_clips
        if self.multi_message_short_clips:
            assert not self.flat
            assert self.max_seconds > 0
        super().__init__(split)

    def _build_example(self, video_path, msg):
        metadata = {
            "points": msg["points"],
            "timestamps": msg["timestamps"],
            "subset": msg["subset"],
            "count": msg["count"],
            "video_path": video_path
        }
        if self.fake_timestamp_fps is not None:
            metadata["fake_timestamp_fps"] = self.fake_timestamp_fps
        if self.max_seconds > 0:
            metadata["clip_start_time"] = msg["clip_start_time"]
            metadata["clip_end_time"] = msg["clip_end_time"]
        msg.update(
            {
                "video": video_path,
                "metadata": metadata,
            }
        )
        return msg

    def load(self):
        random.seed(42)
        points_path = join(self.data_path, "vixmo_points_all_1206.parquet")
        df = pd.read_parquet(points_path)
        df = df[df["split"] == self.split]
        if self.subset != "all":
            df = df[df["subset"] == self.subset]
        if not self.include_unsure:
            df = df[~df["unsure"]]
        if not self.include_nonzero_unanswerable:
            df = df[~((df["unanswerable"]) & (df["count"] != 0))]
        if not self.include_zero:
            df = df[df["count"] != 0]
        if self.capability != "all":
            df = df[df["capability"] == self.capability]
        if self.min_points >= 0:
            df = df[df["count"] >= self.min_points]
        if self.max_points >= 0:
            df = df[df["count"] <= self.max_points]
        if self.max_raw_duration > 0:
            df = df[df["video_duration"] <= self.max_raw_duration]
        video2msgs = {}
        video_durations = {}
        formatted_data = []
        invalid_cnt = 0
        not_found_cnt = 0
        long_ann_cnt = 0
        long_video_cnt = 0
        two_fps_index = list(df.columns).index("2fps_video_path")
        for row in df.itertuples(False):
            video_root_dir = (
                self.gen_video_root if row.subset == "anomaly" else self.video_root
            )
            if self.use_2fps_video:
                if row[two_fps_index] in self.corrupt_videos:
                    continue
                video_path = join(video_root_dir, row[two_fps_index])
            else:
                video_path = join(video_root_dir, row.video_path)

            # if not exists(video_path):
            #     logging.warning(f"Video not found: {video_path}")
            #     continue
            
            ann_start = row.ann_start_timestamp
            ann_end = row.ann_end_timestamp
            ann_seconds = ann_end - ann_start
            video_durations[video_path] = row.video_duration
            if self.max_seconds > 0 and ann_seconds > self.max_seconds:
                long_ann_cnt += 1
                continue
            if self.fake_fps_candidates is not None:
                # randomly assign a fake fps from the candidates for this example
                fake_fps = random.choice(self.fake_fps_candidates)
                self.fake_timestamp_fps = fake_fps
                self.timestamp_step = 1 / fake_fps

            if row.count > 0:
                # divide by 0.5 assuming we use 2fps annotations by default
                start_time = (
                    math.floor(ann_start / 0.5)
                    * self.timestamp_step
                )
                end_time = (
                    math.floor(ann_end / 0.5)
                    * self.timestamp_step
                )
            else:
                # for zero count, use the entire video duration, and set ann times to 0
                start_time = 0.0
                end_time = min(row.video_duration, self.max_seconds) if self.max_seconds > 0 else row.video_duration
                ann_start = 0.0
                ann_end = 0.0
            ann_seconds = ann_end - ann_start
            if self.max_seconds > 0 and ann_seconds > self.max_seconds:
                long_ann_cnt += 1
                continue
            
            msg = {
                "subset": row.subset,
                "example_id": f"{row[two_fps_index]}_{row.query}",
                "label": row.query,
                "answer": str(row.count),
                "count": row.count,
                "unanswerable": row.unanswerable,
                "explanation": row.explanation,
            }
            if self.split == "val":
                msg["question"] = f"How many \"{msg['label']}\" are there in the video?"
            if self.subset == "action/event":
                if random.random() < 0.5:
                    # use gpt-generated action questions half of the time
                    msg["question"] = row.question
            # randomly sample a clip that covers the annotated timestamps
            if self.max_seconds > 0:
                if row.video_duration <= self.max_seconds:
                    msg["clip_start_time"] = 0.0
                    msg["clip_end_time"] = row.video_duration
                elif self.load_clip_times_from_metadata:
                    if msg["example_id"] not in self.clip_metadata:
                        not_found_cnt += 1
                        continue
                    clip_info = self.clip_metadata[msg["example_id"]]
                    msg["clip_start_time"] = clip_info["clip_start_time"]
                    msg["clip_end_time"] = clip_info["clip_end_time"]
                else:
                    rand_start, rand_end = sample_random_clip(
                        video_duration=row.video_duration,
                        start_time=start_time,
                        end_time=end_time, # set min for long videos with zero count annotation
                        min_seconds=self.timestamp_step,  # at least one timestamp step
                        max_seconds=self.max_seconds,
                        timestamp_step=self.timestamp_step,
                        seed=42,
                    )
                    msg["clip_start_time"] = rand_start
                    msg["clip_end_time"] = rand_end
                try:
                    assert msg['clip_start_time'] >= 0, msg['clip_start_time']
                    assert msg['clip_end_time'] <= row.video_duration, (msg['clip_end_time'], row.video_duration)
                    assert msg['clip_start_time'] < msg['clip_end_time'], (msg['clip_start_time'], msg['clip_end_time'])
                    assert msg['clip_end_time'] - msg['clip_start_time'] <= self.max_seconds, (msg['clip_end_time'], msg['clip_start_time'], self.max_seconds)
                    assert msg['clip_start_time'] <= ann_start, (msg['clip_start_time'], ann_start)
                    assert msg['clip_end_time'] >= ann_end, (msg['clip_end_time'], ann_end)
                except AssertionError:
                    invalid_cnt += 1
                    continue

            all_sorted_points = []
            all_timestamps = []
            for i, ts in enumerate(row.timestamps):
                points = row.points[i]
                # assuming we use 2fps annotations by default
                assert ts / 0.5 == int(
                    ts / 0.5
                ), f"Original timestamp {ts} not aligned to 0.5s intervals"

                if self.point_sort_by == "xy":
                    # sort by x first and then by y
                    sorted_points = sorted(points, key=lambda p: (p["x"], p["y"]))
                elif self.point_sort_by == "yx":
                    sorted_points = sorted(points, key=lambda p: (p["y"], p["x"]))
                else:
                    sorted_points = points
                all_sorted_points.append(sorted_points)
                if "clip_start_time" in msg:
                    ts = ts - msg["clip_start_time"]
                # align timestamps to the specified fps intervals
                ts = math.floor(ts / 0.5) * self.timestamp_step
                all_timestamps.append(ts)

            msg["points"] = all_sorted_points
            msg["timestamps"] = all_timestamps

            video2msgs[video_path] = video2msgs.get(video_path, []) + [msg]

        if self.multi_message_short_clips:
            # long videos are flat, short videos are multiply annotated
            n_multi_turn = 0
            for video_path, msgs in video2msgs.items():
                single_message = []
                duration = video_durations[video_path]
                if duration <= self.max_seconds and len(msgs) > 1:
                    start, end = msgs[0]["clip_start_time"], msgs[0]["clip_end_time"]
                    assert all(msg["clip_start_time"] == start and msg["clip_end_time"] == end for msg in msgs)
                    n_multi_turn += 1
                    formatted_data.append(dict(
                        message_list=msgs,
                        video=video_path,
                        metadata=dict(
                            clip_start_time=start,
                            clip_end_time=end
                        )
                    ))
                else:
                    for msg in msgs:
                        formatted_data.append(self._build_example(video_path, msg))
            logging.info(f"Have {n_multi_turn} multi-turn video pointing messages")

        elif self.flat or self.max_seconds > 0:
            for video_path, msgs in video2msgs.items():
                for msg in msgs:
                    formatted_data.append(self._build_example(video_path, msg))
        else:
            for video_path, msgs in video2msgs.items():
                example = {
                    "video": video_path,
                    "message_list": msgs,
                }
                formatted_data.append(example)
        return formatted_data

    def __len__(self):
        return len(self.data)

    def _get_style(self, rng):
        if isinstance(self.mode, str):
            style = self.mode
        else:
            style = rng.choice(self.mode)
        return f"video_{style}"

    def get(self, idx, rng):
        example = dict(self.data[idx])
        if "message_list" in example and len(example["message_list"]) > 1:
            with_style = []
            for message in example.pop("message_list"):
                style = self._get_style(rng)
                if style == "video_point" and "question" in message:
                    # Don't use the "how many" question template for point only style 
                    del message["question"]
                with_style.append(dict(message, style=style))
            assert len(with_style) > 0
            if rng.random() < 0.5:
                # Treat as a multi-turn conversation, not a list of messages
                example["multi_turn_messages"] = with_style
            else:
                example["message_list"] = with_style
            return example
        else:
            style = self._get_style(rng)
            if style == "video_point" and "question" in example:
                del example["question"]
            return set_example_style(example, style)


class VixMoClipPointing(DatasetBase):
    data_path = join(VIDEO_DATA_HOME, "video_points")
    video_root = join(VIDEO_DATA_HOME, "youtube-cc")
    gen_video_root = join(VIDEO_DATA_HOME, "defection_video")

    def __init__(
        self,
        split: Literal["train", "val"] = "train",
        subset: str = "all",
        mode: Literal["point", "count", "point_count", "count_point"] = ("point", "point_count"),
        max_points: int = 60,
        max_duration: float = 7.5,
    ):
        assert split in ["train", "val"], f"Invalid split: {split}"
        self.split = split
        self.subset = subset
        self.mode = mode
        self.max_points = max_points
        self.max_duration = max_duration
        super().__init__(split)

    def load(self):
        points_path = join(self.data_path, "vixmo_points_all_1206.parquet")
        df = pd.read_parquet(points_path)
        df = df[df["split"] == self.split]
        if self.subset != "all":
            df = df[df["subset"] == self.subset]
        df = df[~df["unsure"]]
        df = df[~np.isnan(df["video_duration"])]
        if self.max_points:
            df = df[df["count"] <= self.max_points]
        df = df[(~(df["unanswerable"] & (df["count"] != 0)))]
        df = df[[x not in VixMoPoints.corrupt_videos for x in df["2fps_video_path"]]]
        # if self.split == "val":
        #     df = df[[x == "pieces of raw pork belly" for x in df["query"]]][:1]

        return df

    def __len__(self):
        return len(self.data)

    def _get_style(self, rng):
        if isinstance(self.mode, str):
            style = self.mode
        else:
            style = rng.choice(self.mode)
        return f"video_{style}"

    def get(self, idx, rng):
        row = self.data.iloc[idx]
        video_root_dir = (
            self.gen_video_root if row.subset == "anomaly" else self.video_root
        )
        video_file = row["2fps_video_path"]
        video_path = join(video_root_dir, video_file)
        duration = row.video_duration

        if len(row.timestamps) == 0:
            timestamps = []
            points = []
            if duration < self.max_duration:
                clip = None
            else:
                start = rng.random() * (duration - self.max_duration)
                end = start + self.max_duration
                clip = (start, end)
        else:
            timestamps = np.array(row.timestamps)
            timestamps = np.floor(timestamps / 0.5).astype(np.int64)
            max_duration = int(2*self.max_duration)
            duration = int(np.floor(duration  * 2))
            if duration < max_duration:
                clip = None
                points = row.points
            else:
                first_timestamp = timestamps[0]
                keep = timestamps <= first_timestamp + max_duration
                timestamps = timestamps[keep]
                points = [p for i, p in enumerate(row.points) if keep[i]]
                last_timestamp = timestamps[-1]
                remainder = max_duration - (last_timestamp - first_timestamp)
                min_start = max(first_timestamp - remainder, 0)
                if first_timestamp != min_start:
                    start = rng.randint(min_start, first_timestamp)
                else:
                    start = first_timestamp
                end = min(start + max_duration, duration)
                assert end >= last_timestamp and start <= first_timestamp
                clip = (start / 2.0, end / 2.0)
                timestamps = (timestamps - start) / 2.0
        if clip is None:
            metadata = {}
        else:
            metadata = dict(
                clip_start_time=clip[0],
                clip_end_time=clip[1]
            )

        gt_point_array = []
        for ts, frame_points in zip(timestamps, points):
            for point in frame_points:
                gt_point_array.append([ts, point["x"], point["y"]])
        if gt_point_array:
            gt_point_array = np.array(gt_point_array)

        metadata.update(dict(
            video_path=video_path,
            example_id=f"{video_file}_{row.query}",
            points=points,
            label=row.query,
            timestamps=timestamps,
            subset=row["subset"],
            count=row["count"],
            gt_normalized_point_array=gt_point_array,
        ))
        msg = dict(
            label=row.query,
            points=points,
            timestamps=timestamps,
            style=self._get_style(rng)
        )
        return dict(
            message_list=[msg],
            video=video_path,
            metadata=metadata,
        )


class VixMoSubtitlePoints(DatasetBase):
    corrupt_videos = set(
        [
        ]
    )
    data_path = join(VIDEO_DATA_HOME, "video_points")
    video_root = join(VIDEO_DATA_HOME, "youtube-cc")

    def __init__(
        self,
        split: Literal["train", "val"] = "train",
        mode = ("pointing", "point_count"),
        include_unsure: bool = False,
        include_zero: bool = True,
        include_nonzero_unanswerable: bool = False,
        flat: bool = False,
        max_points: int = -1,
        point_sort_by: Literal["xy", "yx", None] = "xy",
        max_seconds: int = -1,
    ):
        assert split in ["train"], f"Invalid split: {split}"
        self.split = split
        self.include_unsure = include_unsure
        self.include_zero = include_zero
        self.include_nonzero_unanswerable = include_nonzero_unanswerable
        self.mode = mode
        self.flat = flat
        self.max_points = max_points
        self.point_sort_by = point_sort_by
        self.max_seconds = max_seconds
        self.timestamp_step = 0.5
        super().__init__(split)

    def load(self):
        random.seed(42)

        video2msgs = {}
        formatted_data = []
        invalid_cnt = 0

        points_path = join(self.data_path, "all_subsets", "vixmo_points_point_subtitle_1025.parquet")
        df = pd.read_parquet(points_path)
        
        for row in df.itertuples(False):
            video_root_dir = self.video_root
            video_path = join(video_root_dir, row.video_path)
            video_duration = row.video_duration
            if not exists(video_path):
                logging.warning(f"Video not found: {video_path}")
                continue

            if not self.include_nonzero_unanswerable and (
                    row.unanswerable and row.count != 0
            ):
                continue
            if not self.include_zero and row.count == 0:
                continue
            if not self.include_unsure and row.unsure:
                continue
            total_points = row.count
            if self.max_points > 0 and total_points > self.max_points:
                continue

            if row.count > 0:
                ann_start = row.timestamps[0]
                start_time = ann_start
                ann_end = row.timestamps[-1]
                ann_end = min(ann_end, video_duration)
                end_time = ann_end
            else:
                # for zero count, use the entire video duration, and set ann times to 0
                start_time = 0.0
                end_time = video_duration
                ann_start = 0.0
                ann_end = 0.0

            ann_seconds = end_time - start_time
            if self.max_seconds > 0 and ann_seconds > self.max_seconds:
                continue

            if not (0 <= start_time <= end_time <= video_duration):
                logging.warning((video_path, start_time, end_time, video_duration))
                continue


            msg = {
                "subset": "subtitle",
                "example_id": f"{row.video_path}_{row.query}",
                "label": row.query,
                "answer": str(row.count),
                "count": row.count,
                "unanswerable": row.unanswerable,
                "explanation": row.explanation,
                "style": f"video_{self.mode}",
            }
            if self.split != "train":
                # use the same "how many" question template for val/test
                msg["question"] = f"How many \"{msg['label']}\" are there in the video?"
            # randomly sample a clip that covers the annotated timestamps
            if self.max_seconds > 0:
                if video_duration <= self.max_seconds:
                    msg["clip_start_time"] = 0.0
                    msg["clip_end_time"] = video_duration
                else:
                    rand_start, rand_end = sample_random_clip(
                        video_duration=video_duration,
                        start_time=start_time,
                        end_time=end_time,
                        min_seconds=self.timestamp_step,  # at least one timestamp step
                        max_seconds=self.max_seconds,
                        timestamp_step=self.timestamp_step,
                        seed=42,
                    )
                    msg["clip_start_time"] = rand_start
                    msg["clip_end_time"] = rand_end
                try:
                    assert msg['clip_start_time'] >= 0, msg['clip_start_time']
                    assert msg['clip_end_time'] <= video_duration, (msg['clip_end_time'], video_duration)
                    assert msg['clip_start_time'] < msg['clip_end_time'], (msg['clip_start_time'], msg['clip_end_time'])
                    assert msg['clip_end_time'] - msg['clip_start_time'] <= self.max_seconds, (msg['clip_end_time'], msg['clip_start_time'], self.max_seconds)
                    assert msg['clip_start_time'] <= ann_start, (msg['clip_start_time'], ann_start)
                    assert msg['clip_end_time'] >= ann_end, (msg['clip_end_time'], ann_end)
                except AssertionError:
                    invalid_cnt += 1
                    continue

            all_sorted_points = []
            all_timestamps = []
            for i, ts in enumerate(row.timestamps):
                points = row.points[i]

                if self.point_sort_by == "xy":
                    # sort by x first and then by y
                    sorted_points = sorted(points, key=lambda p: (p["x"], p["y"]))
                elif self.point_sort_by == "yx":
                    sorted_points = sorted(points, key=lambda p: (p["y"], p["x"]))
                else:
                    sorted_points = points
                all_sorted_points.append(sorted_points)

                if "clip_start_time" in msg:
                    ts = ts - msg["clip_start_time"]
                # align timestamps to the specified fps intervals
                all_timestamps.append(ts)

            msg["points"] = all_sorted_points
            msg["timestamps"] = all_timestamps

            video2msgs[video_path] = video2msgs.get(video_path, []) + [msg]
            if self.flat or self.max_seconds > 0:
                metadata = {
                    "points": msg["points"],
                    "timestamps": msg["timestamps"],
                    "subset": msg["subset"],
                }
                if self.max_seconds > 0:
                    metadata["clip_start_time"] = msg["clip_start_time"]
                    metadata["clip_end_time"] = msg["clip_end_time"]

                msg.update(
                    {
                        "video": video_path,
                        "metadata": metadata,
                    }
                )

                if row.audio_required:
                    subtitle = {}
                    start = msg.get("clip_start_time", 0.0)
                    for sub in row.subtitle:
                        # offset the subtitle timestamps if a clip is sampled
                        subtitle[(sub['start']-start, sub['end']-start)] = sub['text']
                    msg['subtitle'] = subtitle

                formatted_data.append(msg)

        # only concatenate messages when not flat and max_seconds < 0 (no clip sampling)
        if not self.flat and self.max_seconds < 0:
            for video_path, msgs in video2msgs.items():
                example = {
                    "video": video_path,
                    "message_list": msgs,
                }
                formatted_data.append(example)
        if invalid_cnt > 0 and get_global_rank() == 0:
            logging.warning(f"Total invalid clips due to preprocessed metadata issues: {invalid_cnt}")
        return formatted_data

    def __len__(self):
        return len(self.data)

    def get(self, idx, rng):
        if isinstance(self.mode, str):
            style = self.mode
        else:
            style = rng.choice(self.mode)
        return set_example_style(self.data[idx], f"video_{style}")


class VixMoPointsEval(DatasetBase):
    data_path = join(
        VIDEO_DATA_HOME, "video_points/vixmo_points_val_filtered_1209.parquet"
    )
    video_root = join(VIDEO_DATA_HOME, "youtube-cc")

    def __init__(self, split):
        assert split in ["val"], f"Invalid split: {split}"
        super().__init__(split=split)

    def load(self):
        formatted_data = []
        df = pd.read_parquet(self.data_path)
        for row in df.itertuples():
            row = row._asdict()
            height = row["height"]
            width = row["width"]
            video_duration = row["video_duration"]
            all_masks = row["masks"]
            gt_abs_triplets = []
            gt_abs_masks = []
            for i, masks in enumerate(all_masks):
                gt_point = row['points'][i]
                gt_t = row['timestamps'][i]
                gt_abs_triplets.append((gt_t, gt_point['x'], gt_point['y']))
                time2mask = {}
                for mask in masks:
                    mask_t = mask['frame_id'] / row['fps']
                    rows = mask['mask']
                    mask_arr = np.vstack(rows)
                    time2mask[mask_t] = mask_arr
                gt_abs_masks.append(time2mask)
            # normalizing points and adding to metadata for visualization
            norm_points = [[{'x': round(p['x'] / width * 100, 1), 'y': round(p['y'] / height * 100, 1)}] for p in row['points']]
            msg = {
                "video": join(self.video_root, row["video_path"]),
                "label": row["query"],
                "question": f"How many \"{row['query']}\" are there in the video?", # f'Point to the \"{row["query"]}\" in the video.',
                "style": "video_point_count",
                "question": f"How many \"{row['query']}\" are there in the video?", # f'Point to the \"{row["query"]}\" in the video.',
                "style": "video_point_count",
                "timestamps": row["timestamps"],
                "points": norm_points,
                "metadata": {
                    "gt_abs_triplets": gt_abs_triplets,
                    "gt_abs_masks": gt_abs_masks,
                    "video_duration": video_duration,
                    "video_height": height,
                    "video_width": width,
                },
            }
            formatted_data.append(msg)
        return formatted_data

    def __len__(self):
        return len(self.data)

    def get(self, idx, rng):
        return self.data[idx]



if __name__ == "__main__":
    from olmo.data.get_dataset import get_dataset_by_name
    all_datasets = [
        # "academic_points_clip_63s_2fps", # 44k
        # "vixmo_points_minmax_0_5", # 260k
        # "vixmo_points_minmax_6_26", # 60k
        # "vixmo_points_minmax_26_60", # 10k
        "academic_points_mf384",
        "vixmo_points_mf384_minmax_0_5", # 220k
        "vixmo_points_mf384_minmax_6_26", # 75k
        "vixmo_points_mf384_minmax_26_60", # 14k
        # "vixmo_points_point_eval",
    ]
    splits = ["val"]
    for ds_name in all_datasets:
        for split in splits:
            ds = get_dataset_by_name(ds_name, split=split)
            print(f"Dataset: {ds_name}, Split: {split}, Length: {len(ds)}")
    breakpoint()
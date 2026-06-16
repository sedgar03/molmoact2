import gzip
import json
import logging
import os
import re
import string
from collections import defaultdict
from os.path import exists
from os.path import join
from typing import List, Dict, Any

import datasets
import hashlib
import numpy as np
import hashlib

from olmo.data.dataset import DATA_HOME, DatasetBase, Dataset, HfDataset, \
    MULTI_IMG_DATA_HOME
from olmo.data.pixmo_datasets import save_local_dataset
from olmo.io import resource_path
from olmo.preprocessing.image_preprocessor import save_images
from olmo.hf_datasets.a_okvqa import AOkVqaBuilder
from olmo.hf_datasets.ai2d import Ai2dDatasetBuilder
from olmo.hf_datasets.android_control import AndroidControlBuilder
from olmo.hf_datasets.clock_bench import ClockBenchBuilder
from olmo.hf_datasets.count_qa import CountQaBuilder
from olmo.hf_datasets.dv_qa import DvQaBuilder
from olmo.hf_datasets.figure_qa import FigureQaBuilder
from olmo.hf_datasets.plot_qa import PlotQaBuilder
from olmo.hf_datasets.tabmwp import TabMwpBuilder
from olmo.hf_datasets.tally_qa import TallyQaBuilder
from olmo.hf_datasets.vqa_v2 import VQAv2BuilderMultiQA
from olmo.hf_datasets.mantis_instruct import MantisInstructBuilder
from olmo.hf_datasets.vlm3r import MultiImageVLM3RBuilder

if DATA_HOME is not None:
    DOWNLOADS = join(DATA_HOME, "downloads")
    ACADEMIC_DATASETS = join(DATA_HOME, "academic_datasets")
    ANDROID_IMAGES = join(DATA_HOME, "android_images")
else:
    DOWNLOADS = None
    ACADEMIC_DATASETS = None
    ANDROID_IMAGES = None


class ChartQa(HfDataset):
    """
    ChartQA dataset from HuggingFace M4 project.
    This class loads the ChartQA dataset from HuggingFace (https://huggingface.co/datasets/HuggingFaceM4/ChartQA).

    Args:
        split (str): Dataset split to load. One of "train", "validation", or "test".
        parts (str, optional): Which subset of examples to include. One of:
            - "human": Only human-authored examples
            - "augmented": Only automatically generated examples
            - "both": Both human and augmented examples (default)
        weighted (bool, optional): Whether to apply weighting to balance human/augmented examples. Only valid when parts="both".
            Defaults to False.
    """
    PATH = "HuggingFaceM4/ChartQA"

    def __init__(self, split: str, parts="both", weighted=False, keep_in_memory=False):
        assert split in ["train", "validation", "test"]
        assert parts in ["human", "augmented", "both"]

        if split == "validation":
            split = "val"
        self.updated_split = split
        self.weighted = weighted
        self.parts = parts
        super().__init__(split, keep_in_memory=keep_in_memory)
        if self.parts != "both":
            # Filter out either human or aug datasets
            to_keep = 0 if (self.parts == "human") else 1
            self.dataset = self.dataset.filter(
                lambda x: x == to_keep,
                input_columns=["human_or_machine"]
            )

    def get(self, item, rng):
        ex = self.dataset[item]
        ex = dict(
            image=ex["image"],
            question=ex["query"],
            answers=ex["label"],
            style="chart_qa",
            metadata=dict(
                is_human=ex['human_or_machine'] == 0,
            )
        )
        if self.weighted:
            is_human = ex["metadata"]["is_human"]
            # Weight to balanced human/augmented sets
            if is_human:
                w = 2*20901/(20901+7398)
            else:
                w = 2*7398/(20901+7398)
            ex["weight"] = w
        return ex


class Vqa2(Dataset):
    @classmethod
    def download(cls, n_procs=1):
        VQAv2BuilderMultiQA(DOWNLOADS).download_and_prepare()

    def __init__(self, split, multi_question=False, sample=None):
        assert split in ["train", "validation", "test"]
        self.multi_question = multi_question
        self.dataset = VQAv2BuilderMultiQA(DOWNLOADS).as_dataset(split=split)
        if not self.multi_question:
            flattened_data = []
            for item in self.dataset:
                for q in item["messages"]:
                    flattened_data.append(dict(
                        style=q['style'],
                        question=q["question"],
                        answers=q["answers"],
                        image=item["image"],
                        image_id=item["image_id"],
                        question_id=q["question_id"],
                    ))
            if sample:
                logging.info(f"Sampling {sample} of {len(flattened_data)} ({100*sample/len(flattened_data)}:0.1f)")
                np.random.RandomState(9123).shuffle(flattened_data)
                flattened_data = flattened_data[:sample]
            self.dataset = flattened_data
        else:
            assert sample is None

    def __len__(self):
        return len(self.dataset)

    def get(self, item, rng):
        ex = self.dataset[item]
        if self.multi_question:
            return dict(
                metadata=dict(image_id=ex["image_id"]),
                image=ex["image"],
                message_list=ex["messages"],
            )
        else:
            return dict(
                style="vqa2",
                answers=ex["answers"],
                metadata=dict(image_id=ex["image_id"], example_id=ex["question_id"]),
                image=ex["image"],
                question=ex["question"],
            )


class AOkVqa(Dataset):
    @classmethod
    def download(cls, n_procs=1):
        AOkVqaBuilder(DOWNLOADS).download_and_prepare()

    def __init__(self, split, direct_answer=False):
        self.split = split
        self.direct_answer = direct_answer
        self.dataset = AOkVqaBuilder(DOWNLOADS).as_dataset(split=split)
        self.style = "a_okvqa_" + ("da" if direct_answer else "mc")
        self.loaded_data = self.load()

    def load(self):
        loaded_data = []
        for example in self.dataset:
            if self.direct_answer:
                if example["difficult_direct_answer"] and self.split in ["validation", "test"]:
                    continue
                out = dict(
                    image=example["image"],
                    question=example["question"],
                    answers=example["direct_answers"],
                    metadata=dict(
                        example_id=example["question_id"]
                    )
                )
            else:
                if example["correct_choice_idx"] is None:
                    out = dict(
                        image=example["image"],
                        question=example["question"],
                        options=example["choices"],
                        metadata=dict(example_id=example["question_id"])
                    )
                else:
                    out = dict(
                        image=example["image"],
                        question=example["question"],
                        options=example["choices"],
                        answer_idx=example["correct_choice_idx"],
                        metadata=dict(example_id=example["question_id"])
                    )
            loaded_data.append(out)
        return loaded_data

    def __len__(self):
        return len(self.loaded_data)

    def get(self, item, rng):
        return dict(**self.loaded_data[item], style=self.style)


class OkVqa(Dataset):
    """
    OK-VQA dataset from HuggingFace M4 project.
    This class loads the OK-VQA dataset from HuggingFace (https://huggingface.co/datasets/HuggingFaceM4/OK-VQA).

    Args:
        split (str): Dataset split to load. One of "train", "validation", or "test".
        multi_question (bool, optional): Whether to group questions by image. Defaults to False.
    """

    PATH = "HuggingFaceM4/OK-VQA"

    @classmethod
    def download(cls, n_procs=1):
        local_name = join(ACADEMIC_DATASETS, "okvqa")
        datasets.load_dataset_builder(cls.PATH, trust_remote_code=True).download_and_prepare()
        ds = datasets.load_dataset(cls.PATH, trust_remote_code=True)
        save_local_dataset(ds, local_name, n_procs)

    def __init__(self, split: str, multi_question=False, keep_in_memory=False):
        super().__init__()
        self.multi_question = multi_question
        dataset = datasets.load_from_disk(
            join(ACADEMIC_DATASETS, "okvqa"), keep_in_memory=keep_in_memory
        )[split]
        if self.multi_question:
            grouped_by_image = defaultdict(list)
            for ex in dataset:
                grouped_by_image[ex["image_id"]].append(ex)
            data = []
            for image_id, examples in grouped_by_image.items():
                questions = []
                for ex in examples:
                    questions.append(dict(
                        question=ex["question"],
                        answers=[x["raw_answer"] for x in ex["answers"]],
                    ))
                data.append(dict(
                    image=examples[0]["image"],
                    metadata=dict(image_id=image_id),
                    message_list=questions
                ))
            self.data = data
        else:
            self.data = dataset

    def __len__(self):
        return len(self.data)

    def get(self, item, rng):
        ex = self.data[item]
        if self.multi_question:
            return dict(ex, style="okvqa")
        else:
            return dict(
                image=ex["image"],
                question=ex["question"],
                answers=[x["raw_answer"] for x in ex["answers"]],
                metadata=dict(
                    example_id=ex["question_id"],
                ),
                style="okvqa",
            )


class TextVqa(HfDataset):
    """
    This class loads the TextVQA dataset from HuggingFace (https://huggingface.co/datasets/facebook/textvqa).
    """
    PATH = "facebook/textvqa"

    @classmethod
    def download(cls, n_procs=1):
        datasets.load_dataset_builder(cls.PATH, trust_remote_code=True).download_and_prepare()

    def __init__(self, split: str, identifier=None, keep_in_memory=False):
        super().__init__(
            split=split, keep_in_memory=keep_in_memory, trust_remote_code=True)

    def get(self, item, rng):
        example = self.dataset[item]
        return dict(
            image=example["image"],
            question=example["question"],
            answers=example.get("answers", []),
            metadata=dict(
                image_url=example["flickr_300k_url"],
                image_id=example["image_id"],
                example_id=example["question_id"],
            ),
            style="text_vqa"
        )


class TallyQa(Dataset):

    @classmethod
    def download(cls, n_procs=1):
        TallyQaBuilder().download_and_prepare()

    def __init__(self, split):
        assert split in ["train", "test"]
        self.dataset = TallyQaBuilder().as_dataset(split=split)
        super().__init__()

    def __len__(self):
        return len(self.dataset)

    def get(self, item, rng):
        ex = self.dataset[item]
        messages = []
        questions = ex["questions"]
        for ix, question in enumerate(questions["question"]):
            messages.append(dict(
                question=question,
                answer=str(questions["answer"][ix]),
                style="tally_qa"
            ))
        return dict(
            image=ex["image"],
            message_list=messages,
            metadata=dict( image_id=ex["image_id"])
        )


class AI2D(Dataset):

    @classmethod
    def download(cls, n_procs=1):
        local_name = join(ACADEMIC_DATASETS, "ai2d")
        Ai2dDatasetBuilder().download_and_prepare()
        all_data = datasets.DatasetDict()
        for split in ["train", "validation", "test"]:
            ds = Ai2dDatasetBuilder().as_dataset(split)
            all_data[split] = ds
        save_local_dataset(all_data, local_name, n_procs)

    def __init__(self, split, boxes="both", keep_in_memory=False):
        assert split in ["train", "validation", "test"]
        dataset = datasets.load_from_disk(
            join(ACADEMIC_DATASETS, "ai2d"), keep_in_memory=keep_in_memory)[split]
        if boxes == "transparent":
            dataset = dataset.filter(lambda x: not x["abc_label"] or x["has_transparent_box"])
        elif boxes == "opaque":
            dataset = dataset.filter(lambda x: not x["abc_label"] or not x["has_transparent_box"])
        elif boxes == "both":
            pass
        else:
            raise NotImplementedError(boxes)
        self.dataset = dataset

        self.split = split
        self.boxes = boxes
        super().__init__()

    def __len__(self):
        return len(self.dataset)

    def get(self, item, rng):
        _ex = dict(self.dataset[item])
        ex = dict(
            image=_ex["image"],
            question=_ex["question"],
            answer_idx=_ex["correct_answer"],
            metadata=dict(
                example_id=_ex["question_id"],
                image_id=_ex["image_id"],
                abc_label=_ex["abc_label"],
                has_transparent_box=_ex["has_transparent_box"]
            ),
        )
        options = _ex["answer_texts"]
        if _ex["abc_label"] and sum(_ex["option_is_abc"]) >= (len(options)-1):
            ex["unlabelled_options"] = [
                opt.upper() if abc else opt
                for opt, abc in zip(options, _ex["option_is_abc"])
            ]
            ex["style"] = "ai2_diagram_no_letter"
        else:
            ex["options"] = options
            ex["style"] = "ai2_diagram"
        return ex


class ScienceQAImageOnly(Dataset):
    """
    This class loads the ScienceQA dataset from HuggingFace (https://huggingface.co/datasets/derek-thomas/ScienceQA).
    """
    PATH = "derek-thomas/ScienceQA"

    @classmethod
    def download(self, n_procs=1):
        datasets.load_dataset_builder(self.PATH).download_and_prepare()

    def __init__(self, split):
        assert split in ["train", "validation", "test"]
        self.dataset = datasets.load_dataset(self.PATH, split=split).filter(lambda ex: ex["image"] is not None)
        super().__init__()

    def __len__(self):
        return len(self.dataset)

    def get(self, item, rng):
        ex = self.dataset[item]
        question =  ex["question"]
        hint = ex["hint"]
        if hint:
            question = hint + "\n" + question
        return dict(
            image=ex["image"],
            question=question,
            style="science_qa",
            answer_idx=ex["answer"],
            options=ex["choices"],
        )


class DocQa(HfDataset):
    """
    DocumentVQA dataset from HuggingFace M4 project.
    This class loads the DocumentVQA dataset from HuggingFace (https://huggingface.co/datasets/HuggingFaceM4/DocumentVQA).
    The dataset contains document images paired with questions and answers for visual document understanding tasks.

    Args:
        split (str): Dataset split to load. One of "train", "validation", or "test".
    """
    PATH = "HuggingFaceM4/DocumentVQA"

    def __init__(self, split: str, keep_in_memory=False, **kwargs):
        super().__init__(split, keep_in_memory, **kwargs)

    def get(self, item, rng):
        example = self.dataset[item]
        if self.split == "test":
            for k in ["answers", "question_types"]:
                assert k not in example or example[k] is None
                example[k] = []
        return dict(
                dict(
                image=example["image"],
                question=example["question"],
                answers=example.get("answers"),
                metadata=dict(
                    doc_id=example["docId"],
                    question_types=example.get("question_types"),
                    example_id=example["questionId"],
                )
            ), style="doc_qa")


class CountBenchQa(Dataset):

    @classmethod
    def download(self, n_procs=1):
        local_name = join(ACADEMIC_DATASETS, "countbench_qa")
        CountQaBuilder().download_and_prepare()
        ds = CountQaBuilder().as_dataset("test")
        save_local_dataset(ds, local_name, n_procs)

    def __init__(self):
        self.dataset = datasets.load_from_disk(join(ACADEMIC_DATASETS, "countbench_qa"))

    def __len__(self):
        return len(self.dataset)

    def get(self, item, rng):
        ex = self.dataset[item]
        return {
            'image': ex["image"],
            'question': ex['question'],
            'style': "point_count",
            'metadata': {
                'count': ex['count'],
                'image_id': ex["example_id"],
                'image_url': ex['image_url'],
            }
        }


class TabWMPDirectAnswer(Dataset):

    @classmethod
    def download(cls, n_procs=1):
        local_name = join(ACADEMIC_DATASETS, "tabwmp")
        TabMwpBuilder().download_and_prepare()
        all_data = datasets.DatasetDict()
        for split in ["train", "dev", "test"]:
            ds = TabMwpBuilder().as_dataset(split)
            all_data[split] = ds
        save_local_dataset(all_data, local_name, n_procs)

    def __init__(self, split, include_options: bool, keep_in_memory=False):
        self.include_options = include_options
        self._dataset = datasets.load_from_disk(
            join(ACADEMIC_DATASETS, "tabwmp"), keep_in_memory=keep_in_memory)[split]

    def __len__(self):
        return len(self._dataset)

    def get(self, item, rng):
        ex = self._dataset[item]
        out = dict(
            image=ex["image"],
            question=ex["question"],
            answer=ex["answer"],
            style="tabwmp_da",
            metadata=dict(
                example_id=ex["example_id"]
            )
        )
        if self.include_options and ex["choices"]:
            out["options"] = ex["choices"]
        return out


class FigureQa(Dataset):

    @classmethod
    def download(cls, n_procs=1):
        local_name = join(ACADEMIC_DATASETS, "figure_qa")
        FigureQaBuilder().download_and_prepare()
        all_data = datasets.DatasetDict()
        for split in ["train", "validation1", "test1", "validation2", "test2"]:
            ds = FigureQaBuilder().as_dataset(split)
            all_data[split] = ds
        save_local_dataset(all_data, local_name, n_procs)

    def __init__(self, split, in_memory=False):
        assert split in ["train", "validation1", "test1", "validation2", "test2"]
        self.hf_dataset = datasets.load_from_disk(
            join(ACADEMIC_DATASETS, "figure_qa"), keep_in_memory=in_memory)[split]

    def get(self, item, rng):
        example = self.hf_dataset[int(item)]
        qas = example["questions"]
        messages = []
        for q, a in zip(qas["question"], qas["answer"]):
            messages.append(dict(question=q, answer=str(a), style="figure_qa"))
        return dict(image=example["image"], message_list=messages)

    def __len__(self):
        return len(self.hf_dataset)


class PlotQa(Dataset):

    @classmethod
    def download(cls, n_procs=1):
        local_name = join(ACADEMIC_DATASETS, "plot_qa")
        PlotQaBuilder().download_and_prepare()
        all_data = datasets.DatasetDict()
        for split in ["train", "validation", "test"]:
            ds = PlotQaBuilder().as_dataset(split)
            all_data[split] = ds
        save_local_dataset(all_data, local_name, n_procs)

    @staticmethod
    def _hf_cache_candidates():
        candidates = []
        if "HF_DATASETS_CACHE" in os.environ:
            candidates.append(os.environ["HF_DATASETS_CACHE"])
        if "HF_HOME" in os.environ:
            candidates.append(join(os.environ["HF_HOME"], "datasets"))
        if "MOLMO_DATA_DIR" in os.environ:
            candidates.append(join(os.environ["MOLMO_DATA_DIR"], "hf_datasets"))
        # Preserve order while deduplicating.
        return list(dict.fromkeys(candidates))

    @classmethod
    def _load_from_hf_cache(cls, split, in_memory=False):
        for cache_dir in cls._hf_cache_candidates():
            builder = PlotQaBuilder(cache_dir=cache_dir)
            if exists(builder._output_dir):
                logging.info(f"Loading PlotQA from HF cache at {builder._output_dir}")
                return builder.as_dataset(split, in_memory=in_memory)
        raise FileNotFoundError("missing cached plot_qa dataset")

    def __init__(self, split, in_memory=False):
        assert split in ["train", "validation", "test"]
        try:
            self.hf_dataset = datasets.load_from_disk(
                join(ACADEMIC_DATASETS, "plot_qa"), keep_in_memory=in_memory
            )[split]
        except FileNotFoundError:
            try:
                self.hf_dataset = self._load_from_hf_cache(split, in_memory=in_memory)
            except FileNotFoundError:
                raise FileNotFoundError(
                    f"PlotQA local dataset not found at {join(ACADEMIC_DATASETS, 'plot_qa')}, "
                    f"and no compatible HF cache was found in {self._hf_cache_candidates()}. "
                    "Run `python -c \"from olmo.data.academic_datasets import PlotQa; PlotQa.download()\"` "
                    "with the same MOLMO_DATA_DIR/HF cache environment before training."
                ) from None

    def get(self, item, rng):
        example = self.hf_dataset[int(item)]
        qas = example["questions"]
        messages = []
        for q, a in zip(qas["question"], qas["answer"]):
            messages.append(dict(question=q, answer=a, style="plot_qa"))
        return dict(image=example["image"], message_list=messages)

    def __len__(self):
        return len(self.hf_dataset)


class AndroidControl(Dataset):
    @classmethod
    def download(cls, n_procs=1):
        local_name = join(ACADEMIC_DATASETS, "android_control")
        # AndroidControlBuilder().download_and_prepare(num_proc=n_procs)
        all_data = datasets.DatasetDict()
        for split in ["train", "val", "test"]:
            ds = AndroidControlBuilder().as_dataset(split)
            ds = ds.add_column("id", list(range(len(ds))))
            pil_images = (ex["image"] for ex in ds)
            filenames = [
                join(ANDROID_IMAGES, f"{split}_{example_id:05d}.png")
                for example_id in ds["id"]
            ]
            saved_images = save_images(pil_images, filenames, n_procs)
            assert len(saved_images) == len(filenames)
            def pil_to_path(ex):
                ex["image"] = join(ANDROID_IMAGES, f"{split}_{ex['id']:05d}.png")
                return ex
            new_features = ds.features.copy()
            new_features["image"] = datasets.Value(dtype="string")
            ds = ds.map(pil_to_path, features=new_features)
            ds = ds.remove_columns(["id"])
            all_data[split] = ds
        save_local_dataset(all_data, local_name, n_procs)

    def __init__(self, split, mode="all", in_memory=False):
        self.mode = mode
        self.hf_dataset = datasets.load_from_disk(
            join(ACADEMIC_DATASETS, "android_control"), keep_in_memory=in_memory
        )["val" if split == "validation" else split]

    def __len__(self):
        return len(self.hf_dataset)

    def get(self, item, rng):
        ex = self.hf_dataset[item]
        ll, hl_ll, hl, hl_cot = [
            dict(
                prompt="low_level: " + ex["ll_instruction"],
                text=ex["target_action"],
                style="android_control"
            ),
            dict(
                prompt="high_level: " + ex["hl_instruction"] + " low_level: " + ex["ll_instruction"],
                text=ex["target_action"],
                style="android_control"
            ),
            dict(
                prompt="high_level: " + ex["hl_instruction"],
                text=ex["target_action"],
                style="android_control"
            ),
            dict(
                prompt="high_level_cot: " + ex["hl_instruction"],
                text="Plan: " + ex["ll_instruction"] + " Action: " + ex["target_action"],
                style="android_control"
            )
        ]
        example = dict(
            image=ex["image"],
            metadata=dict(
                target_action=ex["target_action"],
                target_box=ex["target_box"],
                ll_instruction=ex["ll_instruction"],
                hl_instruction=ex["hl_instruction"],
            )
        )
        if self.mode == "ll":
            example.update(ll)
        elif self.mode == "hl":
            example.update(hl)
        elif self.mode == "hl_ll":
            example.update(hl_ll)
        elif self.mode == "hl_cot":
            example.update(hl_cot)
        elif self.mode == "all":
            example["message_list"] = [ll, hl_ll, hl, hl_cot]
        else:
            raise NotImplementedError(self.mode)
        return example


class DvQa(Dataset):
    @classmethod
    def download(cls, n_procs=1):
        local_name = join(ACADEMIC_DATASETS, "dv_qa")
        DvQaBuilder().download_and_prepare()
        all_data = datasets.DatasetDict()
        for split in ["train", "val_hard", "val_easy"]:
            ds = DvQaBuilder().as_dataset(split)
            all_data[split] = ds
        save_local_dataset(all_data, local_name, n_procs)

    def __init__(self, split, in_memory=False):
        self.hf_dataset = datasets.load_from_disk(
            join(ACADEMIC_DATASETS, "dv_qa"), keep_in_memory=in_memory)[split]

    def __len__(self):
        return len(self.hf_dataset)

    def get(self, item, rng):
        example = self.hf_dataset[int(item)]
        qas = example["questions"]
        messages = []
        for q, a in zip(qas["question"], qas["answer"]):
            messages.append(dict(question=q, answer=a, style="dv_qa"))
        return dict(
            image=example["image"],
            message_list=messages,
            metadata=dict(image_id=example["image_id"]),
        )


class MathVista(HfDataset):
    PATH = "AI4Math/MathVista"

    def __init__(self, split, simplify_question=True, **kwargs):
        super().__init__(split, **kwargs)
        self.simplify_question = simplify_question

    def get(self, item, rng):
        ex = self.dataset[item]
        question: str = ex["question"]
        if self.simplify_question:
            question = question.split("Question:")[-1]
            question = question.split("Hint:")[0].strip()
        out = dict(
            question=question,
            image=ex["decoded_image"],
            metadata=dict(
                example_id=ex["pid"],
                answer=ex["answer"],
                precision=ex["precision"],
                query=ex["question"],
                choices=ex["choices"],
                question_type=ex["question_type"],
                answer_type=ex["answer_type"]
            ),
        )
        if ex["question_type"] == "multi_choice":
            out["options"] = ex["choices"]
            out["style"] = "eval_multiple_choice"
        else:
            out["style"] = "eval_short_answer"
        return out


class RealWorldQa(HfDataset):
    PATH = "xai-org/RealworldQA"

    def __init__(self, mode="no_mc_instruction", in_memory=False):
        super().__init__("test", in_memory)
        self.mode = mode

    def get(self, item, rng):
        ex = self.dataset[item]
        prompt: str = ex["question"]
        if "Please answer directly with a single word or number." in prompt:
            question_type = "short_answer"
        else:
            assert "Please answer directly with only the letter of the correct option and nothing else." in prompt
            question_type = "multiple_choice"
        out = dict(
            image=ex["image"],
            metadata=dict(answer=ex["answer"], prompt=ex["question"], question_type=question_type),
        )
        if self.mode == "plain":
            out.update(style="none", prompt=prompt)
        else:
            if question_type == "short_answer":
                style = "eval_short_answer"
            else:
                style = "eval_multiple_choice"
            if self.mode == "no_instruction":
                if question_type == "short_answer":
                    prompt = prompt.split("\n")[0]
            else:
                if self.mode != "vqa_style_tag":
                    raise NotImplementedError(self.mode)
            out.update(style=style, question=prompt)
        return out


class MMMU(Dataset):
    NAMES = [
        'Accounting', 'Agriculture', 'Architecture_and_Engineering', 'Art', 'Art_Theory',
        'Basic_Medical_Science', 'Biology', 'Chemistry', 'Clinical_Medicine', 'Computer_Science',
        'Design', 'Diagnostics_and_Laboratory_Medicine', 'Economics', 'Electronics', 'Energy_and_Power',
        'Finance', 'Geography', 'History', 'Literature', 'Manage', 'Marketing', 'Materials', 'Math',
        'Mechanical_Engineering', 'Music', 'Pharmacy', 'Physics', 'Psychology', 'Public_Health',
        'Sociology'
    ]

    @classmethod
    def download(cls, n_procs=1):
        for name in cls.NAMES:
            if exists(join(DATA_HOME, "mmmu", name)):
                continue
            builder = datasets.load_dataset_builder("MMMU/MMMU", name=name)
            builder.download_and_prepare()

    def __init__(self, split: str, use_multi_image: bool = False):
        all_parts = []
        for name in self.NAMES:
            all_parts.append(datasets.load_dataset("MMMU/MMMU", name=name, split=split))
        self.data = datasets.concatenate_datasets(all_parts)
        self.use_multi_image = use_multi_image

    def __len__(self):
        return len(self.data)
    
    def replace_placeholders(self, all_strings: List[str]) -> List[str]:
        
        replaced = []
        for s in all_strings:
            replaced.append(re.sub(r"<image\s*(\d+)>", r"Image \1", s))
        
        return replaced

    def get(self, item, rng):
        ex = self.data[item]
        mc = ex["question_type"] == "multiple-choice"
        images = [ex[f"image_{i}"] for i in range(1, 8) if ex[f"image_{i}"] is not None]
        if len(images) > 1 and self.use_multi_image:
            style = "mantis_instruct_mc" if mc else "mantis_instruct_da"
            image = images
        else:
            style = 'a_okvqa_mc' if mc else 'vqa2'
            image = ex["image_1"]
        if self.use_multi_image:
            question = self.replace_placeholders([ex["question"]])[0]
        else:
            question = ex["question"]
        out = dict(
            image=image,
            text=ex["answer"],
            question=question,
            metadata=dict(answer=ex["answer"], example_id=ex["id"], question_type=ex["question_type"]),
            style=style
        )
        if mc:
            options = eval(ex["options"])
            if not self.use_multi_image and sum((re.match("<img='(.*?)'>", opt) is not None) for opt in options) > 1:
                # Following LLaVa, don't use any images if there are multiple images paths
                # I think the rationale is that this means the image are answer-options
                del out["image"]
            elif self.use_multi_image:
                options = self.replace_placeholders(options)
            out["options"] = options
        return out


class ClockBench(Dataset):

    @classmethod
    def download(cls, n_procs=1):
        local_name = join(ACADEMIC_DATASETS, "clock_bench")
        ClockBenchBuilder().download_and_prepare()
        all_data = datasets.DatasetDict()
        for split in ["coco", "openimg", "movies"]:
            ds = ClockBenchBuilder().as_dataset(split)
            all_data[split] = ds
        save_local_dataset(all_data, local_name, n_procs)

    def __init__(self, split, keep_in_memory=False):
        assert split in ["coco", "openimg", "movies"]
        dataset = datasets.load_from_disk(
            join(ACADEMIC_DATASETS, "clock_bench"), keep_in_memory=keep_in_memory)[split]
        self.dataset = dataset
        self.split = split

    def __len__(self):
        return len(self.dataset)

    def get(self, item, rng):
        _ex = dict(self.dataset[item])
        hour, minute = [int(_ex[k]) for k in ["hour", "minute"]]
        if hour == 12:
            hour = 0
        second = -1
        return dict(
            image=np.array(_ex["image"]),
            prompt="What time is being shown?",
            metadata=dict(
                hour=hour,
                minute=minute,
                second=second,
                example_id=_ex["image_id"],
            ),
            style="clocks",
        )


def replace_images(question, options, max_images=None):
    all_strings = [question] + options
    image_counter = 1

    total_images = sum(s.count("<image>") for s in all_strings)
    if max_images is not None:
        total_images = min(total_images, max_images)

    replaced = []

    for s in all_strings:
        def repl(match):
            nonlocal image_counter
            if image_counter > total_images:
                return match.group(0)
            replacement = f"Image {image_counter}"
            image_counter += 1
            return replacement

        replaced.append(re.sub(r"<image>", repl, s))

    return replaced[0], replaced[1:]


class MuirBench(Dataset):
    """
    This class loads the MuirBench dataset from HuggingFace (https://huggingface.co/datasets/MUIRBENCH/MUIRBENCH).
    VQA questions that each involve 2-9 images.
    """
    PATH = "MUIRBENCH/MUIRBENCH"

    @classmethod
    def download(cls, n_procs=1):
        local_name = join(ACADEMIC_DATASETS, "muir_bench")
        datasets.load_dataset_builder(cls.PATH).download_and_prepare()
        ds = datasets.load_dataset(cls.PATH)
        save_local_dataset(ds, local_name, n_procs)

    def __init__(self, split: str, format: str = "multiple_choice", legacy=False, keep_in_memory=False):
        if not legacy:
            assert format == "multiple_choice"
        self.format = format
        self.legacy = legacy
        self.dataset = datasets.load_from_disk(
            join(ACADEMIC_DATASETS, "muir_bench"), keep_in_memory=keep_in_memory
        )[split]

    def qo_template(self, question, options, format: str, sep: str = "."):
        question, options = replace_images(question, options)
        option_text = "\n".join(
            f"{chr(ord('A') + idx)}{sep} {options[idx]}" for idx in range(len(options))
        )
        if format == "answer_first":
            question += " Choose the correct option and then explain your reasoning."
        elif format == "answer_last":
            question += " Explain your reasoning and then choose the correct option."
        prompts = [question, option_text]
        if format == "short_answer":
            prompts.append("Select the correct answer from the options above.")
        prompt = "\n".join(prompts)
        return question, prompt, options

    def __len__(self):
        return len(self.dataset)
    
    def get(self, item, rng):
        example = self.dataset[item]
        question, prompt, options = self.qo_template(example['question'], example['options'], self.format)
        out = dict(
            image=example["image_list"],
            metadata=dict(
                example_id=example["idx"],
                task=example["task"],
                image_relation=example["image_relation"],
                image_type=example["image_type"],
                counterpart_id=example["counterpart_idx"],
            )
        )

        answer_idx = ord(example["answer"]) - ord("A")
        if self.format == "multiple_choice":
            out.update(
                question=question,
                options=options,
                answer_idx=answer_idx,
                style="eval_multiple_choice" if self.legacy else "mantis_instruct_mc",
            )
            if not self.legacy:
                out["content_in_mc"] = False
        else:
            out.update(
                question=prompt,
                answer=example["answer"],
                style=f"eval_multi_image_{self.format}",
            )
            out["metadata"]["options"] = options
            out["metadata"]["answer_idx"] = answer_idx
        
        return out


class MantisEval(HfDataset):
    """
    Multi-image reasoning benchmark from the Mantis paper.
    VQA questions that each involve 2-5 images.
    https://huggingface.co/datasets/TIGER-Lab/Mantis-Eval
    """
    PATH = "TIGER-Lab/Mantis-Eval"

    @classmethod
    def download(cls, n_procs=1):
        datasets.load_dataset_builder(cls.PATH).download_and_prepare()
    
    def __init__(self, split: str, question_type: str = "all", legacy=False, keep_in_memory=False):
        assert split in ["test"]
        assert question_type in ["all", "multi-choice", "short-answer"]
        self.legacy = legacy
        super().__init__(split, keep_in_memory=keep_in_memory)
        if question_type != "all":
            self.dataset = self.dataset.filter(lambda ex: ex["question_type"] == question_type)
    
    def question_template(self, question, options, question_type):
        question, _ = replace_images(question, [])
        if question_type == "multi-choice":            
            option_text = "\n".join(options)
            prompts = [
                question,
                option_text,
                "Select the correct answer from the options above.",
            ]
            prompt = "\n".join(prompts)
        else:
            prompt = question
        return prompt

    def get(self, item, rng):
        ex = self.dataset[item]
        prompt = self.question_template(ex["question"], ex["options"], ex["question_type"])

        if self.legacy:
            style = (
                "eval_multi_image_short_answer"
                if ex["question_type"] == "multi-choice"
                else "eval_multi_image_short_answer"
            )
        else:
            style = (
                "mantis_instruct_mc"
                if ex["question_type"] == "multi-choice"
                else "mantis_instruct_da"
            )

        out = dict(
            image=ex["images"],
            question=prompt,
            answer=ex["answer"],
            style=style,
            metadata=dict(
                example_id=ex["id"],
                question_type=ex["question_type"],
                category=ex["category"],
                data_source=ex["data_source"],
                options=[re.sub(r"\([A-Z]\)\s*", "", o.lstrip(), count=1) for o in ex["options"]],
            )
        )

        return out


class MMSIBench(Dataset):
    """
    MMSI-Bench (Multi-Image Spatial Intelligence) — 1 000 multiple-choice
    VQA questions that each involve 2-10 images.
    Hugging Face repo: https://huggingface.co/datasets/RunsenXu/MMSI-Bench
    Paper: arXiv 2505.23764

    ── Splits ────────────────────────────────────────────────────────
      • test   (1 000 rows)  ← the only official split so far
    """
    PATH = "RunsenXu/MMSI-Bench"

    @classmethod
    def download(cls, n_procs=1):
        local_name = join(ACADEMIC_DATASETS, "mmsi_bench")
        datasets.load_dataset_builder(cls.PATH).download_and_prepare()
        ds = datasets.load_dataset(cls.PATH)
        save_local_dataset(ds, local_name, n_procs)
    
    def __init__(self, split: str, format: str = "multiple_choice", keep_in_memory=False):
        assert split in ["test"]
        super().__init__()
        self.format = format
        self.dataset = datasets.load_from_disk(
            join(ACADEMIC_DATASETS, "mmsi_bench"), keep_in_memory=keep_in_memory
        )[split]
    
    def __len__(self):
        return len(self.dataset)
    
    def question_template(self, question, format):
        if format == "answer_first":
            format_instruction = "Choose the correct option and then explain your reasoning."
        elif format == "answer_last":
            format_instruction = "Explain your reasoning and then choose the correct option."
        else:
            format_instruction = "Select the correct answer from the options above."
        prompts = [question, format_instruction]
        prompt = "\n".join(prompts)
        return prompt
    
    def extract_options(self, question):
        matches = re.findall(r'[A-D]:\s*(.*?)(?=\s+[A-D]:|$)', question, flags=re.DOTALL)
        return matches
    
    def get(self, item, rng):
        ex = self.dataset[item]
        prompt = self.question_template(ex["question"], self.format)
        options = self.extract_options(ex["question"])

        format = "mc" if self.format == "multiple_choice" else self.format

        out = dict(
            image=ex["images"],
            question=prompt,
            answer=ex["answer"],
            rationale=ex["thought"],
            style=f"eval_multi_image_{format}",
            metadata=dict(
                example_id=ex["id"],
                question_type=ex["question_type"],
                options=options,
                answer_idx=ord(ex["answer"]) - ord("A"),
            )
        )

        return out


class MMIU(HfDataset):
    """
    MMIU (Multimodal Multi-image Understanding) benchmark
    7 types of multi-image relationships, 52 tasks, 77K images, and 11K meticulously curated multiple-choice questions
    VQA questions that each involve 1-62 images.
    Hugging Face repo: https://huggingface.co/datasets/FanqingM/MMIU-Benchmark
    Paper: arXiv 2408.02718
    """

    PATH = "FanqingM/MMIU-Benchmark"

    # List of image ZIP files in the MMIU repository
    IMAGE_ZIP_FILES = [
        '2D-spatial.zip',
        '3D-spatial.zip', 
        'Continuous-temporal.zip',
        'Discrete-temporal.zip',
        'High-level-obj-semantic.zip',
        'High-level-sub-semantic.zip',
        'Low-level-semantic.zip'
    ]

    @classmethod
    def download(cls, n_procs=1):
        local_name = join(ACADEMIC_DATASETS, "mmiu")
        if exists(local_name):
            return
        from huggingface_hub import hf_hub_download
        import zipfile
        from pathlib import Path

        # Download and unzip the image files
        logging.info("Downloading MMIU images...")
        for zip_file in cls.IMAGE_ZIP_FILES:
            local_zip_file = hf_hub_download(
                repo_id=cls.PATH,
                repo_type="dataset",
                filename=zip_file,
                revision="main",
                local_dir=join(DATA_HOME, "mmiu"),
                local_dir_use_symlinks=False,
            )
            extract_dir = join(DATA_HOME, "mmiu", zip_file.replace(".zip", ""))
            Path(extract_dir).mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(local_zip_file, "r") as zip_ref:
                zip_ref.extractall(extract_dir)
            Path(local_zip_file).unlink()
        
        datasets.load_dataset_builder(cls.PATH).download_and_prepare()
    
    def __init__(self, split: str, format: str = "multiple_choice", legacy=False, keep_in_memory=False):
        assert split in ["test"]
        self.format = format
        self.legacy = legacy
        super().__init__(split, keep_in_memory=keep_in_memory)
    
    def question_template(self, question, options, format: str, sep: str = "."):
        option_text = "\n".join(
            f"{chr(ord('A') + idx)}{sep} {options[idx]}" for idx in range(len(options))
        )
        if format == "answer_first":
            question += " Choose the correct option and then explain your reasoning."
        elif format == "answer_last":
            question += " Explain your reasoning and then choose the correct option."
        prompts = [question, option_text]
        if format == "short_answer":
            prompts.append("Select the correct answer from the options above.")
        prompt = "\n".join(prompts)
        return question, prompt

    def extract_options(self, option_string: str):
        matches = []
        for ix, (_, letter, answer) in enumerate(re.findall(r'(^|\n)([A-Z]):\s?([^\n]+)', option_string, flags=re.DOTALL | re.MULTILINE)):
            assert letter == string.ascii_uppercase[ix]
            matches.append(answer)
        return matches
    
    def get(self, item, rng):
        example_id = str(item)
        ex = self.dataset[item]
        images = [join(DATA_HOME, "mmiu", img[len("./"):]) for img in ex["input_image_path"]]
        relationship = list(set([img.split("/")[1] for img in ex["input_image_path"]]))
        assert len(relationship) == 1, "It should only have one relationship"
        relationship = relationship[0]
        options = self.extract_options(ex["options"])
        question, prompt = self.question_template(ex["question"], options, self.format)

        # Note the ex["options"] is sometimes an option not listed in ex["options"], for
        # example the output will be G but the options will only have A-D
        # Therefore we can't get a true ground-truth answer_idx reliably, although it is not
        # needed given this is an eval set
        # answer_idx = ord(ex["output"]) - ord("A")
        # if len(options) <= answer_idx:
        #     raise ValueError()

        format = "mc" if self.format == "multiple_choice" else self.format

        out = dict(
            image=images,
            question=prompt if self.legacy else question,
            answer=ex["output"],
            style=f"eval_multi_image_{format}" if self.legacy else f"mantis_instruct_{format}",
            metadata=dict(
                example_id=example_id,
                task=ex["task"],
                relationship=relationship,
                context=ex["context"],
                visual_input_component=ex["visual_input_component"],
                source=ex["source"],
                num_images=len(images),
            )
        )

        if self.legacy:
            # out["answer"] = ex["output"]
            out["metadata"]["options"] = options
            # out["metadata"]["answer_idx"] = ord(ex["output"]) - ord("A")
        else:
            out["options"] = options
            # out["metadata"]["answer_idx"] = ord(ex["output"]) - ord("A")
            out["content_in_mc"] = False
        
        return out


class MantisInstruct(Dataset):
    SOURCE = join(DATA_HOME, "mantis-instruct")
    NAMES = [
        "nlvr2",
        "llava_665k_multi",
        "spot-the-diff",
        "nextqa",
        "star",
    ]
    TRAIN_ONLY = [
        "llava_665k_multi",
        "spot-the-diff",
        "nextqa",
        "star",
    ]
    SPLITS = ["train", "validation"]
    PATH = "TIGER-Lab/Mantis-Instruct"

    @classmethod
    def download(cls, n_procs=1, n_val=512):
        from huggingface_hub import snapshot_download
        import zipfile
        from pathlib import Path
        # TODO: Uncomment this before releasing models/datasets
        # import requests

        for name in cls.NAMES:
            local_name = join(ACADEMIC_DATASETS, "mantis-instruct", name)
            if exists(local_name):
                continue

            # Download the dataset
            logging.info(f"Downloading Mantis-Instruct, {name}...")
            snapshot_download(
                repo_id=cls.PATH,
                repo_type="dataset",
                revision="main",
                local_dir=join(DATA_HOME, "mantis-instruct"),
                local_dir_use_symlinks=False,
                allow_patterns=[f"{name}/*", f"{name}/**"],
            )
            splits = ["train"] if name in cls.TRAIN_ONLY else ["train", "val"]
            for split in splits:
                # Unzip the images and remove the zip file
                logging.info(f"Unzipping Mantis-Instruct, {name} images...")
                zip_path = join(DATA_HOME, "mantis-instruct", name, f"{split}_images.zip")
                extract_dir = join(DATA_HOME, "mantis-instruct", name, f"{split}_images")
                Path(extract_dir).mkdir(parents=True, exist_ok=True)
                with zipfile.ZipFile(zip_path, "r") as zip_ref:
                    zip_ref.extractall(extract_dir)
                Path(zip_path).unlink()

                # Download preprocessed oe and mc data
                # TODO: Uncomment this before releasing models/datasets
                # fname = f"{split}.json"
                # url = f"https://storage.googleapis.com/oe-training-public/mantis-instruct/{name}/{fname}"
                # output_path = join(DATA_HOME, "mantis-instruct", name, fname)
                # with requests.get(url, stream=True) as r:
                #     r.raise_for_status()
                #     with open(output_path, "wb") as f:
                #         for chunk in r.iter_content(chunk_size=8192):
                #             f.write(chunk)

            # Prepare the dataset
            builder = MantisInstructBuilder(
                mantis_instruct_source=cls.SOURCE,
                config_name=name,
            )
            logging.info(f"Preparing Mantis-Instruct, {name}...")
            builder.download_and_prepare()
            if name in cls.TRAIN_ONLY:
                dataset = builder.as_dataset(split="train")
                save_local_dataset(dataset, local_name, n_procs, n_val=n_val)
                # all_data = datasets.DatasetDict()
                # all_data["train"] = builder.as_dataset(split="train")
                # save_local_dataset(all_data, local_name, n_procs)
            else:
                all_data = datasets.DatasetDict()
                for split in ["train", "validation"]:
                    all_data[split] = builder.as_dataset(split=split)
                save_local_dataset(all_data, local_name, n_procs)
    
    def __init__(self, name: str, split: str, direct_answer=False, multi_image_only=False, flat=False, sample=None, keep_in_memory=False):
        # TODO: Add support for multi-turn conversations
        assert split in self.SPLITS
        self.split = split
        self.direct_answer = direct_answer
        self.style = "mantis_instruct_" + ("da" if direct_answer else "mc")
        self.data = datasets.load_from_disk(
            join(ACADEMIC_DATASETS, "mantis-instruct", name), keep_in_memory=keep_in_memory
        )[split]
        if multi_image_only:
            self.data = self.data.filter(lambda images: len(images) > 1, input_columns="images")
        self.flat = flat
        if flat:
            flattened_data = []
            for item in self.data:
                for i in range(len(item["mc_question"])):
                    flattened_data.append(dict(
                        subset=item["subset"],
                        example_id=f"{item['example_id']}-{i:03d}",
                        images=item["images"],
                        mc_question=item["mc_question"][i],
                        oe_question=item["oe_question"][i],
                        direct_answer=item["direct_answer"][i],
                        choices=item["choices"][i],
                        correct_choice_idx=item["correct_choice_idx"][i],
                    ))
            if sample:
                logging.info(f"Sampling {sample} of {len(flattened_data)} ({100*sample/len(flattened_data)}:0.1f)")
                np.random.RandomState(9123).shuffle(flattened_data)
                flattened_data = flattened_data[:sample]
            self.data = flattened_data
        else:
            assert sample is None
    
    def __len__(self):
        return len(self.data)
    
    def shuffle_options(self, options: List[str], answer_idx: int, rng: np.random.RandomState):
        perm = rng.permutation(len(options))
        shuffled_options = [options[i] for i in perm]

        inverse_perm = np.empty_like(perm)
        inverse_perm[perm] = np.arange(len(perm))

        shuffled_answer_idx = int(inverse_perm[answer_idx])
        return shuffled_options, shuffled_answer_idx
    
    def get(self, item, rng: np.random.RandomState):
        ex = self.data[item]

        if self.flat:
            question = ex["oe_question"] if self.direct_answer else ex["mc_question"]
            out = dict(
                image=ex["images"],
                question=question,
                metadata=dict(
                    example_id=ex["example_id"],
                    subset=ex["subset"],
                ),
                style=self.style,
            )
            if self.direct_answer:
                out["answer"] = ex["direct_answer"]
            else:
                out["options"], out["answer_idx"] = self.shuffle_options(
                    ex["choices"], ex["correct_choice_idx"], rng
                )
        else:
            questions = ex["oe_question"] if self.direct_answer else ex["mc_question"]
            messages = []
            for i, question in enumerate(questions):
                if self.direct_answer:
                    messages.append(dict(question=question, answer=ex["direct_answer"][i], style=self.style))
                else:
                    options, answer_idx = self.shuffle_options(
                        ex["choices"][i], ex["correct_choice_idx"][i], rng
                    )
                    messages.append(
                        dict(
                            question=question,
                            options=options,
                            answer_idx=answer_idx,
                            style=self.style,
                        )
                    )
            out = dict(
                image=ex["images"],
                message_list=messages,
                metadata=dict(
                    example_id=ex["example_id"],
                    subset=ex["subset"],
                ),
            )
        
        return out


class Ego3dBench(HfDataset):
    """5-7 images per question"""
    PATH = "vbdai/Ego3D-Bench"

    IMAGE_ORDERS = {
        "nuscenes": ['Front_Left','Front','Front_Right','Back_Right','Back','Back_Left'],
        "waymo": ['Front','Front_Left','Side_Left','Front_Right','Side_Right'],
        "argoverse": ['Front_Left','Front','Front_Right','Right','Back_Right','Back_Left','Left'],
    }

    @classmethod
    def download(cls, n_procs=1):
        from huggingface_hub import snapshot_download
        import zipfile
        from pathlib import Path

        # Download the dataset
        logging.info(f"Downloading Ego3D-Bench...")
        snapshot_download(
            repo_id=cls.PATH,
            repo_type="dataset",
            revision="main",
            local_dir=join(DATA_HOME, "Ego3D-Bench"),
            local_dir_use_symlinks=False,
        )

        zip_path = join(DATA_HOME, "Ego3D-Bench", "raw_images.zip")
        extract_dir = join(DATA_HOME, "Ego3D-Bench", "images")
        Path(extract_dir).mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(extract_dir)
        Path(zip_path).unlink()

        datasets.load_dataset_builder(cls.PATH).download_and_prepare()
    
    def __init__(self, split: str, keep_in_memory=False):
        assert split == "test"
        super().__init__(split, keep_in_memory=keep_in_memory)
    
    def replace_placehoders_by_source(self, question: str, source: str) -> str:
        orders = self.IMAGE_ORDERS[source]
        labels = [k.replace("_", " ") for k in orders]
        label_to_num = {label: i+1 for i, label in enumerate(labels)}

        alternation = "|".join(map(re.escape, labels))

        pattern = re.compile(
            rf"(?mi)^({alternation})\s+view\s*:\s*<image>"
        )

        def repl(m):
            view = m.group(1)
            n = label_to_num[view]
            return f"{view} view: Image {n}"  # "<image>" → "Image n"
        
        return pattern.sub(repl, question)
    
    def process_qo(self, question: str, options: List[str] | None) -> str:
        if options is None:
            question += "\nAnswer with a single number only."
        else:
            options = [opt[2:].lstrip() for opt in options]  # Remove prefix "[LETTER]. "
        
        return question, options
    
    def get(self, item, rng):
        ex = self.dataset[item]
        source = ex["source"]
        image_orders = self.IMAGE_ORDERS[source]
        if source == "argoverse":
            image_orders[3] = "Side_Right"
            image_orders[6] = "Side_Left"
        images = [
            join(DATA_HOME, "Ego3D-Bench", "images", ex["images"][view])
            for view in image_orders
        ]
        question = self.replace_placehoders_by_source(ex["question"], source)
        question, options = self.process_qo(question, ex["options"])

        out = dict(
            image=images,
            question=question,
            metadata=dict(
                source=source,
                category=ex["category"],
            )
        )

        if options is None:
            out["answer"] = ex["answer"]
            out["style"] = "eval_multi_image_da"
            out["metadata"]["task"] = "number"
        else:
            out["answer_idx"] = ord(ex["answer"]) - ord("A")
            out["content_in_mc"] = False
            out["options"] = options
            out["style"] = "eval_multi_image_mc"
            out["metadata"]["task"] = "multi_choice"
        
        return out


class BLINK(Dataset):
    """1-4 images per question"""
    NAMES = [
        'Art_Style', 'Functional_Correspondence', 'Multi-view_Reasoning',
        'Relative_Reflectance', 'Visual_Correspondence', 'Counting',
        'IQ_Test', 'Object_Localization', 'Semantic_Correspondence',
        'Visual_Similarity', 'Forensic_Detection', 'Jigsaw',
        'Relative_Depth', 'Spatial_Relation',
    ]

    @classmethod
    def download(cls, n_procs=1):
        for name in cls.NAMES:
            builder = datasets.load_dataset_builder("BLINK-Benchmark/BLINK", name=name)
            builder.download_and_prepare()
    
    def __init__(self, split: str):
        split = "val" if split == "validation" else split
        all_parts = []
        for name in self.NAMES:
            all_parts.append(datasets.load_dataset("BLINK-Benchmark/BLINK", name=name, split=split,
                                                   keep_in_memory=True))
        self.data = datasets.concatenate_datasets(all_parts)

    def __len__(self):
        return len(self.data)
    
    def get(self, item, rng):
        ex = self.data[item]

        images = [ex[f"image_{i}"] for i in range(1, 5) if ex[f"image_{i}"] is not None]
        if len(images) > 1:
            style = "mantis_instruct_mc"
        else:
            style = "eval_multiple_choice"
        
        answer = ex["answer"].replace("(", "").replace(")", "")

        out = dict(
            image=images if len(images) > 1 else images[0],
            question=ex["prompt"],
            answer=answer,
            style=style,
            metadata=dict(
                options=ex["choices"],
                answer_idx=ord(answer) - ord("A"),
                example_id=ex["idx"],
                sub_task=ex["sub_task"],
            )
        )
        return out
    

class Tulu3SftFiltered(Dataset):
    """
    This class loads the MuirBench dataset from HuggingFace (https://huggingface.co/datasets/MUIRBENCH/MUIRBENCH).
    """

    @classmethod
    def format_messages(cls, parts):
        messages = []
        if parts[0]["role"] == "system":
            assert parts[1]["role"] == "user"
            assert "\n" not in parts[0]['content']
            messages.append(f"System: {parts[0]['content']}\n{parts[1]['content']}")
            parts = parts[2:]
        elif parts[0]["role"] == "assistant":
            return None
        else:
            messages.append(parts[0]['content'])
            parts = parts[1:]

        for ix, message in enumerate(parts):
            if ix % 2 == 0:
                if message["role"] != "assistant":
                    return None
            else:
                if message["role"] != "user":
                    return None
            messages.append(message["content"])
        if len(messages) <= 1:
            return None
        return messages

    def __init__(self, split: str):
        assert split == "train"
        src = resource_path(join(ACADEMIC_DATASETS, "tulu3.9-filtered"), "tulu3-filtered.json.gz")
        with gzip.open(src, "r") as f:
            raw_data = json.load(f)
        data = []
        for ex in raw_data:
            messages = self.format_messages(ex["messages"])
            if messages is not None:
                data.append(dict(messages=messages, example_id=ex["id"], source=ex["source"]))
        self.data = data

    def __len__(self):
        return len(self.data)

    def get(self, item, rng):
        ex = self.data[item]
        return dict(
            message_list=[dict(messages=ex["messages"], style="text_sft")],
            metadata=dict(
                example_id=ex["example_id"],
                source=ex["source"],
            )
        )


class Tulu4Filtered(Dataset):
    @classmethod
    def format_messages(cls, parts):
        messages = []
        if parts[0]["role"] == "system":
            assert parts[1]["role"] == "user"
            assert "\n" not in parts[0]['content']
            messages.append(f"System: {parts[0]['content']}\n{parts[1]['content']}")
            parts = parts[2:]
        elif parts[0]["role"] == "assistant":
            return None
        else:
            messages.append(parts[0]['content'])
            parts = parts[1:]

        for ix, message in enumerate(parts):
            if ix % 2 == 0:
                if message["role"] != "assistant":
                    return None
            else:
                if message["role"] != "user":
                    return None
            messages.append(message["content"])
        if len(messages) <= 1:
            return None
        return messages

    def __init__(self, split: str, use_code=False, use_puzzles=False, use_reasoning=False, use_non_english=False, max_first_msg_len=4096):
        self.data = datasets.load_from_disk(
            join(DATA_HOME, "olmo-3-instruct-sft-no-tools-classified-v3"),
            keep_in_memory=False
        )
        def _filter(cls, src, n_tokens, empty_message, has_special_token):
            if empty_message or has_special_token:
                return False
            if src in ["allenai/dino-hardcodes", "allenai/hardcoded-olmo"]:
                return False
            if not use_puzzles and src == "allenai/puzzle_data_160k-ngram-filtered":
                return False
            if not use_reasoning and src in [
                "faezeb/verifiable-reasoning-v3-o4-mini-length-filtered-verified",
                "allenai/verifiable-reasoning-filtered-o4-mini-filtered",
            ]:
                return False
            if not use_code and cls == "code":
                return False
            if not use_non_english and cls == "non-english":
                return False
            if max_first_msg_len and n_tokens > max_first_msg_len:
                return False
            return True
        self.data = self.data.filter(_filter, input_columns=["category", "source", "first_message_qwen3_tokens", "empty_messages", "has_special_token"])

    def __len__(self):
        return len(self.data)

    def get(self, item, rng):
        ex = self.data[item]
        messages = self.format_messages(ex["messages"])
        assert messages is not None
        return dict(
            message_list=[dict(messages=messages, style="text_sft")],
            metadata=dict(
                example_id=ex["id"],
                source=ex["source"],
            )
        )


class MultiImageVLM3R(Dataset):
    """
    Loads:
      v1: /weka/oe-training-default/mm-olmo/multi_image_datasets/3D/VLM-3R/multi-img_vlm3r_4frames.json
      v2: 
        - /weka/oe-training-default/mm-olmo/multi_image_datasets/3D/VLM-3R/multi-img_vlm3r_scannet_mconly.json
        - /weka/oe-training-default/mm-olmo/multi_image_datasets/3D/VLM-3R/multi-img_vlm3r_scannetpp_mconly.json
    v1: 2-6 images per question
    v2: 2-10 images per question

    JSON schema per example:
      {
        "uuid": {
          "paths": [p1, p2, ..., pK],
          "question": "... includes 'Options:' and A./B./C. ...",
          "answer": "B",
          "question_type": "route_planning",
          "scene_name": "...",
          "data_source": "scannet",
          "orig_question": "..."
        },
        ...
      }

    Exposed fields per item:
      - question_type         := question_type
      - example_id            := uuid key
      - images                := paths (list[str])
      - question              := parsed question           
      - choices               := [optA, optB, ...]      
      - correct_choice_idx    := [int]
      - original_question     := original question
      - scene_name            := scene name
      - data_source           := data source
    """
    NAMES = [
        "vlm3r_v1",
        "vlm3r_v2",
    ]
    SPLITS = ["train", "validation"]

    @classmethod
    def download(cls, n_procs: int = 1, val_ratio: float = 0.02):
        for name in cls.NAMES:
            local_name = join(ACADEMIC_DATASETS, "multi_image_vlm3r", name)
            if exists(local_name):
                return
            # Prepare the dataset
            import os
            builder = MultiImageVLM3RBuilder(
                data_source=os.environ["MOLMO_DATA_DIR"], config_name=name
            )
            logging.info(f"Preparing MultiImageVLM3R {name} dataset...")
            builder.download_and_prepare()

            ds = builder.as_dataset(split="train")
            
            # Deterministic split based on example_id hash (stable across runs)
            def split_by_hash(ex: Dict[str, Any], val_ratio: float = val_ratio):
                h = int(hashlib.md5(ex["example_id"].encode("utf-8")).hexdigest(), 16)
                r = (h % 10_000) / 10_000.0
                return {"split": "validation" if r < val_ratio else "train"}
            
            # Add the split column to the dataset
            ds_with_split = ds.map(split_by_hash)

            # Split the dataset into train and validation
            train_ds = ds_with_split.filter(lambda x: x["split"] == "train")
            val_ds = ds_with_split.filter(lambda x: x["split"] == "validation")
            all_data = datasets.DatasetDict(train=train_ds, validation=val_ds)
            
            # Save the dataset to disk
            logging.info(f"Saving MultiImageVLM3R {name} dataset to disk...")
            all_data.save_to_disk(local_name, num_proc=n_procs)
            logging.info("Done")

    def __init__(self, name: str, split: str, keep_in_memory: bool = False, *, seed: int = 9123, sample: int = None):
        assert split in self.SPLITS, f"split must be one of {self.SPLITS}"
        self.split = split
        self.data = datasets.load_from_disk(
            join(ACADEMIC_DATASETS, "multi_image_vlm3r", name), keep_in_memory=keep_in_memory
        )[split]

        # Optional sampling (deterministic)
        if sample is not None and sample < len(self.data):
            logging.info(f"Sampling {sample} of {len(self.data)} ({100*sample/len(self.data):.1f}%)")
            idx = np.arange(len(self.data))
            np.random.RandomState(seed).shuffle(idx)
            idx = idx[:sample]
            self.data = self.data.select(idx)

    def __len__(self):
        return len(self.data)
    
    def get(self, item: int, rng: np.random.RandomState):
        ex = self.data[item]
        return dict(
            image=ex["images"],
            question=ex["question"],
            style="multi_image_mc",
            options=ex["choices"],
            answer_idx=ex["correct_choice_idx"],
            metadata=dict(
                example_id=ex["example_id"],
                question_type=ex["question_type"],
                scene_name=ex["scene_name"],
                data_source=ex["data_source"]
            )
        )

    def shuffle_options(self, options: List[str], answer_idx: int, rng: np.random.RandomState):
        perm = rng.permutation(len(options))
        shuffled = [options[i] for i in perm]
        inverse = np.empty_like(perm)
        inverse[perm] = np.arange(len(perm))
        new_answer_idx = int(inverse[answer_idx])
        return shuffled, new_answer_idx

class Omni3D3DOD(Dataset):
    """
    Omni3D 3D Object Detection dataset in VST format.

    Converted from Omni3D training data with unified FOV (hfov=vfov=69.16°).
    Images resized to 960x960. Contains 6 datasets: SUNRGBD, ARKitScenes,
    Hypersim, KITTI, nuScenes, Objectron.

    Data format:
    - question: Detection prompt with camera FOV parameters
    - answer: JSON list of 3D bounding boxes [x,y,z,w,h,l,pitch,yaw,roll]
    """

    def __init__(self, split: str = "train", sample: int = None):
        """
        Args:
            split: "train" or "validation"
            sample: Optional limit on number of samples (for debugging)
        """
        assert split in ["train", "validation"]
        self.split = split
        self.sample = sample
        self.data = self._load()

    def _load(self):
        """Load JSONL data and split into train/validation."""
        jsonl_path = join(DATA_HOME, "omni3d_3dod", "all_vst.jsonl")

        if not exists(jsonl_path):
            raise FileNotFoundError(
                f"Omni3D 3D detection data not found at {jsonl_path}. "
                f"Please run convert_omni3d_to_vlm.py and link the data to "
                f"$MOLMO_DATA_DIR/torch_datasets/omni3d_3dod/all_vst.jsonl"
            )

        data = []
        with open(jsonl_path, 'r') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line.strip()))

        # Shuffle with fixed seed for reproducibility
        np.random.RandomState(42).shuffle(data)

        # Split: 5% for validation, 95% for training
        n_val = min(5000, len(data) // 20)

        if self.split == "train":
            data = data[n_val:]
        else:  # validation
            data = data[:n_val]

        # Apply sample limit if specified
        if self.sample is not None:
            data = data[:self.sample]

        return data

    def __len__(self):
        return len(self.data)

    def get(self, item, rng):
        """Return a single training example."""
        ex = self.data[item]
        return dict(
            image=ex["image_path"],  # mm_olmo loads from path
            question=ex["question"],
            answer=ex["answers"],    # Keep as string (JSON format)
            style="omni3d_3dod_vst",
            metadata=ex.get("metadata", {})
        )

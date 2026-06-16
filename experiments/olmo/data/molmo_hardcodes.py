import dataclasses
import io
import json
import logging
from os.path import join
import numpy as np
from typing import List

import requests
from PIL import Image

from olmo.io import list_directory, read_file, write_file, file_exists

from olmo.data.dataset import Dataset, DATA_HOME, VIDEO_DATA_HOME
from olmo.util import compute_hash

class Molmo2HardCodes(Dataset):
    HOME = join(DATA_HOME, "molmo2-hardcodes")
    FILE = "hardcodes-v3.json"

    @classmethod
    def download(cls, n_procs=1):
        from olmo.data.vixmo_datasets import VixMoCaptions, VixMoHumanQA
        if not file_exists(join(cls.HOME, "images.json")):
            logging.info("Getting image list")
            images = sorted(list_directory("/weka/oe-training-default/mm-olmo/torch_datasets/pixmo_images"))
            write_file(cls.HOME, "images.json", json.dumps(images), True)
        if not file_exists(join(cls.HOME, "videos.json")):
            logging.info("Getting video list")
            videos = set()
            for ds in [
                lambda: VixMoCaptions("train", subset="all", version="v3", include_merged_caption=True),
                lambda: VixMoHumanQA("train")
            ]:
                ds = ds()
                for ex in ds:
                    videos.add(ex["video"])
            write_file(cls.HOME, "videos.json", json.dumps(sorted(videos)), True)
        for ex in json.loads(read_file(join(cls.HOME, cls.FILE))):
            for url in ex["urls"]:
                key = compute_hash(url)
                src = join(cls.HOME, "images", key)
                if not file_exists(src):
                    logging.info(f"Downloading {url} or {src}")
                    res = requests.get(url)
                    Image.open(io.BytesIO(res.content))
                    write_file(join(cls.HOME, "images"), key, res.content, True)

    def __init__(self, p_video=0.25):
        data = []
        self.p_video = p_video
        self.images = json.loads(read_file(join(self.HOME, "images.json")))
        self.videos = json.loads(read_file(join(self.HOME, "videos.json")))
        raw_data = json.loads(read_file(join(self.HOME, self.FILE)))
        for hardcode in raw_data:
            for question in hardcode["questions"]:
                if hardcode["urls"]:
                    for url in hardcode["urls"]:
                        data.append(dict(
                            question=question,
                            image=join(self.HOME, "images", compute_hash(url)),
                            answer=hardcode["response"]
                        ))
                else:
                    data.append(dict(
                        question=question,
                        answer=hardcode["response"]
                    ))
        self.data = data
        self.options = ["image", "video", "multi-image", "none"]
        self.probs = [0.25, self.p_video, 0.15, 0.35]
        self.probs = np.array(self.probs) / sum(self.probs)

    def __len__(self):
        return len(self.data)

    def get(self, item, rng):
        ex = self.data[item]
        ex = dict(ex, style="user_qa")
        if "image" not in ex:
            src = rng.choice(self.options, p=self.probs)
            if src == "image":
                ex["image"] = rng.choice(self.images)
            elif src == "multi-image":
                n = rng.randint(2, 6)
                ex["image"] = [rng.choice(self.images) for _ in range(n)]
            elif src == "video":
                ex["video"] = join(VIDEO_DATA_HOME, rng.choice(self.videos))
            elif src == "none":
                pass
            else:
                raise RuntimeError()
        return ex



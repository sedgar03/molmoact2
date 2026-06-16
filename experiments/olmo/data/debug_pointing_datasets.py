from collections import defaultdict, Counter

from olmo.data.dataset import Dataset
import numpy as np

from olmo.data.video_loader import VideoFrames

COLORS = [
    [220, 38, 38],
    [37, 99, 235],
    [34, 197, 94],
    [249, 115, 22],
    [168, 85, 247],
    [236, 72, 153],
    [14, 165, 233],
    [234, 179, 8]
]
COLOR_NAMES = [
    "red",
    "blue",
    "green",
    "orange",
    "purple",
    "pink",
    "cyan",
    "yellow",
]


class PointAtTheSquare(Dataset):
    def __init__(self, split, n_examples=100000, min_points=1, max_points=1):
        self.split = split
        self.n_examples = 1000000
        self.min_points = min_points
        self.max_points = max_points

    def __len__(self):
        return self.n_examples

    def get(self, item, rng: np.random.RandomState, seed=None):
        if seed is None:
            seed = rng.randint(0, 2 ** 32 - 1)
            seed += item * 1771
            if self.split == "validation":
                seed += 6923812
            elif self.split != "train":
                raise NotImplementedError()
        rng = np.random.RandomState(seed % (2 ** 32))
        height = rng.randint(256, 512)
        if rng.random() < 0.1:
            width = height
        elif rng.random() < 0.1:
            width = int(height * (4/3))
        else:
            width = rng.randint(256, 512)
        # height, width = 378, 378
        image = np.zeros([height, width, 3], dtype=np.uint8)
        point_size = 14
        point_r = point_size//2

        n_points = rng.randint(self.min_points, self.max_points+1)
        attempts = 0
        cur_points = []
        if n_points != 0:
            while len(cur_points) < n_points and attempts < 5:
                ys = rng.randint(point_r, height - point_r, [n_points])
                xs = rng.randint(point_r, width - point_r, [n_points])
                for point in np.stack([xs, ys], -1):
                    if len(cur_points) == 0:
                        cur_points = point[None, :]
                    else:
                        dist = np.abs(cur_points - point[None, :]).sum(-1)
                        if dist.min() > point_size*3:
                            cur_points = np.insert(cur_points, 0, point[None, :], 0)
                attempts += 1

            for x, y in cur_points:
                assert np.all(image[y-point_r:y+point_r, x-point_r:x+point_r, :] == 0)
                image[y-point_r:y+point_r, x-point_r:x+point_r, :] = 255
        return dict(
            image=image,
            points=cur_points,
            label="square",
            style="pointing",
            metadata=dict(idx=item, seed=seed)
        )


class PointAtTheSquareVideo(Dataset):

    def __init__(
        self, split, n_examples=100000, min_points=1, max_points=1,
        min_frames=8, max_frames=8, unique_colors=False, max_messages=None,
        tracking=False
    ):
        self.split = split
        self.n_examples = 1000000
        self.min_points = min_points
        self.min_frames = min_frames
        self.max_points = max_points
        self.max_frames = max_frames
        self.unique_colors = unique_colors
        self.max_messages = max_messages
        self.tracking = tracking

    def __len__(self):
        return self.n_examples

    def _generate_points(self, msg_ix, image, rng, n_frames, width, height, point_r, check_r):
        n_points = rng.randint(self.min_points, self.max_points+1)
        attempts = 0
        cur_points = []
        if msg_ix is None:
            color = 255
            label = "square"
        else:
            color = COLORS[msg_ix]
            label = f"{COLOR_NAMES[msg_ix]} square"
            assert not self.unique_colors
            assert not self.tracking
        if n_points != 0:
            if self.tracking:
                n_classes = np.random.randint(1, n_points+1)
            while len(cur_points) < n_points and attempts < 10:
                y = rng.randint(point_r, height - point_r)
                x = rng.randint(point_r, width - point_r)
                f = rng.randint(0, n_frames)
                if np.all(image[f, max(y-check_r, 0):y+check_r, max(x-check_r, 0):x+check_r, :] == 0):
                    if self.tracking:
                        current = set(_obj_id for _f, _obj_id, _, _ in cur_points if _f == f)
                        candidates = [i for i in range(n_classes) if i not in current]
                        if len(candidates) == 0:
                            attempts += 1
                            continue
                        obj_id = rng.choice(candidates)
                        image[f, y-point_r:y+point_r, x-point_r:x+point_r, :] = COLORS[obj_id]
                        cur_points.append((f, obj_id, x, y))
                    else:
                        image[f, y-point_r:y+point_r, x-point_r:x+point_r, :] = color
                        cur_points.append((f, x, y))
                else:
                    attempts += 1

        if self.tracking:
            frame_data = {}
            for (f, obj, x, y) in cur_points:
                if f not in frame_data:
                    frame_data[f] = dict(frame=f, time=f/2.0, points={})
                points = frame_data[f]["points"]
                assert obj not in points
                points[obj] = dict(
                    point=(x/width*100, y/height*100),
                    occluded=False
                )
            gt_point_array = np.array(cur_points, dtype=np.float64)
            if cur_points:
                gt_point_array[:, 0] /= 2.0
            return dict(
                points=[frame_data[f] for f in sorted(frame_data)],
                point_scale=100,
                label=label,
                sampling_fps=2.0,
                style="video_point_track_per_frame",
            ), gt_point_array
        else:
            group_by_ts = defaultdict(list)
            for p_ix, (f, x, y) in enumerate(cur_points):
                group_by_ts[f].append((x, y))
            frame_indices = sorted(group_by_ts.keys())
            points = [[dict(x=x/width*100, y=y/height*100) for x, y in group_by_ts[t]] for t in frame_indices]
            gt_point_array = np.array(cur_points, dtype=np.float64)
            if cur_points:
                gt_point_array[:, 0] /= 2.0
            return dict(
                points=points,
                timestamps=[i/2.0 for i in frame_indices],
                label=label,
                style="video_point",
            ), gt_point_array

    def get(self, item, rng: np.random.RandomState, seed=None):
        if seed is None:
            seed = rng.randint(0, 2 ** 32 - 1)
            seed += item * 1771
            if self.split == "validation":
                seed += 6923812
            elif self.split != "train":
                raise NotImplementedError()
        rng = np.random.RandomState(seed % (2 ** 32))
        height = rng.randint(256, 512)
        if rng.random() < 0.1:
            width = height
        elif rng.random() < 0.1:
            width = int(height * (4/3))
        else:
            width = rng.randint(256, 512)
        # height, width = 378, 378
        n_frames = rng.randint(self.min_frames, self.max_frames+1)
        image = np.zeros([n_frames, height, width, 3], dtype=np.uint8)

        point_size = 14
        point_r = point_size//2
        check_r = point_r*3

        if self.max_messages is not None:
            if self.split != "train":
                n_messages = 1
            else:
                n_messages = rng.randint(1, self.max_messages+1)
            messages = []
            for msg_ix in range(n_messages):
                msg, gt_point_array = self._generate_points(msg_ix, image, rng, n_frames, width, height, point_r, check_r)
                messages.append(msg)
            return dict(
                message_list=messages,
                video=VideoFrames(image, np.arange(len(image))/2.0, None),
                metadata=dict(idx=item, seed=seed, gt_point_array=gt_point_array)
            )
        else:
            example, gt_point_array = self._generate_points(None, image, rng, n_frames, width, height, point_r, check_r)
            return dict(
                example,
                video=VideoFrames(image, np.arange(len(image))/2.0, None),
                metadata=dict(idx=item, seed=seed, gt_point_array=gt_point_array)
            )


class PointAtTheSquareMultiImage(Dataset):

    def __init__(self, split, n_examples=100000, min_points=1, max_points=1,
                 min_images=1, max_images=5, max_messages=None):
        self.split = split
        self.n_examples = 1000000
        self.min_points = min_points
        self.min_images = min_images
        self.max_points = max_points
        self.max_images = max_images
        self.max_messages = max_messages

    def __len__(self):
        return self.n_examples

    def _generate_points(self, msg_ix, images, rng, point_r, check_r):
        n_points = rng.randint(self.min_points, self.max_points+1)
        attempts = 0
        cur_points = []
        if msg_ix is None:
            color = 255
            label = "square"
        else:
            color = COLORS[msg_ix]
            label = f"{COLOR_NAMES[msg_ix]} square"
        if n_points != 0:
            while len(cur_points) < n_points and attempts < 10:
                f = rng.randint(0, len(images))
                h, w = images[f].shape[:2]
                y = rng.randint(point_r, h - point_r)
                x = rng.randint(point_r, w - point_r)
                if np.all(images[f][max(y - check_r, 0):y + check_r, max(x - check_r, 0):x + check_r, :] == 0):
                    images[f][y - point_r:y + point_r, x - point_r:x + point_r, :] = color
                    cur_points.append((f, x, y))
                else:
                    attempts += 1
        return dict(
            points=np.array(cur_points, dtype=np.float64),
            label=label,
            style="multi_image_pointing",
        )

    def get(self, item, rng: np.random.RandomState, seed=None):
        if seed is None:
            seed = rng.randint(0, 2 ** 32 - 1)
            seed += item * 1771
            if self.split == "validation":
                seed += 6923812
            elif self.split != "train":
                raise NotImplementedError()
        rng = np.random.RandomState(seed % (2 ** 32))
        n_images = rng.randint(self.min_images, self.max_images+1)
        images = []

        for _ in range(n_images):
            height = rng.randint(256, 512)
            if rng.random() < 0.1:
                width = height
            elif rng.random() < 0.1:
                width = int(height * (4/3))
            else:
                width = rng.randint(256, 512)
            image = np.zeros([height, width, 3], dtype=np.uint8)
            images.append(image)

        point_size = 14
        point_r = point_size//2
        check_r = point_r*3
        if self.max_messages is not None:
            if self.split != "train":
                n_messages = 1
            else:
                n_messages = rng.randint(1, self.max_messages+1)
            messages = []
            for msg_ix in range(n_messages):
                msg = self._generate_points(msg_ix, images, rng, point_r, check_r)
                messages.append(msg)
            return dict(
                message_list=messages,
                image=images,
                metadata=dict(idx=item, seed=seed, image_paths=images)
            )
        else:
            example = self._generate_points(None, images, rng, point_r, check_r)
            return dict(
                example,
                image=images,
                metadata=dict(idx=item, seed=seed, image_paths=images)
            )
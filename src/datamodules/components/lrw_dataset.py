import glob
import os
import random
import sys
from typing import Dict, Literal

import librosa
import numpy as np
import torch
from torch.utils.data import Dataset
from turbojpeg import TJPF_GRAY, TurboJPEG

from ...utils.lipreading_utils import read_txt_lines
from ...utils.video_transform import CenterCrop, HorizontalFlip, RandomCrop

jpeg = TurboJPEG()


# FIXME: Optimize the implementation
class LRWDataset(Dataset):

    # init LRW dataset (cropped mouth)
    def __init__(
        self,
        data_dir: str,
        label_path: str,
        phase: Literal["train", "test", "val"] = "train",
    ):
        self.data_dir = data_dir
        self.label_dir = label_path
        self.phase = phase

        # read label list from file specified in `label_dir`
        with open(label_path, "r") as f:
            self.labels = f.read().splitlines()

        # store list of all video files according to `phase`
        self.list = []

        for (_, label) in enumerate(self.labels):
            # read video list from file specified in `data_dir`
            video_list = glob.glob(os.path.join(data_dir, label, phase, "*.pkl"))
            if len(video_list) == 0:
                continue
            video_list.sort()
            self.list.extend(video_list)

    # get length of dataset
    def __len__(self) -> int:
        return len(self.list)

    # get item from dataset
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # load video data from *.pkl files
        tensor = torch.load(self.list[idx])
        # *.pkl files contains dict of {'video': img, 'label': label, 'duration': duration}
        video = tensor.get("video")
        # video contains 29 frames, convert each frame as grayscale using TJPF_GRAY
        frames = [jpeg.decode(frame, TJPF_GRAY) for frame in video]
        # stack and normalize value between 0 and 1
        frames = np.stack(frames, axis=0) / 255.0
        frames = frames[:, :, :, 0]

        if self.phase == "val" or self.phase == "test":
            # for validation and test phase, only crop region
            img_frames = CenterCrop(frames, (88, 88))
        else:
            img_frames = RandomCrop(frames, (88, 88))
            img_frames = HorizontalFlip(img_frames)

        result = {}
        result["video"] = torch.FloatTensor(img_frames[:, np.newaxis, ...])  # (29, 1, 88, 88)
        # result['label'] = torch.LongTensor([self.labels.index(tensor.get('label'))])
        result["label"] = tensor.get("label")  # label in int
        result["duration"] = 1.0 * tensor.get("duration")  # from bool array to zero or one

        return result

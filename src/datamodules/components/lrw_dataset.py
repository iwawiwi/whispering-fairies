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
class LRWDatasetJPEG(Dataset):

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


class LRWDatasetGray(object):
    def __init__(
        self,
        modality,
        data_partition,
        data_dir,
        label_fp,
        annonation_direc=None,
        preprocessing_func=None,
        data_suffix=".npz",
    ):
        assert os.path.isfile(
            label_fp
        ), "File path provided for the labels does not exist. Path iput: {}".format(label_fp)
        self.__data_partition = data_partition
        self.__data_dir = data_dir
        self.__data_suffix = data_suffix

        self.__label_fp = label_fp  # path to labels file for one video
        self.__annonation_direc = (
            annonation_direc  # path to timestep annotation directory for a word
        )

        self.fps = 25 if modality == "video" else 16000  # frames per second for video and audio
        self.is_var_length = True  # if True, we used augmented version with variable length
        self.__label_idx = -3  # index of label in the path

        self.preprocessing_func = preprocessing_func

        self.__data_files = []

        self.load_dataset()

    def load_dataset(self):

        # -- read the labels file
        self.__labels = read_txt_lines(self._label_fp)

        # -- add examples to self._data_files
        self._get_files_for_partition()

        # -- from self._data_files to self.list
        self.list = dict()
        self.instance_ids = dict()
        for i, x in enumerate(self._data_files):
            label = self._get_label_from_path(x)
            self.list[i] = [x, self._labels.index(label)]
            self.instance_ids[i] = self._get_instance_id_from_path(x)

        print("Partition {} loaded".format(self._data_partition))

    def _get_instance_id_from_path(self, x):
        # for now this works for npz/npys, might break for image folders
        instance_id = x.split("/")[-1]
        return os.path.splitext(instance_id)[0]

    def _get_label_from_path(self, x):
        return x.split("/")[self.label_idx]

    def _get_files_for_partition(self):
        # get rgb/mfcc file paths

        dir_fp = self._data_dir
        if not dir_fp:
            return

        # get npy/npz/mp4 files
        search_str_npz = os.path.join(dir_fp, "*", self._data_partition, "*.npz")
        search_str_npy = os.path.join(dir_fp, "*", self._data_partition, "*.npy")
        search_str_mp4 = os.path.join(dir_fp, "*", self._data_partition, "*.mp4")
        self._data_files.extend(glob.glob(search_str_npz))
        self._data_files.extend(glob.glob(search_str_npy))
        self._data_files.extend(glob.glob(search_str_mp4))

        # If we are not using the full set of labels, remove examples for labels not used
        self._data_files = [
            f for f in self._data_files if f.split("/")[self.label_idx] in self._labels
        ]

    def load_data(self, filename):

        try:
            if filename.endswith("npz"):
                return np.load(filename)["data"]
            elif filename.endswith("mp4"):
                return librosa.load(filename, sr=16000)[0][-19456:]
            else:
                return np.load(filename)
        except IOError:
            print("Error when reading file: {}".format(filename))
            sys.exit()

    def _apply_variable_length_aug(self, filename, raw_data):
        # read info txt file (to see duration of word, to be used to do temporal cropping)
        info_txt = os.path.join(
            self._annonation_direc, *filename.split("/")[self.label_idx :]
        )  # swap base folder
        info_txt = os.path.splitext(info_txt)[0] + ".txt"  # swap extension
        info = read_txt_lines(info_txt)

        utterance_duration = float(info[4].split(" ")[1])
        half_interval = int(utterance_duration / 2.0 * self.fps)  # num frames of utterance / 2

        n_frames = raw_data.shape[0]
        mid_idx = (
            n_frames - 1
        ) // 2  # video has n frames, mid point is (n-1)//2 as count starts with 0
        left_idx = random.randint(
            0, max(0, mid_idx - half_interval - 1)
        )  # random.randint(a,b) chooses in [a,b]
        right_idx = random.randint(min(mid_idx + half_interval + 1, n_frames), n_frames)

        return raw_data[left_idx:right_idx]

    def __getitem__(self, idx):

        raw_data = self.load_data(self.list[idx][0])
        # -- perform variable length on training set
        if (self._data_partition == "train") and self.is_var_length:
            data = self._apply_variable_length_aug(self.list[idx][0], raw_data)
        else:
            data = raw_data
        preprocess_data = self.preprocessing_func(data)
        label = self.list[idx][1]
        return preprocess_data, label

    def __len__(self):
        return len(self._data_files)


def pad_packed_collate(batch):
    if len(batch) == 1:
        data, lengths, labels_np, = zip(
            *[
                (a, a.shape[0], b)
                for (a, b) in sorted(batch, key=lambda x: x[0].shape[0], reverse=True)
            ]
        )
        data = torch.FloatTensor(data)
        lengths = [data.size(1)]

    if len(batch) > 1:
        data_list, lengths, labels_np = zip(
            *[
                (a, a.shape[0], b)
                for (a, b) in sorted(batch, key=lambda x: x[0].shape[0], reverse=True)
            ]
        )

        if data_list[0].ndim == 3:
            max_len, h, w = data_list[
                0
            ].shape  # since it is sorted, the longest video is the first one
            data_np = np.zeros((len(data_list), max_len, h, w))
        elif data_list[0].ndim == 1:
            max_len = data_list[0].shape[0]
            data_np = np.zeros((len(data_list), max_len))
        for idx in range(len(data_np)):
            data_np[idx][: data_list[idx].shape[0]] = data_list[idx]
        data = torch.FloatTensor(data_np)
    labels = torch.LongTensor(labels_np)
    return data, lengths, labels

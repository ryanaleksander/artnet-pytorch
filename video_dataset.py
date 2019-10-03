import torch
import os
import random
from PIL import Image
from torchvision import transforms
import utils


class VideoFramesDataset(torch.utils.data.Dataset):
    """Some Information about VideoFramesDataset"""
    def __init__(self, root_dir, frame_num=16, transform=None):
        super(VideoFramesDataset, self).__init__()

        self.samples = []
        self.transform = transform
        self.frame_num = frame_num

        # Import data in root_dir, each subfolder corresponds to a class label
        cls_lst = os.listdir(root_dir)
        self.num_classes = len(cls_lst)
        for i in range(len(cls_lst)):
            cls_dir = os.path.join(root_dir, cls_lst[i])
            for video in os.listdir(cls_dir):
                video_path = os.path.join(cls_dir, video)
                if len(os.listdir(video_path)) > 0:
                    self.samples.append((os.path.join(cls_dir, video), i))

    def __getitem__(self, index):
        sample = self.samples[index]
        frames = [Image.open(os.path.join(sample[0], f)) for f in os.listdir(sample[0])]

        # Get a random sequence of frames
        frame_index = random.randrange(0, len(frames) - self.frame_num)
        frames = frames[frame_index:frame_index + self.frame_num]

        if self.transform is not None:
            frames = [self.transform(frame) for frame in frames]
        frames = torch.stack(frames)
        return frames, sample[i]

    def __len__(self):
        return len(self.samples)

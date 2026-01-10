from pathlib import Path

import cv2
import imageio.v3 as iio
import torch


class DigitDataSet(torch.utils.data.Dataset):

    def __init__(self, location="./cnn/datasets"):
        all_ims = list(Path(location).rglob('*.png'))
        self.labels = [int(im.parent.name) - 1 for im in all_ims]
        self.images = all_ims
        self.num_classes = len(set(self.labels))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx].as_posix()
        label = self.labels[idx]
        image = iio.imread(image_path, mode='RGB')
        image = cv2.resize(image, (64, 64))
        image = torch.tensor(image).permute(2, 0, 1).float() / 255.0  # HWC to CHW and normalize
        return image, label


class TestDigitDataSet(torch.utils.data.Dataset):

    def __init__(self, location="./cnn/testimages"):
        all_ims = list(Path(location).rglob('*.png'))
        self.images = all_ims

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx].as_posix()
        image = iio.imread(image_path, mode='RGB')
        image = cv2.resize(image, (64, 64))
        image = torch.tensor(image).permute(2, 0, 1).float() / 255.0  # HWC to CHW and normalize
        return image, image_path


def im_transform(image):
    # 灰度图转RGB
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    image = cv2.resize(image, (64, 64))
    image = torch.tensor(image).permute(2, 0, 1).float() / 255.0  # HWC to CHW and normalize
    return image


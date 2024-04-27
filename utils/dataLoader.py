import os
import os.path
import torch.utils.data as data
from PIL import Image
import random
import numpy as np
import cv2


def crop_size(input, depth, gt, scale):
    inner_s = scale - 20
    index_x = 2 * random.randint(0, 9)
    index_y = 2 * random.randint(0, 9)
    new_input = input[index_x : index_x + inner_s, index_y : index_y + inner_s]
    new_depth = depth[index_x : index_x + inner_s, index_y : index_y + inner_s]
    new_gt = gt[index_x : index_x + inner_s, index_y : index_y + inner_s]

    new_input = cv2.resize(new_input, (scale, scale))
    new_depth = cv2.resize(new_depth, (scale, scale))
    new_gt = cv2.resize(new_gt, (scale, scale))

    return new_input, new_depth, new_gt


def RandomHorizontalFlip(input, depth, gt):
    p = 0.5
    if random.random() < p:
        input = np.flip(input, axis=1).copy()
        depth = np.flip(depth, axis=1).copy()
        gt = np.flip(gt, axis=1).copy()
    return input, depth, gt


def make_train_data(data_path):
    print("INFO: Processing Train Data")
    # data_path = os.path.join(data_path, 'train')
    img_list = [
        os.path.splitext(f)[0]
        for f in os.listdir(os.path.join(data_path, "train_gt"))
        if f.endswith(".png")
    ]
    return [
        (
            os.path.join(data_path, "rgb", img_name + ".png"),
            os.path.join(data_path, "train_gt", img_name + ".png"),
            os.path.join(data_path, "nir", img_name + ".png"),
        )
        for img_name in img_list
    ]


def make_test_data(data_path):
    print("INFO: Processing Test Data")
    # data_path = os.path.join(data_path, 'test')
    img_list = [
        os.path.splitext(f)[0]
        for f in os.listdir(os.path.join(data_path, "test_gt"))
        if f.endswith(".png")
    ]
    return [
        (
            os.path.join(data_path, "rgb", img_name + ".png"),
            os.path.join(data_path, "test_gt", img_name + ".png"),
            os.path.join(data_path, "nir", img_name + ".png"),
        )
        for img_name in img_list
    ]


class make_dataSet(data.Dataset):
    def __init__(self, data_path, train=True, rgb_transform=None, grey_transform=None):
        self.train = train
        self.data_path = data_path
        self.rgb_transform = rgb_transform
        self.grey_transform = grey_transform

        if self.train:
            self.images = make_train_data(data_path)
        else:
            self.images = make_test_data(data_path)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        if self.train:
            (
                image_path,
                glass_path,
                nir_path,
            ) = self.images[index]
        else:
            image_path, glass_path, nir_path = self.images[index]

        # image_path, ghost_path = self.images[index]
        image = Image.open(image_path).convert("RGB")
        glass = Image.open(glass_path).convert("L")
        nir = Image.open(nir_path).convert("RGB")

        if self.rgb_transform is not None:
            image = self.rgb_transform(image)

        if self.grey_transform is not None:
            glass = self.grey_transform(glass)
            nir = self.grey_transform(nir)

        if self.train:
            return image, glass, nir
        else:
            return image, glass, nir

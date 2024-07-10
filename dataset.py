# 数据处理
import os
import cv2
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class CSL_Isolated(Dataset):
    def __init__(self, data_path, label_path, frames=16, num_classes=500, train=True, transform=None):
        super(CSL_Isolated, self).__init__()
        self.data_path = data_path
        self.label_path = label_path
        self.train = train
        self.transform = transform
        self.frames = frames
        self.num_classes = num_classes
        # 一个动作有250个样本 五十个人 一人做5次
        self.signers = 50
        self.repetition = 5

        if self.train:
            self.videos_per_folder = int(0.8 * self.signers * self.repetition)
        else:
            self.videos_per_folder = int(0.2 * self.signers * self.repetition)
        # 每个动作的路径
        self.data_folder = []
        try:
            obs_path = [os.path.join(self.data_path, item) for item in os.listdir(self.data_path)]
            self.data_folder = sorted([item for item in obs_path if os.path.isdir(item)])
        except Exception as e:
            print("data path wrong!!!")
            raise
        # 标签
        self.labels = {}
        try:
            label_file = open(self.label_path, 'r', encoding='utf-8')
            for line in label_file.readlines():
                line = line.strip()
                line = line.split('\t')
                self.labels[line[0]] = line[1]
        except Exception as e:
            raise

    def read_images(self, folder_path):
        assert len(os.listdir(folder_path)) >= self.frames, "Too few images in your data folder:" + str(folder_path)
        images = []
        start = 1
        step = int(len(os.listdir(folder_path)) / self.frames)

        for i in range(self.frames):
            index = "{:06d}.jpg".format(start + i * step)
            image = Image.open(os.path.join(folder_path, index))
            if self.transform is not None:
                image = self.transform(image)
            images.append(image)

        # switch dimension to 3d cnn
        images = torch.stack(images, dim=0)
        images = images.permute(1, 0, 2, 3)
        # images = images.permute(0, 1, 2, 3)
        # print(images.shape)
        return images

    def __len__(self):
        return self.num_classes * self.videos_per_folder

    def __getitem__(self, idx):

        # 第几个动作的文件夹 000 001 002...498 499
        top_folder = self.data_folder[int(idx / self.videos_per_folder)]
        selected_folders = [os.path.join(top_folder, item) for item in os.listdir(top_folder)]
        selected_folders = sorted([item for item in selected_folders if os.path.isdir(item)])

        if self.train:
            selected_folder = selected_folders[idx % self.videos_per_folder]
        else:
            selected_folder = selected_folders[idx % self.videos_per_folder + int(0.8 * self.signers * self.repetition)]
        images = self.read_images(selected_folder)
        label = torch.LongTensor([int(idx / self.videos_per_folder)])

        # 返回dict格式的图片和标签
        return {'data': images, 'label': label}

    def label_to_word(self, label):
        if isinstance(label, torch.Tensor):
            return self.labels['{:06d}'.format(label.item())]
        elif isinstance(label, int):
            return self.labels['{:06d}'.format(label)]


if __name__ == '__main__':

    data_path = f"C:\Sign\SLR_Dataset\CSL_Isolated\color_video_125000"
    label_path = f'../SLR_Dataset/CSL_Isolated/dictionary.txt'
    signers = 50
    repetition = 5
    train = True
    if train:
        videos_per_folder = int(0.8 * signers * repetition)
    else:
        videos_per_folder = int(0.2 * signers * repetition)

    # 每个动作的路径
    data_folder = []
    obs_path = [os.path.join(data_path, item) for item in os.listdir(data_path)]
    data_folder = sorted([item for item in obs_path if os.path.isdir(item)])

    print(data_folder)

    # idx = 0-199 200-399
    idx = 199
    top_folder = data_folder[int(idx / videos_per_folder)]
    print(int(idx / videos_per_folder))
    selected_folders = [os.path.join(top_folder, item) for item in os.listdir(top_folder)]
    selected_folders = sorted([item for item in selected_folders if os.path.isdir(item)])

    if train:
        selected_folder = selected_folders[idx % videos_per_folder]
    else:
        selected_folder = selected_folders[idx % videos_per_folder + int(0.8 * signers * repetition)]

    print(selected_folder)

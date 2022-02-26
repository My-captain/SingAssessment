# -*- coding:utf-8 -*-
"""
author: zliu.elliot
@time: 2022-02-20 20-36
@file: dataset.py
"""
import json
import os

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class Quality400Dataset(Dataset):
    """
    400质量检测数据集
    """

    def __init__(self, meta_path, score_id, label_id, feature_dir):
        """
            meta_path:  full path to the file which contains the pitch contour data
            label_id:   the label to use for training
            feature_dir: feature文件夹
        """
        super(Quality400Dataset, self).__init__()
        self.score_id = score_id
        self.feature_dir = feature_dir
        self.performance_data = json.load(open(meta_path, "r"))
        self.label_id = label_id
        self.length = len(self.performance_data)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        performance = self.performance_data[idx]
        label = performance[self.score_id][self.label_id]
        clip_id = performance["clipId"].split(".")[0]
        feature_path = os.path.join(self.feature_dir, f"{clip_id}.npy")
        features = np.load(feature_path)
        return features, label


def collate_fn(batch):
    x = [i[0] for i in batch]
    y = [i[1] for i in batch]
    return torch.tensor(x, dtype=torch.float), torch.tensor(y, dtype=torch.float)


def collate_fn_classifier(batch):
    """
    用于分类任务的Dataloader函数
    """
    def classify_score(score):
        if score <= 60:
            return 0
        elif 60 < score <= 70:
            return 1
        elif 70 < score < 80:
            return 2
        elif score >= 80:
            return 3
    x = [i[0] for i in batch]
    y = [classify_score(i[1]) for i in batch]
    return torch.tensor(x, dtype=torch.float), torch.tensor(y, dtype=torch.long)


def get_dataloader(batch_size, num_workers=0, shuffle=True):
    # TODO: 回归/分类  需要修改collate_fn
    return DataLoader(dataset=Quality400Dataset(
        meta_path="/home/zliu-elliot/workspace/SingAssessment/data/quality_400/clips_metadata_train_国歌.json",
        score_id="songScore", label_id="summary",
        feature_dir="/home/zliu-elliot/workspace/SingAssessment/data/quality_400/clip_feature"),
        batch_size=batch_size, num_workers=num_workers, shuffle=shuffle, collate_fn=collate_fn_classifier), \
           DataLoader(dataset=Quality400Dataset(
               meta_path="/home/zliu-elliot/workspace/SingAssessment/data/quality_400/clips_metadata_valid_国歌.json",
               score_id="songScore", label_id="summary",
               feature_dir="/home/zliu-elliot/workspace/SingAssessment/data/quality_400/clip_feature"),
               batch_size=batch_size, num_workers=num_workers, shuffle=shuffle, collate_fn=collate_fn_classifier), \
           DataLoader(dataset=Quality400Dataset(
               meta_path="/home/zliu-elliot/workspace/SingAssessment/data/quality_400/clips_metadata_test_国歌.json",
               score_id="songScore", label_id="summary",
               feature_dir="/home/zliu-elliot/workspace/SingAssessment/data/quality_400/clip_feature"),
               batch_size=batch_size, num_workers=num_workers, shuffle=shuffle, collate_fn=collate_fn_classifier)


if __name__ == '__main__':
    dataloader = DataLoader(
        Quality400Dataset(meta_path="./quality_400/clips_metadata.json", score_id="songScore", label_id="summary",
                          feature_dir="./quality_400/clip_feature"))
    for i, (data) in enumerate(dataloader):
        print(data)

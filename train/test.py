# -*- coding:utf-8 -*-
"""
author: zliu.elliot
@time: 2022-02-19 13:
@file: test.py
"""
import json

import torch
import torch.nn as nn
from torch.autograd import Variable

import numpy as np
from models.model import CNNSA
from data.dataset import get_dataloader
import tqdm
from sklearn import metrics


def to_var(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)


def collate_fn(clips):
    feature_path = [i["clipId"].replace("wav", "npy") for i in clips]
    features = []
    for i in feature_path:
        f = np.load(f"/home/zliu-elliot/workspace/SingAssessment/data/quality_400/clip_feature/{i}")
        features.append(f)
    return torch.tensor(features, dtype=torch.float), torch.tensor(clips[0]['songScore']['summary'], dtype=torch.float)


def load_and_test(model, model_path, dataloader):
    """
    载入模型并进行测试
    """
    # 载入模型
    S = torch.load(model_path)
    model.load_state_dict(S)
    model = model.cuda()

    # 进行预测
    predictions = []
    targets = []
    for x, y in tqdm.tqdm(dataloader):
        # forward
        x = to_var(x)
        y = to_var(y)
        out = model(x)
        out = out.detach().cpu().numpy()
        predictions.extend(out)
        targets.extend(y.data.cpu().numpy())
    predictions, targets = np.array(predictions), np.array(targets)
    predictions = np.argmax(predictions, axis=-1)

    correct = np.sum(predictions==targets)
    print(f"correct: {correct}")
    print(f"false: {len(predictions)-correct}")
    print(f"Micro Precision : {metrics.precision_score(targets, predictions, average='micro')}")
    delta = predictions - targets
    filter = (delta!=0)
    estimate = np.array((delta[filter], predictions[filter], targets[filter]))
    delta = delta[filter]
    print(f"Delta -3: {np.sum(delta==-3)}")
    print(f"Delta -2: {np.sum(delta==-2)}")
    print(f"Delta -1: {np.sum(delta==-1)}")
    print(f"Delta 1: {np.sum(delta==1)}")
    print(f"Delta 2: {np.sum(delta==2)}")
    print(f"Delta 3: {np.sum(delta==3)}")


def load_and_test_song_level(model, model_path):
    # 载入模型
    S = torch.load(model_path)
    model.load_state_dict(S)
    model = model.cuda()

    # 进行预测
    predictions = []
    targets = []

    metajson = json.load(open("/home/zliu-elliot/workspace/SingAssessment/data/quality_400/clips_metadata_test_国歌_songLevel.json", "r"))

    song_clips = dict()
    for clip in metajson:
        clips = song_clips.get(clip["songId"], list())
        clips.append(clip)
        song_clips[clip["songId"]] = clips

    for song_id, clips in song_clips.items():
        x, y = collate_fn(clips)
        # forward
        x = to_var(x)
        y = to_var(y)
        out = model(x)
        out = out.detach().cpu().numpy()
        predictions.extend(out)
        targets.extend(y.data.cpu().numpy())
    predictions, targets = np.array(predictions), np.array(targets)
    predictions = np.argmax(predictions, axis=-1)

    correct = np.sum(predictions==targets)
    print(f"correct: {correct}")
    print(f"false: {len(predictions)-correct}")
    print(f"Micro Precision : {metrics.precision_score(targets, predictions, average='micro')}")
    delta = predictions - targets
    filter = (delta!=0)
    estimate = np.array((delta[filter], predictions[filter], targets[filter]))
    delta = delta[filter]
    print(f"Delta -3: {np.sum(delta==-3)}")
    print(f"Delta -2: {np.sum(delta==-2)}")
    print(f"Delta -1: {np.sum(delta==-1)}")
    print(f"Delta 1: {np.sum(delta==1)}")
    print(f"Delta 2: {np.sum(delta==2)}")
    print(f"Delta 3: {np.sum(delta==3)}")


if __name__ == '__main__':
    model = CNNSA()
    load_and_test_song_level(model, "/home/zliu-elliot/workspace/SingAssessment/model_serial/CNNSA_四分类_400epoch_无Schedule_国歌_歌曲级分割数据集_cnnsa_lr0.0001.pth")


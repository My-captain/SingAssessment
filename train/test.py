# -*- coding:utf-8 -*-
"""
author: zliu.elliot
@time: 2022-02-19 13:
@file: test.py
"""
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


def load_and_test(model, model_path, dataloader):
    """
    载入模型并进行测试
    """
    # 载入模型
    S = torch.load(model_path)
    model.load_state_dict(S)

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

    # 计算指标
    roc_auc = metrics.roc_auc_score(targets, predictions, average="macro")
    print(f"ROC_AUC:{roc_auc:.2f}")
    return roc_auc


if __name__ == '__main__':
    _, _, test_loader = get_dataloader(128)
    model = CNNSA()
    load_and_test(model, "", test_loader)


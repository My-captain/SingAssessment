import pickle
import os
import time
import numpy as np
import pandas as pd
from sklearn import metrics
from scipy.stats import pearsonr
import datetime
import csv
import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import LabelBinarizer
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable

import models.model as Model


def read_file(tsv_file):
    tracks = {}
    with open(tsv_file) as fp:
        reader = csv.reader(fp, delimiter='\t')
        next(reader, None)  # skip header
        for row in reader:
            track_id = row[0]
            tracks[track_id] = {
                'path': row[3].replace('.mp3', '.npy'),
                'tags': row[5:],
            }
    return tracks


class Solver(object):
    def __init__(self, data_loader, valid_loader, test_loader, config):
        # data loader
        self.test_loader = test_loader
        self.valid_loader = valid_loader
        self.data_loader = data_loader
        self.dataset = config.dataset
        self.data_path = config.data_path
        self.input_length = config.input_length

        # training settings
        self.n_epochs = config.n_epochs
        self.lr = config.lr
        self.use_tensorboard = config.use_tensorboard

        # model path and step size
        self.model_save_path = config.model_save_path
        self.model_load_path = config.model_load_path
        self.log_step = config.log_step
        self.batch_size = config.batch_size
        self.model_type = config.model_type

        # cuda
        self.is_cuda = torch.cuda.is_available()

        # Build model
        self.build_model()

        # Tensorboard
        self.writer = SummaryWriter(f'{config.log_path}_{self.model_type}_lr{self.lr}')

    def get_model(self):
        if self.model_type == 'fcn':
            return Model.FCN()
        elif self.model_type == 'musicnn':
            return Model.Musicnn(dataset=self.dataset)
        elif self.model_type == 'crnn':
            return Model.CRNN()
        elif self.model_type == 'sample':
            return Model.SampleCNN()
        elif self.model_type == 'se':
            return Model.SampleCNNSE()
        elif self.model_type == 'short':
            return Model.ShortChunkCNN()
        elif self.model_type == 'short_res':
            return Model.ShortChunkCNN_Res()
        elif self.model_type == 'cnnsa':
            return Model.CNNSA()
        elif self.model_type == 'hcnn':
            return Model.HarmonicCNN()

    def build_model(self):
        # model
        self.model = self.get_model()

        # cuda
        if self.is_cuda:
            self.model.cuda()

        # load pretrained model
        if len(self.model_load_path) > 1:
            self.load(self.model_load_path)

        # optimizers
        self.optimizer = torch.optim.Adam(self.model.parameters(), self.lr, weight_decay=1e-4)

    def load(self, filename):
        S = torch.load(filename)
        if 'spec.mel_scale.fb' in S.keys():
            self.model.spec.mel_scale.fb = S['spec.mel_scale.fb']
        self.model.load_state_dict(S)

    def to_var(self, x):
        if torch.cuda.is_available():
            x = x.cuda()
        return Variable(x)

    def get_loss_function(self):
        # TODO: 回归/分类
        return nn.MSELoss()
        # return nn.CrossEntropyLoss()

    def train(self):
        # Start training
        start_t = time.time()
        current_optimizer = 'adam'
        loss_function = self.get_loss_function()
        best_metric = -100
        drop_counter = 0

        # Iterate
        for epoch in range(self.n_epochs):
            ctr = 0
            drop_counter += 1
            self.model = self.model.train()
            for x, y in self.data_loader:
                ctr += 1
                # Forward
                x = self.to_var(x)
                y = self.to_var(y)
                out = self.model(x)

                # Backward
                loss = loss_function(out, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # Log
                self.print_log(epoch, ctr, loss, start_t)
            self.writer.add_scalar('Loss/train', loss.item(), epoch)

            # validation
            best_metric = self.validation(best_metric, epoch, self.valid_loader, "Valid")
            self.validation(best_metric, epoch, self.test_loader, "Test")

            # schedule optimizer
            # current_optimizer, drop_counter = self.opt_schedule(current_optimizer, drop_counter)

        print("[%s] Train finished. Elapsed: %s"
              % (datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                 datetime.timedelta(seconds=time.time() - start_t)))

    def opt_schedule(self, current_optimizer, drop_counter):
        # adam to sgd
        if current_optimizer == 'adam' and drop_counter == 80:
            self.load(os.path.join(self.model_save_path, 'best_model.pth'))
            self.optimizer = torch.optim.SGD(self.model.parameters(), 0.001, momentum=0.9, weight_decay=0.0001, nesterov=True)
            current_optimizer = 'sgd_1'
            drop_counter = 0
            print('sgd 1e-3')
        # first drop
        if current_optimizer == 'sgd_1' and drop_counter == 20:
            self.load(os.path.join(self.model_save_path, 'best_model.pth'))
            for pg in self.optimizer.param_groups:
                pg['lr'] = 0.0001
            current_optimizer = 'sgd_2'
            drop_counter = 0
            print('sgd 1e-4')
        # second drop
        if current_optimizer == 'sgd_2' and drop_counter == 20:
            self.load(os.path.join(self.model_save_path, 'best_model.pth'))
            for pg in self.optimizer.param_groups:
                pg['lr'] = 0.00001
            current_optimizer = 'sgd_3'
            print('sgd 1e-5')
        return current_optimizer, drop_counter

    def save(self, filename):
        model = self.model.state_dict()
        torch.save({'model': model}, filename)

    def get_tensor(self, fn):
        # load audio
        if self.dataset == 'mtat':
            npy_path = os.path.join(self.data_path, 'mtat', 'npy', fn.split('/')[1][:-3]) + 'npy'
        elif self.dataset == 'msd':
            msid = fn.decode()
            filename = '{}/{}/{}/{}.npy'.format(msid[2], msid[3], msid[4], msid)
            npy_path = os.path.join(self.data_path, filename)
        elif self.dataset == 'jamendo':
            filename = self.file_dict[fn]['path']
            npy_path = os.path.join(self.data_path, filename)
        raw = np.load(npy_path, mmap_mode='r')

        # split chunk
        length = len(raw)
        hop = (length - self.input_length) // self.batch_size
        x = torch.zeros(self.batch_size, self.input_length)
        for i in range(self.batch_size):
            x[i] = torch.Tensor(raw[i * hop:i * hop + self.input_length]).unsqueeze(0)
        return x

    def print_log(self, epoch, ctr, loss, start_t):
        if (ctr) % self.log_step == 0:
            print("[%s] Epoch [%d/%d] Iter [%d/%d] train loss: %.4f Elapsed: %s" %
                  (datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                   epoch + 1, self.n_epochs, ctr, len(self.data_loader), loss.item(),
                   datetime.timedelta(seconds=time.time() - start_t)))

    def validation(self, best_metric, epoch, dataloader, flag):
        loss, r2 = self.get_validation_score(epoch, dataloader, flag)
        if r2 > best_metric:
            print('best model!')
            best_metric = r2
            torch.save(self.model.state_dict(), os.path.join(self.model_save_path, 'best_model.pth'))
        return best_metric

    def get_validation_score(self, epoch, dataloader, flag):
        self.model = self.model.eval()
        predictions = []
        targets = []
        losses = []
        loss_function = self.get_loss_function()
        for x, y in tqdm.tqdm(dataloader):
            # forward
            x = self.to_var(x)
            y = self.to_var(y)
            out = self.model(x)
            loss = loss_function(out, y)
            losses.append(float(loss.data.cpu()))
            out = out.detach().cpu().numpy()
            predictions.extend(out)
            targets.extend(y.data.cpu().numpy())
        losses = np.mean(np.array(losses))
        predictions, targets = np.array(predictions), np.array(targets)

        # TODO: 回归/分类
        r2 = metrics.r2_score(targets, predictions)
        pearson = pearsonr(predictions, targets)
        print(f"R2:{r2:.3f}\tpearson:{pearson[0]:.3f} p:{pearson[1]:.2f}")
        self.writer.add_scalar(f"Loss/{flag}", losses, epoch)
        self.writer.add_scalar(f"R2/{flag}", r2, epoch)
        self.writer.add_scalar(f"Pearson/{flag}", pearson[0], epoch)
        self.writer.add_scalar(f"Pearson_p/{flag}", pearson[1], epoch)
        return losses, r2
        # roc_auc = metrics.roc_auc_score(targets, predictions, average="macro", multi_class="ovo")
        # predictions = np.argmax(predictions, axis=-1)
        # macro_precision = metrics.precision_score(targets, predictions, average="macro")
        # self.writer.add_scalar(f"Loss/{flag}", losses, epoch)
        # self.writer.add_scalar(f"Macro Precision/{flag}", macro_precision, epoch)
        # self.writer.add_scalar(f"ROC_AUC/{flag}", roc_auc, epoch)
        # return losses, roc_auc

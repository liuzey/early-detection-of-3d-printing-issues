#!/usr/bin/env python
# encoding: utf-8

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F


def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    n_samples = int(source.size()[0])+int(target.size()[0])
    total = torch.cat([source, target], dim=0)
    total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    L2_distance = ((total0-total1)**2).sum(2)
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
    return sum(kernel_val)#/len(kernel_val)


def mmd(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    batch_size = int(source.size()[0])
    kernels = guassian_kernel(source, target,
                              kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
    XX = kernels[:batch_size, :batch_size]
    YY = kernels[batch_size:, batch_size:]
    XY = kernels[:batch_size, batch_size:]
    YX = kernels[batch_size:, :batch_size]
    loss = torch.mean(XX + YY - XY -YX)
    return loss


class ADDneck(nn.Module):

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(ADDneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride

    def forward(self, x):

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu(out)

        return out


class MFSAN(nn.Module):

    def __init__(self, model, num_classes=2):
        super(MFSAN, self).__init__()
        self.sharedNet = model
        self.sonnet1 = ADDneck(2048, 256)
        self.sonnet2 = ADDneck(2048, 256)
        self.sonnet3 = ADDneck(2048, 256)
        self.cls_fc_son1 = nn.Linear(256, num_classes)
        self.cls_fc_son2 = nn.Linear(256, num_classes)
        self.cls_fc_son3 = nn.Linear(256, num_classes)
        self.avgpool = nn.AvgPool2d(7, stride=1)

    def forward(self, data_src, data_tgt=0, label_src=0, mark=1):
        mmd_loss = 0

        if self.training == True:
            data_src = self.sharedNet(data_src)
            data_tgt = self.sharedNet(data_tgt)

            data_tgt_son1 = self.sonnet1(data_tgt)
            data_tgt_son1 = self.avgpool(data_tgt_son1)
            data_tgt_son1 = data_tgt_son1.view(data_tgt_son1.size(0), -1)
            pred_tgt_son1 = self.cls_fc_son1(data_tgt_son1)

            data_tgt_son2 = self.sonnet2(data_tgt)
            data_tgt_son2 = self.avgpool(data_tgt_son2)
            data_tgt_son2 = data_tgt_son2.view(data_tgt_son2.size(0), -1)
            pred_tgt_son2 = self.cls_fc_son2(data_tgt_son2)

            data_tgt_son3 = self.sonnet3(data_tgt)
            data_tgt_son3 = self.avgpool(data_tgt_son3)
            data_tgt_son3 = data_tgt_son3.view(data_tgt_son3.size(0), -1)
            pred_tgt_son3 = self.cls_fc_son3(data_tgt_son3)

            if mark == 1:

                data_src = self.sonnet1(data_src)
                data_src = self.avgpool(data_src)
                data_src = data_src.view(data_src.size(0), -1)
                mmd_loss += mmd(data_src, data_tgt_son1)

                l1_loss = torch.mean( torch.abs(torch.nn.functional.softmax(data_tgt_son1, dim=1)
                                                - torch.nn.functional.softmax(data_tgt_son2, dim=1)) )
                l1_loss += torch.mean( torch.abs(torch.nn.functional.softmax(data_tgt_son1, dim=1)
                                                - torch.nn.functional.softmax(data_tgt_son3, dim=1)) )
                pred_src = self.cls_fc_son1(data_src)

                cls_loss = F.nll_loss(F.log_softmax(pred_src, dim=1), label_src)

                return cls_loss, mmd_loss, l1_loss / 2

            if mark == 2:

                data_src = self.sonnet2(data_src)
                data_src = self.avgpool(data_src)
                data_src = data_src.view(data_src.size(0), -1)
                mmd_loss += mmd(data_src, data_tgt_son2)

                l1_loss = torch.mean(torch.abs(torch.nn.functional.softmax(data_tgt_son2, dim=1)
                                                - torch.nn.functional.softmax(data_tgt_son1, dim=1)))
                l1_loss += torch.mean(torch.abs(torch.nn.functional.softmax(data_tgt_son2, dim=1)
                                                - torch.nn.functional.softmax(data_tgt_son3, dim=1)))
                pred_src = self.cls_fc_son2(data_src)

                cls_loss = F.nll_loss(F.log_softmax(pred_src, dim=1), label_src)

                return cls_loss, mmd_loss, l1_loss / 2

            if mark == 3:

                data_src = self.sonnet3(data_src)
                data_src = self.avgpool(data_src)
                data_src = data_src.view(data_src.size(0), -1)
                mmd_loss += mmd(data_src, data_tgt_son3)

                l1_loss = torch.mean(torch.abs(torch.nn.functional.softmax(data_tgt_son3, dim=1)
                                                - torch.nn.functional.softmax(data_tgt_son1, dim=1)))
                l1_loss += torch.mean(torch.abs(torch.nn.functional.softmax(data_tgt_son3, dim=1)
                                                - torch.nn.functional.softmax(data_tgt_son2, dim=1)))
                pred_src = self.cls_fc_son3(data_src)

                cls_loss = F.nll_loss(F.log_softmax(pred_src, dim=1), label_src)

                return cls_loss, mmd_loss, l1_loss / 2

        else:
            data = self.sharedNet(data_src)

            fea_son1 = self.sonnet1(data)
            fea_son1 = self.avgpool(fea_son1)
            fea_son1 = fea_son1.view(fea_son1.size(0), -1)
            pred1 = self.cls_fc_son1(fea_son1)

            fea_son2 = self.sonnet2(data)
            fea_son2 = self.avgpool(fea_son2)
            fea_son2 = fea_son2.view(fea_son2.size(0), -1)
            pred2 = self.cls_fc_son2(fea_son2)

            fea_son3 = self.sonnet3(data)
            fea_son3 = self.avgpool(fea_son3)
            fea_son3 = fea_son3.view(fea_son3.size(0), -1)
            pred3 = self.cls_fc_son3(fea_son3)

            return pred1, pred2, pred3

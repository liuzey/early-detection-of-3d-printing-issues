import numpy as np
import os
import time
import sys
import copy
import torch
import torch.optim as optim
import torch.utils.data
from torchvision import datasets
from torchvision import transforms
from sklearn.metrics import confusion_matrix


def test(model, loader, epoch, epochs, device, log_interval=20):
    model.eval()
    n_correct = 0
    tn, fn, tp, fp = 0, 0, 0, 0
    len_dataloader = len(loader)
    start = time.time()
    with torch.no_grad():
        for i, (t_img, t_label) in enumerate(loader):
            t_img, t_label = t_img.to(device), t_label.to(device)
            class_output = model(t_img)
            pred = torch.max(class_output.data, 1)
            n_correct += (pred[1] == t_label).sum().item()
            cm = confusion_matrix(pred[1].cpu().data, t_label.cpu().data, labels=[0, 1])
            tn += cm[0][0]
            fn += cm[1][0]
            tp += cm[1][1]
            fp += cm[0][1]
            f1 = 2 * tp / (2 * tp + fp + fn)

            if i % log_interval == 0:
                print(
                    'Batch: [{}/{}], f1: {:.4f}, time used: {:.1f}s'.format(
                        i, len_dataloader, f1, time.time()-start
                        ))

    accu = float(n_correct) / len(loader.dataset) * 100
    print('------Epoch: [{}/{}], accuracy: {:.4f}%, f-1 score: {:.4f}------'.format(epoch, epochs, accu, f1))
    return accu, f1


def vanilla_train(model, optimizer, scheduler, trainloader, testloader,
                  saved_name, epochs, device, log_interval=20):
    loss_class = torch.nn.CrossEntropyLoss()
    best_f1 = 0
    for epoch in range(1, epochs + 1):
        model.train()
        len_dataloader = len(trainloader)
        data_iter = iter(trainloader)
        i = 1
        tn, fn, tp, fp = 0, 0, 0, 0
        start = time.time()
        while i < len_dataloader + 1:
            data_source = data_iter.next()
            optimizer.zero_grad()
            img, label = data_source[0].to(device), data_source[1].to(device)
            class_output = model(img)
            err = loss_class(class_output, label)
            err.backward()
            optimizer.step()
            scheduler.step()

            pred = torch.max(class_output.data, 1)
            cm = confusion_matrix(pred[1].cpu().data, label.cpu().data, labels=[0, 1])
            tn += cm[0][0]
            fn += cm[1][0]
            tp += cm[1][1]
            fp += cm[0][1]
            f1 = 2 * tp / (2 * tp + fp + fn)

            if i % log_interval == 0:
                print(
                    'Epoch: [{}/{}], Batch: [{}/{}], err: {:.4f}, f-1 score: {:.4f}, time used: {:.1f}s'.format(
                        epoch, epochs, i, len_dataloader, err.item(), f1, time.time()-start
                        ))
            i += 1

        acc, f1 = test(model, testloader, epoch, epochs, device, log_interval=log_interval)
        if f1 > best_f1:
            torch.save(model.state_dict(), saved_name)
            best_f1 = f1

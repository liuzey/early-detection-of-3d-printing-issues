import torch
import torch.utils.data
from torchvision import datasets
from torchvision import transforms
from .dataset import CustomDataset


def load_data(label_path="/home/liuzey/disk/intern/TDS/early-detection-of-3d-printing-issues/little_train.csv",
              data_path="/home/liuzey/disk/intern/TDS/early-detection-of-3d-printing-issues/images/Train",
              mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), img_size=224, batch_size=128):
    img_transform = transforms.Compose([
        transforms.RandomRotation(10),
        transforms.RandomPerspective(distortion_scale=0.1, p=0.1),
        transforms.RandomResizedCrop((img_size, img_size), scale=(0.9, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(
            brightness=0.1, contrast=0.1, hue=0.1, saturation=0.1
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    dataset = CustomDataset(label_path, data_path, transform=img_transform)
    dataloader = torch.utils.data.DataLoader(
        dataset=dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=8)
    return dataloader


def load_test_data(label_path="/home/liuzey/disk/intern/TDS/early-detection-of-3d-printing-issues/little_valid.csv",
                   data_path="/home/liuzey/disk/intern/TDS/early-detection-of-3d-printing-issues/images/Valid",
                   mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), img_size=224, batch_size=128):
    img_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    dataset = CustomDataset(label_path, data_path, transform=img_transform)
    dataloader = torch.utils.data.DataLoader(
        dataset=dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=8)
    return dataloader
import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, csv_path, images_folder, transform=None):
        self.df = pd.read_csv(csv_path)
        self.images_folder = images_folder
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        filename = self.df.loc[index, "img_path"]
        label = self.df.loc[index, "has_under_extrusion"]
        image = Image.open(os.path.join(self.images_folder, filename))
        if self.transform is not None:
            image = self.transform(image)
        return image, label


if __name__ == '__main__':
    a = CustomDataset("../train.csv", "../images/Train")
    train_dataloader = DataLoader(a, batch_size=64, shuffle=True)
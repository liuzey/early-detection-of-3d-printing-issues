import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms
from torchinfo import summary
from model.residual_attention_network import ResidualAttentionModel_56
import numpy as np
from utils.dataloader import load_data, load_test_data
from utils.train import vanilla_train, test
from PIL import Image
import pandas as pd

DATASET_MEAN = [0.2925814, 0.2713622, 0.14409496]
DATASET_STD = [0.0680447, 0.06964592, 0.0779964]
IMG_SIZE = 224
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
LEARNING_RATE = 1e-3
TRAIN_BATCH_SIZE = 48
TEST_BATCH_SIZE = 48
N_EPOCH = 50
LOG_INTERVAL = 30
SAVED_MODEL_NAME = "./little.pth"
SEED = 8888

origin = pd.read_csv("./test.csv")
dict_ = dict()
dict_["img_path"] = list()
dict_["has_under_extrusion"] = list()

img_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(DATASET_MEAN, DATASET_STD),
    ])
model = ResidualAttentionModel_56().to(DEVICE)
model.load_state_dict(torch.load("./little.pth"), strict=True)
model.eval()

for ind in origin.index:
    filename = origin.loc[ind, "img_path"]
    image = Image.open(os.path.join("/home/liuzey/disk/intern/TDS/early-detection-of-3d-printing-issues/images/Test",
                                    filename))
    tensor = img_transform(image).to(DEVICE).unsqueeze(0)
    class_output = model(tensor)
    pred = int(torch.max(class_output.data, 1)[1][0].cpu().data)
    dict_["img_path"].append(filename)
    dict_["has_under_extrusion"].append(pred)
    if ind % 100 == 0:
        print(ind)
    # if ind > 200:
    #     break

df = pd.DataFrame(dict_, columns=["img_path", "has_under_extrusion"])
df.to_csv("./little_1.csv", index=False)



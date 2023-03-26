import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchinfo import summary
from model.residual_attention_network import ResidualAttentionModel_56
import numpy as np
from utils.dataloader import load_data, load_test_data
from utils.train import vanilla_train, test
from torchvision import transforms

# a = ResidualAttentionModel_56()
# summary(a, (1, 3, 224, 224))

# b = torchvision.models.vit_b_16(torchvision.models.ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1)
# summary(b, (1, 3, 384, 384))

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

TRAIN_LOADER = load_data(mean=DATASET_MEAN, std=DATASET_STD, img_size=IMG_SIZE, batch_size=TRAIN_BATCH_SIZE)
TEST_LOADER = load_test_data(mean=DATASET_MEAN, std=DATASET_STD, img_size=IMG_SIZE, batch_size=TEST_BATCH_SIZE)

torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True
torch.manual_seed(SEED)
np.random.seed(SEED)

model = ResidualAttentionModel_56().to(DEVICE)
for param in model.parameters():
    param.requires_grad = True

# optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
# scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=3, threshold=0.01)
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, N_EPOCH, eta_min=1e-7)
test(model, TEST_LOADER, -1, N_EPOCH, DEVICE)
vanilla_train(model, optimizer, scheduler, TRAIN_LOADER, TEST_LOADER,
                  SAVED_MODEL_NAME, N_EPOCH, DEVICE, log_interval=LOG_INTERVAL)



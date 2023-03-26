'''
This is from this github repository:
https://github.com/cam-cambridge/caxton
'''

import torch.nn as nn
from .basic_layers import ResidualBlock
from .attention_module import (
    AttentionModule_stage1,
    AttentionModule_stage2,
    AttentionModule_stage3,
)


class ResidualAttentionModel_56(nn.Module):
    # for input size 224 x 224
    def __init__(self, num_class=2):
        super(ResidualAttentionModel_56, self).__init__()
        self.retrieve_layers = False
        self.retrieve_masks = False
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.mpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.residual_block1 = ResidualBlock(64, 256)
        self.attention_module1 = AttentionModule_stage1(
            256, 256, retrieve_mask=self.retrieve_masks
        )
        self.residual_block2 = ResidualBlock(256, 512, 2)
        self.attention_module2 = AttentionModule_stage2(
            512, 512, retrieve_mask=self.retrieve_masks
        )
        self.residual_block3 = ResidualBlock(512, 1024, 2)
        self.attention_module3 = AttentionModule_stage3(
            1024, 1024, retrieve_mask=self.retrieve_masks
        )
        self.residual_block4 = ResidualBlock(1024, 2048, 2)
        self.residual_block5 = ResidualBlock(2048, 2048)
        self.residual_block6 = ResidualBlock(2048, 2048)
        self.mpool2 = nn.Sequential(
            nn.BatchNorm2d(2048),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=7, stride=1),
        )
        self.fc = nn.Linear(2048, num_class)

    def forward(self, x):
        out = self.conv1(x)
        out = self.mpool1(out)
        out = self.residual_block1(out)
        out = self.attention_module1(out)
        out = self.residual_block2(out)
        out = self.attention_module2(out)
        out = self.residual_block3(out)
        out = self.attention_module3(out)
        out = self.residual_block4(out)
        out = self.residual_block5(out)
        out = self.residual_block6(out)
        out = self.mpool2(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

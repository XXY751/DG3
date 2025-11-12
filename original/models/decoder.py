import torch
import torch.nn as nn
import torch.nn.functional as F
from models.transformer import TransformerEncoder


# (不再需要 PSDNorm)

class Decoder(nn.Module):
    def __init__(self, params):
        super(Decoder, self).__init__()
        self.params = params

        # --- Block 1: 换回 BatchNorm1d ---
        self.tconv1 = nn.ConvTranspose1d(512, 512, kernel_size=5, stride=5, padding=0, bias=False)
        self.norm1 = nn.BatchNorm1d(512)  # <--- 换回
        self.drop1 = nn.Dropout(params.dropout)

        # --- Block 2: 换回 BatchNorm1d ---
        self.tconv2 = nn.ConvTranspose1d(512, 256, kernel_size=10, stride=2, padding=4, bias=False)
        self.norm2 = nn.BatchNorm1d(256)  # <--- 换回
        self.drop2 = nn.Dropout(params.dropout)

        # --- Block 3: 换回 BatchNorm1d ---
        self.tconv3 = nn.ConvTranspose1d(256, 128, kernel_size=10, stride=2, padding=4, bias=False)
        self.norm3 = nn.BatchNorm1d(128)  # <--- 换回
        self.drop3 = nn.Dropout(params.dropout)

        # --- Block 4: 换回 BatchNorm1d ---
        self.tconv4 = nn.ConvTranspose1d(128, 64, kernel_size=10, stride=2, padding=4, bias=False)
        self.norm4 = nn.BatchNorm1d(64)  # <--- 换回
        self.drop4 = nn.Dropout(params.dropout)

        # --- Block 5: 换回 BatchNorm1d ---
        self.tconv5 = nn.ConvTranspose1d(64, 64, kernel_size=5, stride=5, padding=0, bias=False)
        self.norm5 = nn.BatchNorm1d(64)  # <--- 换回
        self.drop5 = nn.Dropout(params.dropout)

        # --- Final Block ---
        self.tconv_final = nn.ConvTranspose1d(64, 2, kernel_size=49, stride=15, padding=17, bias=False)

    def forward(self, x):
        bz = x.shape[0]
        x = x.view(bz * 20, 512, 1)

        # (恢复 ConvT -> BatchNorm -> GELU -> Dropout 的原始结构)

        # Block 1
        x = self.tconv1(x)
        x = self.norm1(x)  # BatchNorm 在激活前
        x = F.gelu(x)
        x = self.drop1(x)

        # Block 2
        x = self.tconv2(x)
        x = self.norm2(x)  # BatchNorm 在激活前
        x = F.gelu(x)
        x = self.drop2(x)

        # Block 3
        x = self.tconv3(x)
        x = self.norm3(x)  # BatchNorm 在激活前
        x = F.gelu(x)
        x = self.drop3(x)

        # Block 4
        x = self.tconv4(x)
        x = self.norm4(x)  # BatchNorm 在激活前
        x = F.gelu(x)
        x = self.drop4(x)

        # Block 5
        x = self.tconv5(x)
        x = self.norm5(x)  # BatchNorm 在激活前
        x = F.gelu(x)
        x = self.drop5(x)

        # Final Block
        x = self.tconv_final(x)

        return x.view(bz, 20, 2, 3000)
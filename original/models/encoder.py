import torch
import torch.nn as nn
import torch.nn.functional as F
from models.transformer import TransformerEncoder

# 假设 PSDNorm1d 在一个名为 psdnorm.py 的文件中定义，并位于同一 'models' 目录下
# (Assuming PSDNorm1d is defined in a file named psdnorm.py in the same 'models' directory)
try:
    from .psdnorm import PSDNorm1d
except ImportError:
    print("WARNING: Could not import PSDNorm1d. Ensure psdnorm.py is in the same directory.")
    PSDNorm1d = nn.BatchNorm1d  # Fallback to BatchNorm if PSDNorm isn't found


class Encoder(nn.Module):
    def __init__(self, params):
        super(Encoder, self).__init__()
        self.params = params
        self.epoch_encoder = EpochEncoder(self.params)
        self.seq_encoder = TransformerEncoder(
            seq_length=20,
            num_layers=1,
            num_heads=8,
            hidden_dim=512,
            mlp_dim=512,
            dropout=self.params.dropout,
            attention_dropout=self.params.dropout,
        )
        self.fc_mu = nn.Linear(512, 512)
        # self.fc_log_var  = nn.Linear(512, 512)

    def forward(self, x):
        bz = x.shape[0]

        x = x.view(bz * 20, 2, 3000)
        x = self.epoch_encoder(x)
        x_epoch = x.view(bz, 20, -1)

        x_seq = self.seq_encoder(x_epoch)
        mu = self.fc_mu(x_seq)
        # log_var = self.fc_log_var(x_seq)
        return mu


class EpochEncoder(nn.Module):
    def __init__(self, params):
        super(EpochEncoder, self).__init__()

        # --- Block 1: 使用 PSDNorm (F=5) ---
        self.conv1 = nn.Conv1d(2, 64, kernel_size=49, stride=6, bias=False, padding=24)
        # 按照您的要求，使用 F_len=5
        self.norm1 = PSDNorm1d(num_channels=64, F_len=5, momentum=1e-2)
        self.pool1 = nn.MaxPool1d(kernel_size=9, stride=2, padding=4)
        self.drop1 = nn.Dropout(params.dropout)

        # --- Block 2: 使用 PSDNorm (F=5) ---
        self.conv2 = nn.Conv1d(64, 128, kernel_size=9, stride=1, bias=False, padding=4)
        self.norm2 = PSDNorm1d(num_channels=128, F_len=5, momentum=1e-2)
        self.pool2 = nn.MaxPool1d(kernel_size=9, stride=2, padding=4)

        # --- Block 3: 换回 BatchNorm1d ---
        self.conv3 = nn.Conv1d(128, 256, kernel_size=9, stride=1, bias=False, padding=4)
        self.norm3 = nn.BatchNorm1d(256)  # <--- 换回
        self.pool3 = nn.MaxPool1d(kernel_size=9, stride=2, padding=4)

        # --- Block 4: 换回 BatchNorm1d ---
        self.conv4 = nn.Conv1d(256, 512, kernel_size=9, stride=1, bias=False, padding=4)
        self.norm4 = nn.BatchNorm1d(512)  # <--- 换回
        self.pool4 = nn.MaxPool1d(kernel_size=9, stride=2, padding=4)

        self.avg = nn.AdaptiveAvgPool1d(1)
        # self.layer_norm = LayerNorm(512)

    def forward(self, x: torch.tensor):
        # Block 1 (Conv -> GELU -> PSDNorm)
        x = self.conv1(x)
        x = F.gelu(x)
        x = self.norm1(x)  # PSDNorm 在激活后
        x = self.pool1(x)
        x = self.drop1(x)

        # Block 2 (Conv -> GELU -> PSDNorm)
        x = self.conv2(x)
        x = F.gelu(x)
        x = self.norm2(x)  # PSDNorm 在激活后
        x = self.pool2(x)

        # Block 3 (Conv -> BatchNorm -> GELU - 恢复原始结构)
        x = self.conv3(x)
        x = self.norm3(x)  # BatchNorm 在激活前
        x = F.gelu(x)
        x = self.pool3(x)

        # Block 4 (Conv -> BatchNorm -> GELU - 恢复原始结构)
        x = self.conv4(x)
        x = self.norm4(x)  # BatchNorm 在激活前
        x = F.gelu(x)
        x = self.pool4(x)

        x = self.avg(x).squeeze()
        return x
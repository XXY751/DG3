# improved/models/virtual_modules.py
# (这是修正维度和通道不匹配后的版本)

import torch
import torch.nn as nn
import torch.nn.functional as F


# --- 1. Style Encoder (E_style) ---

class StyleEncoder(nn.Module):
    def __init__(self, input_dim=3000, latent_dim=64, in_channels=2):  # <--- 更改 1: 默认 in_channels=2
        super().__init__()
        self.latent_dim = latent_dim
        self.input_dim = input_dim

        # 简单的 1D CNN 来提取风格
        self.encoder = nn.Sequential(
            # ==================================================================
            # 更改 1: Conv1d(1, 32, ...) -> Conv1d(in_channels, 32, ...)
            # 你的日志显示主模型使用 2 个通道
            nn.Conv1d(in_channels, 32, kernel_size=25, stride=1, padding=12),
            # ==================================================================
            nn.InstanceNorm1d(32), nn.LeakyReLU(0.2),
            nn.AvgPool1d(kernel_size=10, stride=5),  # (B, 32, 599)

            nn.Conv1d(32, 64, kernel_size=11, stride=1, padding=5),
            nn.InstanceNorm1d(64), nn.LeakyReLU(0.2),
            nn.AvgPool1d(kernel_size=10, stride=5),  # (B, 64, 119)

            nn.Conv1d(64, 128, kernel_size=5, stride=1, padding=2),
            nn.InstanceNorm1d(128), nn.LeakyReLU(0.2),
            nn.AvgPool1d(kernel_size=10, stride=5),  # (B, 128, 23)

            nn.Flatten(),  # (B, 128 * 23 = 2944)
            nn.Linear(128 * 22, 256),
            nn.LeakyReLU(0.2)
        )

        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_log_sigma = nn.Linear(256, latent_dim)

    # ==================================================================
    # 更改 2: 替换整个 forward 函数
    # 新函数可以处理 (B, T, C, L) 格式的输入
    # ==================================================================
    def forward(self, x):
        # x 可能是 (B, T, C, L), e.g., [32, 20, 2, 3000]
        # 也可能是 (B, C, L)

        original_shape = x.shape
        B = original_shape[0]

        if x.dim() == 4:
            # (B, T, C, L) -> (B*T, C, L)
            T, C, L = original_shape[1], original_shape[2], original_shape[3]
            x = x.reshape(B * T, C, L)
        elif x.dim() == 2:
            # (B, L) -> (B, 1, L)
            x = x.unsqueeze(1)
        # 3D (B, C, L) 输入保持不变

        # --- 确保 L 维度 (信号长度) 匹配 ---
        if x.size(-1) > self.input_dim:
            x = x[..., :self.input_dim]
        elif x.size(-1) < self.input_dim:
            pad_width = self.input_dim - x.size(-1)
            x = F.pad(x, (0, pad_width))

        # --- 编码 ---
        # h shape: (B*T, 256) 或 (B, 256)
        h = self.encoder(x)

        # mu / log_sigma shape: (B*T, latent_dim) 或 (B, latent_dim)
        mu = self.fc_mu(h)
        log_sigma = self.fc_log_sigma(h)


        # --- 如果输入是4D，我们需要将T维度平均掉 ---
        if len(original_shape) == 4:
            T = original_shape[1]  # e.g., 20
            latent_dim = mu.shape[-1]  # e.g., 64

            # (B*T, D) -> (B, T, D)
            mu = mu.view(B, T, latent_dim)
            log_sigma = log_sigma.view(B, T, latent_dim)

            # (B, T, D) -> (B, D) (取序列的平均风格)
            mu = mu.mean(dim=1)
            log_sigma = log_sigma.mean(dim=1)

            # 7. Return (B, D) (e.g., [32, 64])
        return mu, log_sigma
    # ==================================================================


# --- 2. Generator / Decoder (G) ---
# (这个类保持不变)
class Generator(nn.Module):
    def __init__(self, z_c_dim=256, style_dim=64, num_classes=5, output_dim=3000,
                 out_channels=2):  # <--- 新增 out_channels=2
        super().__init__()
        self.output_dim = output_dim

        self.label_embed = nn.Embedding(num_classes, num_classes)

        combined_dim = z_c_dim + style_dim + num_classes
        self.init_mlp = nn.Sequential(
            nn.Linear(combined_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 128 * 24)
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(128, 64, kernel_size=(10,), stride=(5,), padding=(2,), output_padding=(1,)),
            nn.InstanceNorm1d(64), nn.LeakyReLU(0.2),

            nn.ConvTranspose1d(64, 32, kernel_size=(10,), stride=(5,), padding=(2,)),
            nn.InstanceNorm1d(32), nn.LeakyReLU(0.2),

            # ==================================================================
            # 更改在这里: 1 -> out_channels (即 2)
            nn.ConvTranspose1d(32, out_channels, kernel_size=(10,), stride=(5,), padding=(2,)),
            # ==================================================================

            nn.Tanh()
        )

    def forward(self, z_c, z_s, y_cond):
        if y_cond.dim() == 1:
            y_emb = self.label_embed(y_cond)
        else:
            y_emb = y_cond

        z_combined = torch.cat([z_c, z_s, y_emb], dim=1)

        h = self.init_mlp(z_combined)
        h = h.view(h.size(0), 128, 24)

        x_hat = self.decoder(h)

        if x_hat.size(-1) > self.output_dim:
            x_hat = x_hat[..., :self.output_dim]
        elif x_hat.size(-1) < self.output_dim:
            pad_width = self.output_dim - x_hat.size(-1)
            x_hat = F.pad(x_hat, (0, pad_width))

        return x_hat


# --- 3. Physiological Discriminator (D_physio) ---
# (这个类保持不变)
class PhysioDiscriminator(nn.Module):
    def __init__(self, feature_dim=128, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, physio_features):
        return self.net(physio_features)
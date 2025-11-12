# utils/physiological_utils.py
# (修正了 PhysioFeatureExtractor 的 forward 方法)

import torch
import torch.nn as nn
import torch.nn.functional as F


# --- 1. 生理特征提取器 (Phi) ---
class PhysioFeatureExtractor(nn.Module):
    def __init__(self, fs=100, n_fft=256, physio_feature_dim=128):
        super().__init__()
        self.fs = fs
        self.n_fft = n_fft
        self.physio_feature_dim = physio_feature_dim

        self.bands = {
            'delta': (0.5, 4.0), 'theta': (4.0, 8.0), 'alpha': (8.0, 13.0),
            'beta': (13.0, 30.0), 'gamma': (30.0, 49.0)
        }
        self.register_buffer('freqs', torch.fft.rfftfreq(n_fft, 1.0 / fs))

        # 5 (band features) + 3 (Hjorth) = 8
        self.final_mapper = nn.Linear(8, physio_feature_dim)

    def calculate_psd(self, x):
        # x: (B, L) e.g., [32, 3000]
        X_fft = torch.fft.rfft(x, n=self.n_fft, dim=-1)
        Sxx_log = torch.log1p(X_fft.abs() ** 2)
        return Sxx_log

    def calculate_band_ratios(self, Sxx_log):
        band_powers = []
        for band, (low, high) in self.bands.items():
            idx = (self.freqs >= low) & (self.freqs <= high)
            if idx.sum() == 0:
                band_powers.append(torch.zeros(Sxx_log.size(0), 1, device=Sxx_log.device))
            else:
                band_powers.append(Sxx_log[:, idx].mean(dim=1, keepdim=True))
        powers = torch.cat(band_powers, dim=1)  # (B, 5)
        return powers

    def calculate_hjorth(self, x):
        # x: (B, L) e.g., [32, 3000]
        activity = x.var(dim=1, keepdim=True)
        dx = torch.diff(x, dim=1)
        mobility_num = dx.var(dim=1, keepdim=True)
        mobility = mobility_num / torch.clamp(activity, min=1e-10)
        d2x = torch.diff(dx, dim=1)
        complexity_num = d2x.var(dim=1, keepdim=True)
        complexity = complexity_num / torch.clamp(mobility_num, min=1e-10)
        return torch.cat([
            torch.log1p(activity), torch.log1p(mobility), torch.log1p(complexity)
        ], dim=1)

    # ==================================================================
    # 更改在这里: forward 方法
    # ==================================================================
    def forward(self, x):
        # x 可能是 (B, C, L), e.g., [32, 2, 3000]
        # x 也可能是 (B, T, C, L), e.g., [32, 20, 2, 3000]

        if x.dim() == 4:
            # 如果是4D, 取第一个时间步
            x = x[:, 0, :, :]

        # x 现在是 (B, C, L)
        # 我们只分析第一个通道 (C=0)
        x_squeezed = x[:, 0, :]  # Shape: (B, L), e.g., [32, 3000]

        Sxx_log = self.calculate_psd(x_squeezed)
        band_features = self.calculate_band_ratios(Sxx_log)
        hjorth_params = self.calculate_hjorth(x_squeezed)

        raw_features = torch.cat([band_features, hjorth_params], dim=1)  # (B, 8)
        final_features = self.final_mapper(raw_features)  # (B, 128)

        return final_features


# --- 2. 损失函数 (保持不变) ---

def calculate_physio_loss(x_hat_r, x_r, phi_model):
    with torch.no_grad():
        features_r = phi_model(x_r)
    features_hat = phi_model(x_hat_r)
    loss = F.l1_loss(features_hat, features_r)
    return loss


def calculate_discriminator_loss(real_logits, fake_logits):
    real_loss = 0.5 * torch.mean((real_logits - 1) ** 2)
    fake_loss = 0.5 * torch.mean(fake_logits ** 2)
    return real_loss + fake_loss


def calculate_generator_loss(fake_logits):
    return 0.5 * torch.mean((fake_logits - 1) ** 2)
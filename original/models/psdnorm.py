# spectral_alignment.py
# -*- coding: utf-8 -*-
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _hann_window(F_len: int, device, dtype):
    # 频谱结构编码的窗函数，做成 L2=1，避免能量引入额外尺度
    w = torch.hann_window(F_len, periodic=True, dtype=dtype, device=device)
    w = w / torch.linalg.vector_norm(w)
    return w


def _segment_indices(L: int, F_len: int, hop: Optional[int] = None) -> Tuple[torch.Tensor, int]:
    if hop is None:
        hop = max(1, F_len // 2)  # 默认 50% overlap
    if L < F_len:
        # 序列过短也要至少做一次结构编码
        return torch.tensor([0]), 1
    num = 1 + (L - F_len) // hop
    starts = torch.arange(0, num * hop, hop)
    return starts, hop


class SpectralStructureAlignment1d(nn.Module):
    """
    SpectralStructureAlignment1d
    --------------------------------
    一个面向 1D 时序特征 (N, C, L) 的“频谱结构对齐层”。

    设计理念：
      1) 先把每个样本的通道做成频域结构表征（结构发现）；
      2) 再在一个 batch 内得到一个跨样本的“频谱结构共识”（结构融合）；
      3) 最后把每个样本的频谱用一个传输滤波器拉到这个共识上（结构传输），再回到时域。

    这样就能在进入分类头之前，把来自不同数据集/被试/设备的时序特征在“结构层面”先做一次统一，
    为后面的域不变表示学习打基础。

    参数
    ----
    num_channels : int
        特征通道数 C
    F_len : int
        做频谱结构编码的窗口长度，同时决定频率采样数
    momentum : float
        共识频谱的更新权重，类似“共识记忆”的平滑
    eps : float
        数值稳定项
    hop : Optional[int]
        窗之间的跳步
    """
    def __init__(self, num_channels: int, F_len: int = 9,
                 momentum: float = 1e-2, eps: float = 1e-8,
                 hop: Optional[int] = None):
        super().__init__()
        assert F_len >= 1, "F_len 必须 >= 1"
        self.C = num_channels
        self.F_len = int(F_len)
        self.momentum = float(momentum)
        self.eps = float(eps)
        self.hop = hop

        # “运行中的批间频谱结构共识”，形状 (C, F_r)
        # 初始设成 1，表示各频段能量均衡
        self.register_buffer("running_spectral_consensus", torch.ones(self.C, self.F_len))

        # 窗函数缓存
        self._cached_window = None
        self._cached_window_key = (None, None)

    # ------------------------------------------------
    # 工具：窗缓存
    # ------------------------------------------------
    def _get_window(self, device, dtype):
        key = (device, dtype)
        if self._cached_window is None or self._cached_window_key != key:
            self._cached_window = _hann_window(self.F_len, device, dtype)
            self._cached_window_key = key
        return self._cached_window

    # ------------------------------------------------
    # ① 结构发现：把时序特征编码成频谱结构 (Welch 风格)
    # ------------------------------------------------
    def _encode_spectral_structure(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (N, C, L) —— 已经去掉了时域均值的特征
        return: spectral_repr: (N, C, F_r)
        """
        N, C, L = x.shape
        device, dtype = x.device, x.dtype
        F_len = self.F_len
        w = self._get_window(device, dtype)
        starts, hop = _segment_indices(L, F_len, self.hop)

        if starts.numel() == 1 and L < F_len:
            # 序列太短，右侧补到窗口长度
            pad = F_len - L
            x_pad = F.pad(x, (0, pad))
        else:
            x_pad = x

        seg_list = []
        for s in starts.tolist():
            seg = x_pad[..., s:s + F_len]  # (N, C, F_len)
            if seg.shape[-1] < F_len:
                seg = F.pad(seg, (0, F_len - seg.shape[-1]))
            seg = seg * w  # 上窗
            Xf = torch.fft.rfft(seg, n=F_len, dim=-1)  # (N, C, F_r)
            P = (Xf.abs() ** 2)  # 能量谱
            seg_list.append(P)

        P_stack = torch.stack(seg_list, dim=0)  # (S, N, C, F_r)
        spectral_repr = P_stack.mean(dim=0)     # (N, C, F_r)
        return spectral_repr

    # ------------------------------------------------
    # ② 结构融合：批内构建频谱共识
    # ------------------------------------------------
    def _build_consensus_spectrum(self, P_batch: torch.Tensor) -> torch.Tensor:
        """
        P_batch: (N, C, F_r)
        return: consensus: (C, F_r)
        用的是 ((1/N) sum sqrt(P))^2 这种几何意义更稳的融合，
        我们把它视为“结构共识”而不是统计均值。
        """
        return (P_batch.clamp_min(self.eps).sqrt().mean(dim=0)) ** 2

    @torch.no_grad()
    def _update_running_consensus(self, P_consensus: torch.Tensor):
        """
        把当前 batch 的结构共识，融进到长期的结构共识里
        running <- ((1-α)*sqrt(running) + α*sqrt(batch))^2
        这样可以让测试阶段也有结构锚可用
        """
        alpha = self.momentum
        run_s = self.running_spectral_consensus.clamp_min(self.eps).sqrt()
        cur_s = P_consensus.clamp_min(self.eps).sqrt()
        new_run = ((1.0 - alpha) * run_s + alpha * cur_s) ** 2
        self.running_spectral_consensus.copy_(new_run)

    # ------------------------------------------------
    # ③ 结构传输：把单个样本的频谱搬到共识上，再回时域
    # ------------------------------------------------
    def _transport_to_consensus(self, x_centered: torch.Tensor,
                                P_hat: torch.Tensor,
                                P_ref: torch.Tensor) -> torch.Tensor:
        """
        x_centered: (N, C, L) —— 原始时域特征(已去均值)
        P_hat: (N, C, F_r) —— 当前 batch 的频谱结构
        P_ref: (C, F_r) —— 要对齐到的目标频谱结构（共识）
        return: y: (N, C, L) —— 在统一结构下的时序特征
        """
        N, C, L = x_centered.shape
        F_r = P_hat.shape[-1]
        L_r = (L // 2) + 1  # rFFT 长度

        # 在 Welch 的频点上构造传输系数
        H_welch = torch.sqrt(
            (P_ref.unsqueeze(0).expand_as(P_hat).clamp_min(self.eps)) /
            P_hat.clamp_min(self.eps)
        )  # (N, C, F_r)

        # 插值到整段长度的频率分辨率
        H = F.interpolate(
            H_welch.unsqueeze(1),  # (N, 1, C, F_r) 不行，我们想在最后一维插
            size=L_r,
            mode='linear',
            align_corners=True
        ).squeeze(1)  # (N, C, L_r)

        # 对整段做 rFFT
        Xf_full = torch.fft.rfft(x_centered, n=L, dim=-1)  # (N, C, L_r)

        # 频域调制：只改幅值，不改相位
        Yf = Xf_full * H

        # 回时域
        y = torch.fft.irfft(Yf, n=L, dim=-1)  # (N, C, L)
        return y

    # ------------------------------------------------
    # 前向：三阶段结构对齐
    # ------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (N, C, L)
        return: (N, C, L) —— 已做频谱结构对齐的时序特征
        """
        assert x.dim() == 3 and x.shape[1] == self.C, "输入必须是 (N, C, L)"
        N, C, L = x.shape

        # (1) 先消除全局直流分量，避免它干扰频谱结构
        mu = x.mean(dim=-1, keepdim=True)
        x_centered = x - mu

        # (2) 每个样本做频谱结构编码
        P_hat = self._encode_spectral_structure(x_centered)  # (N, C, F_r)

        # (3) 训练时，用 batch 的结构去更新长期结构共识
        if self.training:
            with torch.no_grad():
                P_consensus = self._build_consensus_spectrum(P_hat)  # (C, F_r)
                # 若首次尺寸不一致，说明 F_len 改了，重建共识
                if self.running_spectral_consensus.shape[-1] != P_consensus.shape[-1]:
                    self.running_spectral_consensus = torch.ones(
                        C, P_consensus.shape[-1],
                        device=x.device, dtype=x.dtype
                    )
                self._update_running_consensus(P_consensus)

        # (4) 用长期结构共识做传输，对齐到统一的频谱结构下
        P_ref = self.running_spectral_consensus  # (C, F_r)
        y = self._transport_to_consensus(x_centered, P_hat, P_ref)

        return y

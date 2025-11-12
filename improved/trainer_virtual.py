# improved/trainer_virtual.py
# [重大修复] 修复了 virtual_aug_stages 的 .all() 逻辑错误
# 现在使用掩码 (mask) 来对特定样本进行损失计算
# [已修改] 引入 utils.ckpt.BestKeeper 来保存模型
# [已修改] 修复了 torch.load 的 weights_only=False 错误

import os
import copy
import logging
from datetime import datetime
from timeit import default_timer as timer
from pathlib import Path
from typing import Dict, Tuple, Optional
from types import SimpleNamespace
import importlib

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm

# --- 导入 (保持不变) ---
from datasets.dataset import LoadDataset
from evaluator import Evaluator
from losses.double_alignment import CORAL
from losses.ae_loss import AELoss
from utils.allutils import (
    ensure_dir, MetricsLogger, build_ratio_loader,
    plot_curves_for_fold, extract_features_for_tsne, tsne_compare_plot,
    write_aggregate_row, _now
)

# [新增] 导入 BestKeeper
try:
    from utils.ckpt import BestKeeper
except ImportError:
    print("[ERROR] 无法从 utils.ckpt 导入 BestKeeper。请确保 utils/ckpt.py 文件存在。")


    # 定义一个假的 BestKeeper 以防止崩溃，但无法保存模型
    class BestKeeper:
        def __init__(self, *args, **kwargs):
            print("[WARN] BestKeeper 未成功导入，将使用假的替代品，模型不会被保存。")
            self.val_best_filename = None

        def update_val(self, *args, **kwargs): return False

        def update_test(self, *args, **kwargs): return False

try:
    from improved.models.virtual_modules import StyleEncoder, Generator, PhysioDiscriminator

    print("[INFO] Imported SleepDG-Virtual modules (StyleEncoder, Generator, PhysioDiscriminator).")
except ImportError as e:
    print(f"[ERROR] Failed to import SleepDG-Virtual modules: {e}. ")
    print("[ERROR] >> 请确保 'improved/models/virtual_modules.py' 包含完整的类定义 (不是伪代码) <<")
    raise

try:
    from utils.physiological_utils import (
        PhysioFeatureExtractor, calculate_physio_loss,
        calculate_discriminator_loss, calculate_generator_loss
    )

    print("[INFO] Imported physiological utils.")
except ImportError as e:
    print(f"[ERROR] Failed to import physiological utils: {e}.")
    print("[ERROR] >> 请确保 'utils/physiological_utils.py' 文件已创建并包含代码 <<")
    raise


# --- 导入结束 ---


class Trainer(object):
    def __init__(self, params: SimpleNamespace):
        self.params = params

        # --- 目录和数据 (保持不变) ---
        self.model_dir = Path(params.model_dir)
        self.fold_id = params.fold
        self.fold_dir = ensure_dir(self.model_dir / f"fold{self.fold_id}")
        self.allfold_dir = ensure_dir(self.model_dir / "allfold")

        self.data_loader, subject_id = LoadDataset(params).get_data_loader()
        self.data_ratio = getattr(params, "data_ratio", 1.0)
        if 0 < self.data_ratio < 1.0:
            print(f"[INFO] Using {self.data_ratio * 100:.1f}% of training data.")
            self.data_loader['train'] = build_ratio_loader(
                self.data_loader['train'],
                self.data_ratio,
                seed=getattr(params, 'seed', 42)
            )
        self.val_eval = Evaluator(params, self.data_loader['val'])
        self.test_eval = Evaluator(params, self.data_loader['test'])

        # --- 模型加载 (与上次修复相同) ---
        try:
            model_module = importlib.import_module('improved.models.models')
            ModelClass = model_module.Model
            print(f"[INFO] Using IMPROVED Model class (E_content + C_task) from improved.models.models.")
            self.model = ModelClass(params)
        except (ImportError, AttributeError, ValueError) as e:
            print(f"[ERROR] Failed to load IMPROVED model class: {e}")
            raise

        # --- 自动加载 LODO 预训练权重 (与上次修复相同) ---
        base_dir_str = getattr(params, "pretrained_original_base_dir", None)

        if base_dir_str and Path(base_dir_str).exists():
            current_fold_id = self.fold_id
            # [修改] 假设原始权重的目录结构也是 'fold{id}'
            original_fold_path = Path(base_dir_str) / f"fold{current_fold_id}"

            if not original_fold_path.exists():
                print(f"[WARN] 预训练目录存在, 但找不到对应的 'original' 折叠目录: {original_fold_path}")
                print("[WARN] 将继续使用随机初始化的 'self.model'.")
            else:
                # [修改] 自动查找 .pth 文件，而不是依赖 config_improved.yaml 中的 hardcoded 名字
                model_files = list(original_fold_path.glob("*.pth"))

                # [修改] 排除 'LAST_EPOCH_fallback.pth'
                model_files = [p for p in model_files if "LAST_EPOCH_fallback" not in p.name]

                if len(model_files) == 0:
                    print(f"[WARN] 找到了 'original' 折叠目录, 但在 {original_fold_path} 中未找到 .pth 权重文件。")
                    print("[WARN] 将继续使用随机初始化的 'self.model'.")

                if len(model_files) >= 1:
                    if len(model_files) > 1:
                        print(
                            f"[WARN] 找到了 'original' 折叠目录, 但在 {original_fold_path} 中找到了多个 .pth 文件。将使用第一个。")

                    pretrained_path = model_files[0]
                    print(f"[INFO] 正在为 'improved' Fold {current_fold_id} 加载 'original' 预训练权重:")
                    print(f"       > {pretrained_path}")
                    try:
                        # [修改] 同样使用 weights_only=False 加载预训练模型
                        checkpoint = torch.load(pretrained_path, map_location='cpu', weights_only=False)

                        if isinstance(checkpoint, dict) and 'model_state' in checkpoint:
                            original_state_dict = checkpoint['model_state']
                        elif isinstance(checkpoint, dict):
                            original_state_dict = checkpoint
                        else:
                            raise ValueError("预训练 checkpoint 格式无法识别。")

                        model_to_load = self.model.module if isinstance(self.model, nn.DataParallel) else self.model

                        missing_keys, unexpected_keys = model_to_load.load_state_dict(original_state_dict, strict=False)

                        print(f"[INFO] 预训练权重加载成功 (strict=False)。")
                        if missing_keys:
                            print(f"       > 缺失的键 (将随机初始化): {missing_keys}")
                        if unexpected_keys:
                            print(f"       > 意外的键 (将被忽略): {unexpected_keys}")

                    except Exception as e:
                        print(f"[ERROR] 加载预训练权重失败 {pretrained_path}: {e}")
                        print("[WARN] 将继续使用随机初始化的 'self.model'.")
        else:
            print("[INFO] 未指定 'pretrained_original_base_dir'。将从头开始初始化 'self.model'.")
        # --- 预训练加载结束 ---

        # --- C-VAE 模块加载 (与上次修复相同) ---
        vp = getattr(params, "virtual_params", SimpleNamespace())
        eeg_dim = getattr(vp, "eeg_dim", 3000)
        z_c_dim = 512
        style_dim = getattr(vp, "style_dim", 64)
        num_classes = getattr(params, "num_of_classes", 5)
        physio_feature_dim = getattr(vp, "physio_feature_dim", 128)
        self.num_classes = num_classes

        self.E_style = StyleEncoder(
            input_dim=eeg_dim,
            latent_dim=style_dim,
            in_channels=2
        )
        self.G = Generator(
            z_c_dim=z_c_dim,
            style_dim=style_dim,
            num_classes=num_classes,
            output_dim=eeg_dim,
            out_channels=2
        )
        self.D_physio = PhysioDiscriminator(
            feature_dim=physio_feature_dim
        )
        # [修改] Phi 也放到 GPU
        self.Phi = PhysioFeatureExtractor(
            fs=100,
            n_fft=256,
            physio_feature_dim=physio_feature_dim
        )
        self.Phi.eval()

        print("[INFO] Loaded SleepDG-Virtual C-VAE modules.")

        # --- GPU setup (保持不变) ---
        if torch.cuda.device_count() > 1:
            print(f"Let's use {torch.cuda.device_count()} GPUs!")
            self.model = nn.DataParallel(self.model)
            self.E_style = nn.DataParallel(self.E_style)
            self.G = nn.DataParallel(self.G)
            self.D_physio = nn.DataParallel(self.D_physio)
            # [修改] Phi 也放到 GPU
            self.Phi = nn.DataParallel(self.Phi)

        self.model.cuda()
        self.E_style.cuda()
        self.G.cuda()
        self.D_physio.cuda()
        self.Phi.cuda()  # [修改] Phi 也放到 GPU

        # --- 损失和 Lambda (保持不变) ---
        self.lambda_ae = getattr(params, "lambda_ae", 1.0)
        self.lambda_coral = getattr(params, "lambda_coral", 0.0)
        # [修改] 蒸馏损失的 CE reduce='none'
        self.ce_loss = CrossEntropyLoss(label_smoothing=getattr(params, "label_smoothing", 0.1)).cuda()
        self.ce_loss_none = CrossEntropyLoss(label_smoothing=getattr(params, "label_smoothing", 0.1),
                                             reduction='none').cuda()
        # ---
        self.ae_loss = AELoss().cuda() if self.lambda_ae > 0 else None
        self.lmb_caa = getattr(params, "lambda_caa", 0.3)
        self.lmb_stat = getattr(params, "lambda_stat", 0.2)
        self.lmb_Areg = getattr(params, "lambda_Areg", 0.1)

        self.l1_loss = nn.L1Loss(reduction='none').cuda()  # [修改] L1 loss reduce='none'
        self.lambda_rec = getattr(params, "lambda_rec", 10.0)
        self.lambda_kl = getattr(params, "lambda_kl", 0.1)
        self.lambda_physio = getattr(params, "lambda_physio", 0.5)
        self.lambda_adv = getattr(params, "lambda_adv", 0.1)
        self.lambda_distill = getattr(params, "lambda_distill", 0.5)

        # --- 控制参数 (保持不变) ---
        self.virtual_aug_start_epoch = getattr(params, "virtual_aug_start_epoch", 0)
        self.virtual_aug_ratio = getattr(params, "virtual_aug_ratio", 1.0)
        aug_stages_list = getattr(params, "virtual_aug_stages", [0, 1, 2, 3, 4])
        self.virtual_aug_stages_tensor = torch.tensor(aug_stages_list, dtype=torch.long)
        print(f"[INFO] Virtual Augmentation Controls:")
        print(f"       Start Epoch: {self.virtual_aug_start_epoch}")
        print(f"       Ratio: {self.virtual_aug_ratio}")
        print(f"       Allowed Stages: {aug_stages_list}")

        self.eval_test_every_epoch = getattr(params, "eval_test_every_epoch", False)
        if self.eval_test_every_epoch:
            print("[INFO] eval_test_every_epoch is ENABLED. Test metrics will be calculated every epoch.")

        # --- 优化器和调度器 (保持不变) ---
        main_model_params = (
                list(self.model.parameters()) +
                list(self.E_style.parameters()) +
                list(self.G.parameters())
        )
        self.optimizer_main = torch.optim.Adam(
            main_model_params,
            lr=params.lr,
            weight_decay=getattr(params, 'weight_decay', params.lr / 10)
        )
        self.optimizer_D = torch.optim.Adam(
            self.D_physio.parameters(),
            lr=getattr(params, "lr_D", params.lr),
            betas=(0.5, 0.999)
        )
        self.data_length = len(self.data_loader['train'])
        self.scheduler_main = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer_main,
            T_max=params.epochs * self.data_length
        )

        self._frozen_uni_built = False

        # --- 日志 (保持不变) ---
        self.logger = self._setup_logger()
        self.metrics_logger = MetricsLogger(self.fold_dir, self.fold_id)

        # [修改] 移除 self.best_model_states = None

        self.result_txt = self.fold_dir / "results.txt"
        self.test_cm_file = self.fold_dir / f"test_confusion_fold{self.fold_id}.txt"

        # [新增] 初始化 BestKeeper
        self.best_keeper = BestKeeper(
            out_dir=str(self.fold_dir),
            val_metric_name="val_acc",  # 使用 val_acc 作为主要跟踪指标
            test_metric_name="test_acc",
            maximize=True,
            save_optimizer=False  # 只保存模型状态
        )
        self.best_model_path_to_load = None  # 用于存储最佳模型的路径以供 test() 使用

    # --- _setup_logger and _log_txt (保持不变) ---
    def _setup_logger(self):
        logger = logging.getLogger(f"TrainerFold{self.fold_id}")
        if logger.hasHandlers():
            logger.handlers.clear()
        logger.setLevel(logging.INFO)
        logger.propagate = False
        log_format = "%(asctime)s | %(levelname)s | fold=%(fold)d | %(message)s"
        date_format = "%Y-%m-%d %H:%M:%S"
        formatter = logging.Formatter(fmt=log_format, datefmt=date_format)
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        self.fold_dir.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(self.fold_dir / "run.log", mode="a", encoding="utf-8")
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        old_factory = logging.getLogRecordFactory()

        def record_factory(*args, **kwargs):
            record = old_factory(*args, **kwargs)
            record.fold = self.fold_id
            return record

        logging.setLogRecordFactory(record_factory)
        return logger

    def _log_txt(self, msg: str):
        self.logger.info(msg)
        try:
            with open(self.result_txt, "a", encoding="utf-8") as f:
                f.write(msg + "\n")
        except Exception as e:
            self.logger.error(f"Failed to write to results.txt: {e}")

    # --- reparameterize (保持不变) ---
    def reparameterize(self, mu, log_sigma):
        std = torch.exp(0.5 * log_sigma)
        eps = torch.randn_like(std)
        return mu + eps * std

    # --- [新增] 安全计算掩码均值的函数 ---
    def safe_mean(self, tensor, mask):
        """计算掩码张量的均值，如果掩码全为0则返回0"""
        masked_tensor = tensor * mask
        mask_sum = mask.sum()
        if mask_sum > 0:
            return masked_tensor.sum() / mask_sum
        return torch.tensor(0.0).to(tensor.device)

    # --- [重大修改] train 方法 ---
    def train(self) -> Dict:
        try:
            with open(self.result_txt, "a", encoding="utf-8") as f:
                f.write(
                    f"[{_now()}] Start fold {self.fold_id}, Target: {getattr(self.params, 'target_domains', 'N/A')}\n")
                f.write(f"Run ID: {getattr(self.params, 'run_name', 'N/A')}\n")
                f.write(f"Data ratio: {self.data_ratio}\n\n")
        except Exception as e:
            self.logger.error(f"Failed to write initial info to results.txt: {e}")

        self._log_txt(
            f"===== [START] Training Fold {self.fold_id} / Target: {getattr(self.params, 'target_domains', 'N/A')} =====")
        acc_best = 0.0
        f1_best = 0.0
        best_f1_epoch = 0

        self.virtual_aug_stages_tensor = self.virtual_aug_stages_tensor.cuda()

        for epoch in range(self.params.epochs):
            self.model.train()
            self.E_style.train()
            self.G.train()
            self.D_physio.train()
            # Phi 保持 eval 模式
            self.Phi.eval()

            start_time = timer()
            losses = []
            task_losses, caa_losses, stat_losses, areg_losses, ae_losses, coral_losses = [], [], [], [], [], []
            rec_losses, kl_losses, physio_losses, adv_g_losses, adv_d_losses, distill_losses = [], [], [], [], [], []

            pbar_desc = f"Fold {self.fold_id} | Epoch {epoch + 1}/{self.params.epochs}"
            pbar = tqdm(self.data_loader['train'], mininterval=5, desc=pbar_desc, leave=False)

            for x_r, y_r, d_r in pbar:
                x_r = x_r.cuda(non_blocking=True)
                y_r = y_r.cuda(non_blocking=True).long()
                d_r = d_r.cuda(non_blocking=True).long()

                self.optimizer_D.zero_grad()
                self.optimizer_main.zero_grad()

                # --- 路径 A ---
                logits_r, recon_r, mu_r, z_c_r, reg_A = self.model(x_r, labels=y_r, domain_ids=d_r)

                # --- D1: Classifier Losses ---
                # [修改] 使用 ce_loss_none 来计算蒸馏掩码
                loss_task_per_sample = self.ce_loss_none(logits_r.permute(0, 2, 1), y_r)  # Shape [B, T]
                loss_task = loss_task_per_sample.mean()  # 聚合
                task_losses.append(loss_task.item())

                model_to_update = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
                loss_caa = torch.tensor(0.0).cuda()
                loss_stat = torch.tensor(0.0).cuda()
                if hasattr(model_to_update, 'anchors'):
                    with torch.no_grad():
                        model_to_update.anchors.update(z_c_r.detach(), y_r, d_r)
                    loss_caa = model_to_update.anchors.caa_loss()
                    loss_stat = model_to_update.anchors.stats_align_loss(z_c_r, d_r)
                caa_losses.append(loss_caa.item())
                stat_losses.append(loss_stat.item())

                current_reg_A = reg_A.mean() if isinstance(reg_A, torch.Tensor) else torch.tensor(0.0).cuda()
                if isinstance(reg_A, torch.Tensor):
                    areg_losses.append(current_reg_A.item())

                loss_orig_ae = torch.tensor(0.0).cuda()
                if self.lambda_ae > 0 and self.ae_loss is not None and recon_r is not None:
                    loss_orig_ae = self.ae_loss(x_r, recon_r)
                ae_losses.append(loss_orig_ae.item())

                loss_coral_val = torch.tensor(0.0).cuda()
                if self.lambda_coral > 0 and self.coral_loss is not None and mu_r is not None:
                    loss_coral_val = self.coral_loss(mu_r, d_r)
                coral_losses.append(loss_coral_val.item())

                L_classifier = (loss_task +
                                self.lmb_caa * loss_caa +
                                self.lmb_stat * loss_stat +
                                self.lmb_Areg * current_reg_A +
                                self.lambda_ae * loss_orig_ae +
                                self.lambda_coral * loss_coral_val)

                # --- 初始化虚拟损失 (与上次修复相同) ---
                L_rec = torch.tensor(0.0).cuda()
                L_kl_s = torch.tensor(0.0).cuda()
                L_physio = torch.tensor(0.0).cuda()
                L_G_adv = torch.tensor(0.0).cuda()
                L_distill = torch.tensor(0.0).cuda()
                loss_D_val = 0.0

                # --- [修改] 虚拟增强控制逻辑 (使用掩码) ---
                y_r_single = y_r[:, 0]  # 取第一个 epoch 的标签 [B,]

                aug_enabled_by_epoch = (epoch >= self.virtual_aug_start_epoch)
                aug_enabled_by_ratio = (torch.rand(1).item() < self.virtual_aug_ratio)

                if aug_enabled_by_epoch and aug_enabled_by_ratio:

                    # --- 路径 B: 总是生成 ---
                    mu_s, log_sigma_s = self.E_style(x_r)
                    z_s_r = self.reparameterize(mu_s, log_sigma_s)
                    z_s_virt = torch.randn_like(z_s_r)
                    y_cond_single = F.one_hot(y_r_single, self.num_classes).float()

                    x_virt_single_epoch = self.G(z_c_r.detach(), z_s_virt, y_cond_single)
                    x_hat_r_single_epoch = self.G(z_c_r, z_s_r, y_cond_single)

                    # --- [新增] 创建掩码 (Mask) ---
                    # 1. 找到允许增强的样本
                    stages_to_aug_device = self.virtual_aug_stages_tensor.to(y_r_single.device)
                    # mask_stage 是 [B,] 的 bool 张量, e.g., [True, False, True, ...]
                    mask_stage = torch.isin(y_r_single, stages_to_aug_device)

                    # 2. 确保掩码至少有一个 True 元素，否则跳过
                    if mask_stage.sum() > 0:
                        # 3. 将掩码 [B,] 扩展到损失所需的维度
                        #    [B,] -> [B, 1]
                        mask_kl = mask_stage.float().unsqueeze(1)
                        #    [B,] -> [B, 1, 1] (用于 L_rec, L_physio)
                        mask_rec = mask_stage.float().view(-1, 1, 1)

                        # --- 路径 C: 训练判别器 (使用掩码) ---
                        with torch.no_grad():
                            phi_r_masked = self.Phi(x_r[mask_stage])  # 只选真实样本
                            phi_virt_fake_masked = self.Phi(x_virt_single_epoch[mask_stage].detach())

                        real_logits = self.D_physio(phi_r_masked)
                        fake_logits = self.D_physio(phi_virt_fake_masked)

                        loss_D = calculate_discriminator_loss(real_logits, fake_logits)

                        loss_D.backward()
                        self.optimizer_D.step()
                        loss_D_val = loss_D.item()

                        # --- D2: Generator (C-VAE) Losses (使用掩码) ---
                        x_r_first_epoch = x_r[:, 0, :, :]

                        # [修改] 使用 'none' reduction，然后应用掩码
                        # self.l1_loss 的 reduction='none'
                        L_rec_per_sample = self.l1_loss(x_hat_r_single_epoch, x_r_first_epoch).mean(dim=[1, 2])  # [B,]
                        L_rec = self.safe_mean(L_rec_per_sample, mask_stage.float())

                        L_kl_s_per_sample = -0.5 * torch.sum(1 + log_sigma_s - mu_s.pow(2) - log_sigma_s.exp(),
                                                             dim=1)  # [B,]
                        L_kl_s = self.safe_mean(L_kl_s_per_sample, mask_stage.float())

                        # L_physio 需要 (B, C, L) 输入, phi(x) 输出 (B, D_phi)
                        # 我们需要手动计算 L1
                        phi_hat = self.Phi(x_hat_r_single_epoch)
                        with torch.no_grad():
                            phi_r = self.Phi(x_r_first_epoch)
                        L_physio_per_sample = torch.mean(torch.abs(phi_hat - phi_r), dim=1)  # [B,]
                        L_physio = self.safe_mean(L_physio_per_sample, mask_stage.float())

                        phi_virt_real_masked = self.Phi(x_virt_single_epoch[mask_stage])
                        L_G_adv = calculate_generator_loss(self.D_physio(phi_virt_real_masked))

                        rec_losses.append(L_rec.item())
                        kl_losses.append(L_kl_s.item())
                        physio_losses.append(L_physio.item())
                        adv_g_losses.append(L_G_adv.item())

                        # --- D3: Distillation Loss (使用掩码) ---
                        T = x_r.shape[1]
                        x_virt_sequence = x_virt_single_epoch.unsqueeze(1).expand(-1, T, -1, -1).contiguous()

                        logits_virt, _, _, _, _ = self.model(x_virt_sequence, labels=y_r, domain_ids=d_r)

                        # [修改] 使用 'none' reduction
                        # L_distill_per_sample shape: [B, T]
                        L_distill_per_sample = self.ce_loss_none(logits_virt.permute(0, 2, 1), y_r)

                        # 将掩码 [B,] 扩展到 [B, T]
                        mask_distill = mask_stage.float().unsqueeze(1).expand(-1, T)

                        L_distill = self.safe_mean(L_distill_per_sample, mask_distill)
                        distill_losses.append(L_distill.item())

                    else:  # 如果掩码全为 False (例如，整个批次都是 N1)
                        rec_losses.append(0.0)
                        kl_losses.append(0.0)
                        physio_losses.append(0.0)
                        adv_g_losses.append(0.0)
                        distill_losses.append(0.0)

                else:  # 如果 epoch 或 ratio 不满足
                    rec_losses.append(0.0)
                    kl_losses.append(0.0)
                    physio_losses.append(0.0)
                    adv_g_losses.append(0.0)
                    distill_losses.append(0.0)

                adv_d_losses.append(loss_D_val)

                # --- D4: 组合总损失 ---
                loss = (L_classifier +
                        self.lambda_rec * L_rec +
                        self.lambda_kl * L_kl_s +
                        self.lambda_physio * L_physio +
                        self.lambda_adv * L_G_adv +
                        self.lambda_distill * L_distill)

                loss.backward()

                clip_val = getattr(self.params, 'clip_value', 0)
                if clip_val > 0:
                    torch.nn.utils.clip_grad_norm_(self.optimizer_main.param_groups[0]['params'], clip_val)

                self.optimizer_main.step()
                self.scheduler_main.step()

                losses.append(loss.item())
                pbar.set_postfix(loss=np.mean(losses[-100:]), task=loss_task.item(), distill=L_distill.item(),
                                 refresh=False)

            # --- Epoch End ---
            optim_state = self.optimizer_main.state_dict()
            lr_now = optim_state['param_groups'][0]['lr']
            time_min = (timer() - start_time) / 60.0

            # --- Validation ---
            val_acc, val_f1, val_cm, val_wake_f1, val_n1_f1, val_n2_f1, \
                val_n3_f1, val_rem_f1, val_kappa, val_report = 0.0, 0.0, None, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ""
            if self.val_eval and len(self.data_loader['val']) > 0:
                with torch.no_grad():
                    model_to_eval = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
                    model_to_eval.eval()
                    if hasattr(model_to_eval, 'freeze_unified_projection'):
                        model_to_eval.freeze_unified_projection(strategy="avg")
                        self._frozen_uni_built = True
                    else:
                        self._frozen_uni_built = False

                    val_acc, val_f1, val_cm, val_wake_f1, val_n1_f1, val_n2_f1, \
                        val_n3_f1, val_rem_f1, val_kappa, val_report = self.val_eval.get_accuracy(self.model)
            else:
                self.logger.warning("Validation loader is empty or not provided. Skipping validation.")

            # --- [新增] 每轮测试集评估 (保持不变) ---
            test_acc, test_f1, test_kappa = 0.0, 0.0, 0.0
            test_wake_f1, test_n1_f1, test_n2_f1, test_n3_f1, test_rem_f1 = 0.0, 0.0, 0.0, 0.0, 0.0

            if self.eval_test_every_epoch and self.test_eval and len(self.data_loader['test']) > 0:
                with torch.no_grad():
                    model_to_eval = self.model.module if isinstance(self.model, nn.DataParallel) else self.model

                    if not self._frozen_uni_built and hasattr(model_to_eval, 'freeze_unified_projection'):
                        self.logger.warning("Unified projection not built during val. Building for test.")
                        model_to_eval.freeze_unified_projection(strategy="avg")
                        self._frozen_uni_built = True

                    test_acc, test_f1, _, test_wake_f1, test_n1_f1, test_n2_f1, \
                        test_n3_f1, test_rem_f1, test_kappa, _ = self.test_eval.get_accuracy(self.model)

            # --- Logging (保持不变) ---
            avg_loss = np.mean(losses) if losses else 0.0
            avg_task = np.mean(task_losses) if task_losses else 0.0
            avg_caa = np.mean(caa_losses) if caa_losses else 0.0
            avg_stat = np.mean(stat_losses) if stat_losses else 0.0
            avg_areg = np.mean(areg_losses) if areg_losses else 0.0
            avg_ae = np.mean(ae_losses) if ae_losses else 0.0
            avg_coral = np.mean(coral_losses) if coral_losses else 0.0
            avg_rec = np.mean(rec_losses) if rec_losses else 0.0
            avg_kl = np.mean(kl_losses) if kl_losses else 0.0
            avg_physio = np.mean(physio_losses) if physio_losses else 0.0
            avg_g_adv = np.mean(adv_g_losses) if adv_g_losses else 0.0
            avg_d_adv = np.mean(adv_d_losses) if adv_d_losses else 0.0
            avg_distill = np.mean(distill_losses) if distill_losses else 0.0

            loss_detail_cls = (f"task={avg_task:.3f}, caa={avg_caa:.3f}, stat={avg_stat:.3f}, "
                               f"Areg={avg_areg:.3f}, ae_orig={avg_ae:.3f}, coral={avg_coral:.3f}")
            loss_detail_gen = (f"rec={avg_rec:.3f}, kl={avg_kl:.3f}, phys={avg_physio:.3f}, "
                               f"adv_g={avg_g_adv:.3f}, adv_d={avg_d_adv:.3f}, distill={avg_distill:.3f}")

            msg = (f"Epoch {epoch + 1:03d} | train_loss={avg_loss:.5f}\n"
                   f"    CLS: {loss_detail_cls}\n"
                   f"    GEN: {loss_detail_gen}\n"
                   f"    VAL: val_acc={val_acc:.5f} | val_f1={val_f1:.5f} | val_kappa={val_kappa:.5f} | "
                   f"W={val_wake_f1:.3f} N1={val_n1_f1:.3f} N2={val_n2_f1:.3f} N3={val_n3_f1:.3f} R={val_rem_f1:.3f}\n"
                   f"    TEST:test_acc={test_acc:.5f} | test_f1={test_f1:.5f} | test_kappa={test_kappa:.5f} | "
                   f"W={test_wake_f1:.3f} N1={test_n1_f1:.3f} N2={test_n2_f1:.3f} N3={test_n3_f1:.3f} R={test_rem_f1:.3f}\n"
                   f"    SYS: lr={lr_now:.6f} | {time_min:.2f} min")
            self._log_txt(msg)

            self.metrics_logger.log_epoch(
                time_str=_now(), epoch=epoch + 1, lr=lr_now, train_loss=avg_loss,
                train_acc=None, train_f1=None,
                val_acc=val_acc, val_f1=val_f1, val_kappa=val_kappa,
                test_acc=test_acc, test_f1=test_f1, test_kappa=test_kappa,
                wake_f1=val_wake_f1, n1_f1=val_n1_f1, n2_f1=val_n2_f1,
                n3_f1=val_n3_f1, rem_f1=val_rem_f1,
                test_wake_f1=test_wake_f1, test_n1_f1=test_n1_f1, test_n2_f1=test_n2_f1,
                test_n3_f1=test_n3_f1, test_rem_f1=test_rem_f1
            )

            # --- [MODIFIED] Best Model Tracking (使用 BestKeeper) ---
            current_epoch_metrics = {
                "val_acc": val_acc, "val_f1": val_f1, "val_kappa": val_kappa,
                "test_acc": test_acc, "test_f1": test_f1, "test_kappa": test_kappa,
                "wake_f1": val_wake_f1, "n1_f1": val_n1_f1, "n2_f1": val_n2_f1,
                "n3_f1": val_n3_f1, "rem_f1": val_rem_f1,
                "test_wake_f1": test_wake_f1, "test_n1_f1": test_n1_f1, "test_n2_f1": test_n2_f1,
                "test_n3_f1": test_n3_f1, "test_rem_f1": test_rem_f1
            }

            # [!! 修复 !!] 调用 update_val 时传入 'metrics' 字典
            updated_val = self.best_keeper.update_val(
                metrics=current_epoch_metrics,
                epoch=epoch + 1,
                model=self.model.module if isinstance(self.model, nn.DataParallel) else self.model,
                # [修改] 传递 C-VAE 模块的状态
                extra_state={
                    'E_style': self.E_style.module.state_dict() if isinstance(self.E_style,
                                                                              nn.DataParallel) else self.E_style.state_dict(),
                    'G': self.G.module.state_dict() if isinstance(self.G, nn.DataParallel) else self.G.state_dict()
                }
            )

            # (可选) 同时保存测试集上最好的模型
            self.best_keeper.update_test(
                metrics=current_epoch_metrics,
                epoch=epoch + 1,
                model=self.model.module if isinstance(self.model, nn.DataParallel) else self.model,
                extra_state={
                    'E_style': self.E_style.module.state_dict() if isinstance(self.E_style,
                                                                              nn.DataParallel) else self.E_style.state_dict(),
                    'G': self.G.module.state_dict() if isinstance(self.G, nn.DataParallel) else self.G.state_dict()
                }
            )

            if updated_val:
                best_f1_epoch = epoch + 1
                acc_best = val_acc
                f1_best = val_f1
                self._log_txt(
                    f"[BEST@{best_f1_epoch:03d}] val_acc={acc_best:.5f} | val_f1={f1_best:.5f} (New checkpoint saved)")

        # --- Training End (保持不变) ---
        self._log_txt(
            f"Training finished. Best val @ Epoch {best_f1_epoch:03d} -> val_acc={acc_best:.5f}, val_f1={f1_best:.5f}")

        try:
            plot_curves_for_fold(self.metrics_logger.path(), out_dir=self.fold_dir)
            self.logger.info(f"Curves plotted for fold {self.fold_id} in {self.fold_dir}")
        except Exception as e:
            self.logger.error(f"Failed to plot curves for fold {self.fold_id}: {e}")

        # [MODIFIED] 检查 BestKeeper 是否保存了模型
        if self.best_keeper.val_best_filename is None:
            self.logger.warning("No best validation model was saved by BestKeeper.")
            if self.params.epochs > 0:
                self.logger.warning("Using last epoch model for testing.")
                # 保存最后一个 epoch 的模型作为备用
                model_state = self.model.module.state_dict() if isinstance(self.model,
                                                                           nn.DataParallel) else self.model.state_dict()
                estyle_state = self.E_style.module.state_dict() if isinstance(self.E_style,
                                                                              nn.DataParallel) else self.E_style.state_dict()
                g_state = self.G.module.state_dict() if isinstance(self.G, nn.DataParallel) else self.G.state_dict()

                fallback_state = {
                    'main_model': model_state,
                    'E_style': estyle_state,
                    'G': g_state
                }
                fallback_filename = f"fold{self.fold_id}_LAST_EPOCH_fallback.pth"
                fallback_path = self.fold_dir / fallback_filename
                try:
                    torch.save(fallback_state, fallback_path)
                    self.logger.info(f"Last epoch fallback model saved to: {fallback_path}")
                    self.best_model_path_to_load = str(fallback_path)
                except Exception as e:
                    self.logger.error(f"Failed to save last epoch fallback model: {e}")
                    return {"error": "No model state available for testing."}
            else:
                self.logger.error("No model state available for testing.")
                return {"error": "No model state available for testing."}
        else:
            self.best_model_path_to_load = str(self.fold_dir / self.best_keeper.val_best_filename)

        test_results_dict = self.test(best_val_acc=acc_best, best_val_f1=f1_best)
        return test_results_dict

    # --- test 方法 (保持不变) ---
    def test(self, best_val_acc: float, best_val_f1: float) -> Dict:
        if not hasattr(self, 'best_model_path_to_load') or not self.best_model_path_to_load or not Path(
                self.best_model_path_to_load).exists():
            self._log_txt(f"[ERROR] No model file found at: {getattr(self, 'best_model_path_to_load', 'N/A')}")
            return {"error": "No model file available for testing."}

        self._log_txt(f"[INFO] Loading best model from: {self.best_model_path_to_load}")

        try:
            model_module = importlib.import_module('improved.models.models')
            ModelClass = model_module.Model
            temp_model = ModelClass(self.params)

            # [!! 修复 !!] 添加 weights_only=False
            checkpoint = torch.load(self.best_model_path_to_load, map_location='cpu', weights_only=False)

            # [修改] 假设 ckpt.py 保存的字典键是 "model_state"
            if isinstance(checkpoint, dict) and 'model_state' in checkpoint:
                best_model_states = checkpoint['model_state']
            elif isinstance(checkpoint, dict) and 'main_model' in checkpoint:
                best_model_states = checkpoint['main_model']  # 兼容旧的 fallback
            elif isinstance(checkpoint, dict):
                best_model_states = checkpoint  # 兼容纯 state_dict
            else:
                raise ValueError("Checkpoint 格式无法识别。")

            temp_model.load_state_dict(best_model_states)

            self.logger.info("Best MAIN_MODEL state loaded into temporary model for testing.")

            # (注意：此 test 方法不加载 E_style 或 G，因为 'improved' 模型的 test 评估
            # 仅在 'main' (improved.models.models.Model) 上进行，该模型不使用 C-VAE)

            if torch.cuda.device_count() > 1 and isinstance(self.model, nn.DataParallel):
                model_to_test = nn.DataParallel(temp_model).cuda()
                self.logger.info("Wrapping loaded model with DataParallel for evaluation.")
            else:
                model_to_test = temp_model.cuda()

        except (RuntimeError, ImportError, AttributeError, KeyError, FileNotFoundError) as e:
            self.logger.error(f"Failed to load best model state (main_model): {e}")
            return {"error": f"Failed to load best model state: {e}"}

        test_acc, test_f1, test_cm, test_wake_f1, test_n1_f1, test_n2_f1, \
            test_n3_f1, test_rem_f1, test_kappa, test_report = 0.0, 0.0, None, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, "N/A"

        if self.test_eval and len(self.data_loader['test']) > 0:
            with torch.no_grad():
                model_to_test.eval()
                eval_model_instance = model_to_test.module if isinstance(model_to_test,
                                                                         nn.DataParallel) else model_to_test
                if not self._frozen_uni_built and hasattr(eval_model_instance, 'freeze_unified_projection'):
                    self.logger.warning("Unified projection not built during training. Building with 'avg'.")
                    eval_model_instance.freeze_unified_projection(strategy="avg")
                    self._frozen_uni_built = True

                self._log_txt("*************************** Test ***************************")
                test_acc, test_f1, test_cm, test_wake_f1, test_n1_f1, test_n2_f1, \
                    test_n3_f1, test_rem_f1, test_kappa, test_report = self.test_eval.get_accuracy(model_to_test)

                self._log_txt(f"Test: acc={test_acc:.5f}, f1={test_f1:.5f}")
                self._log_txt(f"Cohen's Kappa: {test_kappa:.5f}")
                self._log_txt("Confusion Matrix:\n" + str(test_cm))
                self._log_txt(
                    ("Class F1 -> W={:.3f}, N1={:.3f}, N2={:.3f}, N3={:.3f}, R={:.3f}"
                     .format(test_wake_f1, test_n1_f1, test_n2_f1, test_n3_f1, test_rem_f1))
                )
                self._log_txt("\nClassification Report:\n" + str(test_report))
                try:
                    with open(self.test_cm_file, "w", encoding="utf-8") as fcm:
                        fcm.write("Confusion Matrix:\n")
                        np.savetxt(fcm, test_cm, fmt="%d")
                        fcm.write("\n\nClassification Report:\n")
                        fcm.write(str(test_report))
                except Exception as e:
                    self.logger.error(f"Failed to save confusion matrix/report: {e}")
        else:
            self.logger.warning("Test loader is empty or not provided. Skipping testing.")

        # [修改] BestKeeper 已经在 train() 中保存了模型
        model_path = Path(self.best_model_path_to_load)
        self._log_txt(f"Model tested: {model_path.name}")

        try:
            device = next(model_to_test.parameters()).device
            model_for_tsne = model_to_test.module if isinstance(model_to_test, nn.DataParallel) else model_to_test
            model_for_tsne.eval()

            RAW, REP, Y = extract_features_for_tsne(
                model_for_tsne, self.data_loader['test'], device, take="mu_tilde"
            )
            tsne_compare_plot(
                raw_X=RAW, rep_X=REP, y=Y, out_dir=self.fold_dir,
                title_prefix=f"Fold{self.fold_id} Test t-SNE", filename_prefix="tsne"
            )
            self.logger.info(f"t-SNE plots generated in {self.fold_dir}")
        except Exception as e:
            self._log_txt(f"[WARN] Failed to generate t-SNE plot: {e}")
            self.logger.warning(f"t-SNE plot generation failed: {e}", exc_info=True)

        result_dict = {
            "time": _now(),
            "run_id": getattr(self.params, "run_name", f"run_{datetime.now().strftime('%Y%m%d-%H%M%S')}"),
            "fold": self.fold_id,
            "best_val_acc": float(best_val_acc), "best_val_f1": float(best_val_f1),
            "test_acc": float(test_acc), "test_f1": float(test_f1),
            "test_kappa": float(test_kappa),
            "wake_f1": float(test_wake_f1), "n1_f1": float(test_n1_f1),
            "n2_f1": float(test_n2_f1), "n3_f1": float(test_n3_f1),
            "rem_f1": float(test_rem_f1),
            "model_path": str(model_path)
        }

        return result_dict
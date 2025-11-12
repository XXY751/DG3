# original/trainer.py
# [已更新] 增加了在每个 epoch 运行测试集评估的选项
# [已更新] 修复了 forward() 的 5 个返回值问题
# [已修改] 引入 utils.ckpt.BestKeeper 来保存模型，并修复了 TypeError
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

        # --- 模型加载 (保持不变) ---
        try:
            model_module = importlib.import_module('original.models.model')
            ModelClass = model_module.Model
            print(f"[INFO] Using ORIGINAL Model class from original.models.model.")
            model = ModelClass(params)
        except (ImportError, AttributeError, ValueError) as e:
            print(f"[ERROR] Failed to load ORIGINAL model class: {e}")
            raise

        self.model = model
        # --- GPU setup (保持不变) ---
        if torch.cuda.device_count() > 1:
            print(f"Let's use {torch.cuda.device_count()} GPUs!")
            self.model = nn.DataParallel(self.model)
        self.model.cuda()

        # --- 损失和 Lambda (保持不变) ---
        self.lambda_ae = getattr(params, "lambda_ae", 1.0)
        self.lambda_coral = getattr(params, "lambda_coral", 0.0)
        self.ce_loss = CrossEntropyLoss(label_smoothing=getattr(params, "label_smoothing", 0.1)).cuda()
        self.coral_loss = CORAL().cuda() if self.lambda_coral > 0 else None
        self.ae_loss = AELoss().cuda() if self.lambda_ae > 0 else None
        self.lmb_caa = getattr(params, "lambda_caa", 0.0)
        self.lmb_stat = getattr(params, "lambda_stat", 0.0)
        self.lmb_Areg = getattr(params, "lambda_Areg", 0.0)

        # --- 评估控制 (保持不变) ---
        self.eval_test_every_epoch = getattr(params, "eval_test_every_epoch", False)
        if self.eval_test_every_epoch:
            print("[INFO] eval_test_every_epoch is ENABLED. Test metrics will be calculated every epoch.")

        # --- 优化器和调度器 (保持不变) ---
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=params.lr,
            weight_decay=getattr(params, 'weight_decay', params.lr / 10)
        )
        self.data_length = len(self.data_loader['train'])
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=params.epochs * self.data_length
        )

        self._frozen_uni_built = False

        # --- 日志 (保持不变) ---
        self.logger = self._setup_logger()
        self.metrics_logger = MetricsLogger(self.fold_dir, self.fold_id)

        self.result_txt = self.fold_dir / "results.txt"
        self.test_cm_file = self.fold_dir / f"test_confusion_fold{self.fold_id}.txt"

        # [新增] 初始化 BestKeeper
        self.best_keeper = BestKeeper(
            out_dir=str(self.fold_dir),
            val_metric_name="val_acc",  # 根据原始逻辑，使用 val_acc 作为主要跟踪指标
            test_metric_name="test_acc",  # 也跟踪最佳测试
            maximize=True,
            save_optimizer=False  # 原始逻辑只保存模型状态
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

    # --- train 方法 ---
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

        for epoch in range(self.params.epochs):
            self.model.train()
            start_time = timer()
            losses = []
            task_losses, caa_losses, stat_losses, areg_losses, ae_losses, coral_losses = [], [], [], [], [], []

            pbar_desc = f"Fold {self.fold_id} | Epoch {epoch + 1}/{self.params.epochs}"
            pbar = tqdm(self.data_loader['train'], mininterval=5, desc=pbar_desc, leave=False)

            for x, y, z in pbar:
                self.optimizer.zero_grad()
                x = x.cuda(non_blocking=True)
                y = y.cuda(non_blocking=True).long()
                z = z.cuda(non_blocking=True).long()

                logits, recon, mu, _, _ = self.model(x, labels=y, domain_ids=z)

                loss_task = self.ce_loss(logits.permute(0, 2, 1), y)
                task_losses.append(loss_task.item())

                model_to_update = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
                loss_caa = torch.tensor(0.0).cuda()
                loss_stat = torch.tensor(0.0).cuda()
                if hasattr(model_to_update, 'anchors'):
                    pass
                caa_losses.append(loss_caa.item())
                stat_losses.append(loss_stat.item())

                current_reg_A = torch.tensor(0.0).cuda()
                areg_losses.append(current_reg_A.item())

                loss = (loss_task +
                        self.lmb_caa * loss_caa +
                        self.lmb_stat * loss_stat +
                        self.lmb_Areg * current_reg_A)

                loss_ae_val = torch.tensor(0.0).cuda()
                if self.lambda_ae > 0 and self.ae_loss is not None and recon is not None:
                    loss_ae_val = self.ae_loss(x, recon)
                    loss = loss + loss_ae_val * self.lambda_ae
                ae_losses.append(loss_ae_val.item())

                loss_coral_val = torch.tensor(0.0).cuda()
                if self.lambda_coral > 0 and self.coral_loss is not None and mu is not None:
                    loss_coral_val = self.coral_loss(mu, z)
                    loss = loss + loss_coral_val * self.lambda_coral
                coral_losses.append(loss_coral_val.item())

                loss.backward()
                clip_val = getattr(self.params, 'clip_value', 0)
                if clip_val > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip_val)
                self.optimizer.step()
                self.scheduler.step()

                losses.append(loss.item())
                pbar.set_postfix(loss=np.mean(losses[-100:]), task=loss_task.item(), refresh=False)

            # --- Epoch End ---
            optim_state = self.optimizer.state_dict()
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

            # --- 每轮测试集评估 ---
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

            # --- Logging ---
            avg_loss = np.mean(losses) if losses else 0.0
            avg_task = np.mean(task_losses) if task_losses else 0.0
            avg_caa = np.mean(caa_losses) if caa_losses else 0.0
            avg_stat = np.mean(stat_losses) if stat_losses else 0.0
            avg_areg = np.mean(areg_losses) if areg_losses else 0.0
            avg_ae = np.mean(ae_losses) if ae_losses else 0.0
            avg_coral = np.mean(coral_losses) if coral_losses else 0.0
            loss_detail = (f"task={avg_task:.3f}, caa={avg_caa:.3f}, stat={avg_stat:.3f}, "
                           f"Areg={avg_areg:.3f}, ae={avg_ae:.3f}, coral={avg_coral:.3f}")

            msg = (f"Epoch {epoch + 1:03d} | train_loss={avg_loss:.5f} ({loss_detail})\n"
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

            # --- Best Model Tracking (使用 BestKeeper) ---
            current_epoch_metrics = {
                "val_acc": val_acc, "val_f1": val_f1, "val_kappa": val_kappa,
                "test_acc": test_acc, "test_f1": test_f1, "test_kappa": test_kappa,
                "wake_f1": val_wake_f1, "n1_f1": val_n1_f1, "n2_f1": val_n2_f1,
                "n3_f1": val_n3_f1, "rem_f1": val_rem_f1,
                "test_wake_f1": test_wake_f1, "test_n1_f1": test_n1_f1, "test_n2_f1": test_n2_f1,
                "test_n3_f1": test_n3_f1, "test_rem_f1": test_rem_f1
            }

            updated_val = self.best_keeper.update_val(
                metrics=current_epoch_metrics,
                epoch=epoch + 1,
                model=self.model.module if isinstance(self.model, nn.DataParallel) else self.model
            )

            self.best_keeper.update_test(
                metrics=current_epoch_metrics,
                epoch=epoch + 1,
                model=self.model.module if isinstance(self.model, nn.DataParallel) else self.model
            )

            if updated_val:
                best_f1_epoch = epoch + 1
                acc_best = val_acc
                f1_best = val_f1
                self._log_txt(
                    f"[BEST@{best_f1_epoch:03d}] val_acc={acc_best:.5f} | val_f1={f1_best:.5f} (New checkpoint saved)")

        # --- Training End ---
        self._log_txt(
            f"Training finished. Best val @ Epoch {best_f1_epoch:03d} -> val_acc={acc_best:.5f}, val_f1={f1_best:.5f}")

        try:
            plot_curves_for_fold(self.metrics_logger.path(), out_dir=self.fold_dir)
            self.logger.info(f"Curves plotted for fold {self.fold_id} in {self.fold_dir}")
        except Exception as e:
            self.logger.error(f"Failed to plot curves for fold {self.fold_id}: {e}")

        if self.best_keeper.val_best_filename is None:
            self.logger.warning("No best validation model was saved by BestKeeper.")
            if self.params.epochs > 0:
                self.logger.warning("Using last epoch model for testing.")
                last_epoch_state = self.model.module.state_dict() if isinstance(self.model,
                                                                                nn.DataParallel) else self.model.state_dict()
                fallback_filename = f"fold{self.fold_id}_LAST_EPOCH_fallback.pth"
                fallback_path = self.fold_dir / fallback_filename
                try:
                    torch.save(last_epoch_state, fallback_path)
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

    def test(self, best_val_acc: float, best_val_f1: float) -> Dict:
        if not hasattr(self, 'best_model_path_to_load') or not self.best_model_path_to_load or not Path(
                self.best_model_path_to_load).exists():
            self._log_txt(f"[ERROR] No model file found at: {getattr(self, 'best_model_path_to_load', 'N/A')}")
            return {"error": "No model file available for testing."}

        self._log_txt(f"[INFO] Loading best model from: {self.best_model_path_to_load}")

        try:
            # [!! 修复 !!] 添加 weights_only=False
            checkpoint = torch.load(self.best_model_path_to_load, map_location='cpu', weights_only=False)

            if isinstance(checkpoint, dict) and 'model_state' in checkpoint:
                best_model_states = checkpoint['model_state']
            elif isinstance(checkpoint, dict):
                best_model_states = checkpoint
            else:
                raise ValueError("Checkpoint 格式无法识别。")

            model_module = importlib.import_module('original.models.model')
            ModelClass = model_module.Model
            temp_model = ModelClass(self.params)
            temp_model.load_state_dict(best_model_states)
            self.logger.info("Best model state loaded into temporary model for testing.")

            if torch.cuda.device_count() > 1 and isinstance(self.model, nn.DataParallel):
                model_to_test = nn.DataParallel(temp_model).cuda()
                self.logger.info("Wrapping loaded model with DataParallel for evaluation.")
            else:
                model_to_test = temp_model.cuda()

        except (RuntimeError, ImportError, AttributeError, ValueError, KeyError, FileNotFoundError) as e:
            self.logger.error(f"Failed to load best model state or re-initialize model: {e}")
            return {"error": f"Failed to load best model state: {e}"}

        # --- 评估 (保持不变) ---
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

        model_path = Path(self.best_model_path_to_load)
        self._log_txt(f"Model tested: {model_path.name}")

        # --- t-SNE (保持不变) ---
        try:
            device = next(model_to_test.parameters()).device
            model_for_tsne = model_to_test.module if isinstance(model_to_test, nn.DataParallel) else model_to_test
            model_for_tsne.eval()

            RAW, REP, Y = extract_features_for_tsne(
                model_for_tsne, self.data_loader['test'], device, take="mu"  # (原始模型没有 mu_tilde)
            )
            tsne_compare_plot(
                raw_X=RAW, rep_X=REP, y=Y, out_dir=self.fold_dir,
                title_prefix=f"Fold{self.fold_id} Test t-SNE", filename_prefix="tsne"
            )
            self.logger.info(f"t-SNE plots generated in {self.fold_dir}")
        except Exception as e:
            self._log_txt(f"[WARN] Failed to generate t-SNE plot: {e}")
            self.logger.warning(f"t-SNE plot generation failed: {e}", exc_info=True)

        # --- 结果字典 (保持不变) ---
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
            "model_path": str(model_path)  # 记录被测试的模型的路径
        }

        return result_dict
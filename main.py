# main.py
import argparse
import os
import random
import shutil
from datetime import datetime
from pathlib import Path
import yaml
from types import SimpleNamespace
import numpy as np
import torch
import pandas as pd
import importlib  # 用于动态导入


# --- 全局设置 ---
# (可选) 限制线程数
_DEFAULT_THREADS = "8"
for k in ["OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
          "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS"]:
    os.environ.setdefault(k, _DEFAULT_THREADS)

# --- 数据集列表 (LODO) ---
datasets_lodo_order = [
    'sleep-edfx',  # LODO 折 0
    'HMC',         # LODO 折 1
    'ISRUC',       # LODO 折 2
    'SHHS1',       # LODO 折 3
    'P2018',       # LODO 折 4
]


# --- 辅助函数 ---
def setup_seed(seed: int = 0):
    """设置随机种子以确保可复现性"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def backup_code(dst_root: Path):
    """备份当前项目中关键代码到 results/<ts>/code_backup/"""
    code_backup_dir = dst_root / 'code_backup'
    code_backup_dir.mkdir(parents=True, exist_ok=True)
    print(f"[信息] 正在备份代码到 {code_backup_dir}")

    # 备份顶层 .py 文件和 configs 目录
    for item in os.listdir('.'):
        source_path = Path(item)
        dest_path = code_backup_dir / source_path.name
        if source_path.is_file() and source_path.suffix == '.py':
            try:
                shutil.copy(str(source_path), str(dest_path))
            except Exception as e:
                print(f"[警告] 无法备份文件 {item}: {e}")
        elif source_path.is_dir() and item == 'configs':
            try:
                shutil.copytree(str(source_path), str(dest_path), dirs_exist_ok=True)
            except Exception as e:
                print(f"[警告] 无法备份目录 {item}: {e}")

    # 备份核心代码目录 (original, improved, models, losses, datasets, utils)
    for dirname in ['original', 'improved', 'models', 'losses', 'datasets', 'utils']:
        source_dir = Path(dirname)
        if source_dir.is_dir():
            dst_dir = code_backup_dir / dirname
            try:
                # 忽略 __pycache__ 目录
                shutil.copytree(str(source_dir), str(dst_dir), dirs_exist_ok=True,
                                ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
            except Exception as e:
                print(f"[警告] 无法备份目录 {dirname}: {e}")


def load_config(config_path: str) -> dict:
    """加载 YAML 配置文件"""
    if not Path(config_path).is_file():
        raise FileNotFoundError(f"未找到配置文件: {config_path}")
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        raise TypeError(f"配置文件 {config_path} 解析后不是字典类型。")
    return config


def calculate_stats(results_list: list, key: str) -> tuple[float, float]:
    """从结果字典列表中提取指定 key 的值，计算均值和标准差"""
    values = [res[key] for res in results_list if key in res and isinstance(res[key], (int, float))]
    if not values:
        return 0.0, 0.0
    mean_val = np.mean(values)
    std_val = np.std(values)
    return mean_val, std_val


# --- 从 allutils 导入 write_aggregate_row ---
# 确保 utils/allutils.py 文件存在且包含 write_aggregate_row 函数
try:
    from utils.allutils import write_aggregate_row
except ImportError:
    print("[错误] 无法从 'utils.allutils' 导入 'write_aggregate_row'。请确认该文件与函数存在。")
    # 定义一个空函数以避免程序崩溃，但聚合结果将无法保存
    def write_aggregate_row(path, row):
        print(f"[警告] write_aggregate_row 不可用。跳过写入 {path} ：{row}")


def _fmt_metric(v, ndigits=5):
    """安全格式化指标（数值 -> 固定小数；字符串/None -> 原样）"""
    if isinstance(v, (int, float)):
        return f"{v:.{ndigits}f}"
    return str(v)


def main():
    parser = argparse.ArgumentParser(description='SleepDG 运行器（支持 YAML 配置）')
    parser.add_argument('--config', type=str, required=True, help='YAML 配置文件路径')
    args = parser.parse_args()

    # -------- 加载配置 --------
    try:
        config = load_config(args.config)
        print("[信息] 配置文件加载成功。")
        # 如需查看完整配置可取消下行注释
        # print(yaml.dump(config, indent=2, allow_unicode=True))
    except (FileNotFoundError, TypeError, yaml.YAMLError) as e:
        print(f"[错误] 加载或解析配置文件失败 '{args.config}'：{e}")
        return

    # -------- 环境设置 --------
    os.environ['CUDA_VISIBLE_DEVICES'] = config.get('gpus', "")
    setup_seed(config.get('seed', 42))
    torch.backends.cudnn.benchmark = True
    # 若使用较新 PyTorch，可开启下行以优化 matmul
    # torch.set_float32_matmul_precision('high')

    # -------- 结果目录与代码备份 --------
    ts = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    results_basedir = Path(config.get('results_root', './results')) / f"{config.get('run_name', 'run')}_{ts}"
    results_basedir.mkdir(parents=True, exist_ok=True)
    print(f"[信息] 结果将保存到：{results_basedir}")
    # 备份代码
    backup_code(results_basedir)
    print("[信息] 代码备份完成。")

    # -------- GPU 诊断 --------
    print("\n--- GPU 诊断信息 ---")
    print(f"CUDA_VISIBLE_DEVICES: {os.getenv('CUDA_VISIBLE_DEVICES')}")
    print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
    print(f"torch.cuda.device_count(): {torch.cuda.device_count()}")
    if torch.cuda.is_available():
        print(f"当前 CUDA 设备索引: {torch.cuda.current_device()}")
        print(f"当前 CUDA 设备名称: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    print("---------------------\n")

    # -------- 动态加载 Trainer --------
    model_version = config.get('model_version', 'improved')  # 默认为“改进版”
    try:
        if model_version == 'original':
            trainer_module = importlib.import_module('original.trainer')
            TrainerClass = trainer_module.Trainer
            print("[信息] 使用 ORIGINAL 版本的 Trainer。")
        elif model_version == 'improved':
            trainer_module = importlib.import_module('improved.trainer_virtual')
            TrainerClass = trainer_module.Trainer  # 假设改进版类名同为 Trainer
            print("[信息] 使用 IMPROVED 版本的 Trainer。")
        else:
            raise ValueError(f"未知的 model_version: {model_version}。可选值为 'original' 或 'improved'。")
    except ImportError as e:
        print(f"[错误] 无法为 model_version='{model_version}' 导入对应的 Trainer：{e}")
        return
    except AttributeError as e:
        print(f"[错误] 在所导入模块中未找到 Trainer 类：{e}")
        return

    # -------- 初始化结果存储 --------
    all_lodo_results = []
    model_type = config.get('model_type', 'unknown')
    metrics_to_average = [
        'test_acc', 'test_f1',
        # 改进版返回了 kappa
        'test_kappa' if model_version == 'improved' else None,
        'wake_f1', 'n1_f1', 'n2_f1', 'n3_f1', 'rem_f1'
    ]
    metrics_to_average = [m for m in metrics_to_average if m is not None]  # 移除 None

    # -------- LODO 循环 --------
    num_total_datasets = len(datasets_lodo_order)
    expected_num_source = num_total_datasets - 1
    if config.get('num_domains', expected_num_source) != expected_num_source:
        print(f"[警告] 配置项 'num_domains' ({config.get('num_domains')}) 与 LODO 设定不一致 ({expected_num_source})，已自动改为 {expected_num_source}。")
        config['num_domains'] = expected_num_source

    for lodo_fold_index, target_dataset_name in enumerate(datasets_lodo_order):
        print(f"\n{'='*15} LODO 折 {lodo_fold_index}/{num_total_datasets-1}：目标域 = {target_dataset_name} {'='*15}")

        # -- 为本次 LODO 运行构建 params 对象 --
        fold_config = config.copy()  # 复制基础配置
        fold_config['target_domains'] = target_dataset_name
        fold_config['model_dir'] = str(results_basedir)  # Trainer 会在此下创建 fold{i}
        fold_config['fold'] = lodo_fold_index
        # 更新 run_name 以包含 fold 信息
        base_run_name = config.get('run_name', 'run')
        fold_config['run_name'] = f"{base_run_name}_target_{target_dataset_name}_lodo{lodo_fold_index}"

        # 将配置字典转换为 SimpleNamespace 对象，方便 Trainer 使用点号访问
        params = SimpleNamespace(**fold_config)

        # -- 打印参数摘要 --
        print("===== 本次 LODO 折的关键参数 =====")
        print(f"model_version  = {params.model_version}")
        print(f"target_domains = {params.target_domains}")
        print(f"fold (LODO 索引) = {params.fold}")
        print(f"run_name       = {params.run_name}")
        print(f"结果根目录      = {params.model_dir}")
        print(f"epochs         = {params.epochs}")
        print(f"lr             = {params.lr}")
        print(f"batch_size     = {params.batch_size}")
        print(f"num_workers    = {params.num_workers}")
        print(f"num_domains    = {params.num_domains}  # 源域数量")
        # ... 如需可继续打印更多参数 ...
        print("================================\n")

        try:
            # 实例化选择的 Trainer
            trainer = TrainerClass(params)
            # train() 返回测试结果字典
            lodo_result_dict = trainer.train()  # 假设 train 方法返回包含测试指标的字典

            if lodo_result_dict and isinstance(lodo_result_dict, dict):
                all_lodo_results.append(lodo_result_dict)
                # 打印关键指标
                acc = lodo_result_dict.get('test_acc', float('nan'))
                f1 = lodo_result_dict.get('test_f1', float('nan'))
                kappa_val = lodo_result_dict.get('test_kappa', None)
                kappa_str = f"，Kappa：{_fmt_metric(kappa_val)}" if kappa_val is not None else ""
                print(f"LODO 折 {lodo_fold_index}（目标域={target_dataset_name}）测试结果 -> Acc：{_fmt_metric(acc)}，F1：{_fmt_metric(f1)}{kappa_str}")
            else:
                print(f"[警告] LODO 折 {lodo_fold_index} 的 Trainer 返回了无效结果。")
        except Exception as e:
            print(f"[错误] 在 LODO 折 {lodo_fold_index} 运行过程中发生异常：{e}")
            import traceback
            traceback.print_exc()  # 打印详细错误堆栈
            continue  # 继续下一个 LODO 折

    # -------- LODO 结束后的总结 --------
    print(f"\n======== 跨 {len(datasets_lodo_order)} 个 LODO 折的整体均值结果（{model_type}） ========")
    # 聚合 CSV 文件路径在 allfold 目录下
    aggregate_csv_path = results_basedir / "allfold" / "aggregate_results.csv"
    # 确保 allfold 目录存在 (Trainer 可能已创建，但以防万一)
    aggregate_csv_path.parent.mkdir(parents=True, exist_ok=True)

    if all_lodo_results and len(all_lodo_results) == num_total_datasets:
        print("指标名称                | LODO 折均值 ± 标准差")
        print("-----------------------|----------------------")

        avg_row = {
            "time": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "run_id": f"{config.get('run_name', 'run')}_LODO_avg",
            "fold": "mean",
            "best_val_acc": "N/A",
            "best_val_f1": "N/A",
            "model_path": "N/A"
        }

        # 计算并打印均值和标准差
        for key in metrics_to_average:
            mean_val, std_val = calculate_stats(all_lodo_results, key)
            print(f"{key:<23} | {_fmt_metric(mean_val)} ± {_fmt_metric(std_val)}")
            # CSV 保存格式化字符串，便于阅读
            avg_row[key] = f"{_fmt_metric(mean_val)} +/- {_fmt_metric(std_val)}"

        try:
            write_aggregate_row(aggregate_csv_path, row=avg_row)
            print(f"\n已将 LODO 均值结果追加写入：{aggregate_csv_path}")
        except Exception as e:
            print(f"[错误] 写入 LODO 均值结果到 CSV 失败：{e}")
    elif all_lodo_results:
        print(f"[警告] 仅 {len(all_lodo_results)}/{num_total_datasets} 个 LODO 折成功完成。未计算/保存整体均值结果。")
    else:
        print("[警告] 没有成功的 LODO 运行。")

    print("======================================================")
    print(f"\n[信息] 所有 LODO 运行结束。结果目录：{results_basedir}")


if __name__ == '__main__':
    main()

# test.py
import argparse
import os
import random
from pathlib import Path
from types import SimpleNamespace
from collections import OrderedDict
from typing import List, Dict, Any, Union

import yaml
import numpy as np
import torch
import importlib

# ---------------- 必要项目内模块 ----------------
try:
    from datasets.dataset import LoadDataset
    from evaluator import Evaluator
except ImportError as e:
    print(f"[错误] 导入必需模块失败 (LoadDataset, Evaluator): {e}")
    print("请确保 datasets/dataset.py 和 evaluator.py 位于正确的路径下。")
    exit(1)


# ================== 辅助函数 ==================
def setup_seed(seed: int = 0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def load_config(config_path: str) -> dict:
    cfg_file = Path(config_path)
    if not cfg_file.is_file():
        raise FileNotFoundError(f"配置文件未找到: {config_path}")
    with open(cfg_file, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        raise TypeError(f"配置文件 {config_path} 未能加载为字典。")
    return config


def import_model_from_config(model_cfg: dict):
    required = ["import_path", "class_name"]
    for k in required:
        if k not in model_cfg:
            raise KeyError(f"[配置缺失] model.{k} 未提供。")
    try:
        module = importlib.import_module(model_cfg["import_path"])
    except Exception as e:
        raise ImportError(f"[错误] 无法 import 模块 '{model_cfg['import_path']}': {e}")

    try:
        ModelClass = getattr(module, model_cfg["class_name"])
    except AttributeError:
        raise AttributeError(
            f"[错误] 模块 '{model_cfg['import_path']}' 中未找到类 '{model_cfg['class_name']}'。"
        )
    return ModelClass


def normalize_state_dict(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    new_state = OrderedDict()
    for k, v in sd.items():
        name = k[7:] if isinstance(k, str) and k.startswith("module.") else k
        new_state[name] = v
    return new_state


def load_state_dict_from_ckpt(ckpt_path: Path) -> Dict[str, torch.Tensor]:
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"[错误] 模型权重文件不存在: {ckpt_path}")

    # 注意: weights_only=False 解决了您之前遇到的第一个安全加载问题，这里保持不变。
    obj = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    # 1. 尝试直接 state_dict
    if isinstance(obj, dict) and obj and all(isinstance(v, torch.Tensor) for v in obj.values()):
        return normalize_state_dict(obj)

    # 2. 尝试从字典中提取
    if isinstance(obj, dict):
        # ** <<< 这里是新增的逻辑，用于支持您的 'model_state' 键 >>> **
        if "model_state" in obj and isinstance(obj["model_state"], dict):
            print("[信息] 识别到权重格式: 键 'model_state'")
            return normalize_state_dict(obj["model_state"])

        # 检查标准的 'state_dict' 键
        if "state_dict" in obj and isinstance(obj["state_dict"], dict):
            return normalize_state_dict(obj["state_dict"])

        # 检查其他常见的键
        for key in ["model", "net", "ema", "weights"]:
            if key in obj and isinstance(obj[key], dict):
                print(f"[信息] 识别到权重格式: 键 '{key}'")
                return normalize_state_dict(obj[key])

    # 3. 失败并抛出错误
    supported_keys = "'state_dict', 'model', 'net', 'ema', 'weights', 'model_state'"  # 更新支持列表
    raise ValueError(
        f"[错误] 未识别的权重格式: {ckpt_path}\n"
        f"支持：(1) 直接state_dict；(2) 包含 'state_dict' 键；或常见键如 {supported_keys}。"
    )


def resolve_ckpt_paths(model_cfg: dict) -> List[Path]:
    paths: List[Path] = []
    if "ckpts" in model_cfg:
        if not isinstance(model_cfg["ckpts"], list) or not model_cfg["ckpts"]:
            raise ValueError("[配置错误] model.ckpts 必须是非空列表。")
        paths = [Path(p) for p in model_cfg["ckpts"]]
    elif "ckpt_glob" in model_cfg:
        pattern = model_cfg["ckpt_glob"]
        paths = sorted(Path(".").glob(pattern)) if not Path(pattern).is_absolute() else sorted(Path("/").glob(pattern.lstrip("/")))
        if not paths:
            raise FileNotFoundError(f"[配置错误] ckpt_glob 未匹配到任何文件: {pattern}")
    else:
        raise KeyError("[配置缺失] 需要提供 model.ckpts 或 model.ckpt_glob 之一。")
    for p in paths:
        if not p.is_file():
            raise FileNotFoundError(f"[配置错误] 权重文件不存在: {p}")
    return paths


# ================== 集成封装（结构保持版） ==================
class EnsembleWrapper(torch.nn.Module):
    """
    多子模型集成，严格“保持返回结构”：
      * 若子模型返回 Tensor：返回聚合后的 Tensor
      * 若返回 tuple/list：长度与顺序保持一致，仅对“主输出项”做聚合，其余项取第一个子模型（或能聚合则聚合）
      * 若返回 dict：键集合保持一致，仅对“主输出键”做聚合，其余键取第一个子模型（或能聚合则聚合）

    可选配置（放在 config.model 下）：
      - ensemble_method: "logits" | "probs"（默认 logits）
      - primary_index:   int（当返回为 tuple/list 时，指定“主输出”的索引，默认自动猜测为 0）
      - primary_key:     str（当返回为 dict 时，指定“主输出”的键，默认在 ['logits','pred','probs','y_hat','output'] 中猜）
    """
    def __init__(
        self,
        ModelClass,
        params: SimpleNamespace,
        ckpt_paths: List[Path],
        ensemble_method: str = "logits",
        strict_load: bool = True,
        device: torch.device = torch.device("cpu"),
        primary_index: int = None,
        primary_key: str = None,
    ):
        super().__init__()
        assert ensemble_method in ("logits", "probs"), "ensemble_method 只能是 'logits' 或 'probs'。"
        self.ensemble_method = ensemble_method
        self.device = device
        self.models = torch.nn.ModuleList()
        self._sub_has_inference: bool = False
        self._num_models = len(ckpt_paths)
        self.primary_index = primary_index
        self.primary_key = primary_key

        for idx, ckpt in enumerate(ckpt_paths):
            sub_model = ModelClass(params)
            state = load_state_dict_from_ckpt(ckpt)
            res = sub_model.load_state_dict(state, strict=strict_load)
            if not strict_load and isinstance(res, tuple) and len(res) == 2:
                missing, unexpected = res
                if missing:
                    print(f"[警告] 子模型#{idx} 缺失权重: {missing}")
                if unexpected:
                    print(f"[警告] 子模型#{idx} 多余权重: {unexpected}")
            sub_model.to(self.device)
            sub_model.eval()
            self.models.append(sub_model)

        if len(self.models) > 0 and hasattr(self.models[0], "inference") and callable(getattr(self.models[0], "inference")):
            self._sub_has_inference = True

        print(f"[信息] 集成模型已就绪，子模型数量: {len(self.models)}，集成方式: {self.ensemble_method}，"
              f"子模型提供 inference: {self._sub_has_inference}")

    # ---------- 工具函数 ----------
    @staticmethod
    def _to_tensor(x: Any) -> torch.Tensor:
        if isinstance(x, torch.Tensor):
            return x
        if isinstance(x, np.ndarray):
            return torch.from_numpy(x)
        raise TypeError(f"无法将类型 {type(x).__name__} 转为 Tensor。")

    @staticmethod
    def _guess_primary_key(d: Dict[str, Any]) -> str:
        for k in ["logits", "pred", "probs", "y_hat", "output"]:
            if k in d:
                return k
        # 回退：选第一个张量型键
        for k, v in d.items():
            if isinstance(v, (torch.Tensor, np.ndarray)):
                return k
        # 实在不行就选第一个键
        return next(iter(d.keys()))

    def _maybe_to_device(self, t: torch.Tensor) -> torch.Tensor:
        return t if t.device == self.device else t.to(self.device)

    def _to_probs_if_needed(self, t: torch.Tensor) -> torch.Tensor:
        return torch.softmax(t, dim=1) if self.ensemble_method == "probs" else t

    # ---------- 结构保持的聚合 ----------
    @torch.no_grad()
    def _aggregate_like(self, first_out: Any, all_outs: List[Any]) -> Any:
        """
        根据 first_out 的结构，生成“同结构”的聚合结果。
        """
        # 1) Tensor：直接逐模型加和再平均
        if isinstance(first_out, (torch.Tensor, np.ndarray)):
            agg = None
            for o in all_outs:
                t = self._to_tensor(o)
                t = self._maybe_to_device(t)
                t = self._to_probs_if_needed(t)
                agg = t if agg is None else (agg + t)
            return agg / float(self._num_models)

        # 2) tuple/list：只聚合“主输出项”，其余项沿用第一个（或可逐项聚合的张量则聚合）
        if isinstance(first_out, (tuple, list)):
            L = len(first_out)
            idx_main = self.primary_index if self.primary_index is not None else 0
            if not (0 <= idx_main < L):
                idx_main = 0  # 守护

            # 收集每个位置的列表
            per_pos = [[] for _ in range(L)]
            for o in all_outs:
                if not isinstance(o, (tuple, list)) or len(o) != L:
                    raise TypeError(f"[错误] 子模型返回的 tuple/list 结构不一致（期望长度 {L}）。")
                for i in range(L):
                    per_pos[i].append(o[i])

            result = []
            for i in range(L):
                candidates = per_pos[i]
                if i == idx_main:
                    # 主输出：强制做张量聚合
                    agg = None
                    for c in candidates:
                        t = self._to_tensor(c)
                        t = self._maybe_to_device(t)
                        t = self._to_probs_if_needed(t)
                        agg = t if agg is None else (agg + t)
                    out_i = agg / float(self._num_models)
                else:
                    # 非主输出：若都是张量且形状可对齐，则尝试“平均”；否则直接取第一个
                    try:
                        if all(isinstance(c, (torch.Tensor, np.ndarray)) for c in candidates):
                            # 形状一致再平均
                            shapes = [tuple(self._to_tensor(c).shape) for c in candidates]
                            if len(set(shapes)) == 1:
                                agg = None
                                for c in candidates:
                                    t = self._to_tensor(c)
                                    t = self._maybe_to_device(t)
                                    agg = t if agg is None else (agg + t)
                                out_i = agg / float(self._num_models)
                            else:
                                out_i = candidates[0]
                        else:
                            out_i = candidates[0]
                    except Exception:
                        out_i = candidates[0]
                result.append(out_i)
            return tuple(result) if isinstance(first_out, tuple) else result

        # 3) dict：聚合 primary_key，其余键取第一个（或能对齐则聚合）
        if isinstance(first_out, dict):
            key_main = self.primary_key if self.primary_key is not None else self._guess_primary_key(first_out)
            # 收集每个键的候选
            keys = list(first_out.keys())
            per_key: Dict[str, List[Any]] = {k: [] for k in keys}
            for o in all_outs:
                if not isinstance(o, dict):
                    raise TypeError("[错误] 子模型返回结构不一致：有的为 dict，有的不是。")
                # 允许子模型字典包含额外键，但我们只关心 first_out 的键集合
                for k in keys:
                    if k not in o:
                        raise KeyError(f"[错误] 子模型返回的 dict 缺少键：{k}")
                    per_key[k].append(o[k])

            result: Dict[str, Any] = {}
            for k in keys:
                candidates = per_key[k]
                if k == key_main:
                    agg = None
                    for c in candidates:
                        t = self._to_tensor(c)
                        t = self._maybe_to_device(t)
                        t = self._to_probs_if_needed(t)
                        agg = t if agg is None else (agg + t)
                    result[k] = agg / float(self._num_models)
                else:
                    try:
                        if all(isinstance(c, (torch.Tensor, np.ndarray)) for c in candidates):
                            shapes = [tuple(self._to_tensor(c).shape) for c in candidates]
                            if len(set(shapes)) == 1:
                                agg = None
                                for c in candidates:
                                    t = self._to_tensor(c)
                                    t = self._maybe_to_device(t)
                                    agg = t if agg is None else (agg + t)
                                result[k] = agg / float(self._num_models)
                            else:
                                result[k] = candidates[0]
                        else:
                            result[k] = candidates[0]
                    except Exception:
                        result[k] = candidates[0]
            return result

        # 4) 其他类型：无法聚合，直接取第一个子模型输出（并警告）
        print(f"[警告] 未支持的返回类型 {type(first_out).__name__}，将直接沿用第一个子模型输出。")
        return first_out

    # ---------- 对外接口 ----------
    @torch.no_grad()
    def _call_each(self, fn_name: str, *args, **kwargs) -> Any:
        outs = []
        for m in self.models:
            fn = getattr(m, fn_name)
            outs.append(fn(*args, **kwargs))
        # 以第一个的结构为模板，做结构保持聚合
        return self._aggregate_like(outs[0], outs)

    @torch.no_grad()
    def forward(self, *args, **kwargs):
        # 一些 Evaluator 会直接调用 model(x)
        if hasattr(self.models[0], "forward"):
            return self._call_each("forward", *args, **kwargs)
        raise AttributeError("子模型未实现 forward。")

    @torch.no_grad()
    def inference(self, *args, **kwargs):
        if self._sub_has_inference:
            return self._call_each("inference", *args, **kwargs)
        # 退回 forward
        return self._call_each("forward", *args, **kwargs)


# ================== 主流程 ==================
def main():
    parser = argparse.ArgumentParser(description="SleepDG 多折集成评估（完全由配置文件驱动）")
    parser.add_argument("--config", type=str, required=True, help="用于测试的 YAML 配置文件路径")
    args = parser.parse_args()

    try:
        cfg = load_config(args.config)
        print("[信息] 已加载测试配置：")
        print(yaml.dump(cfg, indent=2, allow_unicode=True))
    except (FileNotFoundError, TypeError, yaml.YAMLError) as e:
        print(f"[错误] 加载或解析配置文件失败: {e}")
        return

    params = SimpleNamespace(**cfg)

    required_top = ["datasets_dir", "test_dataset", "batch_size", "num_workers"]
    for k in required_top:
        if not hasattr(params, k):
            print(f"[错误] 配置缺少必要键: {k}")
            return
    if not hasattr(params, "model") or not isinstance(params.model, dict):
        print("[错误] 配置缺少必要键: model（应为字典，包含 import_path/class_name/ckpts 或 ckpt_glob 等）")
        return

    model_cfg = params.model
    model_cfg.setdefault("ensemble_method", "logits")
    model_cfg.setdefault("strict_load", True)
    model_cfg.setdefault("use_dataparallel", False)

    if hasattr(params, "gpus"):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(params.gpus)
    if hasattr(params, "seed"):
        setup_seed(int(params.seed))

    print("\n--- GPU 诊断 ---")
    print(f"CUDA_VISIBLE_DEVICES: {os.getenv('CUDA_VISIBLE_DEVICES')}")
    print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
    print(f"torch.cuda.device_count(): {torch.cuda.device_count()}")
    if torch.cuda.is_available():
        current_device_index = torch.cuda.current_device()
        print(f"当前设备索引: {current_device_index}")
        print(f"设备名称: {torch.cuda.get_device_name(current_device_index)}")
    print("---------------------\n")

    try:
        ckpt_paths = resolve_ckpt_paths(model_cfg)
        print("[信息] 将加载以下权重进行集成：")
        for p in ckpt_paths:
            print(f"  - {p}")
    except Exception as e:
        print(f"[错误] 权重解析失败: {e}")
        return

    try:
        ModelClass = import_model_from_config(model_cfg)
    except Exception as e:
        print(f"[错误] 加载模型类失败: {e}")
        return

    print(f"[信息] 加载测试数据集: {params.test_dataset} ...")
    try:
        load_params = SimpleNamespace(
            datasets_dir=params.datasets_dir,
            test_dataset=params.test_dataset,
            batch_size=params.batch_size,
            num_workers=params.num_workers
        )
        data_loader_dict, _ = LoadDataset(load_params).get_data_loader()
        if "test" in data_loader_dict and len(data_loader_dict["test"]) > 0:
            test_loader = data_loader_dict["test"]
        else:
            if "val" in data_loader_dict and len(data_loader_dict["val"]) > 0:
                print("[警告] 'test' 为空，使用 'val' 加载器作为测试加载器。")
                test_loader = data_loader_dict["val"]
            elif "train" in data_loader_dict and len(data_loader_dict["train"]) > 0:
                print("[警告] 'test' 和 'val' 均为空，使用 'train' 加载器作为测试加载器。")
                test_loader = data_loader_dict["train"]
            else:
                raise ValueError(f"无法为测试数据集加载数据: {params.test_dataset}。所有加载器均为空。")
        print(f"[信息] 测试数据加载完成。批次数: {len(test_loader)}")
    except Exception as e:
        print(f"[错误] 加载数据集 '{params.test_dataset}' 失败: {e}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[信息] 构建集成模型 ...")
    try:
        ensemble = EnsembleWrapper(
            ModelClass=ModelClass,
            params=params,
            ckpt_paths=ckpt_paths,
            ensemble_method=model_cfg["ensemble_method"],
            strict_load=bool(model_cfg["strict_load"]),
            device=device,
            primary_index=model_cfg.get("primary_index", None),
            primary_key=model_cfg.get("primary_key", None),
        )
    except Exception as e:
        print(f"[错误] 构建或加载集成模型失败: {e}")
        return

    if torch.cuda.is_available() and torch.cuda.device_count() > 1 and model_cfg["use_dataparallel"]:
        print("[信息] 使用 DataParallel 进行评估。")
        ensemble = torch.nn.DataParallel(ensemble)

    print("\n--- 开始评估（N 折集成） ---")
    evaluator = Evaluator(params, test_loader)
    try:
        test_acc, test_f1, test_cm, test_wake_f1, test_n1_f1, test_n2_f1, \
            test_n3_f1, test_rem_f1, test_kappa, test_report = evaluator.get_accuracy(ensemble)
    except Exception as e:
        print(f"[错误] 评估阶段出错: {e}")
        return

    print(f"\n***************** 测试结果（{params.test_dataset}｜N={len(ckpt_paths)} 集成） *****************")
    print(f"模型类:             {model_cfg['import_path']}.{model_cfg['class_name']}")
    print(f"集成方式:           {model_cfg['ensemble_method']}")
    print(f"权重文件数:         {len(ckpt_paths)}")
    print("-" * 65)
    print(f"测试准确率 (Acc):    {test_acc:.5f}")
    print(f"测试宏 F1 分数:     {test_f1:.5f}")
    print(f"测试 Cohen's Kappa:  {test_kappa:.5f}")
    print("\n混淆矩阵:")
    print(test_cm)
    print("\n各类别 F1 分数:")
    print(f"  Wake: {test_wake_f1:.5f}")
    print(f"  N1:   {test_n1_f1:.5f}")
    print(f"  N2:   {test_n2_f1:.5f}")
    print(f"  N3:   {test_n3_f1:.5f}")
    print(f"  REM:  {test_rem_f1:.5f}")
    print("\n分类报告:")
    print(test_report)
    print("******************************************************************")

    results_root = Path(getattr(params, "results_log_dir", "./results"))
    results_root.mkdir(parents=True, exist_ok=True)
    results_output_dir = results_root / "test_results"
    results_output_dir.mkdir(parents=True, exist_ok=True)
    results_file = results_output_dir / f"test_on_{params.test_dataset}_ensemble{len(ckpt_paths)}.txt"
    try:
        with open(results_file, "w", encoding="utf-8") as f:
            f.write(f"数据集测试结果: {params.test_dataset}\n")
            f.write(f"模型类: {model_cfg['import_path']}.{model_cfg['class_name']}\n")
            f.write(f"集成方式: {model_cfg['ensemble_method']}\n")
            f.write(f"权重文件数: {len(ckpt_paths)}\n")
            for i, p in enumerate(ckpt_paths):
                f.write(f"  [{i}] {p}\n")
            f.write("-" * 65 + "\n")
            f.write(f"准确率 (Acc): {test_acc:.5f}\n")
            f.write(f"宏 F1 分数:   {test_f1:.5f}\n")
            f.write(f"Kappa 系数:   {test_kappa:.5f}\n\n")
            f.write("混淆矩阵:\n")
        with open(results_file, "a", encoding="utf-8") as f:
            np.savetxt(f, test_cm, fmt="%d")
            f.write("\n\n各类别 F1 分数:\n")
            f.write(f"  Wake: {test_wake_f1:.5f}\n")
            f.write(f"  N1:   {test_n1_f1:.5f}\n")
            f.write(f"  N2:   {test_n2_f1:.5f}\n")
            f.write(f"  N3:   {test_n3_f1:.5f}\n")
            f.write(f"  REM:  {test_rem_f1:.5f}\n\n")
            f.write("分类报告:\n")
            f.write(str(test_report))
        print(f"\n[信息] 测试结果已保存至: {results_file}")
    except Exception as e:
        print(f"[警告] 保存测试结果到文件失败: {e}")


if __name__ == "__main__":
    main()

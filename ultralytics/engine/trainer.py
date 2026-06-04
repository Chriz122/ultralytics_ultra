# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Train a model on a dataset.

Usage:
    $ yolo mode=train model=yolo26n.pt data=coco8.yaml imgsz=640 epochs=100 batch=16
"""

from __future__ import annotations

import gc
import math
import os
import subprocess
import time
import warnings
from copy import copy, deepcopy
from datetime import datetime, timedelta
from functools import partial
from pathlib import Path

import numpy as np
import torch
from torch import distributed as dist
from torch import nn, optim

from ultralytics import __version__
from ultralytics.cfg import get_cfg, get_save_dir
from ultralytics.data.utils import check_cls_dataset, check_det_dataset
from ultralytics.nn.tasks import load_checkpoint
from ultralytics.optim import MuSGD
from ultralytics.utils import (
    DEFAULT_CFG,
    GIT,
    LOCAL_RANK,
    LOGGER,
    RANK,
    TQDM,
    YAML,
    callbacks,
    clean_url,
    colorstr,
    emojis,
)
from ultralytics.utils.autobatch import check_train_batch_size
from ultralytics.utils.lora import (
    LoraTrainingStrategy,
    _is_adapter_param,
    _unfreeze_detection_head,
    apply_lora,
    get_lora_training_stats,
    resolve_adalora_total_step,
    save_lora_adapters,
)
from ultralytics.utils.checks import check_amp, check_file, check_imgsz, check_model_file_from_stem, print_args
from ultralytics.utils.dist import ddp_cleanup, generate_ddp_command
from ultralytics.utils.files import get_latest_run
from ultralytics.utils.plotting import plot_results
from ultralytics.utils.torch_utils import (
    TORCH_2_4,
    EarlyStopping,
    ModelEMA,
    attempt_compile,
    autocast,
    convert_optimizer_state_dict_to_fp16,
    init_seeds,
    one_cycle,
    select_device,
    strip_optimizer,
    torch_distributed_zero_first,
    unset_deterministic,
    unwrap_model,
)

def save_trainer_args_yaml(save_dir: Path, args) -> None:
    """Persist trainer arguments to args.yaml, serializing complex augmentation objects safely."""
    args_dict = vars(args).copy()
    if args_dict.get("augmentations") is not None:
        args_dict["augmentations"] = [repr(t) for t in args_dict["augmentations"]]
    YAML.save(save_dir / "args.yaml", args_dict)


def _hierarchical_hook(storage, key, module, inputs, output):
    """Module-level forward hook for hierarchical distillation feature caching.
    
    Uses a plain function (not a closure) to ensure hooks are picklable when
    saving model checkpoints. Bound via functools.partial in trainer.
    """
    storage[key] = output


def update_args_with_lora_runtime_metadata(args, model) -> None:
    """Copy runtime LoRA metadata from the adapted model onto trainer args."""
    base_model = getattr(model, "module", model)
    metadata = getattr(base_model, "lora_runtime_metadata", {}) or {}
    if not metadata:
        return

    if "requested_backend" in metadata:
        args.requested_lora_backend = metadata["requested_backend"]
    if "effective_backend" in metadata:
        args.effective_lora_backend = metadata["effective_backend"]
    if "requested_variant" in metadata:
        args.requested_lora_variant = metadata["requested_variant"]
    if "effective_variant" in metadata:
        args.effective_lora_variant = metadata["effective_variant"]
    if "peft_type" in metadata:
        args.effective_lora_type = metadata["peft_type"]
    if "requested_init_lora_weights" in metadata:
        args.requested_lora_init_lora_weights = metadata["requested_init_lora_weights"]
    if "effective_init_lora_weights" in metadata:
        args.effective_lora_init_lora_weights = metadata["effective_init_lora_weights"]
    if metadata.get("safety_profile"):
        args.lora_safety_profile = metadata["safety_profile"]
    if metadata.get("safety_overrides"):
        args.lora_safety_overrides = metadata["safety_overrides"]


class AFSSManager:
    def __init__(self, dataset_size, args):
        self.num_images = dataset_size
        self.args = args  # 傳入設定參數
        self.P = np.zeros(dataset_size)
        self.R = np.zeros(dataset_size)
        self.last_epoch = np.zeros(dataset_size, dtype=int) - 1  # 紀錄最後一次參與訓練的 epoch
        self.levels = np.zeros(dataset_size, dtype=int)          # 0: Hard, 1: Moderate, 2: Easy

    def update_states(self, image_indices, precisions, recalls):
        """根據驗證結果更新 P 與 R"""
        self.P[image_indices] = precisions
        self.R[image_indices] = recalls

    def schedule(self, current_epoch):
        """根據 AFSS 論文邏輯，返回當前 Epoch 應參與訓練的圖片 Index"""
        # 為確保 DDP 多卡訓練時各卡抽樣結果一致，綁定隨機種子
        np.random.seed(current_epoch)

        # 從 args 動態獲取 yaml 設定的閾值 (若無則使用預設值 0.85 與 0.55)
        easy_thresh = getattr(self.args, 'afss_easy_thresh', 0.85)
        hard_thresh = getattr(self.args, 'afss_hard_thresh', 0.55)
        easy_ratio = getattr(self.args, 'afss_easy_ratio', 0.02)
        easy_forced_gap = getattr(self.args, 'afss_easy_forced_gap', 10)
        moderate_forced_ratio = getattr(self.args, 'afss_moderate_forced_ratio', 0.4)
        moderate_forced_gap = getattr(self.args, 'afss_moderate_forced_gap', 3)

        # 論文 Eq(1) & Eq(2): 計算 Learning Sufficiency
        suff = np.minimum(self.P, self.R)
        self.levels[suff > easy_thresh] = 2
        self.levels[(suff <= easy_thresh) & (suff >= hard_thresh)] = 1
        self.levels[suff < hard_thresh] = 0

        selected =[]

        # === 🟢 新增：印出當前 Epoch 的難度分佈日誌 ===
        hard_count = np.sum(self.levels == 0)
        mod_count = np.sum(self.levels == 1)
        easy_count = np.sum(self.levels == 2)
        print(f"\n[AFSS] Epoch {current_epoch} 難度分佈 -> Hard: {hard_count}, Moderate: {mod_count}, Easy: {easy_count} (Thresholds: Easy > {easy_thresh}, Hard < {hard_thresh})")
        # ===============================================

        # 1. Continuous Review (Easy 圖片)
        easy_idx = np.where(self.levels == 2)[0]
        unseen_epochs = current_epoch - 1 - self.last_epoch[easy_idx]
        af = easy_idx[unseen_epochs >= easy_forced_gap]  # Long-unseen
        
        total_easy_target = int(easy_ratio * len(easy_idx))
        max_e1 = int(0.5 * total_easy_target)
        
        e1 = np.random.choice(af, min(len(af), max_e1), replace=False) if len(af) > 0 else[]
        rem_easy = np.setdiff1d(easy_idx, e1)
        e2_target = total_easy_target - len(e1)
        e2 = np.random.choice(rem_easy, min(len(rem_easy), e2_target), replace=False) if e2_target > 0 else[]
        
        selected.extend(e1)
        selected.extend(e2)

        # 2. Short-Term Coverage (Moderate 圖片)
        mod_idx = np.where(self.levels == 1)[0]
        unseen_mod = current_epoch - 1 - self.last_epoch[mod_idx]
        bf = mod_idx[unseen_mod >= moderate_forced_gap]  # Forced coverage
        
        total_mod_target = int(moderate_forced_ratio * len(mod_idx))
        rem_mod = np.setdiff1d(mod_idx, bf)
        br_target = total_mod_target - len(bf)
        br = np.random.choice(rem_mod, min(len(rem_mod), br_target), replace=False) if br_target > 0 else[]
        
        selected.extend(bf)
        selected.extend(br)

        # 3. Full Coverage (Hard 圖片)
        hard_idx = np.where(self.levels == 0)[0]
        selected.extend(hard_idx)

        selected = np.array(selected, dtype=int)
        self.last_epoch[selected] = current_epoch  # 更新參與紀錄

        # === 🟢 新增：印出實際抽樣的數量 ===
        print(f"[AFSS] 實際挑選參與訓練的圖片數: {len(selected)} / {self.num_images} (將縮減 Epoch 迭代數)")
        # ===================================

        return selected


class BaseTrainer:
    """A base class for creating trainers.

    This class provides the foundation for training YOLO models, handling the training loop, validation, checkpointing,
    and various training utilities. It supports both single-GPU and multi-GPU distributed training.

    Attributes:
        args (SimpleNamespace): Configuration for the trainer.
        validator (BaseValidator): Validator instance.
        model (nn.Module): Model instance.
        callbacks (defaultdict): Dictionary of callbacks.
        save_dir (Path): Directory to save results.
        wdir (Path): Directory to save weights.
        last (Path): Path to the last checkpoint.
        best (Path): Path to the best checkpoint.
        save_period (int): Save checkpoint every x epochs (disabled if < 1).
        batch_size (int): Batch size for training.
        epochs (int): Number of epochs to train for.
        start_epoch (int): Starting epoch for training.
        device (torch.device): Device to use for training.
        amp (bool): Flag to enable AMP (Automatic Mixed Precision).
        scaler (amp.GradScaler): Gradient scaler for AMP.
        data (str): Path to data.
        ema (nn.Module): EMA (Exponential Moving Average) of the model.
        resume (bool): Resume training from a checkpoint.
        lf (nn.Module): Loss function.
        scheduler (torch.optim.lr_scheduler._LRScheduler): Learning rate scheduler.
        best_fitness (float): The best fitness value achieved.
        fitness (float): Current fitness value.
        loss (float): Current loss value.
        tloss (float): Total loss value.
        loss_names (list): List of loss names.
        csv (Path): Path to results CSV file.
        metrics (dict): Dictionary of metrics.
        plots (dict): Dictionary of plots.

    Methods:
        train: Execute the training process.
        validate: Run validation on the test set.
        save_model: Save model training checkpoints.
        get_dataset: Get train and validation datasets.
        setup_model: Load, create, or download model.
        build_optimizer: Construct an optimizer for the model.

    Examples:
        Initialize a trainer and start training
        >>> trainer = BaseTrainer(cfg="config.yaml")
        >>> trainer.train()
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """Initialize the BaseTrainer class.

        Args:
            cfg (str, optional): Path to a configuration file.
            overrides (dict, optional): Configuration overrides.
            _callbacks (list, optional): List of callback functions.
        """
        self.hub_session = overrides.pop("session", None)  # HUB
        self.args = get_cfg(cfg, overrides)
        self.check_resume(overrides)
        self.device = select_device(self.args.device)
        # Update "-1" devices so post-training val does not repeat search
        self.args.device = os.getenv("CUDA_VISIBLE_DEVICES") if "cuda" in str(self.device) else str(self.device)
        self.validator = None
        self.metrics = None
        self.plots = {}
        init_seeds(self.args.seed + 1 + RANK, deterministic=self.args.deterministic)

        # Dirs
        self.save_dir = get_save_dir(self.args)
        self.args.name = self.save_dir.name  # update name for loggers
        self.wdir = self.save_dir / "weights"  # weights dir
        if RANK in {-1, 0}:
            self.wdir.mkdir(parents=True, exist_ok=True)  # make dir
            self.args.save_dir = str(self.save_dir)
            # Save run args, serializing augmentations as reprs for resume compatibility
            args_dict = vars(self.args).copy()
            if args_dict.get("augmentations") is not None:
                # Serialize Albumentations transforms as their repr strings for checkpoint compatibility
                args_dict["augmentations"] =[repr(t) for t in args_dict["augmentations"]]
            YAML.save(self.save_dir / "args.yaml", args_dict)  # save run args
        self.last, self.best = self.wdir / "last.pt", self.wdir / "best.pt"  # checkpoint paths
        self.save_period = self.args.save_period

        self.batch_size = self.args.batch
        self.epochs = self.args.epochs or 100  # in case users accidentally pass epochs=None with timed training
        self.start_epoch = 0
        if RANK == -1:
            print_args(vars(self.args))

        # Device
        if self.device.type in {"cpu", "mps"}:
            self.args.workers = 0  # faster CPU training as time dominated by inference, not dataloading

        # Callbacks - initialize early so on_pretrain_routine_start can capture original args.data
        self.callbacks = _callbacks or callbacks.get_default_callbacks()

        if isinstance(self.args.device, str) and len(self.args.device):  # i.e. device='0' or device='0,1,2,3'
            world_size = len(self.args.device.split(","))
        elif isinstance(self.args.device, (tuple, list)):  # i.e. device=[0, 1, 2, 3] (multi-GPU from CLI is list)
            world_size = len(self.args.device)
        elif self.args.device in {"cpu", "mps"}:  # i.e. device='cpu' or 'mps'
            world_size = 0
        elif torch.cuda.is_available():  # i.e. device=None or device='' or device=number
            world_size = 1  # default to device 0
        else:  # i.e. device=None or device=''
            world_size = 0

        self.ddp = world_size > 1 and "LOCAL_RANK" not in os.environ
        self.world_size = world_size
        # Run on_pretrain_routine_start before get_dataset() to capture original args.data (e.g., ul:// URIs)
        if RANK in {-1, 0} and not self.ddp:
            callbacks.add_integration_callbacks(self)
            self.run_callbacks("on_pretrain_routine_start")

        # Model and Dataset
        self.model = check_model_file_from_stem(self.args.model)  # add suffix, i.e. yolo26n -> yolo26n.pt
        with torch_distributed_zero_first(LOCAL_RANK):  # avoid auto-downloading dataset multiple times
            self.data = self.get_dataset()

        self.ema = None

        # Optimization utils init
        self.lf = None
        self.scheduler = None

        # Epoch level metrics
        self.best_fitness = None
        self.fitness = None
        self.loss = None
        self.tloss = None
        self.loss_names = ["Loss"]
        self.csv = self.save_dir / "results.csv"
        if self.csv.exists() and not self.args.resume:
            self.csv.unlink()
        self.plot_idx = [0, 1, 2]
        self.nan_recovery_attempts = 0

    def add_callback(self, event: str, callback):
        """Append the given callback to the event's callback list."""
        self.callbacks[event].append(callback)

    def set_callback(self, event: str, callback):
        """Override the existing callbacks with the given callback for the specified event."""
        self.callbacks[event] = [callback]

    def run_callbacks(self, event: str):
        """Run all existing callbacks associated with a particular event."""
        for callback in self.callbacks.get(event,[]):
            callback(self)

    def train(self):
        """Allow device='', device=None on Multi-GPU systems to default to device=0."""
        # Run subprocess if DDP training, else train normally
        if self.ddp:
            # Argument checks
            if self.args.rect:
                LOGGER.warning("'rect=True' is incompatible with Multi-GPU training, setting 'rect=False'")
                self.args.rect = False
            if self.args.batch < 1.0:
                raise ValueError(
                    "AutoBatch with batch<1 not supported for Multi-GPU training, "
                    f"please specify a valid batch size multiple of GPU count {self.world_size}, i.e. batch={self.world_size * 8}."
                )

            # Command
            cmd, file = generate_ddp_command(self)
            try:
                LOGGER.info(f"{colorstr('DDP:')} debug command {' '.join(cmd)}")
                subprocess.run(cmd, check=True)
            except Exception as e:
                raise e
            finally:
                ddp_cleanup(self, str(file))

        else:
            self._do_train()

    # def _setup_scheduler(self):
    #     """
    #     自定義 RTMO 兩階段 Cosine 衰減調度器。
    #     符合論文：Phase 1 (0.004 -> 0.0002) | Phase 2 (0.0005 -> 0.0002)
    #     """
    #     import math
    #     model_unwrapped = unwrap_model(self.model)
    #     criterion = getattr(model_unwrapped, 'criterion', None)

    #     # 檢查是否為 RTMO 系列模型（判斷 Loss 函數是否有 stage2_epoch 屬性）
    #     if criterion is not None and hasattr(criterion, 'stage2_epoch'):
    #         stage2 = criterion.stage2_epoch
    #         total_epochs = self.epochs
    #         lrf = self.args.lrf  # 最終衰減係數 (例如 0.05，即 0.004 * 0.05 = 0.0002)
    #         # 跳轉倍率：從 0.0002 跳到 0.0005，倍率為 2.5
    #         lr_kick = getattr(self.args, 'rtmo_lr_kick', 2.5) 

    #         def rtmo_lr_lambda(epoch):
    #             if epoch < stage2:
    #                 # --- 第一階段：標準 Cosine 衰減 (1.0 -> lrf) ---
    #                 fraction = epoch / stage2
    #                 cos_out = 0.5 * (1.0 + math.cos(math.pi * fraction))
    #                 return cos_out * (1.0 - lrf) + lrf
    #             else:
    #                 # --- 第二階段：跳轉後再次 Cosine 衰減 (lrf * kick -> lrf) ---
    #                 # 計算在剩餘 Epoch 中的進度
    #                 fraction = (epoch - stage2) / max(1, (total_epochs - stage2))
    #                 cos_out = 0.5 * (1.0 + math.cos(math.pi * fraction))
                    
    #                 start_factor = lrf * lr_kick  # 跳轉後的起始點 (0.125)
    #                 return cos_out * (start_factor - lrf) + lrf

    #         self.lf = rtmo_lr_lambda
    #         LOGGER.info(f"{colorstr('RTMO Scheduler:')} Stage 2 Cosine starts at epoch {stage2}. Boosting LR by {lr_kick}x")
        
    #     else:
    #         # --- 原生 YOLO 邏輯 (保持不變，確保不影響其他模型) ---
    #         if self.args.cos_lr:
    #             self.lf = one_cycle(1, self.args.lrf, self.epochs)
    #         else:
    #             self.lf = lambda x: max(1 - x / self.epochs, 0) * (1.0 - self.args.lrf) + self.args.lrf
        
    #     # 套用自定義的衰減函數
    #     self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=self.lf)

    def _setup_scheduler(self):
        """Initialize training learning rate scheduler."""
        if self.args.cos_lr:
            self.lf = one_cycle(1, self.args.lrf, self.epochs)  # cosine 1->hyp['lrf']
        else:
            self.lf = lambda x: max(1 - x / self.epochs, 0) * (1.0 - self.args.lrf) + self.args.lrf  # linear
        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=self.lf)

    def _setup_ddp(self):
        """Initialize and set the DistributedDataParallel parameters for training."""
        torch.cuda.set_device(RANK)
        self.device = torch.device("cuda", RANK)
        os.environ["TORCH_NCCL_BLOCKING_WAIT"] = "1"  # set to enforce timeout
        dist.init_process_group(
            backend="nccl" if dist.is_nccl_available() else "gloo",
            timeout=timedelta(seconds=10800),  # 3 hours
            rank=RANK,
            world_size=self.world_size,
        )

    def _setup_train(self):
        """Build dataloaders and optimizer on correct rank process."""
        ckpt = self.setup_model()
        self.model = self.model.to(self.device)
        self.set_model_attributes()

        # Check imgsz
        gs = max(int(self.model.stride.max() if hasattr(self.model, "stride") else 32), 32)  # grid size (max stride)
        self.args.imgsz = check_imgsz(self.args.imgsz, stride=gs, floor=gs, max_dim=1)
        self.stride = gs  # for multiscale training

        # Batch size
        if self.batch_size < 1 and RANK == -1:  # single-GPU only, estimate best batch size
            self.args.batch = self.batch_size = self.auto_batch()

        # Dataloaders
        batch_size = self.batch_size // max(self.world_size, 1)
        self.train_loader = self.get_dataloader(
            self.data["train"], batch_size=batch_size, rank=LOCAL_RANK, mode="train"
        )
        # Note: When training DOTA dataset, double batch size could get OOM on images with >2000 objects.
        self.test_loader = self.get_dataloader(
            self.data.get("val") or self.data.get("test"),
            batch_size=batch_size if self.args.task == "obb" else batch_size * 2,
            rank=LOCAL_RANK,
            mode="val",
        )
        
        # === 新增：建立用來評估 AFSS 的「無擴增」訓練集 DataLoader ===
        # 僅在 AFSS 開啟時才建立，節省系統資源
        if getattr(self.args, 'afss', False):
            self.train_eval_loader = self.get_dataloader(
                self.data["train"], 
                batch_size=batch_size * 2, 
                rank=LOCAL_RANK, 
                mode="val"   # 關鍵：使用 val 模式關閉 Mosaic 並輸出 ratio_pad
            )
        # ==============================================================

        self.validator = self.get_validator()
        self.ema = ModelEMA(self.model)
        if RANK in {-1, 0}:
            metric_keys = self.validator.metrics.keys + self.label_loss_items(prefix="val")
            self.metrics = dict(zip(metric_keys, [0] * len(metric_keys)))
            if self.args.plots:
                self.plot_training_labels()

        # Optimizer
        self.accumulate = max(round(self.args.nbs / self.batch_size), 1)  # accumulate loss before optimizing
        weight_decay = self.args.weight_decay * self.batch_size * self.accumulate / self.args.nbs  # scale weight_decay
        iterations = math.ceil(len(self.train_loader.dataset) / max(self.batch_size, self.args.nbs)) * self.epochs
        
        resolved_lora_total_step = resolve_adalora_total_step(
            getattr(self.args, "lora_type", "lora"),
            getattr(self.args, "lora_total_step", None),
            iterations,
        )
        if str(getattr(self.args, "lora_type", "lora")).lower() == "adalora":
            self.args.lora_total_step = resolved_lora_total_step
            LOGGER.info(f"[LoRA] AdaLoRA total_step resolved to {resolved_lora_total_step}.")
            if RANK in {-1, 0}:
                save_trainer_args_yaml(self.save_dir, self.args)
        
        self.model = apply_lora(self.model, self.args)
        update_args_with_lora_runtime_metadata(self.args, self.model)
        if RANK in {-1, 0}:
            save_trainer_args_yaml(self.save_dir, self.args)
        self.set_model_attributes()
        
        # MoE Routing Collapse Detector (initialize if model has MoE layers)
        has_moe = any(hasattr(m, 'num_experts') for m in self.model.modules())
        # Persist the detection result so the train loop can gate MoE-only
        # logic (warmup schedule, gain schedule, collapse detector) and avoid
        # printing MoE messages on plain (non-MoE) models.
        self._has_moe = has_moe
        if has_moe:
            from ultralytics.nn.modules.moe.analysis import RoutingCollapseDetector
            collapse_thr = getattr(self.args, 'moe_collapse_threshold', 0.8)
            self._moe_collapse_detector = RoutingCollapseDetector(collapse_threshold=collapse_thr)
            LOGGER.info(f"[MoE] Routing collapse detector initialized (threshold={collapse_thr})")

            # Inject MoE hyperparameters from training config into model modules
            # This bridges the gap: YAML config (moe_balance_loss) → module defaults (balance_loss_coeff)
            balance_loss_coeff = getattr(self.args, 'moe_balance_loss', 0.1)
            router_z_loss_coeff = getattr(self.args, 'moe_router_z_loss', 0.01)
            noise_std = getattr(self.args, 'moe_noise_std', 0.5)
            temperature = getattr(self.args, 'moe_temperature', 1.0)
            weight_threshold = getattr(self.args, 'moe_weight_threshold', 0.01)

            injected = 0
            for m in self.model.modules():
                if hasattr(m, 'balance_loss_coeff'):
                    m.balance_loss_coeff = balance_loss_coeff
                    injected += 1
                if hasattr(m, 'router_z_loss_coeff'):
                    m.router_z_loss_coeff = router_z_loss_coeff
                if hasattr(m, 'routing') and hasattr(m.routing, 'noise_std'):
                    m.routing.noise_std = noise_std
                if hasattr(m, 'routing') and hasattr(m.routing, 'temperature'):
                    m.routing.temperature = temperature
                if hasattr(m, 'weight_threshold'):
                    m.weight_threshold = weight_threshold
                # Propagate to internal MoELoss
                if hasattr(m, 'moe_loss_fn'):
                    m.moe_loss_fn.balance_loss_coeff = balance_loss_coeff
                    m.moe_loss_fn.z_loss_coeff = router_z_loss_coeff
            LOGGER.info(
                f"[MoE] Config injected into {injected} MoE modules: "
                f"balance_loss={balance_loss_coeff}, z_loss={router_z_loss_coeff}, "
                f"noise_std={noise_std}, temperature={temperature}"
            )

        # Few-shot mode: load teacher model for knowledge distillation
        if getattr(self.args, 'lora_few_shot_mode', False):
            teacher_path = getattr(self.args, 'lora_few_shot_teacher', None)
            if teacher_path:
                try:
                    from ultralytics import YOLO
                    self.teacher_model = YOLO(teacher_path).model.to(self.device)
                    self.teacher_model.eval()
                    for p in self.teacher_model.parameters():
                        p.requires_grad = False
                    LOGGER.info(f"[LoRA] 🎓 Teacher model loaded from {teacher_path}")
                except Exception as e:
                    LOGGER.warning(f"[LoRA] Failed to load teacher model: {e}")
                    self.teacher_model = None
            else:
                LOGGER.info("[LoRA] Few-shot mode without teacher — using DropConnect + adaptive rank only")

            # v3: Initialize EMA teacher for progressive self-distillation
            if getattr(self.args, 'lora_few_shot_use_ema_teacher', False):
                try:
                    from copy import deepcopy
                    self.teacher_ema = deepcopy(self.model if self.teacher_model is None else self.teacher_model)
                    self.teacher_ema.eval()
                    for p in self.teacher_ema.parameters():
                        p.requires_grad = False
                    self.teacher_ema_decay = getattr(self.args, 'lora_few_shot_ema_decay', 0.999)
                    LOGGER.info(f"[LoRA] 📊 EMA teacher initialized (decay={self.teacher_ema_decay})")
                except Exception as e:
                    LOGGER.warning(f"[LoRA] Failed to initialize EMA teacher: {e}")
                    self.teacher_ema = None
            else:
                self.teacher_ema = None

            # v3: Initialize hierarchical distillation hook cache
            if getattr(self.args, 'lora_few_shot_hierarchical_distill', False):
                self._init_hierarchical_distill_cache()
        
        # Compile model
        self.model = attempt_compile(self.model, device=self.device, mode=self.args.compile)
        lora_model = unwrap_model(self.model)
        
        # Freeze layers
        freeze_list = (
            self.args.freeze
            if isinstance(self.args.freeze, list)
            else range(self.args.freeze)
            if isinstance(self.args.freeze, int)
            else []
        )
        # Do not freeze .dfl in adapter mode (random init when class mismatch)
        is_lora = getattr(lora_model, "lora_enabled", False)
        always_freeze_names = [] if is_lora else [".dfl"]
        freeze_layer_names = [f"model.{x}." for x in freeze_list] + always_freeze_names
        self.freeze_layer_names = freeze_layer_names
        for k, v in self.model.named_parameters():
            if any(x in k for x in freeze_layer_names):
                LOGGER.info(f"Freezing layer '{k}'")
                v.requires_grad = False
            elif not v.requires_grad and v.dtype.is_floating_point and not is_lora:
                LOGGER.warning(
                    f"setting 'requires_grad=True' for frozen layer '{k}'. "
                    "See ultralytics.engine.trainer for customization of frozen layers."
                )
                v.requires_grad = True

        # Unfreeze detection head in adapter mode (PEFT freezes all by default)
        if is_lora:
            _unfreeze_detection_head(self.model)
            
        # Check AMP
        self.amp = torch.tensor(self.args.amp).to(self.device)  # True or False
        if self.amp and RANK in {-1, 0}:  # Single-GPU and DDP
            callbacks_backup = callbacks.default_callbacks.copy()  # backup callbacks as check_amp() resets them
            self.amp = torch.tensor(check_amp(self.model), device=self.device)
            callbacks.default_callbacks = callbacks_backup  # restore callbacks
        if RANK > -1 and self.world_size > 1:  # DDP
            dist.broadcast(self.amp.int(), src=0)  # broadcast from rank 0 to all other ranks; gloo errors with boolean
        self.amp = bool(self.amp)  # as boolean
        self.scaler = (
            torch.amp.GradScaler("cuda", enabled=self.amp) if TORCH_2_4 else torch.cuda.amp.GradScaler(enabled=self.amp)
        )
        if self.world_size > 1:
            self.model = nn.parallel.DistributedDataParallel(self.model, device_ids=[RANK], find_unused_parameters=True)
            
        self.ema = ModelEMA(self.model)
        self.optimizer = self.build_optimizer(
            model=self.model,
            name=self.args.optimizer,
            lr=self.args.lr0,
            momentum=self.args.momentum,
            decay=weight_decay,
            iterations=iterations,
        )
        
        # print(f"DEBUG: self.model type before build_optimizer: {type(self.model)}")
        self.optimizer = self.build_optimizer(
            model=self.model,
            name=self.args.optimizer,
            lr=self.args.lr0,
            momentum=self.args.momentum,
            decay=weight_decay,
            iterations=iterations,
        )
        # Scheduler
        self._setup_scheduler()
        self.stopper, self.stop = EarlyStopping(patience=self.args.patience), False
        # ── LoRA Training Strategy Engine ──
        self.lora_strategy = None
        if is_lora:
            has_lora = getattr(unwrap_model(self.model), 'lora_enabled', False)
            if has_lora:
                self.lora_strategy = LoraTrainingStrategy(
                    model=self.model,
                    config=getattr(self.model, 'lora_config', None),
                    epochs=self.epochs,
                )
                # Strategy 1: Layer-wise LR decay (apply to optimizer)
                lora_layer_decay = getattr(self.args, 'lora_layer_decay', 0.0)
                if lora_layer_decay > 0:
                    n_before = len(self.optimizer.param_groups)
                    self.lora_strategy.apply_layer_decay_to_optimizer(
                        self.optimizer, decay_rate=lora_layer_decay
                    )
                    n_after = len(self.optimizer.param_groups)
                    # If apply_layer_decay_to_optimizer added new param groups (LoRA
                    # params split by depth), we must rebuild the LR scheduler so that
                    # its internal lr_lambdas list matches the new group count.
                    # Otherwise `scheduler.step()` will raise ValueError in zip(strict=True).
                    if n_after != n_before:
                        self._setup_scheduler()

                # Strategy 2: Alpha warmup preparation
                lora_alpha_warmup = getattr(self.args, 'lora_alpha_warmup', 0)
                if lora_alpha_warmup > 0:
                    if any(hasattr(m, "lora_A") for m in self.model.modules()):
                        self.lora_strategy.prepare_alpha_warmup()
                    else:
                        LOGGER.info("[LoRA] Alpha warmup skipped: active adapter type has no LoRA alpha layers.")
                        lora_alpha_warmup = 0
                        self.args.lora_alpha_warmup = 0

                # Strategy 4: Dynamic dropout scheduling params
                self.lora_dropout_end = getattr(self.args, 'lora_dropout_end', 0.15)
                self.lora_dropout_start_ratio = getattr(self.args, 'lora_dropout_start_ratio', 0.3)

                # Strategy 3: Orthogonal regularization weight
                self.lora_ortho_weight = getattr(self.args, 'lora_ortho_weight', 0.0)
                self.lora_ortho_frequency = getattr(self.args, 'lora_ortho_frequency', 10)
                self.lora_ortho_batch_counter = 0  # Batch counter for orthogonal loss computation

                LOGGER.info(
                    f"[LoRA] 🎯 Training Strategy Engine initialized | "
                    f"layer_decay={lora_layer_decay}, "
                    f"alpha_warmup={lora_alpha_warmup}ep, "
                    f"ortho_weight={self.lora_ortho_weight}, "
                    f"ortho_freq={self.lora_ortho_frequency}, "
                    f"dropout_schedule=[0→{self.lora_dropout_end}]"
                )

        self.resume_training(ckpt)
        self.scheduler.last_epoch = self.start_epoch - 1  # do not move
        self.run_callbacks("on_pretrain_routine_end")

        # === 新增：初始化 AFSS Manager ===
        if getattr(self.args, 'afss', False):
            self.afss_manager = AFSSManager(dataset_size=len(self.train_loader.dataset), args=self.args)
        # =================================


        self.run_callbacks("on_pretrain_routine_end")

    def _do_train(self):
        """Train the model with the specified world size."""
        if self.world_size > 1:
            self._setup_ddp()
        self._setup_train()

        nb = len(self.train_loader)  # number of batches
        nw = max(round(self.args.warmup_epochs * nb), 100) if self.args.warmup_epochs > 0 else -1  # warmup iterations
        last_opt_step = -1
        self.epoch_time = None
        self.epoch_time_start = time.time()
        self.train_time_start = time.time()
        self.run_callbacks("on_train_start")
        LOGGER.info(
            f"Image sizes {self.args.imgsz} train, {self.args.imgsz} val\n"
            f"Using {self.train_loader.num_workers * (self.world_size or 1)} dataloader workers\n"
            f"Logging results to {colorstr('bold', self.save_dir)}\n"
            f"Starting training for " + (f"{self.args.time} hours..." if self.args.time else f"{self.epochs} epochs...")
        )
        if self.args.close_mosaic:
            base_idx = (self.epochs - self.args.close_mosaic) * nb
            self.plot_idx.extend([base_idx, base_idx + 1, base_idx + 2])
        epoch = self.start_epoch
        self.optimizer.zero_grad()  # zero any resumed gradients to ensure stability on train start
        while True:
            self.epoch = epoch

            # === 新增：AFSS 動態排程與更新 Sampler ===
            if hasattr(self, 'afss_manager'):
                # 取得當前 epoch 該訓練的圖片 indices
                selected_indices = self.afss_manager.schedule(epoch)
                current_len = len(selected_indices)
                
                # 初始化長度紀錄 (預設為原始資料集大小)
                if not hasattr(self, 'last_afss_len'):
                    self.last_afss_len = len(self.train_loader.dataset)

                # 更新 Sampler 內部的索引與 Epoch (打亂順序用)
                self.train_loader.sampler.set_indices(selected_indices)
                self.train_loader.sampler.set_epoch(epoch)
                
                # 💡 效能關鍵：只在「抽樣總數」發生變化時，才 Reset DataLoader
                # 這樣修改後，Warm up 期間完全不會重啟 Worker，立刻恢復原本的最快速度！
                if current_len != self.last_afss_len:
                    self.train_loader.reset()
                    self.last_afss_len = current_len
                
                # 更新迴圈的 batch 總數
                nb = len(self.train_loader)
                nw = max(round(self.args.warmup_epochs * nb), 100) if self.args.warmup_epochs > 0 else -1
            # =========================================

            # # === 新增：AFSS 動態排程與更新 Sampler ===
            # if hasattr(self, 'afss_manager'):
            #     # 取得當前 epoch 該訓練的圖片 indices
            #     selected_indices = self.afss_manager.schedule(epoch)
            #     # 更新 Sampler
            #     self.train_loader.sampler.set_indices(selected_indices)
            #     self.train_loader.sampler.set_epoch(epoch)
            #     # 必須 Reset DataLoader 讓它的 Iterator 吃新的長度
            #     self.train_loader.reset()
                
            #     # 更新迴圈的 batch 總數 (nb 變小了！)
            #     nb = len(self.train_loader)
            #     # 修正預熱策略的迭代邊界
            #     nw = max(round(self.args.warmup_epochs * nb), 100) if self.args.warmup_epochs > 0 else -1
            # # =========================================

            # # --- 新增以下這段：將當前的 epoch 推送給 Loss 函數 ---
            # # 1. 獲取原始模型與 Loss 函數
            # model_unwrapped = unwrap_model(self.model)
            # criterion = getattr(model_unwrapped, 'criterion', None)
            # # if hasattr(model_unwrapped, "criterion"):
            # #     # YOLO 把 Loss 放在 model.criterion 裡面
            # #     model_unwrapped.criterion.current_epoch = epoch
            # # ----------------------------------------------------

            # # 2. 自動偵測：只有當 Loss 函數裡有 'stage2_epoch' 屬性時，才執行 RTMO 特有邏輯
            # if criterion is not None and hasattr(criterion, 'stage2_epoch'):
            #     # 同步當前 epoch 給 Loss (這段你本來就有寫)
            #     criterion.current_epoch = epoch
                
            #     # --- RTMO 特有的學習率跳轉 (LR Kick) ---
            #     # 只有在進入第二階段的那一個 Epoch 執行一次
            #     if epoch == criterion.stage2_epoch:
            #         # 獲取跳轉倍率，優先看 args 是否有設定，否則用預設 2.5
            #         lr_kick = getattr(self.args, 'rtmo_lr_kick', 2.5) 
                    
            #         LOGGER.info(f"RTMO Specific Logic: Stage 2 starts at epoch {epoch}. Boosting LR by {lr_kick}x")
                    
            #         for i, param_group in enumerate(self.optimizer.param_groups):
            #             old_lr = param_group['lr']
            #             param_group['lr'] *= lr_kick
            #             # 更新 initial_lr 是關鍵，確保 scheduler.step() 不會把 LR 抓回去
            #             if 'initial_lr' in param_group:
            #                 param_group['initial_lr'] *= lr_kick
                        
            #             # 選項：對不同參數組可以有不同處理 (例如 bias 不跳轉)
            #             # if param_group.get('param_group') == 'bias': continue

            # --- 原有的 Scheduler Step ---
            self.run_callbacks("on_train_epoch_start")

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.scheduler.step() # 先讓原本的 scheduler 跑完

            # =================== RTMO 特有學習率強制改寫 (START) ===================
            model_unwrapped = unwrap_model(self.model)
            criterion = getattr(model_unwrapped, 'criterion', None)
            
            # 偵測是否為 RTMO 兩階段任務
            if criterion is not None and hasattr(criterion, 'stage2_epoch'):
                criterion.current_epoch = epoch # 同步 Epoch 到 Loss
                
                stage2 = criterion.stage2_epoch
                total_epochs = self.epochs
                lr0 = self.args.lr0  # 初始學習率 (例如 0.004)
                lrf = self.args.lrf  # 最終衰減係數 (例如 0.05)
                
                if epoch < stage2:
                    # 第一階段：保持原本 scheduler 的 Cosine 衰減
                    # (這裡不需要動，因為 scheduler.step() 已經算好了)
                    pass
                else:
                    # 第二階段：強制覆蓋 scheduler 的結果
                    # 計算第二階段進度 (0.0 ~ 1.0)
                    progress = (epoch - stage2) / max(1, (total_epochs - stage2))
                    cos_val = 0.5 * (1 + math.cos(math.pi * progress))
                    
                    # 論文邏輯：從 0.0005 (lrf * 2.5) 衰減到 0.0002 (lrf)
                    # 這裡的 factor 是相對於 lr0 的比例
                    kick_start_factor = lrf * 2.5 # 0.05 * 2.5 = 0.125
                    current_factor = cos_val * (kick_start_factor - lrf) + lrf
                    
                    new_lr = lr0 * current_factor
                    
                    # 強制寫入 optimizer
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = new_lr

            # with warnings.catch_warnings():
            #     warnings.simplefilter("ignore")  # suppress 'Detected lr_scheduler.step() before optimizer.step()'
            #     self.scheduler.step()

            # MoE Strategy: Freeze experts for initial epochs while router learns balanced routing
            # Key fix: only freeze expert WEIGHTS, not shared_expert/routing — and use shorter warmup
            # P0 FIX: gate the entire MoE warmup / gain-schedule / collapse-detection
            # block behind `self._has_moe` (set during _setup_train). Previously this
            # block ran unconditionally — it would still scan named_parameters() for
            # 'experts' on every iteration on a plain (non-MoE) YOLO model and
            # unconditionally print '[MoE] Unfreezing expert weights ...' at
            # epoch == warmup, even though there were no expert weights at all.
            if getattr(self, "_has_moe", False):
                moe_warmup_epochs = getattr(self.args, 'moe_expert_warmup_epochs', 3)
                expert_params = [p for n, p in self.model.named_parameters()
                                 if "experts" in n and "routing" not in n and "router" not in n and "shared" not in n]
                if expert_params:
                    if epoch < moe_warmup_epochs:
                        for p in expert_params:
                            p.requires_grad = False
                    elif epoch == moe_warmup_epochs:
                        LOGGER.info(f"[MoE] Unfreezing expert weights after {moe_warmup_epochs}-epoch router warmup...")
                        for p in expert_params:
                            p.requires_grad = True

                # MoE Gain Schedule (fixed: was cosine 0.3→0.05, too aggressive decay)
                # New: warmup → plateau → gentle decay. Prevents late-training routing collapse.
                if hasattr(self.args, 'moe'):
                    initial_moe = getattr(self.args, 'moe', 0.3)
                    progress = epoch / self.epochs
                    if progress < 0.1:
                        # Phase 1: warmup — ramp from 0.5x to 1x initial gain
                        moe_gain = initial_moe * (0.5 + 0.5 * (progress / 0.1))
                    elif progress < 0.7:
                        # Phase 2: plateau — full gain to maintain balanced routing
                        moe_gain = initial_moe
                    else:
                        # Phase 3: gentle linear decay to 0.3x — avoid sudden collapse
                        decay_progress = (progress - 0.7) / 0.3
                        moe_gain = initial_moe * (1.0 - 0.7 * decay_progress)
                    self.args.moe = moe_gain
                    # Also update model args/hyp if needed (propagate to loss)
                    if hasattr(self.model, 'args') and isinstance(self.model.args, dict):
                        self.model.args['moe'] = moe_gain
                    elif hasattr(self.model, 'args') and hasattr(self.model.args, 'moe'):
                        self.model.args.moe = moe_gain

                # MoE Routing Collapse Detection (every 5 epochs after warmup)
                if epoch > 0 and epoch % 5 == 0 and hasattr(self, '_moe_collapse_detector'):
                    diag = self._moe_collapse_detector.diagnose(self.model)
                    collapsed_layers = [n for n, d in diag.items() if d['collapsed']]
                    if collapsed_layers:
                        collapse_thr = getattr(self.args, 'moe_collapse_threshold', 0.8)
                        LOGGER.warning(
                            f"[MoE] ⚠️ Routing collapse detected at epoch {epoch}: "
                            f"layers {collapsed_layers} have max_usage > {collapse_thr}. "
                            f"Auto-increasing noise_std for recovery..."
                        )
                        applied = self._moe_collapse_detector.apply_recovery(self.model, diag)
                        if applied > 0:
                            LOGGER.info(f"[MoE] Applied {applied} recovery actions.")
                        # Also boost balance_loss if not already high
                        if hasattr(self.args, 'moe_balance_loss'):
                            old_bl = self.args.moe_balance_loss
                            self.args.moe_balance_loss = min(old_bl * 2.0, 0.5)
                        LOGGER.info(f"[MoE] balance_loss boosted: {old_bl:.4f} → {self.args.moe_balance_loss:.4f}")

            # ── LoRA Training Strategies (per-epoch) ──
            if self.lora_strategy is not None:
                # Strategy 2: Alpha warmup (gradually ramp up scaling)
                alpha_warmup_ep = getattr(self.args, 'lora_alpha_warmup', 0)
                if alpha_warmup_ep > 0 and epoch < alpha_warmup_ep:
                    scale = self.lora_strategy.step_alpha_warmup(epoch, warmup_epochs=alpha_warmup_ep)
                    LOGGER.debug(f"[LoRA] Alpha warmup: epoch={epoch}, scale={scale:.4f}")
                elif alpha_warmup_ep > 0 and epoch == alpha_warmup_ep:
                    self.lora_strategy.finalize_alpha_warmup()

                # Strategy 4: Dynamic dropout schedule
                self.lora_strategy.update_dropout_schedule(
                    self.model, epoch=epoch, epochs_total=self.epochs,
                    end_dropout=self.lora_dropout_end,
                    schedule_start_ratio=self.lora_dropout_start_ratio,
                )               

            self._model_train()
            if RANK != -1:
                self.train_loader.sampler.set_epoch(epoch)
            pbar = enumerate(self.train_loader)
            # Update dataloader attributes (optional)
            if epoch == (self.epochs - self.args.close_mosaic):
                self._close_dataloader_mosaic()
                self.train_loader.reset()
                
            # ── Few-Shot LoRA: Update scheduled DropConnect progress ──
            if getattr(self.args, 'lora_few_shot_mode', False):
                progress = epoch / max(self.epochs - 1, 1)
                from ultralytics.utils.lora import FewShotLoRAConv
                updated = 0
                for module in self.model.modules():
                    if isinstance(module, FewShotLoRAConv):
                        # No direct attribute setting needed; progress is computed on-the-fly in forward
                        # But we can log the scheduled rate for monitoring
                        updated += 1
                if updated > 0 and epoch % 5 == 0 and RANK in {-1, 0}:
                    sample = next(m for m in self.model.modules() if isinstance(m, FewShotLoRAConv))
                    scheduled_rate = sample.get_scheduled_dropconnect_rate(progress)
                    LOGGER.info(f"[LoRA] 📉 Scheduled DropConnect rate: {scheduled_rate:.3f} (progress={progress:.2f})")

            if RANK in {-1, 0}:
                LOGGER.info(self.progress_string())
                pbar = TQDM(enumerate(self.train_loader), total=nb)
            self.tloss = None
            for i, batch in pbar:
                self.run_callbacks("on_train_batch_start")
                # Warmup
                ni = i + nb * epoch
                if ni <= nw:
                    xi = [0, nw]  # x interp
                    self.accumulate = max(1, int(np.interp(ni, xi,[1, self.args.nbs / self.batch_size]).round()))
                    for x in self.optimizer.param_groups:
                        # Bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                        x["lr"] = np.interp(
                            ni,
                            xi,[
                                self.args.warmup_bias_lr if x.get("param_group") == "bias" else 0.0,
                                x["initial_lr"] * self.lf(epoch),
                            ],
                        )
                        if "momentum" in x:
                            x["momentum"] = np.interp(ni, xi,[self.args.warmup_momentum, self.args.momentum])

                # Forward
                with autocast(self.amp):
                    batch = self.preprocess_batch(batch)
                    if self.args.compile:
                        # Decouple inference and loss calculations for improved compile performance
                        preds = self.model(batch["img"])
                        loss, self.loss_items = unwrap_model(self.model).loss(batch, preds)
                    else:
                        loss, self.loss_items = self.model(batch)
                        
                    # ── LoRA Orthogonal Regularization (Strategy 3) ──
                    # Optimized: compute every N batches instead of every batch.
                    # P1 FIX: cast ortho_loss to the main `loss` dtype before
                    # adding so AMP runs with bf16/fp16 do not crash on the
                    # `+` between fp32 ortho and the lower-precision detection
                    # loss tensor.
                    if self.lora_strategy is not None and self.lora_ortho_weight > 0:
                        self.lora_ortho_batch_counter += 1
                        if self.lora_ortho_batch_counter % self.lora_ortho_frequency == 0:
                            ortho_loss = LoraTrainingStrategy.compute_orthogonal_loss(
                                self.model, weight=self.lora_ortho_weight
                            )
                            if ortho_loss.dtype != loss.dtype:
                                ortho_loss = ortho_loss.to(loss.dtype)
                            loss = loss + ortho_loss

                    # ── Few-Shot LoRA: Knowledge Distillation Loss ──
                    if getattr(self.args, 'lora_few_shot_mode', False) and hasattr(self, 'teacher_model') and self.teacher_model is not None:
                        # v3: Dynamic distillation weight scheduling
                        progress = epoch / max(self.epochs - 1, 1)
                        distill_schedule = getattr(self.args, 'lora_few_shot_distill_schedule', 'constant')
                        distill_max = getattr(self.args, 'lora_few_shot_distill_weight_max', 1.0)
                        distill_min = getattr(self.args, 'lora_few_shot_distill_weight_min', 0.1)

                        if distill_schedule == 'constant':
                            distill_weight = distill_max
                        elif distill_schedule == 'linear':
                            distill_weight = distill_max - (distill_max - distill_min) * progress
                        elif distill_schedule == 'cosine':
                            distill_weight = distill_min + (distill_max - distill_min) * 0.5 * (1 + math.cos(math.pi * progress))
                        elif distill_schedule == 'exponential':
                            distill_weight = distill_min + (distill_max - distill_min) * math.exp(-5 * progress)
                        else:
                            distill_weight = distill_max
                        distill_weight = max(distill_min, min(distill_max, distill_weight))

                        hierarchical_distill = getattr(self.args, 'lora_few_shot_hierarchical_distill', False)
                        distill_layers = getattr(self.args, 'lora_few_shot_distill_layers', None)
                        adaptive_temp = getattr(self.args, 'lora_few_shot_adaptive_temperature', False)
                        use_ema = getattr(self.args, 'lora_few_shot_use_ema_teacher', False)
                        response_distill = getattr(self.args, 'lora_few_shot_response_distill', False)
                        response_weight = getattr(self.args, 'lora_few_shot_response_distill_weight', 0.3)

                        # v3: Select teacher source (static or EMA)
                        teacher_source = self.teacher_ema if (use_ema and hasattr(self, 'teacher_ema') and self.teacher_ema is not None) else self.teacher_model

                        with torch.no_grad():
                            student_preds = self.model(batch["img"])
                            teacher_preds = teacher_source(batch["img"])

                        # Compute distillation loss
                        distill_loss = self._compute_distillation_loss(student_preds, teacher_preds, adaptive_temp=adaptive_temp)

                        # v3: Response distillation (detection head alignment)
                        if response_distill:
                            resp_loss = self._compute_response_distillation_loss(student_preds, teacher_preds)
                            distill_loss = distill_loss + response_weight * resp_loss

                        # Hierarchical distillation: intermediate layer alignment
                        if hierarchical_distill and distill_layers:
                            layer_loss = self._compute_hierarchical_distillation_loss(
                                batch["img"], distill_layers
                            )
                            distill_loss = distill_loss + 0.3 * layer_loss

                        loss = loss + distill_weight * distill_loss

                    # ── Few-Shot LoRA: Variational Rank KL Regularization ──
                    if getattr(self.args, 'lora_few_shot_mode', False) and getattr(self.args, 'lora_few_shot_variational_rank', False):
                        from ultralytics.utils.lora import FewShotLoRAConv
                        kl_loss = 0.0
                        num_modules = 0
                        budget = getattr(self.args, 'lora_few_shot_rank_budget', 0.5)
                        for module in self.model.modules():
                            if isinstance(module, FewShotLoRAConv) and module.variational_rank:
                                # KL divergence between Gumbel-Softmax and target Bernoulli(budget)
                                # Encourage sparsity while maintaining budget
                                probs = torch.sigmoid(module.rank_logits)
                                # KL(q||p) where p = Bernoulli(budget), q = Bernoulli(probs)
                                p = budget
                                kl = probs * torch.log((probs + 1e-8) / (p + 1e-8)) + (1 - probs) * torch.log((1 - probs + 1e-8) / (1 - p + 1e-8))
                                kl_loss += kl.mean()
                                num_modules += 1
                        if num_modules > 0:
                            loss = loss + 0.01 * (kl_loss / num_modules)         
                        
                    self.loss = loss.sum()
                    if RANK != -1:
                        self.loss *= self.world_size
                    self.tloss = self.loss_items if self.tloss is None else (self.tloss * i + self.loss_items) / (i + 1)

                # Backward
                self.scaler.scale(self.loss).backward()
                
                # LoRA collapse early detection:
                # warn when loss stays zero/NaN for many consecutive iterations
                # (typical symptom when adapter injection destabilizes attention or deformable transformer paths).
                if self.lora_strategy is not None and RANK in {-1, 0}:
                    loss_val = float(self.loss.detach().item()) if self.loss is not None else 0.0
                    if not (loss_val == loss_val) or loss_val == 0.0:  # NaN or all-zero
                        self._lora_zero_loss_streak = getattr(self, "_lora_zero_loss_streak", 0) + 1
                        if self._lora_zero_loss_streak == 10:
                            LOGGER.warning(
                                f"[LoRA] Detected {self._lora_zero_loss_streak} consecutive "
                                f"zero/NaN losses — gradients may have collapsed. "
                                f"Suggestion: reduce lora_lr_mult, exclude attn.{{qkv,proj,pe}} "
                                f"from target_modules, enable lora_alpha_warmup >= 3, retry "
                                f"with lora_use_dora=False, or compare an amp=False debug run."
                            )
                    else:
                        self._lora_zero_loss_streak = 0

                # ── Few-Shot LoRA: Update gradient importance after backward ──
                if getattr(self.args, 'lora_few_shot_mode', False) and getattr(self.args, 'lora_few_shot_gradient_importance_weighted', False):
                    from ultralytics.utils.lora import FewShotLoRAConv
                    for module in self.model.modules():
                        if isinstance(module, FewShotLoRAConv) and module.gradient_importance_weighted:
                            module._update_importance()
                
                if ni - last_opt_step >= self.accumulate:
                    self.optimizer_step()
                    last_opt_step = ni

                    # Timed stopping
                    if self.args.time:
                        self.stop = (time.time() - self.train_time_start) > (self.args.time * 3600)
                        if RANK != -1:  # if DDP training
                            broadcast_list = [self.stop if RANK == 0 else None]
                            dist.broadcast_object_list(broadcast_list, 0)  # broadcast 'stop' to all ranks
                            self.stop = broadcast_list[0]
                        if self.stop:  # training time exceeded
                            break

                # Log
                if RANK in {-1, 0}:
                    loss_length = self.tloss.shape[0] if len(self.tloss.shape) else 1
                    pbar.set_description(
                        ("%11s" * 2 + "%11.4g" * (2 + loss_length))
                        % (
                            f"{epoch + 1}/{self.epochs}",
                            f"{self._get_memory():.3g}G",  # (GB) GPU memory util
                            *(self.tloss if loss_length > 1 else torch.unsqueeze(self.tloss, 0)),  # losses
                            batch["cls"].shape[0],  # batch size, i.e. 8
                            batch["img"].shape[-1],  # imgsz, i.e 640
                        )
                    )
                    self.run_callbacks("on_batch_end")
                    if self.args.plots and ni in self.plot_idx:
                        self.plot_training_samples(batch, ni)

                self.run_callbacks("on_train_batch_end")

            if hasattr(unwrap_model(self.model).criterion, "update"):
                unwrap_model(self.model).criterion.update()

            self.lr = {f"lr/pg{ir}": x["lr"] for ir, x in enumerate(self.optimizer.param_groups)}  # for loggers
            
            # ── LoRA Training Stats (per-epoch logging) ──
            if self.lora_strategy is not None and RANK in {-1, 0} and (epoch % 5 == 0 or epoch == self.epochs - 1):
                lora_stats = get_lora_training_stats(self.model)
                if lora_stats['lora_modules'] > 0:
                    LOGGER.info(
                        f"[LoRA] 📊 Epoch {epoch+1} stats: "
                        f"modules={lora_stats['lora_modules']}, "
                        f"eff_rank={lora_stats['effective_rank_avg']:.2%}, "
                        f"|A|_F={lora_stats['norm_A_frobenius']:.4f}, "
                        f"|B|_F={lora_stats['norm_B_frobenius']:.4f}"
                    )

            # ── Few-Shot LoRA Stats ──
            if getattr(self.args, 'lora_few_shot_mode', False) and RANK in {-1, 0} and (epoch % 5 == 0 or epoch == self.epochs - 1):
                from ultralytics.utils.lora import FewShotLoRAConv
                num_fewshot = 0
                total_active_rank = 0
                for module in self.model.modules():
                    if isinstance(module, FewShotLoRAConv):
                        num_fewshot += 1
                        if hasattr(module, 'rank_mask'):
                            active = (module.rank_mask > 0.1).float().mean().item()
                            total_active_rank += active
                        elif module.variational_rank and hasattr(module, 'rank_logits'):
                            active = (torch.sigmoid(module.rank_logits) > 0.5).float().mean().item()
                            total_active_rank += active
                if num_fewshot > 0:
                    avg_active = total_active_rank / num_fewshot
                    LOGGER.info(
                        f"[LoRA] 🎯 FewShot stats: modules={num_fewshot}, "
                        f"avg_active_rank={avg_active:.2%}"
                    )

            self.run_callbacks("on_train_epoch_end")
            if RANK in {-1, 0}:
                self.ema.update_attr(self.model, include=["yaml", "nc", "args", "names", "stride", "class_weights"])

            # Validation
            # === 新增：AFSS 狀態刷新機制 ===
            # 從 yaml 取得狀態刷新頻率 (預設為 5)
            refresh_rate = getattr(self.args, 'afss_refresh_rate', 5)
            
            tau = self.args.warmup_epochs
            # 確認 AFSS 已啟動，且大於等於預熱期，並符合自定義的 N Epoch 刷新頻率
            if getattr(self.args, 'afss', False) and hasattr(self, 'afss_manager') and epoch >= tau and (epoch - tau) % refresh_rate == 0:
                LOGGER.info(f"AFSS: Evaluating learning sufficiency on train set (Epoch {epoch}, Refresh Rate: {refresh_rate})...")
                train_val = self.get_validator()
                train_val.dataloader = self.train_eval_loader  # 修正：使用無擴增的 Loader
                train_val.args.plots = False              
                train_val.args.save_json = False
                train_val(trainer=self)                     
                
                # 提取結果回傳給 AFSSManager
                if hasattr(train_val, 'image_metrics'):
                    # 處理 DDP 多卡訓練：同步所有 GPU 的評估結果
                    from torch import distributed as dist
                    if self.world_size > 1:
                        gathered_metrics = [None] * self.world_size
                        dist.all_gather_object(gathered_metrics, train_val.image_metrics)
                        merged_metrics = {}
                        for m in gathered_metrics:
                            if m is not None:
                                merged_metrics.update(m)
                        train_val.image_metrics = merged_metrics
                        
                    # 建立 im_file 對應到 dataset index 的字典
                    from pathlib import Path
                    im_files = self.train_loader.dataset.im_files
                    path_to_idx = {str(Path(p).resolve()): i for i, p in enumerate(im_files)}
                    
                    indices_to_update = []
                    ps = []
                    rs =[]
                    for path_str, (img_p, img_r) in train_val.image_metrics.items():
                        resolved_path = str(Path(path_str).resolve())
                        if resolved_path in path_to_idx:
                            indices_to_update.append(path_to_idx[resolved_path])
                            ps.append(img_p)
                            rs.append(img_r)
                            
                    if indices_to_update:
                        self.afss_manager.update_states(indices_to_update, ps, rs)

            final_epoch = epoch + 1 >= self.epochs
            if self.args.val or final_epoch or self.stopper.possible_stop or self.stop:
                self._clear_memory(threshold=0.5)  # prevent VRAM spike
                self.metrics, self.fitness = self.validate()

            # NaN recovery
            if self._handle_nan_recovery(epoch):
                continue

            self.nan_recovery_attempts = 0
            if RANK in {-1, 0}:
                self.save_metrics(metrics={**self.label_loss_items(self.tloss), **self.metrics, **self.lr})
                self.stop |= self.stopper(epoch + 1, self.fitness) or final_epoch
                if self.args.time:
                    self.stop |= (time.time() - self.train_time_start) > (self.args.time * 3600)

                # Save model
                if self.args.save or final_epoch:
                    self.save_model()
                    self.run_callbacks("on_model_save")

            # Scheduler
            t = time.time()
            self.epoch_time = t - self.epoch_time_start
            self.epoch_time_start = t
            if self.args.time:
                mean_epoch_time = (t - self.train_time_start) / (epoch - self.start_epoch + 1)
                self.epochs = self.args.epochs = math.ceil(self.args.time * 3600 / mean_epoch_time)
                self._setup_scheduler()
                self.scheduler.last_epoch = self.epoch  # do not move
                self.stop |= epoch >= self.epochs  # stop if exceeded epochs
            self.run_callbacks("on_fit_epoch_end")
            self._clear_memory(0.5)  # clear if memory utilization > 50%

            # Early Stopping
            if RANK != -1:  # if DDP training
                broadcast_list =[self.stop if RANK == 0 else None]
                dist.broadcast_object_list(broadcast_list, 0)  # broadcast 'stop' to all ranks
                self.stop = broadcast_list[0]
            if self.stop:
                break  # must break all DDP ranks
            epoch += 1

        seconds = time.time() - self.train_time_start
        LOGGER.info(f"\n{epoch - self.start_epoch + 1} epochs completed in {seconds / 3600:.3f} hours.")
        # Do final val with best.pt
        self.final_eval()
        if RANK in {-1, 0}:
            if self.args.plots:
                self.plot_metrics()
            self.run_callbacks("on_train_end")
        self._clear_memory()
        unset_deterministic()
        self.run_callbacks("teardown")

    def auto_batch(self, max_num_obj=0):
        """Calculate optimal batch size based on model and device memory constraints."""
        return check_train_batch_size(
            model=self.model,
            imgsz=self.args.imgsz,
            amp=self.amp,
            batch=self.batch_size,
            max_num_obj=max_num_obj,
        )  # returns batch size

    def _get_memory(self, fraction=False):
        """Get accelerator memory utilization in GB or as a fraction of total memory."""
        memory, total = 0, 0
        if self.device.type == "mps":
            memory = torch.mps.driver_allocated_memory()
            if fraction:
                return __import__("psutil").virtual_memory().percent / 100
        elif self.device.type != "cpu":
            memory = torch.cuda.memory_reserved()
            if fraction:
                total = torch.cuda.get_device_properties(self.device).total_memory
        return ((memory / total) if total > 0 else 0) if fraction else (memory / 2**30)

    def _clear_memory(self, threshold: float | None = None):
        """Clear accelerator memory by calling garbage collector and emptying cache."""
        if threshold:
            assert 0 <= threshold <= 1, "Threshold must be between 0 and 1."
            if self._get_memory(fraction=True) <= threshold:
                return
        gc.collect()
        if self.device.type == "mps":
            torch.mps.empty_cache()
        elif self.device.type == "cpu":
            return
        else:
            torch.cuda.empty_cache()

    def read_results_csv(self):
        """Read results.csv into a dictionary using polars."""
        import polars as pl  # scope for faster 'import ultralytics'

        try:
            return pl.read_csv(self.csv, infer_schema_length=None).to_dict(as_series=False)
        except Exception:
            return {}

    def _model_train(self):
        """Set model in training mode."""
        self.model.train()
        # Freeze BN stat
        for n, m in self.model.named_modules():
            if any(filter(lambda f: f in n, self.freeze_layer_names)) and isinstance(m, nn.BatchNorm2d):
                m.eval()

    def save_model(self):
        """Save model training checkpoints with additional metadata."""
        import io

        # Serialize ckpt to a byte buffer once (faster than repeated torch.save() calls)
        buffer = io.BytesIO()
        torch.save(
            {
                "epoch": self.epoch,
                "best_fitness": self.best_fitness,
                "model": None,  # resume and final checkpoints derive from EMA
                "ema": deepcopy(unwrap_model(self.ema.ema)).half(),
                "updates": self.ema.updates,
                "optimizer": convert_optimizer_state_dict_to_fp16(deepcopy(self.optimizer.state_dict())),
                "scaler": self.scaler.state_dict(),
                "train_args": vars(self.args),  # save as dict
                "train_metrics": {**self.metrics, **{"fitness": self.fitness}},
                "train_results": self.read_results_csv(),
                "date": datetime.now().isoformat(),
                "version": __version__,
                "git": {
                    "root": str(GIT.root),
                    "branch": GIT.branch,
                    "commit": GIT.commit,
                    "origin": GIT.origin,
                },
                "license": "AGPL-3.0 (https://ultralytics.com/license)",
                "docs": "https://docs.ultralytics.com",
            },
            buffer,
        )
        serialized_ckpt = buffer.getvalue()  # get the serialized content to save

        # Save checkpoints
        self.wdir.mkdir(parents=True, exist_ok=True)  # ensure weights directory exists
        self.last.write_bytes(serialized_ckpt)  # save last.pt
        if self.best_fitness == self.fitness:
            self.best.write_bytes(serialized_ckpt)  # save best.pt
            
        # Save LoRA adapters if enabled
        lora_model = unwrap_model(self.model)
        if getattr(lora_model, "lora_enabled", False) and getattr(self.args, 'lora_save_adapters', True):
            adapter_dir = self.wdir / (getattr(self.args, 'lora_adapter_dir', 'lora_adapter') + f"_epoch{self.epoch}")
            if self.best_fitness == self.fitness:
                best_adapter_dir = self.wdir / (getattr(self.args, 'lora_adapter_dir', 'lora_adapter') + "_best")
                save_lora_adapters(lora_model, best_adapter_dir)

            if (self.save_period > 0) and (self.epoch % self.save_period == 0):
                save_lora_adapters(lora_model, adapter_dir)     
            
        if (self.save_period > 0) and (self.epoch % self.save_period == 0):
            (self.wdir / f"epoch{self.epoch}.pt").write_bytes(serialized_ckpt)  # save epoch, i.e. 'epoch3.pt'

    def get_dataset(self):
        """Get train and validation datasets from data dictionary.

        Returns:
            (dict): A dictionary containing the training/validation/test dataset and category names.
        """
        try:
            # Convert ul:// platform URIs and NDJSON files to local dataset format first
            data_str = str(self.args.data)
            if data_str.endswith(".ndjson") or (data_str.startswith("ul://") and "/datasets/" in data_str):
                import asyncio

                from ultralytics.data.converter import convert_ndjson_to_yolo
                from ultralytics.utils.checks import check_file

                self.args.data = str(asyncio.run(convert_ndjson_to_yolo(check_file(self.args.data))))

            # Task-specific dataset checking
            if self.args.task == "classify":
                data = check_cls_dataset(self.args.data)
            elif str(self.args.data).rsplit(".", 1)[-1] in {"yaml", "yml"} or self.args.task in {
                "detect",
                "segment",
                "pose",
                "obb",
            }:
                data = check_det_dataset(self.args.data)
                if "yaml_file" in data:
                    self.args.data = data["yaml_file"]  # for validating 'yolo train data=url.zip' usage
        except Exception as e:
            raise RuntimeError(emojis(f"Dataset '{clean_url(self.args.data)}' error ❌ {e}")) from e
        if self.args.single_cls:
            LOGGER.info("Overriding class names with single class.")
            data["names"] = {0: "item"}
            data["nc"] = 1
        return data

    def setup_model(self):
        """Load, create, or download model for any task.

        Returns:
            (dict): Optional checkpoint to resume training from.
        """
        if isinstance(self.model, torch.nn.Module):  # if model is loaded beforehand. No setup needed
            return

        cfg, weights = self.model, None
        ckpt = None
        if str(self.model).endswith(".pt"):
            weights, ckpt = load_checkpoint(self.model)
            cfg = weights.yaml
        elif isinstance(self.args.pretrained, (str, Path)):
            weights, _ = load_checkpoint(self.args.pretrained)
        self.model = self.get_model(cfg=cfg, weights=weights, verbose=RANK == -1)  # calls Model(cfg, weights)
        return ckpt

    def optimizer_step(self):
        """Perform a single step of the training optimizer with gradient clipping and EMA update."""
        self.scaler.unscale_(self.optimizer)  # unscale gradients
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()
        if self.ema:
            self.ema.update(self.model)

    def preprocess_batch(self, batch):
        """Allow custom preprocessing model inputs and ground truths depending on task type."""
        return batch
    
    def _compute_distillation_loss(self, student_preds, teacher_preds, adaptive_temp=False):
        """Compute knowledge distillation loss between student and teacher predictions.
        
        Args:
            student_preds: Student model predictions
            teacher_preds: Teacher model predictions
            adaptive_temp: Whether to use task-adaptive temperature
        """
        # Handle list/tuple formats (YOLO multi-scale predictions)
        if isinstance(student_preds, (list, tuple)):
            student_preds = student_preds[0] if len(student_preds) > 0 else student_preds
        if isinstance(teacher_preds, (list, tuple)):
            teacher_preds = teacher_preds[0] if len(teacher_preds) > 0 else teacher_preds

        # Ensure we have tensors
        if not isinstance(student_preds, torch.Tensor) or not isinstance(teacher_preds, torch.Tensor):
            return torch.tensor(0.0, device=next(self.model.parameters()).device)

        # Adaptive temperature: based on teacher prediction entropy
        if adaptive_temp:
            with torch.no_grad():
                teacher_entropy = self._compute_prediction_entropy(teacher_preds)
                # High entropy (uncertain) -> high temperature; Low entropy (certain) -> low temperature
                temperature = torch.clamp(2.0 + teacher_entropy * 4.0, 1.0, 8.0)
        else:
            temperature = 4.0

        # Handle different spatial dimensions / formats
        # YOLO predictions can be [B, C, H, W] or [B, N, C] (flattened)
        if student_preds.dim() == 3 and teacher_preds.dim() == 3:
            # Both are [B, N, C] format — match sequence length
            if student_preds.shape[1] != teacher_preds.shape[1]:
                min_len = min(student_preds.shape[1], teacher_preds.shape[1])
                student_preds = student_preds[:, :min_len, :]
                teacher_preds = teacher_preds[:, :min_len, :]
            # MSE on flattened predictions
            return torch.nn.functional.mse_loss(student_preds, teacher_preds)

        if student_preds.dim() == 4 and teacher_preds.dim() == 4:
            # Both are [B, C, H, W] format
            if student_preds.shape != teacher_preds.shape:
                teacher_preds = torch.nn.functional.interpolate(
                    teacher_preds, size=student_preds.shape[2:], mode='bilinear', align_corners=False
                )
            # Temperature-scaled KL divergence on spatial features
            if adaptive_temp and isinstance(temperature, torch.Tensor):
                # Use average temperature for batch
                t = temperature.mean().item()
            else:
                t = temperature
            student_soft = torch.nn.functional.log_softmax(student_preds / t, dim=1)
            teacher_soft = torch.nn.functional.softmax(teacher_preds / t, dim=1)
            return torch.nn.functional.kl_div(
                student_soft, teacher_soft, reduction='batchmean', log_target=False
            ) * (t ** 2)

        # Fallback: MSE on whatever we can match
        if student_preds.shape != teacher_preds.shape:
            min_len = min(student_preds.numel(), teacher_preds.numel())
            return torch.nn.functional.mse_loss(
                student_preds.flatten()[:min_len], teacher_preds.flatten()[:min_len]
            )
        return torch.nn.functional.mse_loss(student_preds, teacher_preds)

    def _compute_response_distillation_loss(self, student_preds, teacher_preds):
        """v3: Compute response distillation loss on detection head outputs.
        
        For YOLO detection models, predictions are typically tuples/lists containing
        cls_logits and bbox predictions. This loss aligns the final detection outputs
        between student and teacher, providing task-specific distillation.
        
        Args:
            student_preds: Student model predictions (can be list/tuple of tensors)
            teacher_preds: Teacher model predictions (can be list/tuple of tensors)
            
        Returns:
            torch.Tensor: Response distillation loss
        """
        device = next(self.model.parameters()).device

        # Handle various prediction formats
        if isinstance(student_preds, (list, tuple)):
            # YOLO typically returns [pred_train, pred_val] or list of scale outputs
            student_preds = [p for p in student_preds if isinstance(p, torch.Tensor)]
        else:
            student_preds = [student_preds] if isinstance(student_preds, torch.Tensor) else []

        if isinstance(teacher_preds, (list, tuple)):
            teacher_preds = [p for p in teacher_preds if isinstance(p, torch.Tensor)]
        else:
            teacher_preds = [teacher_preds] if isinstance(teacher_preds, torch.Tensor) else []

        if not student_preds or not teacher_preds:
            return torch.tensor(0.0, device=device)

        total_loss = 0.0
        valid_pairs = 0

        # Pair up predictions by matching shapes
        for s_pred in student_preds:
            for t_pred in teacher_preds:
                if s_pred.shape != t_pred.shape:
                    continue

                # Detect format: [B, N, C] flattened predictions vs [B, C, H, W] feature maps
                if s_pred.dim() == 3 and t_pred.dim() == 3:
                    # Flattened predictions: split into cls (last 80 dims) and bbox (first 4 dims)
                    # YOLO format: [x, y, w, h, obj, cls0, cls1, ...]
                    if s_pred.shape[-1] >= 84:  # 4 bbox + 1 obj + 80 classes (COCO)
                        # Bbox regression: first 4 channels
                        s_bbox = s_pred[..., :4]
                        t_bbox = t_pred[..., :4]
                        bbox_loss = torch.nn.functional.l1_loss(s_bbox, t_bbox)

                        # Classification: remaining channels (skip obj for simplicity)
                        s_cls = s_pred[..., 5:]
                        t_cls = t_pred[..., 5:]
                        if s_cls.numel() > 0 and t_cls.numel() > 0:
                            # Temperature-scaled KL for classification
                            T = 4.0
                            s_soft = torch.nn.functional.log_softmax(s_cls / T, dim=-1)
                            t_soft = torch.nn.functional.softmax(t_cls / T, dim=-1)
                            cls_loss = torch.nn.functional.kl_div(
                                s_soft, t_soft, reduction='batchmean', log_target=False
                            ) * (T ** 2)
                        else:
                            cls_loss = torch.tensor(0.0, device=device)

                        total_loss += (bbox_loss + cls_loss)
                        valid_pairs += 1
                    else:
                        # Generic 3D: MSE
                        total_loss += torch.nn.functional.mse_loss(s_pred, t_pred)
                        valid_pairs += 1

                elif s_pred.dim() == 4 and t_pred.dim() == 4:
                    # Feature map format: spatial distillation
                    total_loss += torch.nn.functional.mse_loss(s_pred, t_pred)
                    valid_pairs += 1

        if valid_pairs > 0:
            return total_loss / valid_pairs
        return torch.tensor(0.0, device=device)

    def _compute_prediction_entropy(self, preds):
        """Compute normalized prediction entropy for adaptive temperature."""
        if isinstance(preds, (list, tuple)):
            preds = preds[0] if len(preds) > 0 else preds
        if not isinstance(preds, torch.Tensor):
            return torch.tensor(1.0)
        if preds.dim() == 4:
            # [B, C, H, W] -> compute entropy over channels
            probs = torch.nn.functional.softmax(preds, dim=1)
            entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1).mean(dim=(1, 2))
            # Normalize to [0, 1]
            max_entropy = math.log(probs.shape[1])
            return (entropy / max_entropy).mean()
        return torch.tensor(1.0)

    def _init_hierarchical_distill_cache(self):
        """v3: Initialize persistent hook cache for hierarchical distillation.
        
        Registers forward hooks once at training start and reuses them across batches,
        eliminating per-batch hook registration/removal overhead.
        
        Uses a module-level hook function (not a nested closure) to allow pickling
        during model checkpoint saving.
        """
        from ultralytics.utils.torch_utils import unwrap_model
        distill_layers = getattr(self.args, 'lora_few_shot_distill_layers', None)
        if not distill_layers:
            self._hierarchical_cache = None
            return

        student_model = unwrap_model(self.model)
        teacher_model = getattr(self, 'teacher_model', None)

        # Use shared dicts attached to the trainer (not nested closures) to keep hooks picklable
        self._hierarchical_cache = {
            'student_features': {},
            'teacher_features': {},
            'student_hooks': [],
            'teacher_hooks': [],
            'layer_indices': list(distill_layers),
        }

        # Register hooks using module-level function + functools.partial (picklable)
        from functools import partial
        for idx in distill_layers:
            if idx < len(student_model.model):
                hook_fn = partial(_hierarchical_hook, self._hierarchical_cache['student_features'], idx)
                h = student_model.model[idx].register_forward_hook(hook_fn)
                self._hierarchical_cache['student_hooks'].append(h)
            if teacher_model is not None and idx < len(teacher_model.model):
                hook_fn = partial(_hierarchical_hook, self._hierarchical_cache['teacher_features'], idx)
                h = teacher_model.model[idx].register_forward_hook(hook_fn)
                self._hierarchical_cache['teacher_hooks'].append(h)

        LOGGER.info(f"[LoRA] 📌 Hierarchical distillation hook cache initialized "
                    f"({len(self._hierarchical_cache['student_hooks'])} student + "
                    f"{len(self._hierarchical_cache['teacher_hooks'])} teacher hooks)")

    def _compute_hierarchical_distillation_loss(self, images, layer_indices):
        """v3: Compute hierarchical distillation loss using cached hooks.
        
        Uses pre-registered hooks from _init_hierarchical_distill_cache,
        avoiding per-batch hook registration overhead.
        """
        if not layer_indices:
            return torch.tensor(0.0, device=images.device)

        # v3: Use cached hooks if available
        if getattr(self, '_hierarchical_cache', None) is not None:
            from ultralytics.utils.torch_utils import unwrap_model
            cache = self._hierarchical_cache
            # Clear previous features
            cache['student_features'].clear()
            cache['teacher_features'].clear()

            student_model = unwrap_model(self.model)
            teacher_model = self.teacher_model

            # Forward pass to populate cached features
            with torch.no_grad():
                _ = student_model(images)
                if teacher_model is not None:
                    _ = teacher_model(images)

            total_loss = 0.0
            valid_layers = 0
            for idx in layer_indices:
                s_feat = cache['student_features'].get(idx)
                t_feat = cache['teacher_features'].get(idx)
                if s_feat is None or t_feat is None:
                    continue

                if s_feat.shape != t_feat.shape:
                    if s_feat.dim() == 4 and t_feat.dim() == 4:
                        t_feat = torch.nn.functional.interpolate(
                            t_feat, size=s_feat.shape[2:], mode='bilinear', align_corners=False
                        )
                    else:
                        continue

                s_attention = torch.abs(s_feat).sum(dim=1, keepdim=True)
                t_attention = torch.abs(t_feat).sum(dim=1, keepdim=True)
                s_attention = s_attention / (s_attention.norm(2, dim=(2,3), keepdim=True) + 1e-8)
                t_attention = t_attention / (t_attention.norm(2, dim=(2,3), keepdim=True) + 1e-8)
                total_loss += torch.nn.functional.mse_loss(s_attention, t_attention)
                valid_layers += 1

            if valid_layers > 0:
                return total_loss / valid_layers
            return torch.tensor(0.0, device=images.device)

        # Fallback to original implementation if cache not available
        student_features = {}
        teacher_features = {}

        def make_hook(storage, key):
            def hook(module, input, output):
                storage[key] = output.detach() if not torch.is_grad_enabled() else output
            return hook

        student_model = unwrap_model(self.model)
        teacher_model = self.teacher_model

        hooks = []
        try:
            for idx in layer_indices:
                if idx < len(student_model.model):
                    hooks.append(student_model.model[idx].register_forward_hook(
                        make_hook(student_features, idx)
                    ))
                if teacher_model is not None and idx < len(teacher_model.model):
                    hooks.append(teacher_model.model[idx].register_forward_hook(
                        make_hook(teacher_features, idx)
                    ))

            with torch.no_grad():
                _ = student_model(images)
                if teacher_model is not None:
                    _ = teacher_model(images)

            total_loss = 0.0
            valid_layers = 0
            for idx in layer_indices:
                if idx in student_features and idx in teacher_features:
                    s_feat = student_features[idx]
                    t_feat = teacher_features[idx]
                    if s_feat.shape != t_feat.shape:
                        if s_feat.dim() == 4 and t_feat.dim() == 4:
                            t_feat = torch.nn.functional.interpolate(
                                t_feat, size=s_feat.shape[2:], mode='bilinear', align_corners=False
                            )
                        else:
                            continue
                    s_attention = torch.abs(s_feat).sum(dim=1, keepdim=True)
                    t_attention = torch.abs(t_feat).sum(dim=1, keepdim=True)
                    s_attention = s_attention / (s_attention.norm(2, dim=(2,3), keepdim=True) + 1e-8)
                    t_attention = t_attention / (t_attention.norm(2, dim=(2,3), keepdim=True) + 1e-8)
                    total_loss += torch.nn.functional.mse_loss(s_attention, t_attention)
                    valid_layers += 1

            if valid_layers > 0:
                return total_loss / valid_layers
            return torch.tensor(0.0, device=images.device)
        finally:
            for hook in hooks:
                hook.remove()
        """Compute hierarchical distillation loss at intermediate layers.
        
        Extracts features at specified layer indices from both student and teacher
        and computes attention transfer loss.
        
        Args:
            images: Input batch images
            layer_indices: List of layer indices to extract features from
        
        Returns:
            torch.Tensor: Hierarchical distillation loss
        """
        if not layer_indices:
            return torch.tensor(0.0, device=images.device)

        # Register forward hooks to extract intermediate features
        student_features = {}
        teacher_features = {}

        def make_hook(storage, key):
            def hook(module, input, output):
                storage[key] = output.detach() if not torch.is_grad_enabled() else output
            return hook

        # Get student model (unwrap DDP if needed)
        student_model = unwrap_model(self.model)
        teacher_model = self.teacher_model

        hooks = []
        try:
            # Register hooks on student layers
            for idx in layer_indices:
                if idx < len(student_model.model):
                    hooks.append(student_model.model[idx].register_forward_hook(
                        make_hook(student_features, idx)
                    ))
                if idx < len(teacher_model.model):
                    hooks.append(teacher_model.model[idx].register_forward_hook(
                        make_hook(teacher_features, idx)
                    ))

            # Forward pass to extract features
            with torch.no_grad():
                _ = student_model(images)
                _ = teacher_model(images)

            # Compute attention transfer loss for each layer pair
            total_loss = 0.0
            valid_layers = 0
            for idx in layer_indices:
                if idx in student_features and idx in teacher_features:
                    s_feat = student_features[idx]
                    t_feat = teacher_features[idx]

                    # Ensure same shape
                    if s_feat.shape != t_feat.shape:
                        if s_feat.dim() == 4 and t_feat.dim() == 4:
                            t_feat = torch.nn.functional.interpolate(
                                t_feat, size=s_feat.shape[2:], mode='bilinear', align_corners=False
                            )
                        else:
                            continue

                    # Attention transfer: convert features to attention maps
                    # Sum over channels, normalize
                    s_attention = torch.abs(s_feat).sum(dim=1, keepdim=True)
                    t_attention = torch.abs(t_feat).sum(dim=1, keepdim=True)

                    # Normalize
                    s_attention = s_attention / (s_attention.norm(2, dim=(2,3), keepdim=True) + 1e-8)
                    t_attention = t_attention / (t_attention.norm(2, dim=(2,3), keepdim=True) + 1e-8)

                    # MSE loss on attention maps
                    layer_loss = torch.nn.functional.mse_loss(s_attention, t_attention)
                    total_loss += layer_loss
                    valid_layers += 1

            if valid_layers > 0:
                return total_loss / valid_layers
            return torch.tensor(0.0, device=images.device)
        finally:
            # Clean up hooks
            for hook in hooks:
                hook.remove()

    def validate(self):
        """Run validation on val set using self.validator.

        Returns:
            metrics (dict): Dictionary of validation metrics.
            fitness (float): Fitness score for the validation.
        """
        if self.ema and self.world_size > 1:
            # Sync EMA buffers from rank 0 to all ranks
            for buffer in self.ema.ema.buffers():
                dist.broadcast(buffer, src=0)
        metrics = self.validator(self)
        if metrics is None:
            return None, None
        fitness = metrics.pop("fitness", -self.loss.detach().cpu().numpy())  # use loss as fitness measure if not found
        if not self.best_fitness or self.best_fitness < fitness:
            self.best_fitness = fitness
        return metrics, fitness

    def get_model(self, cfg=None, weights=None, verbose=True):
        """Get model and raise NotImplementedError for loading cfg files."""
        raise NotImplementedError("This task trainer doesn't support loading cfg files")

    def get_validator(self):
        """Raise NotImplementedError (must be implemented by subclasses)."""
        raise NotImplementedError("get_validator function not implemented in trainer")

    def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode="train"):
        """Raise NotImplementedError (must return a `torch.utils.data.DataLoader` in subclasses)."""
        raise NotImplementedError("get_dataloader function not implemented in trainer")

    def build_dataset(self, img_path, mode="train", batch=None):
        """Build dataset."""
        raise NotImplementedError("build_dataset function not implemented in trainer")

    def label_loss_items(self, loss_items=None, prefix="train"):
        """Return a loss dict with labeled training loss items tensor.

        Notes:
            This is not needed for classification but necessary for segmentation & detection
        """
        return {"loss": loss_items} if loss_items is not None else ["loss"]

    def set_model_attributes(self):
        """Set or update model parameters before training."""
        self.model.names = self.data["names"]

    def build_targets(self, preds, targets):
        """Build target tensors for training YOLO model."""
        pass

    def progress_string(self):
        """Return a string describing training progress."""
        return ""

    # TODO: may need to put these following functions into callback
    def plot_training_samples(self, batch, ni):
        """Plot training samples during YOLO training."""
        pass

    def plot_training_labels(self):
        """Plot training labels for YOLO model."""
        pass

    def save_metrics(self, metrics):
        """Save training metrics to a CSV file."""
        keys, vals = list(metrics.keys()), list(metrics.values())
        n = len(metrics) + 2  # number of cols
        t = time.time() - self.train_time_start
        self.csv.parent.mkdir(parents=True, exist_ok=True)  # ensure parent directory exists
        s = "" if self.csv.exists() else ("%s," * n % ("epoch", "time", *keys)).rstrip(",") + "\n"
        with open(self.csv, "a", encoding="utf-8") as f:
            f.write(s + ("%.6g," * n % (self.epoch + 1, t, *vals)).rstrip(",") + "\n")

    def plot_metrics(self):
        """Plot metrics from a CSV file."""
        plot_results(file=self.csv, on_plot=self.on_plot)  # save results.png

    def on_plot(self, name, data=None):
        """Register plots (e.g. to be consumed in callbacks)."""
        path = Path(name)
        self.plots[path] = {"data": data, "timestamp": time.time()}

    def final_eval(self):
        """Perform final evaluation and validation for object detection YOLO model."""
        model = self.best if self.best.exists() else None
        with torch_distributed_zero_first(LOCAL_RANK):  # strip only on GPU 0; other GPUs should wait
            if RANK in {-1, 0}:
                ckpt = strip_optimizer(self.last) if self.last.exists() else {}
                if model:
                    # update best.pt train_metrics from last.pt
                    strip_optimizer(self.best, updates={"train_results": ckpt.get("train_results")})
        if model:
            LOGGER.info(f"\nValidating {model}...")
            self.validator.args.plots = self.args.plots
            self.validator.args.compile = False  # disable final val compile as too slow
            self.metrics = self.validator(model=model)
            self.metrics.pop("fitness", None)
            self.run_callbacks("on_fit_epoch_end")

    def check_resume(self, overrides):
        """Check if resume checkpoint exists and update arguments accordingly."""
        resume = self.args.resume
        if resume:
            try:
                exists = isinstance(resume, (str, Path)) and Path(resume).exists()
                last = Path(check_file(resume) if exists else get_latest_run())

                # Check that resume data YAML exists, otherwise strip to force re-download of dataset
                ckpt_args = load_checkpoint(last)[0].args
                if not isinstance(ckpt_args["data"], dict) and not Path(ckpt_args["data"]).exists():
                    ckpt_args["data"] = self.args.data

                resume = True
                self.args = get_cfg(ckpt_args)
                self.args.model = self.args.resume = str(last)  # reinstate model
                for k in (
                    "imgsz",
                    "batch",
                    "device",
                    "close_mosaic",
                    "augmentations",
                    "save_period",
                    "workers",
                    "cache",
                    "patience",
                    "time",
                    "freeze",
                    "val",
                    "plots",
                ):  # allow arg updates to reduce memory or update device on resume
                    if k in overrides:
                        setattr(self.args, k, overrides[k])

                # Handle augmentations parameter for resume: check if user provided custom augmentations
                if ckpt_args.get("augmentations") is not None:
                    # Augmentations were saved in checkpoint as reprs but can't be restored automatically
                    LOGGER.warning(
                        "Custom Albumentations transforms were used in the original training run but are not "
                        "being restored. To preserve custom augmentations when resuming, you need to pass the "
                        "'augmentations' parameter again to get expected results. Example: \n"
                        f"model.train(resume=True, augmentations={ckpt_args['augmentations']})"
                    )

            except Exception as e:
                raise FileNotFoundError(
                    "Resume checkpoint not found. Please pass a valid checkpoint to resume from, "
                    "i.e. 'yolo train resume model=path/to/last.pt'"
                ) from e
        self.resume = resume

    def _load_checkpoint_state(self, ckpt):
        """Load optimizer, scaler, EMA, and best_fitness from checkpoint."""
        if ckpt.get("optimizer") is not None:
            self.optimizer.load_state_dict(ckpt["optimizer"])
        if ckpt.get("scaler") is not None:
            self.scaler.load_state_dict(ckpt["scaler"])
        if self.ema and ckpt.get("ema"):
            self.ema = ModelEMA(self.model)  # validation with EMA creates inference tensors that can't be updated
            self.ema.ema.load_state_dict(ckpt["ema"].float().state_dict())
            self.ema.updates = ckpt["updates"]
        self.best_fitness = ckpt.get("best_fitness", 0.0)

    def _handle_nan_recovery(self, epoch):
        """Detect and recover from NaN/Inf loss and fitness collapse by loading last checkpoint."""
        loss_nan = self.loss is not None and not self.loss.isfinite()
        fitness_nan = self.fitness is not None and not np.isfinite(self.fitness)
        fitness_collapse = self.best_fitness and self.best_fitness > 0 and self.fitness == 0
        corrupted = RANK in {-1, 0} and loss_nan and (fitness_nan or fitness_collapse)
        reason = "Loss NaN/Inf" if loss_nan else "Fitness NaN/Inf" if fitness_nan else "Fitness collapse"
        if RANK != -1:  # DDP: broadcast to all ranks
            broadcast_list = [corrupted if RANK == 0 else None]
            dist.broadcast_object_list(broadcast_list, 0)
            corrupted = broadcast_list[0]
        if not corrupted:
            return False
        if epoch == self.start_epoch or not self.last.exists():
            LOGGER.warning(f"{reason} detected but can not recover from last.pt...")
            return False  # Cannot recover on first epoch, let training continue
        self.nan_recovery_attempts += 1
        if self.nan_recovery_attempts > 3:
            raise RuntimeError(f"Training failed: NaN persisted for {self.nan_recovery_attempts} epochs")
        LOGGER.warning(f"{reason} detected (attempt {self.nan_recovery_attempts}/3), recovering from last.pt...")
        self._model_train()  # set model to train mode before loading checkpoint to avoid inference tensor errors
        _, ckpt = load_checkpoint(self.last)
        ema_state = ckpt["ema"].float().state_dict()
        if not all(torch.isfinite(v).all() for v in ema_state.values() if isinstance(v, torch.Tensor)):
            raise RuntimeError(f"Checkpoint {self.last} is corrupted with NaN/Inf weights")
        unwrap_model(self.model).load_state_dict(ema_state)  # Load EMA weights into model
        self._load_checkpoint_state(ckpt)  # Load optimizer/scaler/EMA/best_fitness
        del ckpt, ema_state
        self.scheduler.last_epoch = epoch - 1
        return True

    def resume_training(self, ckpt):
        """Resume YOLO training from given epoch and best fitness."""
        if ckpt is None or not self.resume:
            return
        start_epoch = ckpt.get("epoch", -1) + 1
        assert start_epoch > 0, (
            f"{self.args.model} training to {self.epochs} epochs is finished, nothing to resume.\n"
            f"Start a new training without resuming, i.e. 'yolo train model={self.args.model}'"
        )
        LOGGER.info(f"Resuming training {self.args.model} from epoch {start_epoch + 1} to {self.epochs} total epochs")
        if self.epochs < start_epoch:
            LOGGER.info(
                f"{self.model} has been trained for {ckpt['epoch']} epochs. Fine-tuning for {self.epochs} more epochs."
            )
            self.epochs += ckpt["epoch"]  # finetune additional epochs
        self._load_checkpoint_state(ckpt)
        self.start_epoch = start_epoch
        if start_epoch > (self.epochs - self.args.close_mosaic):
            self._close_dataloader_mosaic()

    def _close_dataloader_mosaic(self):
        """Update dataloaders to stop using mosaic augmentation."""
        if hasattr(self.train_loader.dataset, "mosaic"):
            self.train_loader.dataset.mosaic = False
        if hasattr(self.train_loader.dataset, "close_mosaic"):
            LOGGER.info("Closing dataloader mosaic")
            self.train_loader.dataset.close_mosaic(hyp=copy(self.args))

    def build_optimizer(self, model, name="auto", lr=0.001, momentum=0.9, decay=1e-5, iterations=1e5):
        """Construct an optimizer for the given model."""
        # LoRA-aware parameter group separation
        lora_lr_mult = getattr(self.args, "lora_lr_mult", 1.0)
        has_lora_param = any(_is_adapter_param(n) for n, _ in model.named_parameters())
        
        g = [{}, {}, {}, {}, {}, {}]  # optimizer parameter groups
        bn = tuple(v for k, v in nn.__dict__.items() if "Norm" in k)  # normalization layers, i.e. BatchNorm2d()
        if name == "auto":
            LOGGER.info(
                f"{colorstr('optimizer:')} 'optimizer=auto' found, "
                f"ignoring 'lr0={self.args.lr0}' and 'momentum={self.args.momentum}' and "
                f"determining best 'optimizer', 'lr0' and 'momentum' automatically... "
            )
            nc = self.data.get("nc", 10)  # number of classes
            lr_fit = round(0.002 * 5 / (4 + nc), 6)  # lr0 fit equation to 6 decimal places
            name, lr, momentum = ("MuSGD", 0.01 if iterations > 10000 else lr_fit, 0.9)
            self.args.warmup_bias_lr = 0.0  # no higher than 0.01 for Adam

        use_muon = name == "MuSGD"
        for module_name, module in unwrap_model(model).named_modules():
            for param_name, param in module.named_parameters(recurse=False):
                fullname = f"{module_name}.{param_name}" if module_name else param_name
                if param.ndim >= 2 and use_muon:
                    g[3][fullname] = param  # muon params
                elif "routing" in fullname or "router" in fullname:  # MoE Router parameters
                    g[4][fullname] = param
                elif _is_adapter_param(fullname):  # Adapter params (LoRA/LoHa/LoKr/OFT/BOFT/IA3/HRA) -> separate group
                    g[5][fullname] = param
                elif "bias" in fullname:  # bias (no decay)
                    g[2][fullname] = param
                elif isinstance(module, bn) or "logit_scale" in fullname:  # weight (no decay)
                    # ContrastiveHead and BNContrastiveHead included here with 'logit_scale'
                    g[1][fullname] = param
                else:  # weight (with decay)
                    g[0][fullname] = param
                    
        if not use_muon:
            g =[list(x.values()) for x in g]  # convert to list of params

        optimizers = {"Adam", "Adamax", "AdamW", "NAdam", "RAdam", "RMSProp", "SGD", "MuSGD", "auto"}
        name = {x.lower(): x for x in optimizers}.get(name.lower())
        if name in {"Adam", "Adamax", "AdamW", "NAdam", "RAdam"}:
            optim_args = dict(lr=lr, betas=(momentum, 0.999), weight_decay=0.0)
        elif name == "RMSProp":
            optim_args = dict(lr=lr, momentum=momentum)
        elif name == "SGD" or name == "MuSGD":
            optim_args = dict(lr=lr, momentum=momentum, nesterov=True)
        else:
            raise NotImplementedError(
                f"Optimizer '{name}' not found in list of available optimizers {optimizers}."
            )

        # ------------------ 統一在此處完成參數組包裝 ------------------
        g[2] = {"params": g[2], **optim_args, "param_group": "bias"}
        g[0] = {"params": g[0], **optim_args, "weight_decay": decay, "param_group": "weight"}
        g[1] = {"params": g[1], **optim_args, "weight_decay": 0.0, "param_group": "bn"}
        
        # MoE Router: 修正重複包裝問題並正確保留 optim_args
        moe_router_lr_scale = getattr(self.args, 'moe_router_lr_scale', 0.5)
        router_lr = lr * moe_router_lr_scale
        g[4] = {"params": g[4], **optim_args, "weight_decay": decay, "lr": router_lr, "initial_lr": router_lr, "param_group": "router"}

        # LoRA: 提前完成包裝避免在 use_muon 中引發 KeyError
        lora_log = ""
        lora_lr = lr * lora_lr_mult
        if len(g[5]) > 0:
            g[5] = {"params": g[5], **optim_args, "weight_decay": 0.0, "lr": lora_lr, "initial_lr": lora_lr, "param_group": "lora"}
            lora_log = f", {len(g[5]['params'])} LoRA(lr={lora_lr:.6f}, mult={lora_lr_mult})"
        else:
            g[5] = {"params": g[5], **optim_args, "param_group": "lora"}
            if has_lora_param and lora_lr_mult != 1.0:
                lora_log = " [WARN] lora_lr_mult set but no LoRA params found"

        muon, sgd = (0.1, 1.0) if iterations > 10000 else (0.5, 0.5)  # scale factor for MuSGD
        
        if use_muon:
            g[3] = {"params": g[3], **optim_args, "weight_decay": decay, "use_muon": True, "param_group": "muon"}
        else:
            g[3] = {"params": g[3], **optim_args, "weight_decay": 0.0, "param_group": "muon"}

        # --------- 在 g 被切割前先取得 Logger 用的各群組參數數量 ---------
        c_w_nd = len(g[1]['params'])
        c_w_d = len(g[0]['params']) if len(g[0]['params']) > 0 else len(g[3]['params'])
        c_bias = len(g[2]['params'])
        c_router = len(g[4]['params'])

        if use_muon:
            import re
            pattern = re.compile(r"(?=.*23)(?=.*cv3)|proto\.semseg|flow_model")
            g_ = []  # new param groups
            for x in g:
                p = x["params"]
                if isinstance(p, dict):
                    p1 =[v for k, v in p.items() if pattern.search(k)]
                    p2 =[v for k, v in p.items() if not pattern.search(k)]
                    if p1:
                        g_.append({"params": p1, **{k: v for k, v in x.items() if k != "params"}, "lr": x.get("lr", lr) * 3})
                    if p2:
                        g_.append({"params": p2, **{k: v for k, v in x.items() if k != "params"}})
                elif len(p) > 0:  # 處理被轉換為 list 但尚未切割的部分
                    g_.append(x)
            optim_groups = g_
        else:
            # PyTorch 嚴格要求不能傳入空的 param_group，所以要先過濾長度 > 0 的群組
            optim_groups =[x for x in g if len(x["params"]) > 0]
            
        optimizer = getattr(optim, name, partial(MuSGD, muon=muon, sgd=sgd))(params=optim_groups)

        LOGGER.info(
            f"{colorstr('optimizer:')} {type(optimizer).__name__}(lr={lr}, momentum={momentum}) with parameter groups "
            f"{c_w_nd} weight(decay=0.0), {c_w_d} weight(decay={decay}), {c_bias} bias(decay=0.0), {c_router} router(lr={moe_router_lr_scale}x){lora_log}"
        )
        return optimizer
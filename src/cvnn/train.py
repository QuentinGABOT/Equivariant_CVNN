# Standard library imports
import math
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from collections import defaultdict

# Third-party imports
import torch
import torch.nn as nn
import wandb
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import numpy as np
from torchcvnn.nn.modules import ComplexMSELoss
from torch.nn import MSELoss

# Local imports
from cvnn.callbacks import ModelCheckpoint
from cvnn.losses import compute_class_weights, ELBOLoss, ComplexELBOLoss
from cvnn.models.utils import get_loss_function
from cvnn.schedulers import build_schedulers, step_schedulers
from cvnn.utils import setup_logging, kl_sanity_checks, KLAnnealing
from cvnn.models.blocks import Down

# initialize module-level logger
logger = setup_logging(__name__)

def train_one_epoch(
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    loss_fn: Callable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    warmup_scheduler: Optional[Any] = None,
    scheduler: Optional[Any] = None,
    step_on_batch: bool = False,
    anneal=None,
    max_norm: Optional[float] = None,
) -> dict:
    """
    Run one training epoch.
    Optimized to handle Tensor metrics without CPU/GPU synchronization bottlenecks.
    """
    model.train()
    loss_fn.train()

    # Use a simple dictionary to accumulate sums as Tensors (not floats!)
    # This prevents synchronization overhead during the loop.
    running_sums = defaultdict(float)
    total_samples = 0

    pbar = tqdm(train_loader, desc="Training")
    for batch in pbar:
        # ---- 1. Data Prep ----
        if isinstance(batch, (tuple, list)) and len(batch) == 2:
            inputs, targets = batch[0], batch[1]
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            if targets.dtype == torch.uint8:
                targets = targets.long()
        elif isinstance(batch, (tuple, list)) and len(batch) == 1:
            inputs = batch[0].to(device, non_blocking=True)
            targets = inputs
        else: 
            inputs = batch.to(device, non_blocking=True)
            targets = inputs

        bs = inputs.size(0)
        
        # ---- 2. Forward & Backward ----
        optimizer.zero_grad()
        outputs = model(inputs)
            
        if isinstance(loss_fn, (ELBOLoss, ComplexELBOLoss)):
            # Pass anneal if your loss supports it
            loss_output = loss_fn(outputs, inputs, anneal=anneal)
        elif isinstance(loss_fn, (ComplexMSELoss, MSELoss)):
            loss_output = loss_fn(outputs, inputs)
        else:
            _, outputs_projected = outputs if isinstance(outputs, (tuple, list)) and len(outputs) == 2 else (None, outputs)
            loss_output = loss_fn(outputs_projected, targets)

        # Unpack
        if isinstance(loss_output, tuple):
            loss, batch_metrics = loss_output
        else:
            loss = loss_output
            batch_metrics = {}

        loss.backward()
        
        # Optional: Gradient Clipping
        if max_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
        
        optimizer.step()

        # ---- 3. Schedulers ----
        if warmup_scheduler:
            warmup_scheduler.step()
        
        if scheduler and step_on_batch and not isinstance(scheduler, ReduceLROnPlateau):
            scheduler.step()

        # ---- 4. Optimized Metric Tracking ----
        # We perform ONE synchronization per batch to update the progress bar.
        # For everything else, we stay async.
        current_loss_val = loss.item()
        
        # Update progress bar immediately
        pbar.set_postfix({"loss": f"{current_loss_val:.4f}"})

        # Accumulate metrics (CPU Tensors, No Sync)
        # We perform the accumulation on CPU to save GPU VRAM, 
        # but we use .detach() to ensure we don't drag the graph with us.
        with torch.no_grad():
            total_samples += bs
            
            # Accumulate main loss
            running_sums["loss"] += current_loss_val * bs
            
            # Accumulate VAE metrics
            for k, v in batch_metrics.items():
                if torch.is_tensor(v):
                    # .cpu() is a copy, but it doesn't block execution like .item()
                    running_sums[k] += v.detach().cpu() * bs
                else:
                    running_sums[k] += v * bs

    # ---- 5. Finalize Epoch ----
    # Compute averages only ONCE at the very end
    avg_metrics = {}
    for k, v in running_sums.items():
        # Convert final tensors to python floats here
        val = v / total_samples
        avg_metrics[k] = val.item() if torch.is_tensor(val) else val

    return avg_metrics

def validate_one_epoch(
    model: torch.nn.Module,
    valid_loader: torch.utils.data.DataLoader,
    loss_fn: Callable,
    device: torch.device,
    anneal=None,
) -> dict:
    """
    Run one validation epoch.
    Optimized for speed: uses non-blocking accumulation to prevent GPU stalls.
    """
    model.eval()
    loss_fn.eval()

    # Accumulators
    running_sums = defaultdict(float)
    total_samples = 0

    # No gradients needed for validation
    with torch.no_grad():
        pbar = tqdm(valid_loader, desc="Validation")
        
        for batch in pbar:
            if isinstance(batch, (tuple, list)) and len(batch) == 2:
                inputs, targets = batch[0], batch[1]
                inputs = inputs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                if targets.dtype == torch.uint8:
                    targets = targets.long()
            elif isinstance(batch, (tuple, list)) and len(batch) == 1:
                inputs = batch[0].to(device, non_blocking=True)
                targets = inputs
            else:
                inputs = batch.to(device, non_blocking=True)
                targets = inputs
            
            bs = inputs.size(0)
            
            # ---- 2. Forward Pass ----
            outputs = model(inputs)

            # Compute Loss
            if isinstance(loss_fn, (ELBOLoss, ComplexELBOLoss)):
                loss_output = loss_fn(outputs, inputs, anneal=anneal) 
            else:
                _, outputs_projected = outputs if isinstance(outputs, (tuple, list)) and len(outputs) == 2 else (None, outputs)
                loss_output = loss_fn(outputs_projected, targets)
        
            if isinstance(loss_output, tuple):
                loss, batch_metrics = loss_output
            else:
                loss = loss_output
                batch_metrics = {}

            # ---- 3. Optimized Accumulation ----
            total_samples += bs
            
            # Get scalar loss ONCE for both logging and accumulation
            # (We pay the sync cost here, but it's necessary for the pbar)
            current_loss_val = loss.item()
            running_sums["loss"] += current_loss_val * bs
            
            # Update pbar
            pbar.set_postfix({"loss": f"{current_loss_val:.4f}"})

            # Accumulate other metrics without further syncing
            for k, v in batch_metrics.items():
                if torch.is_tensor(v):
                    # .cpu() moves data without waiting for GPU to finish queue
                    running_sums[k] += v.cpu() * bs
                else:
                    running_sums[k] += v * bs

    # ---- 4. Finalize Averages ----
    avg_metrics = {}
    for k, v in running_sums.items():
        val = v / total_samples
        # Ensure everything returned is a standard python float
        avg_metrics[k] = val.item() if torch.is_tensor(val) else val
    return avg_metrics

def setup_loss_optimizer(
    model: torch.nn.Module,
    cfg: Dict[str, Any],
    dataset: torch.utils.data.Dataset,
    device: torch.device,
) -> Tuple[Callable, torch.optim.Optimizer]:
    """Instantiate loss function and optimizer based on config."""
    loss_name = cfg["loss"]["name"]

    # Get the model's layer_mode for mode-aware loss selection
    layer_mode = getattr(model, "layer_mode", "complex")

    # Get ignore_index from config if available
    ignore_index = cfg["data"].get("ignore_index", None)

    task = cfg["task"]

    if task in ["classification", "segmentation"] and cfg["loss"]["use_class_weights"]:
        targets = torch.tensor(np.array([data[1] for data in dataset]))
        class_weights = compute_class_weights(
            targets,
            cfg["model"]["inferred_num_classes"],
            weight_mode=cfg["loss"]["weight_mode"],
            ignore_index=ignore_index,
            device=device,
        )
    else:
        class_weights = None
    
    schedule = cfg["loss"].get("schedule")
    if schedule == "beta":
        beta_max = float(cfg["loss"].get("beta_max"))
        bpd_target = None
    else:
        beta_max = None
        if schedule == "capacity":
            bpd_target = float(cfg["loss"].get("bpd_target"))
        else:
            bpd_target = None
                
    cov_mode = cfg["model"].get("cov_mode")
    standard_reparam = cfg["model"].get("standard_reparam")
    if "decoder_variance" in cfg["model"]:
        learned_variance = cfg["model"]["decoder_variance"].get("learned_variance")
        min_log_sigma = cfg["model"]["decoder_variance"].get("min_log_sigma")
    else:
        learned_variance = None
        min_log_sigma = None

    # Try to use mode-aware loss selection first
    loss_fn = get_loss_function(loss_name, layer_mode, ignore_index, class_weights, schedule, bpd_target, 
                                beta_max, cov_mode, standard_reparam, learned_variance, min_log_sigma)

    optim_cls = getattr(torch.optim, cfg["optim"]["algo"])
    base = dict(cfg["optim"]["params"])              # e.g., {"lr": 3e-4, "weight_decay": 0.01, ...}

    # group 1: model params, untouched
    groups = [{"params": model.parameters(), **base}]

    # group 2: loss params (e.g., log_sigma2_dec), no WD; skip if none
    loss_params = list(loss_fn.parameters())
    if len(loss_params) > 0:
        loss_group = {**base}
        loss_group["weight_decay"] = 0.0            # critical: don't decay log_sigma2_dec
        # optional: different LR for loss param
        # loss_group["lr"] = base["lr"] * cfg["optim"].get("loss_lr_mult", 1.0)
        groups.append({"params": loss_params, **loss_group})

    optimizer = optim_cls(groups)
    return loss_fn, optimizer


def train_model(
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    valid_loader: torch.utils.data.DataLoader,
    cfg: Dict[str, Any],
    logdir: Union[str, Path],
    device: Optional[torch.device] = None,
    loss_fn: Optional[Callable] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    warmup_scheduler: Optional[Any] = None,
    scheduler: Optional[Any] = None,
    start_epoch: int = 0,
    gumbel_experiment: Optional[Any] = None,
) -> Dict[str, List[float]]:
    """Run full training over epochs and return history of losses.

    Args:
        start_epoch: Starting epoch for training (0 for new training, >0 for retrain)
        gumbel_experiment: Optional BaseExperiment instance for Gumbel tau handling
    """
    if device is None:
        device = next(model.parameters()).device

    # logging configuration
    logger.info(f"Config: {cfg}")

    # build warmup and main schedulers if not provided
    if warmup_scheduler is None or scheduler is None:
        w, s = build_schedulers(optimizer, cfg, len(train_loader))
        warmup_scheduler = warmup_scheduler or w
        scheduler = scheduler or s
    # determine scheduler stepping granularity
    sched_cfg = cfg.get("scheduler", {})
    if sched_cfg:
        step_on_batch = sched_cfg.get("step_on", "epoch") == "batch"
    else:
        step_on_batch = False
    # initialize checkpoint
    # determine input dimensions for export
    sample_batch = next(iter(train_loader))
    sample = (
        sample_batch[0] if isinstance(sample_batch, (tuple, list)) else sample_batch
    )
    num_input_dims = sample.ndim
    total_epochs = cfg["nepochs"]
    max_norm = cfg["optim"].get("max_norm")
    is_elbo = isinstance(loss_fn, (ELBOLoss, ComplexELBOLoss))

    # Use custom checkpoint if gumbel_experiment is provided
    if gumbel_experiment is not None:
        checkpoint = ModelCheckpointWithGumbel(
            model,
            optimizer,
            logdir,
            num_input_dims,
            min_is_best=True,
            warmup_scheduler=warmup_scheduler,
            scheduler=scheduler,
            gumbel_experiment=gumbel_experiment,
        )
    else:
        checkpoint = ModelCheckpoint(
            model,
            optimizer,
            logdir,
            num_input_dims,
            min_is_best=True,
            warmup_scheduler=warmup_scheduler,
            scheduler=scheduler,
        )
    if is_elbo:
        anneal = KLAnnealing(kind="linear", warmup_steps=15 * len(train_loader))
        # run kl sanity checks
        kl_sanity_checks()
    else:
        anneal = None

    history = defaultdict(list)

    # Calculate actual epoch range for training
    if start_epoch >= total_epochs:
        logger.warning(
            f"Start epoch {start_epoch} >= total epochs {total_epochs}, no training needed"
        )
        return history

    logger.info(f"Training from epoch {start_epoch} to {total_epochs}")

    for epoch in range(start_epoch, total_epochs):
        # Update Gumbel tau if experiment is provided
        if gumbel_experiment is not None:
            gumbel_experiment.current_gumbel_tau = update_gumbel_tau(
                model, 
                gumbel_experiment.gumbel_tau_config, 
                gumbel_experiment.current_gumbel_tau, 
                epoch
            )
        
        current_warmup_scheduler = warmup_scheduler if epoch < cfg["warmup"]["epochs"] else None            

        # 1. Get the single dictionary result
        train_metrics = train_one_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
            anneal=anneal,
            warmup_scheduler=current_warmup_scheduler,
            step_on_batch=step_on_batch,
            max_norm=max_norm,
        )
        valid_metrics = validate_one_epoch(
            model=model,
            valid_loader=valid_loader,
            loss_fn=loss_fn,
            device=device,
            anneal=anneal,
        )

        # 2. Extract loss specifically for history and schedulers
        train_loss = train_metrics["loss"]
        valid_loss = valid_metrics["loss"]

        # 3. Update History (cleaner now)
        history["train_loss"].append(train_loss)
        history["valid_loss"].append(valid_loss)

        # Add all other metrics dynamically
        for k, v in train_metrics.items():
            if k != "loss": # avoid duplicating the loss list
                history[f"train_{k}"].append(v)
                
        for k, v in valid_metrics.items():
            if k != "loss":
                history[f"valid_{k}"].append(v)
        # 4. Step Scheduler using the extracted valid_loss
        step_schedulers(
            warmup=None, 
            scheduler=scheduler, 
            metric=valid_loss, 
            step_on_batch=step_on_batch
        )
        
        # 5. Log to wandb if initialized
        if wandb.run:
            # 1. Basic fixed metrics
            log_dict = {
                "epoch": epoch,
                "training/loss": train_loss,
                "validation/loss": valid_loss,
                "training/learning_rate": optimizer.param_groups[0]["lr"],
            }

            # 2. Dynamic Training Metrics
            # Automatically prefixes any metric found in train_metrics
            for k, v in train_metrics.items():
                # specific check to avoid logging 'loss' twice if it's in the dict
                if k != "loss" and v is not None:
                    log_dict[f"training/{k}"] = v

            # 3. Dynamic Validation Metrics
            for k, v in valid_metrics.items():
                if k != "loss" and v is not None:
                    log_dict[f"validation/{k}"] = v

            # 4. Special cases (e.g. Gumbel Tau)
            # These are outside the loss function, so we keep them explicit
            if gumbel_experiment and hasattr(gumbel_experiment, "current_gumbel_tau"):
                log_dict["training/gumbel_tau"] = gumbel_experiment.current_gumbel_tau.item()

            wandb.log(log_dict)

        # 6. Update checkpoint with validation score
        checkpoint.update(valid_loss, epoch)

        # save a snapshot after each epoch for crash recovery
        last_state = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }
        if warmup_scheduler:
            last_state["warmup_state_dict"] = warmup_scheduler.state_dict()
        if scheduler:
            last_state["scheduler_state_dict"] = scheduler.state_dict()

        # Add Gumbel tau state if experiment is provided
        if gumbel_experiment is not None:
            gumbel_state = gumbel_experiment.get_gumbel_tau_state()
            if gumbel_state:
                last_state["gumbel_tau_state"] = gumbel_state

            # Add projection state if experiment has it
            if hasattr(gumbel_experiment, "get_projection_state"):
                projection_state = gumbel_experiment.get_projection_state()
                if projection_state:
                    last_state["projection_state"] = projection_state

        torch.save(last_state, Path(logdir) / "last_model.pt")

    return history


def update_gumbel_tau(model: torch.nn.Module, gumbel_config: Dict[str, Any], current_tau: torch.Tensor, epoch: int) -> torch.Tensor:
    """Update Gumbel tau based on epoch and decay schedule.
    
    Args:
        model: The model containing encoder blocks with Gumbel tau
        gumbel_config: Configuration dict with start_value, gamma, start_decay_epoch, min_value
        current_tau: Current tau value
        epoch: Current training epoch
        
    Returns:
        Updated tau value
    """
    if epoch >= gumbel_config["start_decay_epoch"]:
        decay_epochs = epoch - gumbel_config["start_decay_epoch"]
        new_tau_value = gumbel_config["start_value"] * (gumbel_config["gamma"] ** decay_epochs)
        new_tau = torch.tensor(
            max(new_tau_value, gumbel_config["min_value"]), 
            dtype=torch.float32
        )
        
        # Update all encoder blocks
        if hasattr(model, 'encoder'):
            for enc in model.convnet.encoder[1:]:
                if hasattr(enc, 'downsampling_method') and hasattr(enc.down, 'component_selection'):
                    enc.down.component_selection.gumbel_tau = new_tau
        
        return new_tau
    
    return current_tau

class ModelCheckpointWithGumbel(ModelCheckpoint):
    """Extended ModelCheckpoint that also saves Gumbel tau state."""
    
    def __init__(self, *args, gumbel_experiment=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.gumbel_experiment = gumbel_experiment
    
    def update(self, score: float, epoch: int) -> bool:
        """Update checkpoint if score improved, including Gumbel tau state."""
        if self.is_better(score):
            self.model.eval()
            state = {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "loss": score,
            }
            if self.warmup_scheduler is not None:
                state["warmup_state_dict"] = self.warmup_scheduler.state_dict()
            if self.scheduler is not None:
                state["scheduler_state_dict"] = self.scheduler.state_dict()
            
            # Add Gumbel tau state if experiment has it
            if self.gumbel_experiment and hasattr(self.gumbel_experiment, 'get_gumbel_tau_state'):
                gumbel_state = self.gumbel_experiment.get_gumbel_tau_state()
                if gumbel_state:
                    state["gumbel_tau_state"] = gumbel_state
                    
                # Add projection state if experiment has it
                if hasattr(self.gumbel_experiment, 'get_projection_state'):
                    projection_state = self.gumbel_experiment.get_projection_state()
                    if projection_state:
                        state["projection_state"] = projection_state
            
            torch.save(state, self.savepath / "best_model.pt")
            self.best_score = score
            return True
        return False
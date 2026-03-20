"""
cvnn.schedulers

Factory functions for building and stepping learning rate schedulers (including warmup).
"""

# Standard library imports
from typing import Any, Dict, Optional, Tuple

# Third-party imports
import torch.optim.lr_scheduler as lr_scheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau, _LRScheduler
from torch.optim.optimizer import Optimizer


def build_schedulers(
    optimizer: Optimizer, cfg: Dict[str, Any], train_loader_len: Optional[int] = None
) -> Tuple[Optional[_LRScheduler], Optional[_LRScheduler]]:
    """Instantiate warmup and main schedulers from config."""
    warmup = None
    warmup_cfg = cfg.get("warmup")
    warmup_epochs = warmup_cfg.get("epochs") if warmup_cfg else 0
    if warmup_epochs > 0 and train_loader_len is not None:
        # Total warmup steps = warmup_epochs * batches_per_epoch
        total_warmup_steps = warmup_epochs * train_loader_len
        warmup = lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step: min(1.0, float(step + 1) / total_warmup_steps),
        )

    scheduler = None
    sched_cfg = cfg.get("scheduler", {})
    if sched_cfg:
        SchedulerCls = getattr(lr_scheduler, sched_cfg.get("name"))
        scheduler = SchedulerCls(optimizer, **sched_cfg.get("params", {}))

    return warmup, scheduler


def step_schedulers(
    warmup: Optional[_LRScheduler],
    scheduler: Optional[_LRScheduler],
    metric: Any = None,
    step_on_batch: bool = False,
) -> None:
    """Step warmup and main schedulers at the EPOCH level."""
    if warmup:
        warmup.step()
        
    if not scheduler:
        return
        
    if isinstance(scheduler, ReduceLROnPlateau):
        scheduler.step(metric)
    elif not step_on_batch:
        scheduler.step()

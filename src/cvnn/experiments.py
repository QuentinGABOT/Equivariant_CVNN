"""
Helper functions to run full CVNN experiments in scripts or notebooks.
"""

# Standard library imports
import importlib
import pkgutil
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

# Third-party imports
import yaml

# Local imports
from cvnn.plugins import get_plugins
from cvnn.utils import setup_logging

# initialize module-level logger
logger = setup_logging(__name__)

# auto-discover task plugins from top-level 'projects' package
# determine project root (two levels up from this file)
_repo_root = Path(__file__).resolve().parents[2]
_projects_dir = _repo_root / "projects"
# ensure the repo root is in Python path for imports
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

# Auto-discovery is done lazily to avoid circular imports
_discovery_done = False


def _ensure_plugins_discovered():
    """Ensure plugin auto-discovery has been performed."""
    global _discovery_done
    if _discovery_done:
        return

    # Auto-discover and import task plugins
    if _projects_dir.is_dir():
        for _, module_name, _ in pkgutil.iter_modules([str(_projects_dir)]):
            try:
                importlib.import_module(f"projects.{module_name}.experiment")
            except ImportError as e:
                # log warning but continue - some projects might not have experiment.py
                logger.warning(
                    f"Could not import projects.{module_name}.experiment: {e}"
                )
            except Exception as e:
                logger.error(f"Error importing projects.{module_name}.experiment: {e}")

    _discovery_done = True


def run_experiment(
    config_path: Union[str, Path],
    resume_logdir: Optional[Union[str, Path]] = False,
    mode_override: Optional[str] = None,
) -> Tuple[Optional[Dict[str, Any]], Any, Path]:
    # load and override config
    if resume_logdir:
        resume_path = Path(resume_logdir).resolve()
        if not resume_path.exists():
            raise FileNotFoundError(
                f"Resume log directory does not exist: {resume_path}"
            )
        config_path = resume_path / "config.yml"
        if not config_path.exists():
            raise FileNotFoundError(
                f"Config file not found in resume log directory: {config_path}"
            )

    # Read raw config and dispatch by task before full schema validation
    with open(config_path, "r") as f:
        raw_cfg = yaml.safe_load(f) or {}
    task = raw_cfg.get("task")

    # Ensure plugins are discovered before attempting dispatch
    _ensure_plugins_discovered()

    # Get current plugins (populated dynamically after auto-discovery)
    available_tasks = get_plugins()

    if not task or task not in available_tasks:
        raise ValueError(
            f"Config must specify a valid 'task', got: {task}. Available tasks: {list(available_tasks.keys())}"
        )

    # dispatch via plugin registry
    exp_cls = available_tasks[task]
    exp = exp_cls(config_path, resume_logdir, mode_override)
    exp.run()
    return exp.history, exp.metrics, exp.logdir, exp.model
"""Project-local locations and unique names for experiment outputs."""

from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEST_EXPERIMENTS_ROOT = PROJECT_ROOT / "experiments" / "test"
TRAIN_EXPERIMENTS_ROOT = PROJECT_ROOT / "experiments" / "train"
DEFAULT_CHECKPOINT = PROJECT_ROOT / "checkpoints" / "best.pt"


def validate_experiment_name(value):
    name = value.strip()
    if (not name or name in {".", ".."}
            or any(char in "/\\" or ord(char) < 32 for char in name)):
        raise ValueError("Experiment name must be a nonempty folder name, not a path.")
    return name


def create_experiment_directory(root, name):
    """Reserve a timestamped directory without overwriting an earlier run."""
    name = validate_experiment_name(name)
    timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    stem = f"{name}_{timestamp}"
    suffix = 1
    while True:
        directory = root / (stem if suffix == 1 else f"{stem}_{suffix}")
        try:
            directory.mkdir()
        except FileExistsError:
            suffix += 1
        else:
            return directory

"""文件保护、校验、随机状态和原子写入工具。"""
from contextlib import contextmanager
import hashlib
import json
import os
import random
import struct
from pathlib import Path
import subprocess
import numpy as np
import torch
from .config import BLOCK_SIZE

CRC32 = struct.Struct("<I")
MAX_BATCH_SIZE = 2 ** 32 - 1


def validate_batch_size(batch_size):
    if type(batch_size) is not int or not 1 <= batch_size <= MAX_BATCH_SIZE:
        raise ValueError(
            f"batch_size must be an integer in 1..{MAX_BATCH_SIZE} (uint32)."
        )
    return batch_size


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@contextmanager
def _new_output(path):
    """Never overwrite a file, and remove this call's incomplete output."""
    path = Path(path)
    handle = path.open("xb")
    try:
        with handle:
            yield handle
    except BaseException:
        path.unlink()
        raise


def _read_exact(handle, count):
    data = handle.read(count)
    if len(data) != count:
        raise ValueError("Truncated input or compressed file.")
    return data


def verify_files(original_path, decoded_path):
    """Compare every byte, independently of the container's hash check."""
    with Path(original_path).open("rb") as original, Path(decoded_path).open(
        "rb"
    ) as decoded:
        while True:
            left, right = original.read(1024 * 1024), decoded.read(1024 * 1024)
            if left != right:
                raise ValueError("Original and decoded files differ.")
            if not left:
                return True


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def capture_rng_state():
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and "torch_cuda" in state:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def _atomic_torch_save(data, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(data, temporary)
    os.replace(temporary, path)


def _plot_bpb_history(training_metrics, validation_metrics, path):
    """Write a compact linear BPB curve for visual convergence checks."""
    from PIL import Image, ImageDraw, ImageFont

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    width, height = 1600, 960
    left, top, right, bottom = 145, 100, 70, 135
    plot_left, plot_top = left, top
    plot_right, plot_bottom = width - right, height - bottom
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 20)
        small_font = ImageFont.truetype("DejaVuSans.ttf", 17)
        title_font = ImageFont.truetype("DejaVuSans.ttf", 28)
    except OSError:
        font = small_font = title_font = ImageFont.load_default()

    training_points = [
        (int(record["global_update"]), float(record["bpb_batch"]))
        for record in training_metrics
    ]
    validation_points = [
        (int(record["global_update"]), float(record["bpb"]))
        for record in validation_metrics
    ]
    # Validation BPB is the convergence signal. Mixing the much larger early
    # batch BPB into the same y-axis would visually flatten its late plateau.
    plotted_points = validation_points if validation_points else training_points
    series_name = "Validation BPB" if validation_points else "Training batch BPB"
    if not plotted_points:
        raise ValueError("Cannot plot an empty BPB history.")
    x_min = min(step for step, _ in plotted_points)
    x_max = max(step for step, _ in plotted_points)
    y_min = min(bpb for _, bpb in plotted_points)
    y_max = max(bpb for _, bpb in plotted_points)
    if x_min == x_max:
        x_min, x_max = 0, max(1, x_max)
    if y_min == y_max:
        padding = max(abs(y_min) * 0.05, 0.1)
    else:
        padding = (y_max - y_min) * 0.08
    y_min -= padding
    y_max += padding

    def chart_point(point):
        step, bpb = point
        x = plot_left + (step - x_min) / (x_max - x_min) * (plot_right - plot_left)
        y = plot_bottom - (bpb - y_min) / (y_max - y_min) * (plot_bottom - plot_top)
        return round(x), round(y)

    for tick in range(6):
        fraction = tick / 5
        x = round(plot_left + fraction * (plot_right - plot_left))
        y = round(plot_bottom - fraction * (plot_bottom - plot_top))
        draw.line((x, plot_top, x, plot_bottom), fill="#e3e7eb", width=1)
        draw.line((plot_left, y, plot_right, y), fill="#e3e7eb", width=1)
        step_label = f"{round(x_min + fraction * (x_max - x_min)):,}"
        bpb_label = f"{y_min + fraction * (y_max - y_min):.4f}"
        draw.text(
            (x - 28, plot_bottom + 14), step_label, fill="#4a5560", font=small_font
        )
        draw.text((plot_left - 80, y - 9), bpb_label, fill="#4a5560", font=small_font)

    draw.line((plot_left, plot_top, plot_left, plot_bottom), fill="#202830", width=2)
    draw.line(
        (plot_left, plot_bottom, plot_right, plot_bottom), fill="#202830", width=2
    )

    if not validation_points:
        points = [chart_point(point) for point in plotted_points]
        if len(points) > 1:
            draw.line(points, fill="#8fb9df", width=3)
        for x, y in points:
            draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill="#8fb9df")
    if validation_points:
        points = [chart_point(point) for point in validation_points]
        if len(points) > 1:
            draw.line(points, fill="#d1495b", width=5)
        for x, y in points:
            draw.ellipse((x - 7, y - 7, x + 7, y + 7), fill="#d1495b")
        best_point = min(validation_points, key=lambda point: point[1])
        best_x, best_y = chart_point(best_point)
        draw.ellipse(
            (best_x - 11, best_y - 11, best_x + 11, best_y + 11),
            fill="#2a9d8f",
            outline="white",
            width=3,
        )
        recent_values = [bpb for _, bpb in validation_points[-5:]]
        recent_change = recent_values[0] - recent_values[-1]
        draw.text(
            (plot_right - 310, plot_top + 12),
            f"Recent BPB improvement: {recent_change:.6f}",
            fill="#202830",
            font=small_font,
        )

    draw.text(
        (width // 2 - 145, 38),
        "Validation BPB vs. Training Steps",
        fill="#202830",
        font=title_font,
    )
    draw.text(
        (width // 2 - 55, height - 55), "Training step", fill="#202830", font=font
    )
    draw.text(
        (25, plot_top - 40),
        "Bits per byte (lower is better)",
        fill="#202830",
        font=font,
    )
    legend_y = height - 95
    series_color = "#d1495b" if validation_points else "#8fb9df"
    draw.line(
        (plot_left, legend_y, plot_left + 35, legend_y), fill=series_color, width=5
    )
    draw.text((plot_left + 45, legend_y - 10), series_name, fill="#202830", font=font)
    if validation_points:
        best_bpb = min(bpb for _, bpb in validation_points)
        draw.ellipse(
            (plot_left + 235, legend_y - 7, plot_left + 249, legend_y + 7),
            fill="#2a9d8f",
        )
        draw.text(
            (plot_left + 260, legend_y - 10),
            f"Best validation BPB: {best_bpb:.6f}",
            fill="#202830",
            font=font,
        )

    temporary = path.with_suffix(path.suffix + ".tmp")
    image.save(temporary, format="PNG")
    os.replace(temporary, path)


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n"
    )
    os.replace(temporary, path)


def load_json(path):
    return json.loads(Path(path).read_text())


def git_info():
    from .config import PROJECT_ROOT

    def query(*args):
        return subprocess.check_output(
            ["git", *args], cwd=PROJECT_ROOT, text=True
        ).strip()

    return {
        "commit": query("rev-parse", "HEAD"),
        "dirty": bool(query("status", "--porcelain")),
    }

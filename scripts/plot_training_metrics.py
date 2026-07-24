#!/usr/bin/env python3
"""Plot SaMoye training metrics from text logs and TensorBoard events.

The trainer logs the main loss terms as text and writes validation/score
metrics to TensorBoard. This script combines both sources when available.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path


LOG_RE = re.compile(
    r"(?P<time>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d+)"
    r".*?epoch (?P<epoch>\d+) \| "
    r"g (?P<g_loss>[-+0-9.eE]+) "
    r"m (?P<mel_loss>[-+0-9.eE]+) "
    r"s (?P<stft_loss>[-+0-9.eE]+) "
    r"d (?P<d_loss>[-+0-9.eE]+) "
    r"k (?P<kl_f_loss>[-+0-9.eE]+) "
    r"r (?P<kl_r_loss>[-+0-9.eE]+) "
    r"i (?P<spk_loss>[-+0-9.eE]+) \| "
    r"step (?P<step>\d+)"
)

TRAIN_METRICS = [
    "g_loss",
    "d_loss",
    "score_loss",
    "mel_loss",
    "stft_loss",
    "kl_f_loss",
    "kl_r_loss",
    "spk_loss",
]

TB_TAGS = {
    "train/g_loss": "g_loss",
    "train/d_loss": "d_loss",
    "train/score_loss": "score_loss",
    "train/mel_loss": "mel_loss",
    "train/stft_loss": "stft_loss",
    "train/kl_f_loss": "kl_f_loss",
    "train/kl_r_loss": "kl_r_loss",
    "validation/mel_loss": "validation_mel_loss",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-name",
        required=True,
        help="Experiment name, e.g. sovits_spk_expand_v1_core.",
    )
    parser.add_argument(
        "--model-dir",
        default="SaMoye-Model",
        help="Directory containing logs/, train logs, and trainer outputs.",
    )
    parser.add_argument(
        "--log",
        action="append",
        default=[],
        help="Text training log path. Can be repeated. Defaults to latest logs/<run>/*.log.",
    )
    parser.add_argument(
        "--event-dir",
        default=None,
        help="TensorBoard event directory. Defaults to <model-dir>/logs/<run-name>.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/training_plots",
        help="Directory where plots and CSV summaries are written.",
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=101,
        help="Moving average window for raw step curves.",
    )
    return parser.parse_args()


def resolve_path(path: str | Path, base: Path) -> Path:
    p = Path(path)
    if p.is_absolute() or p.exists():
        return p
    candidate = base / p
    if candidate.exists():
        return candidate
    return p


def discover_logs(model_dir: Path, run_name: str) -> list[Path]:
    run_dir = model_dir / "logs" / run_name
    logs = sorted(run_dir.glob(f"{run_name}-*.log"), key=lambda p: p.stat().st_mtime)
    if logs:
        return [logs[-1]]

    candidates = [
        model_dir / f"train_{run_name}.log",
        model_dir / "train_v1_core.log",
        model_dir / "train_v1_core_continue.log",
        model_dir / "train_v2_mix.log",
    ]
    return [p for p in candidates if p.exists()]


def parse_text_logs(log_paths: list[Path]) -> list[dict[str, float | int | str]]:
    rows = []
    seen = set()
    for log_path in log_paths:
        if not log_path.exists():
            print(f"[plot] skip missing log: {log_path}")
            continue
        with log_path.open("r", errors="ignore") as f:
            for line in f:
                for match in LOG_RE.finditer(line):
                    data = match.groupdict()
                    step = int(data["step"])
                    key = (log_path, step, int(data["epoch"]))
                    if key in seen:
                        continue
                    seen.add(key)
                    row: dict[str, float | int | str] = {
                        "source": str(log_path),
                        "timestamp": data["time"],
                        "epoch": int(data["epoch"]),
                        "step": step,
                    }
                    for metric in (
                        "g_loss",
                        "mel_loss",
                        "stft_loss",
                        "d_loss",
                        "kl_f_loss",
                        "kl_r_loss",
                        "spk_loss",
                    ):
                        row[metric] = float(data[metric])
                    rows.append(row)
    rows.sort(key=lambda r: (int(r["step"]), int(r["epoch"])))
    add_elapsed_and_throughput(rows)
    return rows


def add_elapsed_and_throughput(rows: list[dict[str, float | int | str]]) -> None:
    if not rows:
        return
    parsed_times: list[datetime | None] = []
    for row in rows:
        try:
            parsed_times.append(datetime.strptime(str(row["timestamp"]), "%Y-%m-%d %H:%M:%S,%f"))
        except ValueError:
            parsed_times.append(None)

    start = next((t for t in parsed_times if t is not None), None)
    prev_t = None
    prev_step = None
    for row, t in zip(rows, parsed_times):
        if t is not None and start is not None:
            row["elapsed_hours"] = (t - start).total_seconds() / 3600.0
        else:
            row["elapsed_hours"] = ""

        sec_per_step = ""
        if t is not None and prev_t is not None and prev_step is not None:
            step_delta = int(row["step"]) - prev_step
            sec_delta = (t - prev_t).total_seconds()
            if step_delta > 0 and 0 < sec_delta < 300:
                sec_per_step = sec_delta / step_delta
        row["sec_per_step"] = sec_per_step

        if t is not None:
            prev_t = t
            prev_step = int(row["step"])


def read_tensorboard_scalars(event_dir: Path) -> dict[str, dict[int, float]]:
    scalars: dict[str, dict[int, float]] = defaultdict(dict)
    if not event_dir.exists():
        return scalars
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except Exception as exc:  # pragma: no cover - optional dependency
        print(f"[plot] TensorBoard unavailable, skip events: {exc}")
        return scalars

    event_files = sorted(event_dir.glob("events.out.tfevents.*"), key=lambda p: p.stat().st_mtime)
    for event_file in event_files:
        try:
            acc = EventAccumulator(str(event_file), size_guidance={"scalars": 0})
            acc.Reload()
        except Exception as exc:
            print(f"[plot] skip unreadable event file {event_file}: {exc}")
            continue
        for tag in acc.Tags().get("scalars", []):
            metric = TB_TAGS.get(tag)
            if metric is None:
                continue
            for item in acc.Scalars(tag):
                scalars[metric][int(item.step)] = float(item.value)
    return scalars


def merge_tensorboard(rows: list[dict[str, float | int | str]], scalars: dict[str, dict[int, float]]) -> None:
    by_step = {int(row["step"]): row for row in rows}
    for metric, values in scalars.items():
        if metric.startswith("validation_"):
            continue
        for step, value in values.items():
            if step in by_step and metric not in by_step[step]:
                by_step[step][metric] = value


def validation_rows(scalars: dict[str, dict[int, float]]) -> list[dict[str, float | int]]:
    values = scalars.get("validation_mel_loss", {})
    return [{"step": step, "validation_mel_loss": value} for step, value in sorted(values.items())]


def mean(values: list[float]) -> float:
    clean = [v for v in values if not math.isnan(v)]
    if not clean:
        return float("nan")
    return sum(clean) / len(clean)


def epoch_summary(rows: list[dict[str, float | int | str]]) -> list[dict[str, float | int]]:
    grouped: dict[int, list[dict[str, float | int | str]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["epoch"])].append(row)
    summary = []
    for epoch in sorted(grouped):
        items = grouped[epoch]
        out: dict[str, float | int] = {
            "epoch": epoch,
            "n_steps": len(items),
            "first_step": int(items[0]["step"]),
            "last_step": int(items[-1]["step"]),
        }
        for metric in TRAIN_METRICS + ["sec_per_step"]:
            vals = []
            for item in items:
                value = item.get(metric, "")
                if value == "":
                    continue
                vals.append(float(value))
            if vals:
                out[metric] = mean(vals)
        summary.append(out)
    return summary


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    keys = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def to_xy(rows: list[dict], x_key: str, y_key: str) -> tuple[list[float], list[float]]:
    xs, ys = [], []
    for row in rows:
        if y_key not in row or row[y_key] == "":
            continue
        try:
            y = float(row[y_key])
            x = float(row[x_key])
        except (TypeError, ValueError):
            continue
        if math.isfinite(x) and math.isfinite(y):
            xs.append(x)
            ys.append(y)
    return xs, ys


def moving_average(values: list[float], window: int) -> list[float]:
    if window <= 1 or len(values) < 3:
        return values
    window = min(window, len(values))
    if window % 2 == 0:
        window -= 1
    half = window // 2
    smoothed = []
    for idx in range(len(values)):
        lo = max(0, idx - half)
        hi = min(len(values), idx + half + 1)
        smoothed.append(mean(values[lo:hi]))
    return smoothed


def setup_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 180,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 10,
        }
    )
    return plt


def plot_line(ax, rows: list[dict], x_key: str, y_key: str, label: str, smooth_window: int) -> bool:
    xs, ys = to_xy(rows, x_key, y_key)
    if not xs:
        return False
    if len(xs) > 400:
        ax.plot(xs, ys, alpha=0.18, linewidth=0.8)
    ax.plot(xs, moving_average(ys, smooth_window), linewidth=1.8, label=label)
    return True


def plot_overview(
    rows: list[dict],
    validation: list[dict],
    out_path: Path,
    run_name: str,
    smooth_window: int,
) -> None:
    plt = setup_matplotlib()
    fig, axes = plt.subplots(4, 2, figsize=(14, 14), constrained_layout=True)
    panels = [
        ("g_loss", "Generator Total Loss"),
        ("d_loss", "Discriminator Loss"),
        ("mel_loss", "Train Mel L1 Loss"),
        ("stft_loss", "Train Multi-Resolution STFT Loss"),
        ("kl_f_loss", "KL Forward Loss"),
        ("kl_r_loss", "KL Reverse Loss"),
        ("spk_loss", "Speaker Cosine Loss"),
        ("sec_per_step", "Seconds Per Step"),
    ]

    for ax, (metric, title) in zip(axes.flat, panels):
        plotted = plot_line(ax, rows, "step", metric, metric, smooth_window)
        if metric == "d_loss":
            plot_line(ax, rows, "step", "score_loss", "score_loss", smooth_window)
        if metric == "mel_loss" and validation:
            twin = ax.twinx()
            plot_line(twin, validation, "step", "validation_mel_loss", "validation_mel_loss", 1)
            twin.set_ylabel("validation_mel_loss")
            twin.legend(loc="upper right", frameon=False)
        if plotted or metric == "mel_loss":
            ax.legend(loc="best", frameon=False)
        ax.set_title(title)
        ax.set_xlabel("step")
        ax.set_ylabel(metric)

    fig.suptitle(f"Training Metrics: {run_name}", fontsize=14)
    fig.savefig(out_path)
    plt.close(fig)


def plot_epoch_summary(summary: list[dict], out_path: Path, run_name: str) -> None:
    if not summary:
        return
    plt = setup_matplotlib()
    fig, axes = plt.subplots(3, 2, figsize=(13, 10), constrained_layout=True)
    panels = [
        (["g_loss"], "Generator Loss"),
        (["d_loss", "score_loss"], "Discriminator / Score"),
        (["mel_loss"], "Mel Loss"),
        (["stft_loss"], "STFT Loss"),
        (["kl_f_loss", "kl_r_loss"], "KL Losses"),
        (["spk_loss"], "Speaker Loss"),
    ]
    for ax, (metrics, title) in zip(axes.flat, panels):
        for metric in metrics:
            xs, ys = to_xy(summary, "epoch", metric)
            if xs:
                ax.plot(xs, ys, marker="o", linewidth=1.8, label=metric)
        ax.set_title(title)
        ax.set_xlabel("epoch")
        ax.legend(loc="best", frameon=False)
    fig.suptitle(f"Epoch Mean Metrics: {run_name}", fontsize=14)
    fig.savefig(out_path)
    plt.close(fig)


def write_summary(path: Path, run_name: str, rows: list[dict], epochs: list[dict], validation: list[dict]) -> None:
    lines = [f"run_name={run_name}", f"train_points={len(rows)}", f"epochs={len(epochs)}"]
    if rows:
        lines.append(f"first_step={rows[0]['step']}")
        lines.append(f"last_step={rows[-1]['step']}")
    if epochs:
        first = epochs[0]
        last = epochs[-1]
        for metric in ["g_loss", "mel_loss", "stft_loss", "d_loss", "kl_f_loss", "kl_r_loss", "spk_loss"]:
            if metric in first and metric in last:
                delta = float(last[metric]) - float(first[metric])
                lines.append(
                    f"{metric}: first_epoch={float(first[metric]):.6f} "
                    f"last_epoch={float(last[metric]):.6f} delta={delta:.6f}"
                )
    if validation:
        lines.append(
            f"validation_mel_loss: first={validation[0]['validation_mel_loss']:.6f} "
            f"last={validation[-1]['validation_mel_loss']:.6f} "
            f"best={min(float(v['validation_mel_loss']) for v in validation):.6f}"
        )
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    args = parse_args()
    model_dir = Path(args.model_dir)
    log_paths = [resolve_path(p, model_dir) for p in args.log]
    if not log_paths:
        log_paths = discover_logs(model_dir, args.run_name)
    event_dir = Path(args.event_dir) if args.event_dir else model_dir / "logs" / args.run_name
    out_dir = Path(args.output_dir) / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = parse_text_logs(log_paths)
    scalars = read_tensorboard_scalars(event_dir)
    merge_tensorboard(rows, scalars)
    validation = validation_rows(scalars)
    epochs = epoch_summary(rows)

    if not rows and not validation:
        print(f"[plot] no metrics found for {args.run_name}")
        return 2

    write_csv(out_dir / "raw_metrics.csv", rows)
    write_csv(out_dir / "epoch_metrics.csv", epochs)
    write_csv(out_dir / "validation_metrics.csv", validation)
    write_summary(out_dir / "summary.txt", args.run_name, rows, epochs, validation)

    if rows:
        plot_overview(rows, validation, out_dir / "metrics_overview.png", args.run_name, args.smooth_window)
        plot_epoch_summary(epochs, out_dir / "epoch_metrics.png", args.run_name)

    print(f"[plot] wrote training plots to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

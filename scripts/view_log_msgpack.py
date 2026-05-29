import argparse
import os
import time
from collections.abc import Mapping, Sequence
from pathlib import Path

import msgpack
import msgpack_numpy
import numpy as np

msgpack_numpy.patch()

PLOT_CHOICES = ("all", "velocity", "joint-target", "joint-velocity", "base")
DEFAULT_JOINT_PLOT_COUNT = 6


def get_latest_folder(path, index=-1):
    folder_list = os.listdir(path)
    folder_list = [os.path.join(path, f) for f in folder_list]
    folder_list = list(filter(lambda f: os.path.isdir(f), folder_list))
    folder_list.sort(key=lambda f: os.path.getmtime(f))
    return folder_list[index]


def read_msgpack(folder_path):
    log_files = sorted([f for f in os.listdir(folder_path) if f.endswith(".msgpack")])
    if not log_files:
        exit()
    log_file = log_files[0]
    path = os.path.join(folder_path, log_file)

    records = []
    with open(path, "rb") as f:
        unpacker = msgpack.Unpacker(f, raw=False)
        for obj in unpacker:
            records.append(obj)
    return records


def _get_nested(data, path):
    value = data
    for key in path:
        if isinstance(value, Mapping):
            value = value.get(key)
        else:
            value = getattr(value, key, None)
        if value is None:
            return None
    return value


def _as_numeric_array(value):
    if value is None:
        return None
    try:
        arr = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return None
    if arr.ndim == 0:
        arr = arr.reshape(1)
    return arr.reshape(-1)


def _frame_value(frame, paths, index):
    for path in paths:
        arr = _as_numeric_array(_get_nested(frame, path))
        if arr is None or index >= arr.size:
            continue
        return float(arr[index])
    return np.nan


def _series_from_paths(frames, paths, index):
    return np.asarray([_frame_value(frame, paths, index) for frame in frames], dtype=float)


def _has_data(values):
    return np.any(np.isfinite(values))


def build_time_axis(frames):
    if not frames:
        return np.asarray([], dtype=float)

    times = _series_from_paths(frames, [("time",)], 0)
    if _has_data(times):
        first_time = times[np.isfinite(times)][0]
        return times - first_time

    timesteps = _series_from_paths(frames, [("timestep",)], 0)
    if _has_data(timesteps):
        return timesteps

    return np.arange(len(frames), dtype=float)


def _default_joint_indices(frames):
    for frame in frames:
        for path in (("env_data", "dof_pos"), ("pd_target",), ("env_data", "dof_vel")):
            arr = _as_numeric_array(_get_nested(frame, path))
            if arr is not None:
                return list(range(min(DEFAULT_JOINT_PLOT_COUNT, arr.size)))
    return []


def build_velocity_tracking_curves(frames):
    x = build_time_axis(frames)
    command_paths = [
        ("extras", "commands"),
        ("extras", "velocity_commands"),
        ("extras", "command"),
    ]
    curves = {
        "cmd_vx": _series_from_paths(frames, command_paths, 0),
        "actual_vx": _series_from_paths(frames, [("env_data", "base_lin_vel")], 0),
        "cmd_vy": _series_from_paths(frames, command_paths, 1),
        "actual_vy": _series_from_paths(frames, [("env_data", "base_lin_vel")], 1),
        "cmd_yaw": _series_from_paths(frames, command_paths, 2),
        "actual_yaw": _series_from_paths(
            frames,
            [("env_data", "torso_ang_vel"), ("env_data", "base_ang_vel")],
            2,
        ),
    }
    return x, curves


def build_joint_target_curves(frames, joint_indices: Sequence[int] | None = None):
    x = build_time_axis(frames)
    if joint_indices is None:
        joint_indices = _default_joint_indices(frames)

    curves = {}
    for joint_idx in joint_indices:
        curves[f"joint_{joint_idx}_pos"] = _series_from_paths(frames, [("env_data", "dof_pos")], joint_idx)
        curves[f"joint_{joint_idx}_target"] = _series_from_paths(frames, [("pd_target",)], joint_idx)
    return x, curves


def build_joint_velocity_curves(frames, joint_indices: Sequence[int] | None = None):
    x = build_time_axis(frames)
    if joint_indices is None:
        joint_indices = _default_joint_indices(frames)

    curves = {}
    for joint_idx in joint_indices:
        curves[f"joint_{joint_idx}_vel"] = _series_from_paths(frames, [("env_data", "dof_vel")], joint_idx)
    return x, curves


def build_base_motion_curves(frames):
    x = build_time_axis(frames)
    curves = {
        "base_vx": _series_from_paths(frames, [("env_data", "base_lin_vel")], 0),
        "base_vy": _series_from_paths(frames, [("env_data", "base_lin_vel")], 1),
        "base_vz": _series_from_paths(frames, [("env_data", "base_lin_vel")], 2),
        "base_roll_rate": _series_from_paths(frames, [("env_data", "base_ang_vel")], 0),
        "base_pitch_rate": _series_from_paths(frames, [("env_data", "base_ang_vel")], 1),
        "base_yaw_rate": _series_from_paths(frames, [("env_data", "base_ang_vel")], 2),
    }
    return x, curves


def _plot_curve_group(ax, x, curves, title, ylabel):
    plotted = False
    for label, values in curves.items():
        if not _has_data(values):
            continue
        ax.plot(x, values, label=label)
        plotted = True
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(True)
    if plotted:
        ax.legend(loc="best")
    else:
        ax.text(0.5, 0.5, "No numeric data", ha="center", va="center", transform=ax.transAxes)
    return plotted


def _ensure_matplotlib_config_dir():
    if "MPLCONFIGDIR" in os.environ:
        return
    default_config_dir = Path.home() / ".config" / "matplotlib"
    if default_config_dir.exists() and os.access(default_config_dir, os.W_OK):
        return
    fallback_config_dir = Path("/tmp") / "robojudo-matplotlib"
    fallback_config_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(fallback_config_dir)


def plot_velocity_tracking(frames):
    _ensure_matplotlib_config_dir()
    import matplotlib.pyplot as plt

    x, curves = build_velocity_tracking_curves(frames)
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    groups = [
        ("Velocity X Tracking", "m/s", {"cmd_vx": curves["cmd_vx"], "actual_vx": curves["actual_vx"]}),
        ("Velocity Y Tracking", "m/s", {"cmd_vy": curves["cmd_vy"], "actual_vy": curves["actual_vy"]}),
        ("Yaw Rate Tracking", "rad/s", {"cmd_yaw": curves["cmd_yaw"], "actual_yaw": curves["actual_yaw"]}),
    ]
    for ax, (title, ylabel, group_curves) in zip(axes, groups, strict=True):
        _plot_curve_group(ax, x, group_curves, title, ylabel)
    axes[-1].set_xlabel("Time (s)")
    fig.tight_layout()
    return fig


def plot_joint_targets(frames, joint_indices: Sequence[int] | None = None):
    _ensure_matplotlib_config_dir()
    import matplotlib.pyplot as plt

    x, curves = build_joint_target_curves(frames, joint_indices)
    fig, ax = plt.subplots(figsize=(12, 6))
    _plot_curve_group(ax, x, curves, "Joint Position vs PD Target", "rad")
    ax.set_xlabel("Time (s)")
    fig.tight_layout()
    return fig


def plot_joint_velocities(frames, joint_indices: Sequence[int] | None = None):
    _ensure_matplotlib_config_dir()
    import matplotlib.pyplot as plt

    x, curves = build_joint_velocity_curves(frames, joint_indices)
    fig, ax = plt.subplots(figsize=(12, 6))
    _plot_curve_group(ax, x, curves, "Joint Velocities", "rad/s")
    ax.set_xlabel("Time (s)")
    fig.tight_layout()
    return fig


def plot_base_motion(frames):
    _ensure_matplotlib_config_dir()
    import matplotlib.pyplot as plt

    x, curves = build_base_motion_curves(frames)
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    lin_curves = {name: values for name, values in curves.items() if name.startswith("base_v")}
    ang_curves = {name: values for name, values in curves.items() if name.endswith("_rate")}
    _plot_curve_group(axes[0], x, lin_curves, "Base Linear Velocity", "m/s")
    _plot_curve_group(axes[1], x, ang_curves, "Base Angular Velocity", "rad/s")
    axes[-1].set_xlabel("Time (s)")
    fig.tight_layout()
    return fig


def _normalize_plot_names(plot_names):
    names = set(plot_names)
    if "all" in names:
        return {"velocity", "joint-target", "joint-velocity", "base"}
    return names


def _save_figures(figures, save_dir):
    if save_dir is None:
        return
    output_dir = Path(save_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, fig in figures:
        fig.savefig(output_dir / f"{name}.png", dpi=160)


def view_log(folder_path):
    from robojudo.config.g1.env.g1_dummy_env_cfg import G1DummyEnvCfg
    from robojudo.environment.dummy_env import DummyEnv
    from robojudo.tools.tool_cfgs import ForwardKinematicCfg

    log_frames = read_msgpack(folder_path)
    log_frames = log_frames[:]

    fk_cfg = ForwardKinematicCfg(
        xml_path=G1DummyEnvCfg.model_fields["xml"].default,
        debug_viz=True,
    )
    env_cfg = G1DummyEnvCfg(forward_kinematic=fk_cfg, odometry_type="DUMMY")

    env = DummyEnv(cfg_env=env_cfg)

    for _i, log_frame in enumerate(log_frames):
        env_data = log_frame["env_data"]
        ctrl_data = log_frame["ctrl_data"]
        extras = log_frame["extras"]
        pd_target = log_frame["pd_target"]
        timestep = log_frame["timestep"]
        time_then = log_frame["time"]

        joint_pos = env_data["dof_pos"]
        base_pos = env_data["base_pos"]
        base_quat = env_data["base_quat"]
        env.kinematics.forward(
            joint_pos=joint_pos,
            base_pos=base_pos,
            base_quat=base_quat,
        )
        for command in ctrl_data.get("COMMANDS", []):
            print("----->" + command)
        print(f"Step {timestep}")
        time.sleep(0.005)


def plot_log(
    folder_path,
    plot_names: Sequence[str] = ("all",),
    start: int = 0,
    end: int | None = None,
    joint_indices: Sequence[int] | None = None,
    save_dir: str | None = None,
    show: bool = True,
):
    log_frames = read_msgpack(folder_path)
    log_frames = log_frames[start:end]

    _ensure_matplotlib_config_dir()
    import matplotlib.pyplot as plt

    figures = []
    selected_plots = _normalize_plot_names(plot_names)
    if "velocity" in selected_plots:
        figures.append(("velocity_tracking", plot_velocity_tracking(log_frames)))
    if "joint-target" in selected_plots:
        figures.append(("joint_targets", plot_joint_targets(log_frames, joint_indices)))
    if "joint-velocity" in selected_plots:
        figures.append(("joint_velocities", plot_joint_velocities(log_frames, joint_indices)))
    if "base" in selected_plots:
        figures.append(("base_motion", plot_base_motion(log_frames)))

    _save_figures(figures, save_dir)
    if show:
        plt.show()
    else:
        for _, fig in figures:
            plt.close(fig)
    return figures


def parse_args():
    parser = argparse.ArgumentParser(description="View and plot RoboJuDo msgpack debug logs.")
    parser.add_argument(
        "folder",
        nargs="?",
        default=None,
        help="Log folder containing log.msgpack. Defaults to the latest folder under logs/.",
    )
    parser.add_argument(
        "--plot",
        nargs="+",
        choices=PLOT_CHOICES,
        default=["all"],
        help="Curve groups to plot.",
    )
    parser.add_argument("--start", type=int, default=0, help="Start frame index.")
    parser.add_argument("--end", type=int, default=None, help="End frame index.")
    parser.add_argument(
        "--joint-indices",
        type=int,
        nargs="+",
        default=None,
        help="Joint indices for joint-target and joint-velocity plots. Defaults to first 6 joints.",
    )
    parser.add_argument("--save-dir", default=None, help="Directory to save plotted PNG files.")
    parser.add_argument("--no-show", action="store_true", help="Save or build plots without opening windows.")
    parser.add_argument("--view", action="store_true", help="Replay the log in DummyEnv after plotting.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    folder_path = args.folder or get_latest_folder("logs", -1)
    print(folder_path)
    plot_log(
        folder_path,
        plot_names=args.plot,
        start=args.start,
        end=args.end,
        joint_indices=args.joint_indices,
        save_dir=args.save_dir,
        show=not args.no_show,
    )
    if args.view:
        view_log(folder_path)

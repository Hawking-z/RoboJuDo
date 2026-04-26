import logging
from typing import Any

import cv2
import numpy as np

from robojudo.environment.utils.mujoco_viz import MujocoVisualizer
from robojudo.policy import policy_registry
from robojudo.policy.custom_policy import CustomPolicy
from robojudo.policy.policy_cfgs import ParkourPolicyCfg
from robojudo.utils.util_func import get_gravity_orientation

logger = logging.getLogger(__name__)


@policy_registry.register
class ParkourPolicy(CustomPolicy):
    cfg_policy: ParkourPolicyCfg

    def __init__(self, cfg_policy: ParkourPolicyCfg, device):
        self.depth_sensor_name = cfg_policy.depth_sensor_name
        self.depth_input_name = cfg_policy.depth_input_name
        self.depth_source_name = cfg_policy.depth_source_name
        self.depth_resolution = tuple(int(v) for v in cfg_policy.depth_resolution)
        self.depth_range = np.asarray(cfg_policy.depth_range, dtype=np.float32)
        self.depth_output_range = np.asarray(cfg_policy.depth_output_range, dtype=np.float32)
        self.depth_invalid_threshold = float(cfg_policy.depth_invalid_threshold)
        self.depth_inpaint_radius = float(cfg_policy.depth_inpaint_radius)
        self.depth_crop_region = tuple(int(v) for v in cfg_policy.depth_crop_region)
        self.depth_blind_spot_crop = tuple(int(v) for v in cfg_policy.depth_blind_spot_crop)
        self.depth_gaussian_kernel_size = int(cfg_policy.depth_gaussian_kernel_size)
        self.depth_gaussian_sigma = float(cfg_policy.depth_gaussian_sigma)
        super().__init__(cfg_policy=cfg_policy, device=device)

        if self.depth_input_name not in self.ort_input_names:
            raise RuntimeError(
                f"ParkourPolicy expected ONNX depth input '{self.depth_input_name}', "
                f"got inputs: {self.ort_input_names}"
            )

    def _get_raw_depth(self, env_data) -> np.ndarray:
        return np.asarray(self._env_value(env_data, self.depth_source_name), dtype=np.float32)

    def _process_depth(self, env_data) -> np.ndarray:
        raw_depth = self._get_raw_depth(env_data)
        if raw_depth.ndim == 3 and raw_depth.shape[-1] == 1:
            raw_depth = raw_depth[..., 0]
        if raw_depth.ndim != 2:
            raise ValueError(
                f"ParkourPolicy depth source '{self.depth_source_name}' must be a 2D depth image, "
                f"got shape {raw_depth.shape}."
            )

        depth = cv2.resize(raw_depth, self.depth_resolution, interpolation=cv2.INTER_NEAREST)
        if any(self.depth_crop_region):
            top, bottom, left, right = self.depth_crop_region
            h, w = depth.shape
            depth = depth[top : h - bottom, left : w - right]

        invalid_mask = ((depth < self.depth_invalid_threshold) | ~np.isfinite(depth)).astype(np.uint8)
        if np.any(invalid_mask):
            depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
            depth = cv2.inpaint(depth, invalid_mask, self.depth_inpaint_radius, cv2.INPAINT_NS)

        if any(self.depth_blind_spot_crop):
            top, bottom, left, right = self.depth_blind_spot_crop
            h, w = depth.shape
            depth[:top, :] = 0.0
            depth[h - bottom :, :] = 0.0
            depth[:, :left] = 0.0
            depth[:, w - right :] = 0.0

        if self.depth_gaussian_kernel_size > 0:
            kernel = self.depth_gaussian_kernel_size
            if kernel % 2 == 0:
                kernel += 1
            depth = cv2.GaussianBlur(depth, (kernel, kernel), self.depth_gaussian_sigma, self.depth_gaussian_sigma)

        depth = np.clip(depth, self.depth_range[0], self.depth_range[1])
        depth = (depth - self.depth_range[0]) / max(self.depth_range[1] - self.depth_range[0], 1e-6)
        depth = depth * (self.depth_output_range[1] - self.depth_output_range[0]) + self.depth_output_range[0]
        return depth.astype(np.float32, copy=False)

    def _sensor_inputs(self, env_data, ctrl_data):
        commands = self._get_commands(ctrl_data) * self.max_cmd
        clock_phase = self._get_clock_phase()

        def projected_gravity():
            return get_gravity_orientation(self._env_value(env_data, "base_quat"))

        def height_scan():
            base_height = self._env_value(env_data, "base_pos")[2]
            scan = self._env_value(env_data, "height_scan")
            return np.clip(base_height - scan - 0.77, -1.0, 1.0)

        value_getters: dict[str, Any] = {
            "base_ang_vel": lambda: self._env_value(env_data, "base_ang_vel"),
            "projected_gravity": projected_gravity,
            "base_gravity": projected_gravity,
            "gravity": projected_gravity,
            "command_lin_vel": lambda: commands[:2],
            "command_ang_vel": lambda: commands[2:3],
            "commands": lambda: commands,
            "velocity_commands": lambda: commands,
            "command_stand": lambda: self.command_stand,
            "clock_phase": lambda: clock_phase,
            "dof_pos": lambda: self._env_value(env_data, "dof_pos") - self.default_dof_pos,
            "joint_pos": lambda: self._env_value(env_data, "dof_pos") - self.default_dof_pos,
            "dof_vel": lambda: self._env_value(env_data, "dof_vel"),
            "joint_vel": lambda: self._env_value(env_data, "dof_vel"),
            "actions": lambda: self.last_action,
            "last_action": lambda: self.last_action,
            "height_scan": height_scan,
            self.depth_sensor_name: lambda: self._process_depth(env_data),
        }

        inputs = {}
        for name, spec in self.robot_cfg.sensors.items():
            if name in value_getters:
                value = value_getters[name]()
            else:
                value = self._env_value(env_data, name)
            inputs[name] = np.asarray(value, dtype=np.float32).reshape(spec.shape)
        return inputs, commands, clock_phase

    def debug_viz(self, visualizer: MujocoVisualizer, env_data, ctrl_data, extras):
        super().debug_viz(visualizer, env_data, ctrl_data, extras)

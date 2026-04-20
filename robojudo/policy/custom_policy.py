import logging
from pathlib import Path

import numpy as np
import torch

from robojudo.environment.utils.mujoco_viz import MujocoVisualizer
from robojudo.policy import Policy, policy_registry
from robojudo.policy.policy_cfgs import CustomPolicyCfg
from robojudo.policy.utils.obs_assembler import ObsAssembler
from robojudo.policy.utils.robot_config import RobotConfig
from robojudo.utils.util_func import command_remap, get_gravity_orientation

logger = logging.getLogger(__name__)


@policy_registry.register
class CustomPolicy(Policy):
    cfg_policy: CustomPolicyCfg

    def __init__(self, cfg_policy: CustomPolicyCfg, device):
        if not cfg_policy.robot_config_file:
            raise FileNotFoundError("CustomPolicyCfg.robot_config_file must be set.")
        if not cfg_policy.disable_autoload and not cfg_policy.policy_file:
            raise FileNotFoundError("CustomPolicyCfg.model_file or model_dir must be set.")
        if not cfg_policy.disable_autoload and not Path(cfg_policy.policy_file).is_file():
            raise FileNotFoundError(f"Model file not found at {cfg_policy.policy_file}")

        self.robot_cfg = RobotConfig.from_yaml_file(cfg_policy.robot_config_file)
        self.model_backend = cfg_policy.model_backend
        self.obs_heads = cfg_policy.obs_heads
        if not self.obs_heads:
            self.obs_heads = list(self.robot_cfg.obs_map.keys())
        if not self.obs_heads:
            raise ValueError("robot_config obs_config must define at least one obs head.")

        self.obs_head = self.obs_heads[0]
        self.onnx_input_name = cfg_policy.onnx_input_name or self.obs_head
        self.onnx_output_name = cfg_policy.onnx_output_name

        if self.model_backend == "onnx" and not cfg_policy.disable_autoload:
            import onnxruntime as ort

            logger.debug(f"Loading ONNX policy '{cfg_policy.policy_name}' from {cfg_policy.policy_file}")
            self.session = ort.InferenceSession(cfg_policy.policy_file)
            self.input_names = [i.name for i in self.session.get_inputs()]
            self.output_names = [o.name for o in self.session.get_outputs()]
            cfg_policy = cfg_policy.model_copy(update={"disable_autoload": True})

        super().__init__(cfg_policy=cfg_policy, device=device)

        self.obs_assembler = self.robot_cfg.make_obs_assembler(ObsAssembler)
        self.obs_assembler.print_info()
        self.action_scale = np.asarray(self.cfg_policy.action_scale, dtype=np.float32)
        self.max_cmd = np.asarray(self.cfg_policy.max_cmd, dtype=np.float32)
        self.commands_map = self.cfg_policy.commands_map
        self.use_command_stand = (
            "command_stand" in self.robot_cfg.sensors
            and any("command_stand" in obs.sources for obs in self.robot_cfg.obs_map.values())
        )

        self.reset()

    def reset(self):
        self.timestep = 0
        self.last_action = np.zeros(self.num_actions, dtype=np.float32)
        self.command_stand = np.array([1.0 if self.use_command_stand else 0.0], dtype=np.float32)
        self.obs_assembler.reset()

    def post_step_callback(self, commands=None):
        self.timestep += 1

    def _get_commands(self, ctrl_data):
        commands = np.zeros(3, dtype=np.float32)
        for key in ctrl_data.keys():
            if key in ["JoystickCtrl", "UnitreeCtrl"]:
                axes = ctrl_data[key]["axes"]
                lx, ly, rx = axes["LeftX"], axes["LeftY"], axes["RightX"]

                commands[0] = command_remap(ly, self.commands_map[0])
                commands[1] = command_remap(lx, self.commands_map[1])
                commands[2] = command_remap(rx, self.commands_map[2])

                for event in ctrl_data[key].get("button_event", []):
                    if self.use_command_stand and event["type"] == "button" and event["pressed"]:
                        match event["name"]:
                            case "Left":
                                self.command_stand = 1.0 - self.command_stand
                break
            if key in ["KeyboardCtrl"]:
                keys = ctrl_data[key]["keyboard_event"]
                for event in keys:
                    if event["type"] != "keyboard":
                        continue
                    value = event["pressed"] * 1.5
                    match event["name"]:
                        case "w":
                            commands[0] = command_remap(value, self.commands_map[0])
                        case "s":
                            commands[0] = command_remap(-value, self.commands_map[0])
                        case "a":
                            commands[1] = command_remap(-value, self.commands_map[1])
                        case "d":
                            commands[1] = command_remap(value, self.commands_map[1])
                        case "e":
                            commands[2] = command_remap(value, self.commands_map[2])
                        case "q":
                            commands[2] = command_remap(-value, self.commands_map[2])
                        case "=" if self.use_command_stand and event["pressed"]:
                            self.command_stand = 1.0 - self.command_stand
                break
        if self.use_command_stand:
            commands *= 1.0 - self.command_stand[0]
        return commands

    def _get_clock_phase(self):
        gait_cfg = self.robot_cfg.gait_config
        cycle = float(gait_cfg.get("gait_cycle", 1.0))
        cycle = max(cycle, 1e-6)
        phase = (self.timestep * self.dt / cycle) % 1.0
        phase_l = (phase + float(gait_cfg.get("gait_phase_offset_l", 0.0))) % 1.0
        phase_r = (phase + float(gait_cfg.get("gait_phase_offset_r", 0.5))) % 1.0
        clock_phase = np.asarray(
            [
                np.sin(2 * np.pi * phase_l),
                np.sin(2 * np.pi * phase_r),
                np.cos(2 * np.pi * phase_l),
                np.cos(2 * np.pi * phase_r),
            ],
            dtype=np.float32,
        )
        return clock_phase * (1.0 - self.command_stand[0])

    def _env_value(self, env_data, name):
        if hasattr(env_data, name):
            return getattr(env_data, name)
        if isinstance(env_data, dict) and name in env_data:
            return env_data[name]
        raise KeyError(f"env_data missing required field '{name}'")

    def _sensor_inputs(self, env_data, ctrl_data):
        commands = self._get_commands(ctrl_data) * self.max_cmd
        clock_phase = self._get_clock_phase()

        def projected_gravity():
            return get_gravity_orientation(self._env_value(env_data, "base_quat"))

        value_getters = {
            "base_ang_vel": lambda: self._env_value(env_data, "base_ang_vel"),
            "projected_gravity": projected_gravity,
            "base_gravity": projected_gravity,
            "gravity": projected_gravity,
            "command_lin_vel": lambda: commands[:2],
            "command_ang_vel": lambda: commands[2:3],
            "commands": lambda: commands,
            "command_stand": lambda: self.command_stand,
            "clock_phase": lambda: clock_phase,
            "dof_pos": lambda: self._env_value(env_data, "dof_pos") - self.default_dof_pos,
            "dof_vel": lambda: self._env_value(env_data, "dof_vel"),
            "actions": lambda: self.last_action,
            "last_action": lambda: self.last_action,
        }

        inputs = {}
        for name, spec in self.robot_cfg.sensors.items():
            if name in value_getters:
                value = value_getters[name]()
            elif name.startswith("command_"):
                value = np.zeros(spec.shape, dtype=np.float32)
            else:
                value = self._env_value(env_data, name)
            inputs[name] = np.asarray(value, dtype=np.float32).reshape(spec.shape)
        return inputs, commands, clock_phase

    def get_observation(self, env_data, ctrl_data):
        inputs, commands, clock_phase = self._sensor_inputs(env_data, ctrl_data)
        obs_outputs = self.obs_assembler.step(inputs)
        obs = obs_outputs[self.obs_head].reshape(-1).astype(np.float32, copy=False)

        extras = {
            "commands": commands,
            "command_stand": self.command_stand.copy(),
            "clock_phase": clock_phase,
            "obs_head": self.obs_head,
            "obs_heads": self.obs_heads,
            "obs_outputs": obs_outputs,
        }
        return obs, extras

    def _process_actions(self, actions):
        actions = np.asarray(actions, dtype=np.float32).reshape(self.num_actions)
        if self.action_clip is not None:
            actions = np.clip(actions, -self.action_clip, self.action_clip)

        self.last_action = actions.copy()
        return actions * self.action_scale

    def get_action(self, obs: np.ndarray) -> np.ndarray:
        if self.model_backend == "onnx":
            if not hasattr(self, "session"):
                raise RuntimeError("ONNX session is not loaded. Set disable_autoload=False to run inference.")

            ort_inputs = {
                self.onnx_input_name: np.expand_dims(obs, axis=0).astype(np.float32),
            }
            output_names = None if self.onnx_output_name is None else [self.onnx_output_name]
            outputs = self.session.run(output_names, ort_inputs)
            return self._process_actions(np.asarray(outputs[0]).squeeze())

        obs_tensor = torch.from_numpy(obs).unsqueeze(0).float().to(self.device)
        with torch.no_grad():
            actions_tensor = self.model(obs_tensor).cpu()
        return self._process_actions(actions_tensor.numpy().squeeze())

    def debug_viz(self, visualizer: MujocoVisualizer, env_data, ctrl_data, extras):
        base_pos = self._env_value(env_data, "base_pos")
        base_quat = self._env_value(env_data, "base_quat")
        command_x = extras["commands"][0]
        command_y = extras["commands"][1]
        command_yaw = extras["commands"][2]

        visualizer.draw_arrow(
            base_pos,
            base_quat,
            [command_x, 0, 0],
            color=[1, 0, 0, 1],
            scale=2,
            horizontal_only=True,
            id=0,
        )
        visualizer.draw_arrow(
            base_pos,
            base_quat,
            [0, command_y, 0],
            color=[0, 1, 0, 1],
            scale=2,
            horizontal_only=True,
            id=1,
        )
        visualizer.draw_arrow(
            base_pos + np.array([0.0, 0, 0.6]),
            base_quat,
            [0, command_yaw, 0],
            color=[1, 1, 1, 1],
            scale=2,
            horizontal_only=True,
            id=2,
        )

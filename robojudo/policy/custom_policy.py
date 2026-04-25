import logging
from pathlib import Path
from typing import Any

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
        if not cfg_policy.policy_file:
            raise FileNotFoundError("CustomPolicyCfg.model_file or model_dir must be set.")
        if not Path(cfg_policy.policy_file).is_file():
            raise FileNotFoundError(f"Model file not found at {cfg_policy.policy_file}")

        self.robot_cfg = RobotConfig.from_yaml_file(cfg_policy.robot_config_file)
        self.model_backend = cfg_policy.model_backend
        all_obs_heads = list(self.robot_cfg.obs_map.keys())
        if not all_obs_heads:
            raise ValueError("robot_config obs_config must define at least one obs head.")
        self.deploy_obs_heads = list(cfg_policy.deploy_obs_heads or self.robot_cfg.deploy_obs_heads)
        if not self.deploy_obs_heads:
            raise ValueError("CustomPolicyCfg.deploy_obs_heads must define at least one deploy obs head.")
        missing_deploy_obs = [name for name in self.deploy_obs_heads if name not in self.robot_cfg.obs_map]
        if missing_deploy_obs:
            raise ValueError(
                f"CustomPolicyCfg.deploy_obs_heads contains unknown obs heads: {', '.join(missing_deploy_obs)}"
            )

        if self.model_backend == "onnx":
            import onnxruntime as ort

            logger.debug(f"Loading ONNX policy '{cfg_policy.policy_name}' from {cfg_policy.policy_file}")
            self.ort_session = ort.InferenceSession(cfg_policy.policy_file)
            self.ort_input_names = [i.name for i in self.ort_session.get_inputs()]
            self.ort_output_names = [o.name for o in self.ort_session.get_outputs()]
            if not self.ort_input_names:
                raise RuntimeError(f"ONNX model has no inputs: {cfg_policy.policy_file}")
            if not self.ort_output_names:
                raise RuntimeError(f"ONNX model has no outputs: {cfg_policy.policy_file}")
            missing_onnx_inputs = [name for name in self.ort_input_names if name not in self.deploy_obs_heads]
            if missing_onnx_inputs:
                raise RuntimeError(
                    "ONNX model inputs are not covered by deploy_obs_heads: "
                    + ", ".join(missing_onnx_inputs)
                )
        elif self.model_backend == "torchscript":
            logger.debug(f"Loading TorchScript policy '{cfg_policy.policy_name}' from {cfg_policy.policy_file}")
            self.model = torch.jit.load(cfg_policy.policy_file, map_location=device)
            schema = self.model._c._get_method("forward").schema
            self.torch_input_names = [arg.name for arg in schema.arguments if arg.name != "self"]
            if not self.torch_input_names:
                raise RuntimeError(f"TorchScript model has no forward inputs: {cfg_policy.policy_file}")
            if len(self.torch_input_names) == 1:
                if len(self.deploy_obs_heads) != 1:
                    raise RuntimeError(
                        "TorchScript model expects exactly 1 deploy obs head, got "
                        f"{len(self.deploy_obs_heads)}: {self.deploy_obs_heads}"
                    )
                self.torch_single_input_head = self.deploy_obs_heads[0]
            else:
                missing_torch_inputs = [name for name in self.torch_input_names if name not in self.deploy_obs_heads]
                if missing_torch_inputs:
                    raise RuntimeError(
                        "TorchScript forward inputs are not covered by deploy_obs_heads: "
                        + ", ".join(missing_torch_inputs)
                    )
                extra_torch_inputs = [name for name in self.deploy_obs_heads if name not in self.torch_input_names]
                if extra_torch_inputs:
                    raise RuntimeError(
                        "deploy_obs_heads contains unused TorchScript inputs: "
                        + ", ".join(extra_torch_inputs)
                    )
        else:
            raise ValueError(f"Unsupported CustomPolicy model_backend: {self.model_backend}")

        super().__init__(cfg_policy=cfg_policy, device=device)

        self.obs_assembler = self.robot_cfg.make_obs_assembler(ObsAssembler)
        self.obs_assembler.print_info()
        self.action_scale = np.asarray(self.cfg_policy.action_scale, dtype=np.float32)
        self.max_cmd = np.asarray(self.cfg_policy.max_cmd, dtype=np.float32)
        self.commands_map = self.cfg_policy.commands_map
        self.use_command_stand = self.cfg_policy.use_command_stand
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


        def height_scan():
            base_height = self._env_value(env_data, "base_pos")[2]
            height_scan = self._env_value(env_data, "height_scan")
            
            return np.clip(base_height - height_scan - 0.77, -1.0, 1.0)

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
            "height_scan": height_scan,
        }
        

        inputs = {}
        for name, spec in self.robot_cfg.sensors.items():
            if name in value_getters:
                value = value_getters[name]()
            else:
                raise KeyError(f"Sensor '{name}' has no defined value getter in CustomPolicy.")
            inputs[name] = np.asarray(value, dtype=np.float32).reshape(spec.shape)
        return inputs, commands, clock_phase

    def get_observation(self, env_data, ctrl_data):
        inputs, commands, clock_phase = self._sensor_inputs(env_data, ctrl_data)
        obs_outputs = self.obs_assembler.step(inputs)
        obs = {}
        for name in self.deploy_obs_heads:
            if name not in obs_outputs:
                raise KeyError(f"ObsAssembler output missing deploy obs head '{name}'")
            obs[name] = np.asarray(obs_outputs[name], dtype=np.float32)

        extras = {
            "commands": commands,
            "command_stand": self.command_stand.copy(),
            "clock_phase": clock_phase,
            "deploy_obs_heads": self.deploy_obs_heads,
            "obs_outputs": obs_outputs,
        }
        return obs, extras

    def _process_actions(self, actions):
        actions = np.asarray(actions, dtype=np.float32).reshape(self.num_actions)
        if self.action_clip is not None:
            actions = np.clip(actions, -self.action_clip, self.action_clip)

        self.last_action = actions.copy()
        return actions * self.action_scale

    def _batch_numpy(self, obs: np.ndarray) -> np.ndarray:
        return np.expand_dims(np.asarray(obs, dtype=np.float32), axis=0)

    def _batch_tensor(self, obs: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(self._batch_numpy(obs)).float().to(self.device)

    def _unwrap_torch_output(self, output: Any) -> torch.Tensor:
        if isinstance(output, (tuple, list)):
            if not output:
                raise RuntimeError("TorchScript model returned no outputs.")
            output = output[0]
        if not torch.is_tensor(output):
            raise RuntimeError(f"TorchScript model must return a Tensor or tuple/list with Tensor, got {type(output)}")
        return output

    def get_action(self, obs: dict[str, np.ndarray]) -> np.ndarray:
        if not isinstance(obs, dict):
            raise TypeError("CustomPolicy expects observation dict keyed by deploy_obs_heads.")

        missing_obs = [name for name in self.deploy_obs_heads if name not in obs]
        if missing_obs:
            raise KeyError(f"Missing deploy observations for inference: {', '.join(missing_obs)}")

        if self.model_backend == "onnx":
            ort_inputs = {
                name: self._batch_numpy(obs[name])
                for name in self.deploy_obs_heads
                if name in self.ort_input_names
            }
            missing_ort_inputs = [name for name in self.ort_input_names if name not in ort_inputs]
            if missing_ort_inputs:
                raise KeyError(f"Missing ONNX inputs for inference: {', '.join(missing_ort_inputs)}")
            outputs = self.ort_session.run(self.ort_output_names, ort_inputs)
            return self._process_actions(np.asarray(outputs[0]).squeeze())

        with torch.no_grad():
            if len(self.torch_input_names) == 1:
                actions_tensor = self.model(self._batch_tensor(obs[self.torch_single_input_head]))
            else:
                actions_tensor = self.model(
                    **{name: self._batch_tensor(obs[name]) for name in self.torch_input_names}
                )
            actions_tensor = self._unwrap_torch_output(actions_tensor).cpu()
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

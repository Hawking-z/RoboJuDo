import importlib.util
import unittest
from pathlib import Path
from tempfile import NamedTemporaryFile

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_module(module_name: str, relative_path: str):
    module_path = REPO_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


robot_config_module = _load_module(
    "robot_config_for_test",
    "robojudo/policy/utils/robot_config.py",
)
obs_assembler_module = _load_module(
    "obs_assembler_for_test",
    "robojudo/policy/utils/obs_assembler.py",
)
RobotConfig = robot_config_module.RobotConfig
ObsAssembler = obs_assembler_module.ObsAssembler


def _nested_obs_head_config() -> dict:
    return {
        "num_actions": 1,
        "dof_config": {
            "isaac_order": ["joint_0"],
            "joint_0": {
                "kp": 1.0,
                "kd": 0.1,
                "action_scale": 1.0,
                "torque_limits": 10.0,
                "default_pos": 0.0,
            },
        },
        "obs_config": {
            "sensors": {
                "base_ang_vel": {"shape": [3]},
                "projected_gravity": {"shape": [3]},
                "command_lin_vel": {"shape": [2]},
                "command_ang_vel": {"shape": [1]},
                "clock_phase": {"shape": [4]},
                "dof_pos": {"shape": [1]},
                "dof_vel": {"shape": [1]},
                "actions": {"shape": [1]},
                "height_scan": {"shape": [5]},
            },
            "encoder_prop_obs": {
                "sources": [
                    "base_ang_vel_noise",
                    "projected_gravity_noise",
                    "command_lin_vel",
                    "command_ang_vel",
                    "clock_phase",
                    "dof_pos_noise",
                    "dof_vel_noise",
                    "actions",
                ],
                "history_len": 5,
                "tail_ndim": 0,
                "history_mode": "merge",
            },
            "encoder_obs": {
                "sources": [
                    "encoder_prop_obs",
                    "height_scan_noise",
                ],
                "history_len": 1,
                "tail_ndim": 0,
                "history_mode": "merge",
            },
            "prop_obs": {
                "sources": [
                    "base_ang_vel_noise",
                    "projected_gravity_noise",
                    "command_lin_vel",
                    "command_ang_vel",
                    "clock_phase",
                    "dof_pos_noise",
                    "dof_vel_noise",
                    "actions",
                    "height_scan_noise",
                ],
                "history_len": 1,
                "tail_ndim": 0,
                "history_mode": "merge",
            },
        },
        "deploy_heads": ["prop_obs", "encoder_obs"],
    }


class TestRobotConfig(unittest.TestCase):
    def test_robot_config_supports_nested_obs_heads_and_legacy_deploy_heads(self):
        robot_cfg = _nested_obs_head_config()

        with NamedTemporaryFile("w", suffix=".yaml", delete=False) as tmp:
            yaml.safe_dump(robot_cfg, tmp, sort_keys=False)
            tmp_path = Path(tmp.name)

        try:
            cfg = RobotConfig.from_yaml_file(tmp_path.as_posix())
            sensors, obs_heads = cfg.obs_assembler_schema()
            assembler = cfg.make_obs_assembler(obs_assembler_cls=ObsAssembler)
        finally:
            tmp_path.unlink(missing_ok=True)

        self.assertEqual(cfg.deploy_obs_heads, ["prop_obs", "encoder_obs"])

        self.assertEqual(
            cfg.obs_map["encoder_prop_obs"].sources,
            [
                "base_ang_vel",
                "projected_gravity",
                "command_lin_vel",
                "command_ang_vel",
                "clock_phase",
                "dof_pos",
                "dof_vel",
                "actions",
            ],
        )
        self.assertEqual(cfg.obs_map["encoder_prop_obs"].per_step_dim, 16)
        self.assertEqual(cfg.obs_map["encoder_prop_obs"].final_dim, 80)

        self.assertEqual(
            cfg.obs_map["encoder_obs"].sources,
            ["encoder_prop_obs", "height_scan"],
        )
        self.assertEqual(cfg.obs_map["encoder_obs"].per_step_dim, 85)
        self.assertEqual(cfg.obs_map["encoder_obs"].final_dim, 85)

        self.assertEqual(cfg.obs_map["prop_obs"].per_step_dim, 21)
        self.assertEqual(cfg.obs_map["prop_obs"].final_dim, 21)

        self.assertEqual(sensors["height_scan"]["shape"], [5])
        self.assertEqual(obs_heads["encoder_obs"]["sources"], ["encoder_prop_obs", "height_scan"])
        self.assertEqual(obs_heads["encoder_obs"]["tail_ndim"], 0)
        self.assertEqual(obs_heads["encoder_obs"]["history_mode"], "merge")

        output_spec = assembler.head_output_spec()
        self.assertEqual(output_spec["encoder_prop_obs"]["output_shape"], (80,))
        self.assertEqual(output_spec["encoder_obs"]["output_shape"], (85,))
        self.assertEqual(output_spec["prop_obs"]["output_shape"], (21,))


if __name__ == "__main__":
    unittest.main()

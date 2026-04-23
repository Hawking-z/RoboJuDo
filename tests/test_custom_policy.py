import unittest
from pathlib import Path
from tempfile import NamedTemporaryFile
from types import SimpleNamespace

import numpy as np
import yaml

from robojudo.policy.custom_policy import CustomPolicy
from robojudo.policy.policy_cfgs import CustomPolicyCfg
from robojudo.policy.utils.robot_config import RobotConfig


class TestCustomPolicy(unittest.TestCase):
    def test_robot_config_supports_regex_dof_rules_with_last_match_wins(self):
        source = Path("assets/models/g1/custom/exported_1/robot_config.yaml")
        robot_cfg = yaml.safe_load(source.read_text())
        robot_cfg["dof_config"] = {
            "isaac_order": robot_cfg["dof_config"]["isaac_order"],
            ".*": {
                "kp": 1.0,
                "kd": 2.0,
                "action_scale": 3.0,
                "torque_limits": 4.0,
                "default_pos": 5.0,
            },
            ".*_hip_pitch_joint": {
                "kp": 10.0,
                "kd": 20.0,
                "action_scale": 30.0,
                "torque_limits": 40.0,
                "default_pos": 50.0,
            },
            "left_hip_pitch_joint": {
                "kp": 100.0,
                "kd": 200.0,
                "action_scale": 300.0,
                "torque_limits": 400.0,
                "default_pos": 500.0,
            },
        }

        with NamedTemporaryFile("w", suffix=".yaml", delete=False) as tmp:
            yaml.safe_dump(robot_cfg, tmp, sort_keys=False)
            tmp_path = Path(tmp.name)

        try:
            cfg = RobotConfig.from_yaml_file(tmp_path.as_posix())
            left_idx = cfg.dof.name_to_index["left_hip_pitch_joint"]
            right_idx = cfg.dof.name_to_index["right_hip_pitch_joint"]
            waist_idx = cfg.dof.name_to_index["waist_yaw_joint"]

            self.assertEqual(cfg.dof.kp[left_idx], 100.0)
            self.assertEqual(cfg.dof.kd[left_idx], 200.0)
            self.assertEqual(cfg.dof.scale[left_idx], 300.0)
            self.assertEqual(cfg.dof.torque_limits[left_idx], 400.0)
            self.assertEqual(cfg.dof.default_pos[left_idx], 500.0)

            self.assertEqual(cfg.dof.kp[right_idx], 10.0)
            self.assertEqual(cfg.dof.kd[right_idx], 20.0)
            self.assertEqual(cfg.dof.scale[right_idx], 30.0)
            self.assertEqual(cfg.dof.torque_limits[right_idx], 40.0)
            self.assertEqual(cfg.dof.default_pos[right_idx], 50.0)

            self.assertEqual(cfg.dof.kp[waist_idx], 1.0)
            self.assertEqual(cfg.dof.kd[waist_idx], 2.0)
            self.assertEqual(cfg.dof.scale[waist_idx], 3.0)
            self.assertEqual(cfg.dof.torque_limits[waist_idx], 4.0)
            self.assertEqual(cfg.dof.default_pos[waist_idx], 5.0)
        finally:
            tmp_path.unlink(missing_ok=True)

    def test_custom_policy_cfg_paths_are_empty_by_default(self):
        cfg = CustomPolicyCfg()

        self.assertEqual(cfg.robot_config_file, "")
        self.assertEqual(cfg.model_dir, "")
        self.assertEqual(cfg.model_file, "")
        self.assertEqual(cfg.policy_file, "")

    def test_custom_policy_cfg_uses_explicit_robot_config(self):
        cfg = CustomPolicyCfg(
            model_backend="onnx",
            model_dir="exported_1",
            policy_name="policy_0",
            robot_config_file="exported_1/robot_config.yaml",
        )
        robot_cfg = RobotConfig.from_yaml_file("exported_1/robot_config.yaml")

        self.assertEqual(cfg.policy_type, "CustomPolicy")
        self.assertTrue(cfg.policy_file.endswith("exported_1/policy_0.onnx"))
        self.assertTrue(cfg.robot_config_file.endswith("exported_1/robot_config.yaml"))
        self.assertEqual(cfg.freq, 50)
        self.assertEqual(cfg.action_clip, 10.0)
        self.assertEqual(cfg.action_beta, 1.0)
        self.assertEqual(cfg.obs_heads, list(robot_cfg.obs_map.keys()))
        self.assertEqual(cfg.obs_dof.joint_names, robot_cfg.dof.isaac_order)
        self.assertEqual(cfg.obs_dof.default_pos, robot_cfg.dof.default_pos.tolist())
        self.assertEqual(cfg.obs_dof.stiffness, robot_cfg.dof.kp.tolist())
        self.assertEqual(cfg.action_scale, robot_cfg.dof.scale.tolist())
        self.assertEqual(cfg.action_dof, cfg.obs_dof)

    def test_custom_policy_cfg_selects_torchscript_and_onnx_model_files(self):
        torch_cfg = CustomPolicyCfg(
            model_backend="torchscript",
            model_dir="exported",
            policy_name="policy_0",
            robot_config_file="exported/robot_config.yaml",
        )
        onnx_cfg = CustomPolicyCfg(
            model_backend="onnx",
            model_dir="exported",
            policy_name="policy_0",
            robot_config_file="exported/robot_config.yaml",
        )
        explicit_cfg = CustomPolicyCfg(
            model_backend="onnx",
            model_file="/tmp/custom_policy.onnx",
            robot_config_file="exported/robot_config.yaml",
        )
        pt_cfg = CustomPolicyCfg(
            model_backend="torchscript",
            model_dir="exported",
            policy_name="policy_0",
            model_suffix=".pt",
            robot_config_file="exported/robot_config.yaml",
        )

        self.assertTrue(torch_cfg.policy_file.endswith("exported/policy_0.jit"))
        self.assertTrue(onnx_cfg.policy_file.endswith("exported/policy_0.onnx"))
        self.assertEqual(explicit_cfg.policy_file, "/tmp/custom_policy.onnx")
        self.assertTrue(pt_cfg.policy_file.endswith("exported/policy_0.pt"))

    def test_custom_policy_requires_robot_config_file_at_runtime(self):
        cfg = CustomPolicyCfg(disable_autoload=True)

        with self.assertRaises(FileNotFoundError):
            CustomPolicy(cfg_policy=cfg, device="cpu")

    def test_custom_policy_builds_actor_obs_from_robot_config_schema(self):
        cfg = CustomPolicyCfg(
            model_backend="onnx",
            model_dir="exported_1",
            policy_name="policy_0",
            robot_config_file="exported_1/robot_config.yaml",
            disable_autoload=True,
        )
        policy = CustomPolicy(cfg_policy=cfg, device="cpu")
        obs_shape = policy.obs_assembler.head_output_spec()[cfg.obs_heads[0]]["output_shape"]

        env_data = SimpleNamespace(
            base_quat=np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
            base_ang_vel=np.array([1.0, 2.0, 3.0], dtype=np.float32),
            dof_pos=np.asarray(cfg.obs_dof.default_pos, dtype=np.float32) + 0.1,
            dof_vel=np.ones(cfg.obs_dof.num_dofs, dtype=np.float32),
        )
        obs, extras = policy.get_observation(env_data, {})

        self.assertEqual(obs.dtype, np.float32)
        self.assertEqual(obs.shape, obs_shape)
        self.assertEqual(extras["obs_head"], cfg.obs_heads[0])
        self.assertEqual(extras["obs_heads"], cfg.obs_heads)
        self.assertIn(cfg.obs_heads[0], extras["obs_outputs"])
        np.testing.assert_allclose(extras["commands"], np.zeros(3, dtype=np.float32))
        self.assertEqual(extras["clock_phase"].shape, (4,))
        np.testing.assert_allclose(extras["command_stand"], np.array([1.0], dtype=np.float32))
        np.testing.assert_allclose(extras["clock_phase"], np.zeros(4, dtype=np.float32))

    def test_custom_policy_walk_command_ungates_clock_phase(self):
        cfg = CustomPolicyCfg(
            model_backend="onnx",
            model_dir="exported_1",
            policy_name="policy_0",
            robot_config_file="exported_1/robot_config.yaml",
            disable_autoload=True,
        )
        policy = CustomPolicy(cfg_policy=cfg, device="cpu")
        env_data = SimpleNamespace(
            base_quat=np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
            base_ang_vel=np.zeros(3, dtype=np.float32),
            dof_pos=np.asarray(cfg.obs_dof.default_pos, dtype=np.float32),
            dof_vel=np.zeros(cfg.obs_dof.num_dofs, dtype=np.float32),
        )
        ctrl_data = {
            "KeyboardCtrl": {
                "keyboard_event": [
                    {"type": "keyboard", "name": "=", "pressed": True},
                ]
            }
        }

        _, extras = policy.get_observation(env_data, ctrl_data)

        np.testing.assert_allclose(extras["command_stand"], np.array([0.0], dtype=np.float32))
        self.assertGreater(np.linalg.norm(extras["clock_phase"]), 0.0)

    def test_custom_policy_action_processing_matches_export_action_transform(self):
        cfg = CustomPolicyCfg(
            model_backend="onnx",
            model_dir="exported_1",
            policy_name="policy_0",
            robot_config_file="exported_1/robot_config.yaml",
            disable_autoload=True,
        )
        policy = CustomPolicy(cfg_policy=cfg, device="cpu")
        raw_actions = np.full(cfg.action_dof.num_dofs, cfg.action_clip + 2.0, dtype=np.float32)

        processed_actions = policy._process_actions(raw_actions)

        expected = np.clip(raw_actions, -cfg.action_clip, cfg.action_clip) * np.asarray(cfg.action_scale)
        np.testing.assert_allclose(processed_actions, expected)
        np.testing.assert_allclose(policy.last_action, np.clip(raw_actions, -cfg.action_clip, cfg.action_clip))

    def test_custom_policy_rejects_unsupported_exp_avg_decay(self):
        source = Path("exported_1/robot_config.yaml")
        bad_cfg_path = Path("/tmp/custom_policy_exp_avg_decay.yaml")
        robot_cfg = yaml.safe_load(source.read_text())
        robot_cfg["exp_avg_decay"] = 0.05
        bad_cfg_path.write_text(yaml.safe_dump(robot_cfg))

        with self.assertRaisesRegex(ValueError, "exp_avg_decay != 1.0"):
            CustomPolicyCfg(robot_config_file=bad_cfg_path.as_posix())

    def test_custom_policy_wo_gait_config_matches_unitree_wo_gait_shape(self):
        from robojudo.config.g1.policy.g1_custom_policy_cfg import G1CustomPolicy2Cfg

        cfg = G1CustomPolicy2Cfg(disable_autoload=True)
        policy = CustomPolicy(cfg_policy=cfg, device="cpu")
        obs_shape = policy.obs_assembler.head_output_spec()[cfg.obs_heads[0]]["output_shape"]

        self.assertEqual(cfg.policy_file, "/home/zyc/RoboJuDo/exported/policy_wo_gait.pt")
        self.assertEqual(cfg.action_dof.num_dofs, 29)
        self.assertEqual(obs_shape, (480,))

    def test_custom_policy_wo_gait_commands_are_not_stand_gated(self):
        cfg = CustomPolicyCfg(
            model_backend="torchscript",
            model_dir="exported",
            policy_name="policy_wo_gait",
            model_suffix=".pt",
            robot_config_file="exported/robot_config.yaml",
            disable_autoload=True,
        )
        policy = CustomPolicy(cfg_policy=cfg, device="cpu")
        env_data = SimpleNamespace(
            base_quat=np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
            base_ang_vel=np.zeros(3, dtype=np.float32),
            dof_pos=np.asarray(cfg.obs_dof.default_pos, dtype=np.float32),
            dof_vel=np.zeros(cfg.obs_dof.num_dofs, dtype=np.float32),
        )
        ctrl_data = {
            "KeyboardCtrl": {
                "keyboard_event": [
                    {"type": "keyboard", "name": "w", "pressed": True},
                ]
            }
        }

        _, extras = policy.get_observation(env_data, ctrl_data)

        np.testing.assert_allclose(extras["commands"], np.array([1.2, 0.0, 0.0], dtype=np.float32))

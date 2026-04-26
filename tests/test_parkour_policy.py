import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

from robojudo.policy.parkour_policy import ParkourPolicy
from robojudo.policy.policy_cfgs import ParkourPolicyCfg
from robojudo.policy.utils.robot_config import RobotConfig


class _FakeOrtIO:
    def __init__(self, name: str):
        self.name = name


class _CapturingOrtSession:
    instances = []

    def __init__(self, *_args, **_kwargs):
        self._inputs = [_FakeOrtIO("enc_input"), _FakeOrtIO("proprio_obs")]
        self._outputs = [_FakeOrtIO("actor_output")]
        self.last_inputs = None
        _CapturingOrtSession.instances.append(self)

    def get_inputs(self):
        return self._inputs

    def get_outputs(self):
        return self._outputs

    def run(self, output_names, input_dict):
        self.last_inputs = input_dict
        return [np.zeros((1, 29), dtype=np.float32) for _ in output_names]


class TestParkourPolicy(unittest.TestCase):
    def test_parkour_robot_config_matches_exported_policy_report(self):
        cfg = ParkourPolicyCfg()
        robot_cfg = RobotConfig.from_yaml_file(cfg.robot_config_file)

        expected_joint_order = [
            "left_shoulder_pitch_joint",
            "right_shoulder_pitch_joint",
            "waist_pitch_joint",
            "left_shoulder_roll_joint",
            "right_shoulder_roll_joint",
            "waist_roll_joint",
            "left_shoulder_yaw_joint",
            "right_shoulder_yaw_joint",
            "waist_yaw_joint",
            "left_elbow_joint",
            "right_elbow_joint",
            "left_hip_pitch_joint",
            "right_hip_pitch_joint",
            "left_wrist_roll_joint",
            "right_wrist_roll_joint",
            "left_hip_roll_joint",
            "right_hip_roll_joint",
            "left_wrist_pitch_joint",
            "right_wrist_pitch_joint",
            "left_hip_yaw_joint",
            "right_hip_yaw_joint",
            "left_wrist_yaw_joint",
            "right_wrist_yaw_joint",
            "left_knee_joint",
            "right_knee_joint",
            "left_ankle_pitch_joint",
            "right_ankle_pitch_joint",
            "left_ankle_roll_joint",
            "right_ankle_roll_joint",
        ]
        expected_default_pos = [
            0.2,
            0.2,
            0.0,
            0.2,
            -0.2,
            0.0,
            0.0,
            0.0,
            0.0,
            0.6,
            0.6,
            -0.312,
            -0.312,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.669,
            0.669,
            -0.363,
            -0.363,
            0.0,
            0.0,
        ]
        expected_scale = [
            0.4385773241519928,
            0.4385773241519928,
            0.4385773241519928,
            0.4385773241519928,
            0.4385773241519928,
            0.4385773241519928,
            0.4385773241519928,
            0.4385773241519928,
            0.548,
            0.4385773241519928,
            0.4385773241519928,
            0.548,
            0.548,
            0.4385773241519928,
            0.4385773241519928,
            0.351,
            0.351,
            0.075,
            0.075,
            0.548,
            0.548,
            0.075,
            0.075,
            0.351,
            0.351,
            0.4385773241519928,
            0.4385773241519928,
            0.4385773241519928,
            0.4385773241519928,
        ]

        self.assertEqual(robot_cfg.dof.isaac_order, expected_joint_order)
        np.testing.assert_allclose(robot_cfg.dof.default_pos, expected_default_pos, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(robot_cfg.dof.scale, expected_scale, rtol=1e-6, atol=1e-6)
        self.assertAlmostEqual(robot_cfg.dof.kp[0], 14.251, places=3)
        self.assertAlmostEqual(robot_cfg.dof.kd[0], 0.907, places=3)
        self.assertAlmostEqual(robot_cfg.dof.kp[15], 99.098, places=3)
        self.assertAlmostEqual(robot_cfg.dof.kd[15], 6.309, places=3)

        self.assertEqual(
            list(robot_cfg.sensors),
            ["base_ang_vel", "projected_gravity", "velocity_commands", "joint_pos", "joint_vel", "actions", "depth"],
        )
        self.assertAlmostEqual(robot_cfg.sensors["base_ang_vel"].scale, 0.25)
        self.assertEqual(robot_cfg.sensors["depth"].shape, [18, 32])
        self.assertEqual(robot_cfg.sensors["depth"].frames, 8)
        self.assertEqual(robot_cfg.sensors["depth"].stride, 5)
        self.assertEqual(robot_cfg.deploy_obs_heads, ["enc_input", "proprio_obs"])
        self.assertEqual(robot_cfg.obs_map["proprio_obs"].final_dim, 768)
        self.assertEqual(robot_cfg.obs_map["enc_input"].output_shape, [8, 18, 32])

        self.assertEqual(cfg.depth_resolution, [64, 36])
        self.assertEqual(cfg.depth_crop_region, [18, 0, 16, 16])
        self.assertEqual(cfg.depth_range, [0.0, 2.5])

    def test_parkour_policy_builds_merged_onnx_inputs_from_mujoco_depth(self):
        cfg = ParkourPolicyCfg()
        _CapturingOrtSession.instances.clear()
        with mock.patch("onnxruntime.InferenceSession", side_effect=_CapturingOrtSession):
            policy = ParkourPolicy(cfg_policy=cfg, device="cpu")

        env_data = SimpleNamespace(
            base_quat=np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
            base_ang_vel=np.zeros(3, dtype=np.float32),
            dof_pos=np.asarray(cfg.obs_dof.default_pos, dtype=np.float32),
            dof_vel=np.zeros(cfg.obs_dof.num_dofs, dtype=np.float32),
            camera_d435i_depth=np.ones((270, 480), dtype=np.float32),
        )

        obs, extras = policy.get_observation(env_data, {})
        action = policy.get_action(obs)

        self.assertEqual(action.shape, (29,))
        self.assertEqual(extras["deploy_obs_heads"], ["enc_input", "proprio_obs"])
        session = _CapturingOrtSession.instances[0]
        self.assertEqual(session.last_inputs["enc_input"].shape, (1, 8, 18, 32))
        self.assertEqual(session.last_inputs["proprio_obs"].shape, (1, 768))


if __name__ == "__main__":
    unittest.main()

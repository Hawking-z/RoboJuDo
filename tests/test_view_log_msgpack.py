import unittest

import numpy as np

from scripts.view_log_msgpack import build_joint_target_curves, build_velocity_tracking_curves


class TestViewLogMsgpack(unittest.TestCase):
    def test_velocity_tracking_curves_pair_commands_with_robot_velocity(self):
        frames = [
            {
                "time": 10.0,
                "timestep": 5,
                "env_data": {
                    "base_lin_vel": np.array([0.1, 0.2, 0.0], dtype=np.float32),
                    "base_ang_vel": np.array([0.0, 0.0, 0.3], dtype=np.float32),
                },
                "extras": {"commands": np.array([1.0, -0.5, 0.7], dtype=np.float32)},
            },
            {
                "time": 10.2,
                "timestep": 6,
                "env_data": {
                    "base_lin_vel": np.array([0.4, 0.5, 0.0], dtype=np.float32),
                    "base_ang_vel": np.array([0.0, 0.0, 0.6], dtype=np.float32),
                },
                "extras": {"commands": np.array([1.5, -0.2, 0.9], dtype=np.float32)},
            },
        ]

        x, curves = build_velocity_tracking_curves(frames)

        np.testing.assert_allclose(x, np.array([0.0, 0.2]))
        np.testing.assert_allclose(curves["cmd_vx"], np.array([1.0, 1.5]))
        np.testing.assert_allclose(curves["actual_vx"], np.array([0.1, 0.4]))
        np.testing.assert_allclose(curves["cmd_vy"], np.array([-0.5, -0.2]))
        np.testing.assert_allclose(curves["actual_vy"], np.array([0.2, 0.5]))
        np.testing.assert_allclose(curves["cmd_yaw"], np.array([0.7, 0.9]))
        np.testing.assert_allclose(curves["actual_yaw"], np.array([0.3, 0.6]))

    def test_joint_target_curves_pair_dof_pos_with_pd_target_for_selected_joints(self):
        frames = [
            {
                "timestep": 1,
                "env_data": {"dof_pos": np.array([0.1, 0.2, 0.3], dtype=np.float32)},
                "pd_target": np.array([0.4, 0.5, 0.6], dtype=np.float32),
            },
            {
                "timestep": 2,
                "env_data": {"dof_pos": np.array([1.1, 1.2, 1.3], dtype=np.float32)},
                "pd_target": np.array([1.4, 1.5, 1.6], dtype=np.float32),
            },
        ]

        x, curves = build_joint_target_curves(frames, joint_indices=[1])

        np.testing.assert_allclose(x, np.array([1.0, 2.0]))
        np.testing.assert_allclose(curves["joint_1_pos"], np.array([0.2, 1.2]))
        np.testing.assert_allclose(curves["joint_1_target"], np.array([0.5, 1.5]))


if __name__ == "__main__":
    unittest.main()

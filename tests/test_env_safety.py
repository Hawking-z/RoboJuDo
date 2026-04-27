import unittest

import numpy as np

from robojudo.environment.base_env import Environment
from robojudo.environment.env_cfgs import EnvCfg
from robojudo.tools.tool_cfgs import DoFConfig


class DummyEnvironment(Environment):
    def self_check(self):
        pass

    def reset(self):
        pass

    def update(self):
        pass

    def step(self, pd_target, hand_pose=None):
        pass

    def shutdown(self):
        pass

    def set_gains(self, stiffness, damping):
        self.stiffness = np.asarray(stiffness)
        self.damping = np.asarray(damping)


class TestEnvSafety(unittest.TestCase):
    def test_pd_target_safety_clips_position_and_torque_in_env_dof_order(self):
        dof = DoFConfig(
            joint_names=["a", "b", "c"],
            default_pos=[0.0, 0.0, 0.0],
            stiffness=[10.0, 20.0, 0.0],
            damping=[1.0, 2.0, 3.0],
            torque_limits=[5.0, 4.0, 2.0],
            position_limits=[[-1.0, 1.0], [-0.5, 0.5], [-2.0, 2.0]],
        )
        env = DummyEnvironment(
            EnvCfg(
                env_type="DummyEnv",
                xml="unused.xml",
                dof=dof,
                clip_torque_limits=True,
            )
        )
        env._dof_pos = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        env._dof_vel = np.array([1.0, -1.0, 0.5], dtype=np.float32)

        target = np.array([2.0, -1.0, 10.0], dtype=np.float32)
        clipped = env.apply_pd_target_safety(target)

        np.testing.assert_allclose(clipped, np.array([0.6, -0.3, 2.0], dtype=np.float32))

    def test_unsafe_dof_position_indices_use_expanded_position_limits(self):
        dof = DoFConfig(
            joint_names=["a", "b"],
            default_pos=[0.0, 0.0],
            stiffness=[1.0, 1.0],
            damping=[0.1, 0.1],
            position_limits=[[-1.0, 1.0], [-2.0, 2.0]],
        )
        env = DummyEnvironment(
            EnvCfg(
                env_type="DummyEnv",
                xml="unused.xml",
                dof=dof,
                joint_pos_protect_ratio=1.5,
            )
        )
        env._dof_pos = np.array([1.4, -3.1], dtype=np.float32)

        np.testing.assert_array_equal(env.unsafe_dof_position_indices(), np.array([1], dtype=np.int64))


if __name__ == "__main__":
    unittest.main()

import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np
from box import Box

import robojudo.policy
from robojudo.pipeline.rl_pipeline import PolicyWrapper
from robojudo.tools.dof import DoFAdapter
from robojudo.tools.tool_cfgs import DoFConfig


class FakePolicy:
    last_instance = None

    def __init__(self, cfg_policy, device):
        self.cfg_obs_dof = cfg_policy.obs_dof
        self.cfg_action_dof = cfg_policy.action_dof
        self.default_pos = np.asarray(self.cfg_action_dof.default_pos, dtype=np.float32)
        self.observed_env_data = None
        FakePolicy.last_instance = self

    def get_observation(self, env_data, ctrl_data):
        self.observed_env_data = env_data
        return "obs", {}

    def get_action(self, obs):
        return np.array([1.0, 2.0], dtype=np.float32)

    def get_init_dof_pos(self):
        return np.array([3.0, 4.0], dtype=np.float32)


class TestDoFAdapter(unittest.TestCase):
    def test_fit_maps_joint_names_and_applies_target_joint_signs(self):
        adapter = DoFAdapter(
            src_joint_names=["a", "b"],
            tar_joint_names=["b", "a"],
            tar_joint_signs=[-1.0, 1.0],
        )

        fitted = adapter.fit(np.array([2.0, 3.0], dtype=np.float32))

        np.testing.assert_allclose(fitted, np.array([-3.0, 2.0], dtype=np.float32))

    def test_fit_does_not_apply_joint_signs_to_non_numeric_data(self):
        adapter = DoFAdapter(
            src_joint_names=["a", "b"],
            tar_joint_names=["b", "a"],
            tar_joint_signs=[-1.0, 1.0],
        )

        fitted = adapter.fit(["a", "b"], dim=0)

        self.assertEqual(fitted.tolist(), ["b", "a"])

    def test_policy_wrapper_uses_sign_aware_adapters_for_observation_and_actions(self):
        env_dof = DoFConfig(
            joint_names=["a", "b"],
            default_pos=[10.0, 20.0],
            stiffness=[1.0, 1.0],
            damping=[0.1, 0.1],
        )
        policy_dof = DoFConfig(
            joint_names=["b", "a"],
            default_pos=[200.0, 100.0],
            joint_signs=[-1.0, 1.0],
        )
        cfg_policy = SimpleNamespace(
            policy_type="FakePolicy",
            obs_dof=policy_dof,
            action_dof=policy_dof,
        )

        with mock.patch.object(robojudo.policy, "FakePolicy", FakePolicy, create=True):
            wrapper = PolicyWrapper(cfg_policy=cfg_policy, env_dof_cfg=env_dof, device="cpu")

        env_data = Box(
            {
                "dof_pos": np.array([10.0, 20.0], dtype=np.float32),
                "dof_vel": np.array([1.0, 2.0], dtype=np.float32),
            }
        )
        wrapper.get_observation(env_data, Box({}))

        fake_policy = FakePolicy.last_instance
        np.testing.assert_allclose(fake_policy.observed_env_data.dof_pos, np.array([-20.0, 10.0]))
        np.testing.assert_allclose(fake_policy.observed_env_data.dof_vel, np.array([-2.0, 1.0]))

        np.testing.assert_allclose(wrapper.get_action("obs"), np.array([2.0, -1.0]))
        np.testing.assert_allclose(wrapper.get_pd_target("obs"), np.array([102.0, -201.0]))
        np.testing.assert_allclose(wrapper.get_init_dof_pos(), np.array([4.0, -3.0]))


if __name__ == "__main__":
    unittest.main()

import unittest
from pathlib import Path
from tempfile import NamedTemporaryFile
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import mock

import numpy as np
import yaml
import mujoco

from robojudo.environment.base_env import Environment
from robojudo.environment.env_cfgs import EnvCfg, MujocoEnvCfg
from robojudo.policy.custom_policy import CustomPolicy
from robojudo.policy.policy_cfgs import CustomPolicyCfg
from robojudo.policy.utils.robot_config import RobotConfig
from robojudo.tools.tool_cfgs import DoFConfig


def _minimal_dof_cfg():
    return DoFConfig(
        joint_names=["joint_0"],
        default_pos=[0.0],
        stiffness=[1.0],
        damping=[0.1],
        torque_limits=[10.0],
        position_limits=[[-1.0, 1.0]],
    )


class _FakeTorchscriptModule:
    class _CompiledModule:
        @staticmethod
        def _get_method(name):
            schema = SimpleNamespace(
                arguments=[SimpleNamespace(name="self"), SimpleNamespace(name="x")]
            )
            return SimpleNamespace(schema=schema)

    _c = _CompiledModule()


class _FakeOrtIO:
    def __init__(self, name: str):
        self.name = name


class _FakeOrtSession:
    def __init__(self, input_names: list[str], output_names: list[str]):
        self._inputs = [_FakeOrtIO(name) for name in input_names]
        self._outputs = [_FakeOrtIO(name) for name in output_names]

    def get_inputs(self):
        return self._inputs

    def get_outputs(self):
        return self._outputs

    def run(self, output_names, input_dict):
        return [np.zeros((1, 1), dtype=np.float32) for _ in output_names]


def _external_sensor_robot_config(sensor_name: str, sensor_shape: list[int]) -> Path:
    source = Path("assets/models/g1/custom/exported_1/robot_config.yaml")
    robot_cfg = yaml.safe_load(source.read_text())
    robot_cfg["obs_config"] = {
        "sensors": {
            sensor_name: {
                "shape": sensor_shape,
            }
        },
        "external_obs": {
            "sources": [sensor_name],
            "history_len": 1,
            "flatten": True,
        },
    }

    with NamedTemporaryFile("w", suffix=".yaml", delete=False) as tmp:
        yaml.safe_dump(robot_cfg, tmp, sort_keys=False)
        return Path(tmp.name)


class _DummyEnvironmentForExternalData(Environment):
    def __init__(self):
        super().__init__(
            cfg_env=EnvCfg(
                env_type="DummyTestEnv",
                xml="robot.xml",
                dof=_minimal_dof_cfg(),
            ),
            device="cpu",
        )

    def self_check(self):
        return None

    def reset(self):
        return None

    def update(self):
        return None

    def step(self, pd_target, hand_pose=None):
        return None

    def shutdown(self):
        return None

    def set_gains(self, stiffness, damping):
        return None


class TestCustomPolicy(unittest.TestCase):
    def test_environment_get_data_merges_external_sensor_outputs(self):
        env = _DummyEnvironmentForExternalData()

        env.set_extra_env_data(
            {
                "camera_front_depth": np.ones((4, 4), dtype=np.float32),
                "height_scan": np.arange(6, dtype=np.float32),
            }
        )
        data = env.get_data()

        np.testing.assert_allclose(data["camera_front_depth"], np.ones((4, 4), dtype=np.float32))
        np.testing.assert_allclose(data["height_scan"], np.arange(6, dtype=np.float32))

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

        self.assertTrue(cfg.disable_autoload)
        self.assertEqual(cfg.robot_config_file, "")
        self.assertEqual(cfg.model_dir, "")
        self.assertEqual(cfg.model_file, "")
        self.assertEqual(cfg.policy_file, "")

    def test_custom_policy_cfg_rejects_disable_autoload_false(self):
        with self.assertRaisesRegex(ValueError, "disable_autoload"):
            CustomPolicyCfg(
                disable_autoload=False,
                robot_config_file="assets/models/g1/custom/exported/robot_config.yaml",
            )

    def test_custom_policy_loads_torchscript_model_even_when_base_autoload_is_disabled(self):
        with NamedTemporaryFile(suffix=".pt") as tmp:
            cfg = CustomPolicyCfg(
                model_backend="torchscript",
                model_file=tmp.name,
                robot_config_file="assets/models/g1/custom/exported/robot_config.yaml",
                disable_autoload=True,
            )

            with mock.patch(
                "robojudo.policy.custom_policy.torch.jit.load",
                return_value=_FakeTorchscriptModule(),
            ) as jit_load:
                policy = CustomPolicy(cfg_policy=cfg, device="cpu")

        jit_load.assert_called_once_with(tmp.name, map_location="cpu")
        self.assertTrue(policy.cfg_policy.disable_autoload)
        self.assertTrue(hasattr(policy, "model"))

    def test_custom_policy_cfg_uses_explicit_robot_config(self):
        cfg = CustomPolicyCfg(
            model_backend="onnx",
            model_dir="assets/models/g1/custom/exported_1",
            policy_name="policy_0",
            robot_config_file="assets/models/g1/custom/exported_1/robot_config.yaml",
        )
        robot_cfg = RobotConfig.from_yaml_file("assets/models/g1/custom/exported_1/robot_config.yaml")

        self.assertEqual(cfg.policy_type, "CustomPolicy")
        self.assertTrue(cfg.policy_file.endswith("exported_1/policy_0.onnx"))
        self.assertTrue(cfg.robot_config_file.endswith("exported_1/robot_config.yaml"))
        self.assertEqual(cfg.freq, 50)
        self.assertEqual(cfg.action_clip, 10.0)
        self.assertEqual(cfg.action_beta, 1.0)
        self.assertEqual(cfg.deploy_obs_heads, robot_cfg.deploy_obs_heads)
        self.assertEqual(cfg.obs_dof.joint_names, robot_cfg.dof.isaac_order)
        self.assertEqual(cfg.obs_dof.default_pos, robot_cfg.dof.default_pos.tolist())
        self.assertEqual(cfg.obs_dof.stiffness, robot_cfg.dof.kp.tolist())
        self.assertEqual(cfg.action_scale, robot_cfg.dof.scale.tolist())
        self.assertEqual(cfg.action_dof, cfg.obs_dof)

    def test_custom_policy_cfg_selects_torchscript_and_onnx_model_files(self):
        torch_cfg = CustomPolicyCfg(
            model_backend="torchscript",
            model_dir="assets/models/g1/custom/exported",
            policy_name="policy_0",
            robot_config_file="assets/models/g1/custom/exported/robot_config.yaml",
        )
        onnx_cfg = CustomPolicyCfg(
            model_backend="onnx",
            model_dir="assets/models/g1/custom/exported",
            policy_name="policy_0",
            robot_config_file="assets/models/g1/custom/exported/robot_config.yaml",
        )
        explicit_cfg = CustomPolicyCfg(
            model_backend="onnx",
            model_file="/tmp/custom_policy.onnx",
            robot_config_file="assets/models/g1/custom/exported/robot_config.yaml",
        )
        pt_cfg = CustomPolicyCfg(
            model_backend="torchscript",
            model_dir="assets/models/g1/custom/exported",
            policy_name="policy_0",
            model_suffix=".pt",
            robot_config_file="assets/models/g1/custom/exported/robot_config.yaml",
        )

        self.assertTrue(torch_cfg.policy_file.endswith("exported/policy_0.jit"))
        self.assertTrue(onnx_cfg.policy_file.endswith("exported/policy_0.onnx"))
        self.assertEqual(explicit_cfg.policy_file, "/tmp/custom_policy.onnx")
        self.assertTrue(pt_cfg.policy_file.endswith("exported/policy_0.pt"))

    def test_custom_policy_requires_robot_config_file_at_runtime(self):
        cfg = CustomPolicyCfg(disable_autoload=True)

        with self.assertRaises(FileNotFoundError):
            CustomPolicy(cfg_policy=cfg, device="cpu")

    def test_custom_policy_requires_model_file_even_when_disable_autoload_is_true(self):
        cfg = CustomPolicyCfg(
            model_backend="torchscript",
            robot_config_file="assets/models/g1/custom/exported/robot_config.yaml",
            disable_autoload=True,
        )

        with self.assertRaises(FileNotFoundError):
            CustomPolicy(cfg_policy=cfg, device="cpu")

    def test_custom_policy_builds_actor_obs_from_robot_config_schema(self):
        cfg = CustomPolicyCfg(
            model_backend="onnx",
            model_dir="assets/models/g1/custom/exported_1",
            policy_name="policy_0",
            robot_config_file="assets/models/g1/custom/exported_1/robot_config.yaml",
            disable_autoload=True,
        )
        policy = CustomPolicy(cfg_policy=cfg, device="cpu")
        deploy_head = cfg.deploy_obs_heads[0]
        obs_shape = policy.obs_assembler.head_output_spec()[deploy_head]["output_shape"]

        env_data = SimpleNamespace(
            base_quat=np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
            base_ang_vel=np.array([1.0, 2.0, 3.0], dtype=np.float32),
            dof_pos=np.asarray(cfg.obs_dof.default_pos, dtype=np.float32) + 0.1,
            dof_vel=np.ones(cfg.obs_dof.num_dofs, dtype=np.float32),
        )
        obs, extras = policy.get_observation(env_data, {})

        self.assertEqual(set(obs.keys()), set(cfg.deploy_obs_heads))
        self.assertEqual(obs[deploy_head].dtype, np.float32)
        self.assertEqual(obs[deploy_head].shape, obs_shape)
        self.assertEqual(extras["deploy_obs_heads"], cfg.deploy_obs_heads)
        self.assertIn(deploy_head, extras["obs_outputs"])
        np.testing.assert_allclose(extras["commands"], np.zeros(3, dtype=np.float32))
        self.assertEqual(extras["clock_phase"].shape, (4,))
        np.testing.assert_allclose(extras["command_stand"], np.array([1.0], dtype=np.float32))
        np.testing.assert_allclose(extras["clock_phase"], np.zeros(4, dtype=np.float32))

    def test_custom_policy_onnx_get_action_accepts_deploy_obs_dict(self):
        cfg = CustomPolicyCfg(
            model_backend="onnx",
            model_dir="assets/models/g1/custom/exported_1",
            policy_name="policy_0",
            robot_config_file="assets/models/g1/custom/exported_1/robot_config.yaml",
            disable_autoload=True,
        )
        policy = CustomPolicy(cfg_policy=cfg, device="cpu")
        env_data = SimpleNamespace(
            base_quat=np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
            base_ang_vel=np.zeros(3, dtype=np.float32),
            dof_pos=np.asarray(cfg.obs_dof.default_pos, dtype=np.float32),
            dof_vel=np.zeros(cfg.obs_dof.num_dofs, dtype=np.float32),
        )

        obs, _ = policy.get_observation(env_data, {})
        action = policy.get_action(obs)

        self.assertEqual(set(obs.keys()), set(cfg.deploy_obs_heads))
        self.assertEqual(action.shape, (cfg.action_dof.num_dofs,))

    def test_custom_policy_walk_command_ungates_clock_phase(self):
        cfg = CustomPolicyCfg(
            model_backend="onnx",
            model_dir="assets/models/g1/custom/exported_1",
            policy_name="policy_0",
            robot_config_file="assets/models/g1/custom/exported_1/robot_config.yaml",
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
            model_dir="assets/models/g1/custom/exported_1",
            policy_name="policy_0",
            robot_config_file="assets/models/g1/custom/exported_1/robot_config.yaml",
            disable_autoload=True,
        )
        policy = CustomPolicy(cfg_policy=cfg, device="cpu")
        raw_actions = np.full(cfg.action_dof.num_dofs, cfg.action_clip + 2.0, dtype=np.float32)

        processed_actions = policy._process_actions(raw_actions)

        expected = np.clip(raw_actions, -cfg.action_clip, cfg.action_clip) * np.asarray(cfg.action_scale)
        np.testing.assert_allclose(processed_actions, expected)
        np.testing.assert_allclose(policy.last_action, np.clip(raw_actions, -cfg.action_clip, cfg.action_clip))

    def test_custom_policy_rejects_unsupported_exp_avg_decay(self):
        source = Path("assets/models/g1/custom/exported_1/robot_config.yaml")
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
        obs_shape = policy.obs_assembler.head_output_spec()[cfg.deploy_obs_heads[0]]["output_shape"]

        self.assertEqual(cfg.policy_file, "assets/models/g1/custom/exported/policy_wo_gait.pt")
        self.assertEqual(cfg.action_dof.num_dofs, 29)
        self.assertEqual(obs_shape, (480,))

    def test_custom_policy_torchscript_get_action_accepts_deploy_obs_dict(self):
        from robojudo.config.g1.policy.g1_custom_policy_cfg import G1CustomPolicy2Cfg

        cfg = G1CustomPolicy2Cfg(disable_autoload=True)
        policy = CustomPolicy(cfg_policy=cfg, device="cpu")
        env_data = SimpleNamespace(
            base_quat=np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
            base_ang_vel=np.zeros(3, dtype=np.float32),
            dof_pos=np.asarray(cfg.obs_dof.default_pos, dtype=np.float32),
            dof_vel=np.zeros(cfg.obs_dof.num_dofs, dtype=np.float32),
        )

        obs, _ = policy.get_observation(env_data, {})
        action = policy.get_action(obs)

        self.assertEqual(set(obs.keys()), set(cfg.deploy_obs_heads))
        self.assertEqual(action.shape, (cfg.action_dof.num_dofs,))

    def test_custom_policy_wo_gait_commands_are_not_stand_gated(self):
        cfg = CustomPolicyCfg(
            model_backend="torchscript",
            model_dir="assets/models/g1/custom/exported",
            policy_name="policy_wo_gait",
            model_suffix=".pt",
            robot_config_file="assets/models/g1/custom/exported/robot_config.yaml",
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

    def test_custom_policy_reads_external_sensor_from_mapping_env_data(self):
        robot_config_path = _external_sensor_robot_config("camera_front_depth", [2, 2])
        try:
            cfg = CustomPolicyCfg(
                model_backend="onnx",
                model_dir="assets/models/g1/custom/exported_1",
                policy_name="policy_0",
                robot_config_file=robot_config_path.as_posix(),
                disable_autoload=True,
            )
            cfg.deploy_obs_heads = ["external_obs"]
            with mock.patch(
                "onnxruntime.InferenceSession",
                return_value=_FakeOrtSession(["external_obs"], ["actions"]),
            ):
                policy = CustomPolicy(cfg_policy=cfg, device="cpu")

            env_data = {
                "camera_front_depth": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
            }
            obs, extras = policy.get_observation(env_data, {})

            np.testing.assert_allclose(obs["external_obs"], np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))
            self.assertEqual(extras["deploy_obs_heads"], ["external_obs"])
            self.assertIn("external_obs", extras["obs_outputs"])
        finally:
            robot_config_path.unlink(missing_ok=True)

    def test_mujoco_env_cfg_accepts_external_perception_camera_and_height_sampler(self):
        cfg = MujocoEnvCfg(
            xml="robot.xml",
            dof=_minimal_dof_cfg(),
            external_perception={
                "enabled": True,
                "cameras": {
                    "front": {
                        "link_name": "head",
                        "resolution": [64, 48],
                        "render_mode": "depth",
                    }
                },
                "terrain": {
                    "height_samplers": {
                        "height_scan": {
                            "link": "torso_link",
                            "points": {
                                "type": "grid",
                                "size": [0.4, 0.2],
                                "resolution": [0.2, 0.2],
                            },
                        }
                    }
                },
            },
        )

        self.assertTrue(cfg.external_perception.enabled)
        self.assertIn("front", cfg.external_perception.cameras)
        self.assertIn("height_scan", cfg.external_perception.terrain.height_samplers)

    def test_mujoco_env_cfg_accepts_terrain_type(self):
        cfg = MujocoEnvCfg(
            xml="robot.xml",
            dof=_minimal_dof_cfg(),
            terrain={
                "type": "plane",
            },
        )

        self.assertEqual(cfg.terrain.type, "plane")

    def test_mujoco_terrain_generator_supports_empty_and_default_complex_terrain(self):
        from robojudo.environment.utils.mujoco_terrain_generator import (
            make_default_complex_terrain,
            make_empty_terrain,
        )

        with TemporaryDirectory() as tmpdir:
            terrain_dir = Path(tmpdir)
            plane_file = terrain_dir / "plane.xml"
            complex_file = terrain_dir / "complex.xml"

            make_empty_terrain(plane_file.as_posix())
            make_default_complex_terrain(complex_file.as_posix())

            plane_xml = plane_file.read_text(encoding="utf-8")
            complex_xml = complex_file.read_text(encoding="utf-8")

        self.assertIn("<worldbody>", plane_xml)
        self.assertNotIn('type="box"', plane_xml)
        self.assertIn('name="stairs_col_1"', complex_xml)
        self.assertIn('name="platform_col"', complex_xml)

    def test_g1_custom_perception_pipeline_uses_camera_and_height_debug_env(self):
        from robojudo.config.g1.g1_custom_cfg import g1_custom_policy_perception

        cfg = g1_custom_policy_perception()

        self.assertTrue(cfg.env.external_perception.enabled)
        self.assertTrue(cfg.env.external_perception.debug.show_camera_windows)
        self.assertEqual(cfg.env.external_perception.render_visible_geom_groups, [0, 1, 2])
        self.assertIn("front", cfg.env.external_perception.cameras)
        self.assertEqual(cfg.env.external_perception.cameras["front"].render_mode, "both")
        self.assertIn("height_scan", cfg.env.external_perception.terrain.height_samplers)
        self.assertEqual(cfg.env.external_perception.terrain.raycast.geom_groups, [3])

    def test_mujoco_camera_find_body_supports_nested_body_lookup(self):
        from robojudo.environment.perception.mujoco_camera import _find_body

        spec = mujoco.MjSpec.from_string(
            "<mujoco><worldbody><body name='pelvis'><body name='torso_link'/></body><body name='head_link'/></worldbody></mujoco>"
        )

        body = _find_body(spec, "head_link")

        self.assertIsNotNone(body)
        self.assertEqual(body.name, "head_link")

    def test_terrain_height_yaw_transform_supports_many_points(self):
        from robojudo.environment.env_cfgs import ExternalPerceptionDebugCfg, TerrainPerceptionCfg
        from robojudo.environment.perception.terrain_height import TerrainHeightProvider

        provider = TerrainHeightProvider(TerrainPerceptionCfg(), ExternalPerceptionDebugCfg())
        local_points = np.zeros((1, 63, 3), dtype=np.float64)
        local_points[0, :, 0] = np.linspace(-1.0, 1.0, 63)
        link_pos = np.array([[0.0, 0.0, 1.0]], dtype=np.float64)
        link_quat = np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float64)

        world = provider._transform_local_points(
            local_points=local_points,
            link_pos=link_pos,
            link_quat=link_quat,
            follow="yaw",
            offset=[0.0, 0.0, 0.0],
        )

        self.assertEqual(world.shape, (1, 63, 3))
        np.testing.assert_allclose(world[0, :, 0], local_points[0, :, 0])
        np.testing.assert_allclose(world[0, :, 1], 0.0)
        np.testing.assert_allclose(world[0, :, 2], 1.0)

    def test_terrain_height_raycast_ignores_robot_geom_groups(self):
        from robojudo.environment.env_cfgs import (
            ExternalPerceptionDebugCfg,
            TerrainHeightSamplerCfg,
            TerrainPerceptionCfg,
            TerrainRaycastCfg,
        )
        from robojudo.environment.perception.terrain_height import TerrainHeightProvider

        model = mujoco.MjModel.from_xml_string(
            """
            <mujoco>
              <worldbody>
                <geom name='ground' type='plane' size='2 2 0.1'/>
                <body name='torso' pos='0 0 0.4'>
                  <freejoint/>
                  <geom name='robot' type='sphere' size='0.2'/>
                </body>
              </worldbody>
            </mujoco>
            """
        )
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        provider = TerrainHeightProvider(
            TerrainPerceptionCfg(
                raycast=TerrainRaycastCfg(geom_groups=[3], miss_value=100.0),
                height_samplers={
                    "height_scan": TerrainHeightSamplerCfg(
                        link="torso",
                        follow="none",
                        points=[[0.0, 0.0, 0.0]],
                    )
                },
            ),
            ExternalPerceptionDebugCfg(),
        )
        provider.bind(model, data)

        outputs = provider.refresh()

        self.assertIn("height_scan", outputs)
        self.assertAlmostEqual(float(np.asarray(outputs["height_scan"]).reshape(-1)[0]), 0.0, places=4)

    def test_mujoco_env_marks_exit_when_viewer_is_closed(self):
        from robojudo.environment.mujoco_env import MujocoEnv

        class _ClosedViewer:
            is_alive = False

            def close(self):
                return None

        env = object.__new__(MujocoEnv)
        env.viewer = _ClosedViewer()
        env._shutdown_requested = False

        should_continue = env._handle_viewer_state()

        self.assertFalse(should_continue)
        self.assertTrue(env.should_exit)

    def test_mujoco_env_render_visible_groups_include_terrain_groups(self):
        from robojudo.environment.mujoco_env import MujocoEnv

        env = object.__new__(MujocoEnv)
        env.cfg_env = MujocoEnvCfg(
            xml="robot.xml",
            dof=_minimal_dof_cfg(),
            external_perception={
                "enabled": True,
                "terrain": {
                    "raycast": {
                        "geom_groups": [3],
                    }
                },
                "render_visible_geom_groups": [0, 1, 2],
            },
        )

        groups = env._render_visible_geom_groups()

        self.assertEqual(groups, [0, 1, 2, 3])

    def test_mujoco_env_writes_plane_terrain_include_file_from_config(self):
        from robojudo.environment.mujoco_env import MujocoEnv

        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            robot_dir = root / "robots" / "g1"
            terrain_dir = root / "robots" / "terrain"
            robot_dir.mkdir(parents=True)
            terrain_dir.mkdir(parents=True)

            xml_path = robot_dir / "robot.xml"
            xml_path.write_text("<mujoco/>", encoding="utf-8")

            env = object.__new__(MujocoEnv)
            env.cfg_env = MujocoEnvCfg(
                xml=xml_path.as_posix(),
                dof=_minimal_dof_cfg(),
                terrain={"type": "plane"},
            )

            env._prepare_terrain_file()

            terrain_xml = (terrain_dir / "complex_terrain.xml").read_text(encoding="utf-8")

        self.assertIn("<worldbody>", terrain_xml)
        self.assertNotIn('type="box"', terrain_xml)

    def test_mujoco_env_writes_default_complex_terrain_include_file_from_config(self):
        from robojudo.environment.mujoco_env import MujocoEnv

        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            robot_dir = root / "robots" / "g1"
            terrain_dir = root / "robots" / "terrain"
            robot_dir.mkdir(parents=True)
            terrain_dir.mkdir(parents=True)

            xml_path = robot_dir / "robot.xml"
            xml_path.write_text("<mujoco/>", encoding="utf-8")

            env = object.__new__(MujocoEnv)
            env.cfg_env = MujocoEnvCfg(
                xml=xml_path.as_posix(),
                dof=_minimal_dof_cfg(),
                terrain={"type": "complex"},
            )

            env._prepare_terrain_file()

            terrain_xml = (terrain_dir / "complex_terrain.xml").read_text(encoding="utf-8")

        self.assertIn('name="stairs_col_1"', terrain_xml)
        self.assertIn('name="platform_col"', terrain_xml)

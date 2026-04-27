import logging
import time
from pathlib import Path

import mujoco
import mujoco_viewer
import numpy as np

from robojudo.environment import Environment, env_registry
from robojudo.environment.env_cfgs import MujocoEnvCfg
from robojudo.environment.perception import MujocoCameraProvider, PerceptionManager, TerrainHeightProvider
from robojudo.environment.utils.mujoco_terrain_generator import make_default_complex_terrain, make_empty_terrain
from robojudo.environment.utils.mujoco_viz import MujocoVisualizer
from robojudo.config.global_path import ROOT_DIR
from robojudo.utils.util_func import quat_rotate_inverse_np, quatToEuler

logger = logging.getLogger(__name__)


@env_registry.register
class MujocoEnv(Environment):
    cfg_env: MujocoEnvCfg

    def __init__(self, cfg_env: MujocoEnvCfg, device="cpu"):
        super().__init__(cfg_env=cfg_env, device=device)

        self.sim_duration = cfg_env.sim_duration
        self.sim_dt = cfg_env.sim_dt
        self.sim_decimation = cfg_env.sim_decimation
        self.control_dt = self.sim_dt * self.sim_decimation

        self._prepare_terrain_file()
        self.perception_manager = self._build_perception_manager()
        self.model = self._build_model(cfg_env.xml)
        self.model.opt.timestep = self.sim_dt
        self.data = mujoco.MjData(self.model)  # pyright: ignore[reportAttributeAccessIssue]
        # mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        mujoco.mj_step(self.model, self.data)  # pyright: ignore[reportAttributeAccessIssue]

        self.viewer = mujoco_viewer.MujocoViewer(
            self.model,
            self.data,
            width=1200,
            height=900,
            hide_menus=True,
            diable_key_callbacks=True,
        )
        self.viewer.cam.distance = 3.0
        self.viewer.cam.elevation = -10.0
        self.viewer.cam.azimuth = 180.0
        self.viewer.vopt.geomgroup[:] = 0
        for group in self._render_visible_geom_groups():
            self.viewer.vopt.geomgroup[group] = 1
        # self.viewer._paused = True
        if cfg_env.visualize_extras:
            self.visualizer = MujocoVisualizer(self.viewer)
        else:
            self.visualizer = None
        self.perception_manager.bind(self.model, self.data, viewer=self.viewer)

        self.last_time = time.time()

        self.update()  # get initial state

    def _build_perception_manager(self) -> PerceptionManager:
        perception_cfg = self.cfg_env.external_perception
        if not perception_cfg.has_enabled_outputs:
            return PerceptionManager()

        providers = []
        if perception_cfg.cameras:
            providers.append(
                MujocoCameraProvider(
                    perception_cfg.cameras,
                    perception_cfg.debug,
                    visible_geom_groups=self._render_visible_geom_groups(),
                )
            )
        if perception_cfg.terrain.height_samplers:
            providers.append(TerrainHeightProvider(perception_cfg.terrain, perception_cfg.debug))
        return PerceptionManager(providers)

    def _render_visible_geom_groups(self) -> list[int]:
        groups = list(self.cfg_env.external_perception.render_visible_geom_groups)
        if self.cfg_env.external_perception.enabled:
            groups.extend(self.cfg_env.external_perception.terrain.raycast.geom_groups)
        return sorted(set(int(group) for group in groups))

    def _resolve_xml_path(self, xml_path: str) -> Path:
        path = Path(xml_path)
        if path.is_absolute():
            return path
        return ROOT_DIR / path

    def _terrain_include_path(self) -> Path:
        xml_path = self._resolve_xml_path(self.cfg_env.xml)
        return xml_path.parent.parent / "terrain" / "complex_terrain.xml"

    def _prepare_terrain_file(self):
        terrain_file = self._terrain_include_path()
        terrain_file.parent.mkdir(parents=True, exist_ok=True)

        if self.cfg_env.terrain.type == "plane":
            make_empty_terrain(terrain_file.as_posix())
            return

        make_default_complex_terrain(terrain_file.as_posix())

    def _build_model(self, xml_path: str):
        xml_file = self._resolve_xml_path(xml_path)
        if not self.perception_manager.enabled:
            return mujoco.MjModel.from_xml_path(xml_file.as_posix())  # pyright: ignore[reportAttributeAccessIssue]

        try:
            spec = mujoco.MjSpec.from_file(xml_file.as_posix())
        except TypeError:
            spec = mujoco.MjSpec()
            spec.from_file(xml_file.as_posix())
        self.perception_manager.attach_to_spec(spec)
        return spec.compile()

    def reborn(self, init_qpos=None):
        if init_qpos is not None:
            self.data.qpos[0:7] = init_qpos
            self.data.qvel[:] = 0.0
            self.data.ctrl[:] = 0.0
        else:
            mujoco.mj_resetDataKeyframe(self.model, self.data, 0)  # pyright: ignore[reportAttributeAccessIssue]
        mujoco.mj_forward(self.model, self.data)  # pyright: ignore[reportAttributeAccessIssue]

    def reset(self):
        if self.born_place_align:  # TODO: merge
            self.born_place_align = False  # disable during reset
            self.update()
            self.born_place_align = True  # enable after reset
            self.set_born_place()
            self.update()

    def set_gains(self, stiffness, damping):
        assert len(stiffness) == self.num_dofs and len(damping) == self.num_dofs
        self.stiffness = np.asarray(stiffness)
        self.damping = np.asarray(damping)

    def self_check(self):
        pass

    def set_born_place(self, quat: np.ndarray | None = None, pos: np.ndarray | None = None):
        quat_ = self.base_quat if quat is None else quat
        pos_ = self.base_pos if pos is None else pos
        super().set_born_place(quat_, pos_)

    def update(self, simple=False):  # TODO: clean sensors in xml
        """simple: only update dof pos & vel"""
        dof_pos = self.data.qpos.astype(np.float32)[-self.num_dofs :]
        dof_vel = self.data.qvel.astype(np.float32)[-self.num_dofs :]

        self._dof_pos = dof_pos.copy()
        self._dof_vel = dof_vel.copy()

        if simple:
            return

        quat = self.data.qpos.astype(np.float32)[3:7][[1, 2, 3, 0]]
        ang_vel = self.data.qvel.astype(np.float32)[3:6]
        base_pos = self.data.qpos.astype(np.float32)[:3]
        lin_vel = self.data.qvel.astype(np.float32)[0:3]

        if self.born_place_align:
            quat, base_pos = self.base_align.align_transform(quat, base_pos)

        lin_vel = quat_rotate_inverse_np(quat, lin_vel)
        rpy = quatToEuler(quat)

        self._base_rpy = rpy.copy()
        self._base_quat = quat.copy()
        self._base_ang_vel = ang_vel.copy()

        self._base_pos = base_pos.copy()
        self._base_lin_vel = lin_vel.copy()

        if self.update_with_fk:
            fk_info = self.fk()
            self._fk_info = fk_info.copy()
            
            self._torso_ang_vel = fk_info[self._torso_name]["ang_vel"]
            self._torso_quat = fk_info[self._torso_name]["quat"]
            self._torso_pos = fk_info[self._torso_name]["pos"]

        self.set_extra_env_data(self.perception_manager.refresh())

    def _handle_viewer_state(self) -> bool:
        if self.viewer is None:
            return True
        if self.viewer.is_alive:
            return True
        self.shutdown()
        return False

    def step(self, pd_target, hand_pose=None):
        assert len(pd_target) == self.num_dofs, "pd_target len should be num_dofs of env"

        if hand_pose is not None:
            logger.info("Hand pose-->", hand_pose)

        pd_target = self.apply_pd_target_safety(pd_target)

        for _ in range(self.sim_decimation):
            torque = (pd_target - self.dof_pos) * self.stiffness - self.dof_vel * self.damping
            if self.torque_limits is not None:
                torque = np.clip(torque, -self.torque_limits, self.torque_limits)

            self.data.ctrl = torque

            mujoco.mj_step(self.model, self.data)  # pyright: ignore[reportAttributeAccessIssue]
            self.update(simple=True)
        self.update(simple=False)
        self.perception_manager.render_debug(viewer=self.viewer)
        if not self._handle_viewer_state():
            return
        self.viewer.cam.lookat = self.data.qpos.astype(np.float32)[:3]
        self.viewer.render()
        self._handle_viewer_state()

    def shutdown(self):
        if self.should_exit:
            return
        self.request_shutdown()
        perception_manager = getattr(self, "perception_manager", None)
        if perception_manager is not None:
            perception_manager.close()
        viewer = getattr(self, "viewer", None)
        if viewer is not None:
            viewer.close()


if __name__ == "__main__":
    from robojudo.config.g1.env.g1_mujuco_env_cfg import G1MujocoEnvCfg

    mujoco_env = MujocoEnv(cfg_env=G1MujocoEnvCfg())
    mujoco_env.viewer._paused = False

    while True:
        # mujoco_env.update()
        mujoco_env.step(np.zeros(mujoco_env.num_dofs))
        time.sleep(0.02)

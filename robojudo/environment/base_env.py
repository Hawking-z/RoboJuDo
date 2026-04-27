from abc import ABC, abstractmethod

import numpy as np
from box import Box

from robojudo.tools.dof import merge_dof_cfgs
from robojudo.tools.kinematics import MujocoKinematics
from robojudo.tools.tool_cfgs import DoFConfig
from robojudo.utils.rotation import TransformAlignment

from .env_cfgs import EnvCfg


class Environment(ABC):
    def __init__(self, cfg_env: EnvCfg, device: str = "cpu"):
        self.cfg_env = cfg_env
        self.device = device

        self.kinematics = None
        if self.cfg_env.forward_kinematic is not None:
            self.kinematics = MujocoKinematics(cfg=self.cfg_env.forward_kinematic)
        self.update_with_fk = self.cfg_env.update_with_fk
        self._torso_name = self.cfg_env.torso_name

        # init env dofs config
        self.update_dof_cfg()

        # Feedback variables, all in radian
        self._dof_pos = np.zeros(self.num_dofs)
        self._dof_vel = np.zeros(self.num_dofs)
        self._base_rpy = np.zeros(3)
        self._base_quat = np.array([0.0, 0.0, 0.0, 1.0])  # as x, y, z, w
        self._base_ang_vel = np.zeros(3)

        # optional variables, may not be available in all envs
        self._base_pos: np.ndarray | None = None
        self._base_lin_vel: np.ndarray | None = None
        self._base_lin_acc: np.ndarray | None = None
        self._torso_pos: np.ndarray | None = None
        self._torso_quat: np.ndarray | None = None
        self._torso_ang_vel: np.ndarray | None = None
        self._fk_info: dict | None = None
        self._extra_env_data: dict[str, object] = {}
        self._shutdown_requested = False

        # born place alignment
        self.born_place_align = self.cfg_env.born_place_align
        self.base_align = TransformAlignment(yaw_only=True, xy_only=True)

        self.visualizer = None  # for sim viz debug plot

    def update_dof_cfg(self, override_cfg: DoFConfig | None = None):
        dof_config: DoFConfig = self.cfg_env.dof
        if override_cfg is not None:
            dof_config = merge_dof_cfgs(self.cfg_env.dof, override_cfg)

        self.dof_cfg = dof_config
        self.joint_names = dof_config.joint_names
        self.num_dofs = dof_config.num_dofs
        self.default_pos = np.asarray(dof_config.default_pos)
        self.stiffness = np.asarray(dof_config.stiffness)
        self.damping = np.asarray(dof_config.damping)
        self.torque_limits = None if dof_config.torque_limits is None else np.asarray(dof_config.torque_limits)
        self.position_limits = (
            None if dof_config.position_limits is None else np.asarray(dof_config.position_limits)
        )

        self.set_gains(self.stiffness, self.damping)  # TODO: temp solution

        # if self.kinematics is not None: # TODO: check usage
        #     self.kinematics.update_joint_names_subset(self.joint_names)

    def clip_position_target(self, pd_target) -> np.ndarray:
        target = np.asarray(pd_target).copy()
        if not np.issubdtype(target.dtype, np.floating):
            target = target.astype(np.float32)
        if self.position_limits is None:
            return target
        return np.clip(target, self.position_limits[:, 0], self.position_limits[:, 1])

    def clip_torque_target(self, pd_target, *, dof_pos, dof_vel, stiffness, damping) -> np.ndarray:
        target = np.asarray(pd_target).copy()
        if not np.issubdtype(target.dtype, np.floating):
            target = target.astype(np.float32)
        if self.torque_limits is None:
            return target

        dof_pos = np.asarray(dof_pos, dtype=target.dtype)
        dof_vel = np.asarray(dof_vel, dtype=target.dtype)
        stiffness = np.asarray(stiffness, dtype=target.dtype)
        damping = np.asarray(damping, dtype=target.dtype)
        torque_limits = np.asarray(self.torque_limits, dtype=target.dtype) * float(self.cfg_env.torque_limits_ratio)

        active = np.abs(stiffness) > np.finfo(target.dtype).eps
        if not np.any(active):
            return target

        p_limits_low = -torque_limits + damping * dof_vel
        p_limits_high = torque_limits + damping * dof_vel
        action_low = np.empty_like(target)
        action_high = np.empty_like(target)
        action_low[active] = p_limits_low[active] / stiffness[active] + dof_pos[active]
        action_high[active] = p_limits_high[active] / stiffness[active] + dof_pos[active]

        lower = np.minimum(action_low[active], action_high[active])
        upper = np.maximum(action_low[active], action_high[active])
        target[active] = np.clip(target[active], lower, upper)
        return target

    def position_safety_violations(self, dof_pos, *, protect_ratio: float) -> np.ndarray:
        if self.position_limits is None:
            return np.array([], dtype=np.int64)

        pos = np.asarray(dof_pos)
        joint_pos_mid = (self.position_limits[:, 1] + self.position_limits[:, 0]) / 2.0
        joint_pos_range = (self.position_limits[:, 1] - self.position_limits[:, 0]) / 2.0
        high = joint_pos_mid + joint_pos_range * protect_ratio
        low = joint_pos_mid - joint_pos_range * protect_ratio
        return np.flatnonzero((pos > high) | (pos < low))

    def set_born_place(self, quat: np.ndarray | None = None, pos: np.ndarray | None = None):
        """Need to be called with real quat and pos from subclass"""
        self.base_align.set_base(quat, pos)

    @abstractmethod
    def self_check(self):
        raise NotImplementedError

    @abstractmethod
    def reset(self):
        raise NotImplementedError

    @abstractmethod
    def update(self):
        raise NotImplementedError

    def fk(self):
        if self.kinematics is None:
            raise ValueError("Kinematics model not initialized.")
        fk_info = self.kinematics.forward(
            joint_pos=self.dof_pos,
            joint_vel=self.dof_vel,
            base_pos=self.base_pos,
            base_quat=self.base_quat,
            base_ang_vel=self.base_ang_vel,
            base_lin_vel=self.base_lin_vel,
        )
        return fk_info

    def apply_pd_target_safety(self, pd_target) -> np.ndarray:
        target = np.asarray(pd_target).copy()
        if self.cfg_env.clip_position_limits:
            target = self.clip_position_target(target)
        if self.cfg_env.clip_torque_limits:
            target = self.clip_torque_target(
                target,
                dof_pos=self.dof_pos,
                dof_vel=self.dof_vel,
                stiffness=self.stiffness,
                damping=self.damping,
            )
        return target

    def unsafe_dof_position_indices(self) -> np.ndarray:
        if self.cfg_env.joint_pos_protect_ratio is None:
            return np.array([], dtype=np.int64)
        return self.position_safety_violations(
            self.dof_pos,
            protect_ratio=self.cfg_env.joint_pos_protect_ratio,
        )

    @abstractmethod
    def step(self, pd_target, hand_pose=None):
        assert len(pd_target) == self.num_dofs, "pd_target len should be num_dofs of env"
        raise NotImplementedError

    @abstractmethod
    def shutdown(self):
        raise NotImplementedError

    @abstractmethod
    def set_gains(self, stiffness, damping):
        raise NotImplementedError

    # === Properties ===
    # ALL in radian
    @property
    def dof_pos(self):
        return self._dof_pos.copy()

    @property
    def dof_vel(self):
        return self._dof_vel.copy()

    # @property # TODO: disabled due to conflict with base_quat
    # def base_rpy(self):
    #     return self._base_rpy

    @property
    def base_quat(self):
        return self._base_quat.copy()

    @property
    def base_ang_vel(self):
        return self._base_ang_vel.copy()

    # == Optional Properties ==
    @property
    def base_pos(self):
        return self._base_pos.copy() if self._base_pos is not None else None

    @property
    def base_lin_vel(self):
        return self._base_lin_vel.copy() if self._base_lin_vel is not None else None

    @property
    def base_lin_acc(self):
        return self._base_lin_acc.copy() if self._base_lin_acc is not None else None

    @property
    def torso_pos(self):
        return self._torso_pos.copy() if self._torso_pos is not None else None

    @property
    def torso_quat(self):
        return self._torso_quat.copy() if self._torso_quat is not None else None

    @property
    def torso_ang_vel(self):
        return self._torso_ang_vel.copy() if self._torso_ang_vel is not None else None

    @property
    def fk_info(self):
        return self._fk_info.copy() if self._fk_info is not None else None

    def get_data(self):
        env_data = {
            "dof_pos": self.dof_pos,
            "dof_vel": self.dof_vel,
            # "base_rpy": self.base_rpy,
            "base_quat": self.base_quat,
            "base_ang_vel": self.base_ang_vel,
            "base_lin_acc": self.base_lin_acc,
            "base_pos": self.base_pos,
            "base_lin_vel": self.base_lin_vel,
            "torso_pos": self.torso_pos,
            "torso_quat": self.torso_quat,
            "torso_ang_vel": self.torso_ang_vel,
            "fk_info": self.fk_info,
        }
        env_data.update(self._extra_env_data)
        return Box(env_data)

    def set_extra_env_data(self, data: dict[str, object] | None):
        self._extra_env_data = {} if data is None else dict(data)

    def request_shutdown(self):
        self._shutdown_requested = True

    @property
    def should_exit(self) -> bool:
        return self._shutdown_requested

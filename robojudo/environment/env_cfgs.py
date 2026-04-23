from typing import Any, Literal

from pydantic import model_validator

from robojudo.config import Config
from robojudo.tools.tool_cfgs import DoFConfig, ForwardKinematicCfg, ZedOdometryCfg


class ExternalPerceptionDebugCfg(Config):
    show_camera_windows: bool = False
    draw_height_points: bool = True


class MujocoCameraPerceptionCfg(Config):
    enabled: bool = True
    link_name: str
    resolution: list[int] = [64, 48]
    hfov: float = 58.0
    pos: list[float] = [0.0, 0.0, 0.0]
    rot: list[float] = [0.0, 0.0, 0.0]
    near: float = 0.3
    far: float = 3.0
    render_mode: Literal["depth", "color", "both"] = "depth"
    depth_output: str | None = None
    color_output: str | None = None

    def depth_output_key(self, name: str) -> str:
        return self.depth_output or f"camera_{name}_depth"

    def color_output_key(self, name: str) -> str:
        return self.color_output or f"camera_{name}_color"

    @model_validator(mode="after")
    def validate_camera(self):
        if len(self.resolution) != 2 or any(int(dim) <= 0 for dim in self.resolution):
            raise ValueError("camera resolution must be [width, height] with positive integers")
        if len(self.pos) != 3:
            raise ValueError("camera pos must contain 3 values")
        if len(self.rot) != 3:
            raise ValueError("camera rot must contain 3 values")
        if self.near <= 0.0:
            raise ValueError("camera near must be positive")
        if self.far <= self.near:
            raise ValueError("camera far must be greater than near")
        return self


class TerrainRaycastCfg(Config):
    origin_z_offset: float = 5.0
    geom_groups: list[int] = [3]
    miss_value: float = 100.0

    @model_validator(mode="after")
    def validate_raycast(self):
        if any(group < 0 or group >= 6 for group in self.geom_groups):
            raise ValueError("terrain geom_groups must be within [0, 5]")
        return self


class TerrainHeightSamplerCfg(Config):
    enabled: bool = True
    link: str
    follow: Literal["none", "yaw", "full"] = "yaw"
    offset: list[float] = [0.0, 0.0, 0.0]
    points: Any
    output_key: str | None = None

    def resolved_output_key(self, name: str) -> str:
        return self.output_key or name

    @model_validator(mode="after")
    def validate_sampler(self):
        if len(self.offset) != 3:
            raise ValueError("height sampler offset must contain 3 values")
        return self


class TerrainPerceptionCfg(Config):
    raycast: TerrainRaycastCfg = TerrainRaycastCfg()
    height_samplers: dict[str, TerrainHeightSamplerCfg] = {}


class MujocoTerrainCfg(Config):
    type: Literal["plane", "complex"] = "plane"


class ExternalPerceptionCfg(Config):
    enabled: bool = False
    render_visible_geom_groups: list[int] = [0, 1, 2]
    cameras: dict[str, MujocoCameraPerceptionCfg] = {}
    terrain: TerrainPerceptionCfg = TerrainPerceptionCfg()
    debug: ExternalPerceptionDebugCfg = ExternalPerceptionDebugCfg()

    @property
    def has_enabled_outputs(self) -> bool:
        if not self.enabled:
            return False
        if any(camera.enabled for camera in self.cameras.values()):
            return True
        return any(sampler.enabled for sampler in self.terrain.height_samplers.values())


class EnvCfg(Config):
    env_type: str  # name of the environment class
    is_sim: bool = False

    urdf: str | None = None
    xml: str
    body_names: list[str] | None = None

    dof: DoFConfig

    forward_kinematic: ForwardKinematicCfg | None = None
    update_with_fk: bool = False
    """Whether to update info from fk"""
    torso_name: str = "torso_link"
    """Name of the torso link, used in fk info extraction"""

    born_place_align: bool = True
    """Whether to align the born place to zero position and heading"""


class MujocoEnvCfg(EnvCfg):
    env_type: str = "MujocoEnv"
    is_sim: bool = True
    # ====== ENV CONFIGURATION ======
    sim_duration: float = 60.0
    sim_dt: float = 0.001
    sim_decimation: int = 20

    visualize_extras: bool = True  # TODO: remove
    terrain: MujocoTerrainCfg = MujocoTerrainCfg()
    external_perception: ExternalPerceptionCfg = ExternalPerceptionCfg()


class RobotEnvCfg(EnvCfg):
    env_type: str = "DummyEnv"
    is_sim: bool = False
    # ====== ENV CONFIGURATION ======
    act: bool = True

    odometry_type: Literal["NONE", "DUMMY", "ZED"] = "NONE"
    zed_cfg: ZedOdometryCfg | None = None
    """ZED odometry config, if odometry_type is "ZED", this must be set"""

    @model_validator(mode="after")
    def check_zed_config(self):
        if self.odometry_type == "ZED" and self.zed_cfg is None:
            raise ValueError("zed_cfg must be set if odometry_type is 'ZED'")
        return self


class UnitreeEnvCfg(RobotEnvCfg):
    """
    Configuration for Unitree Robot environment.
    """

    class UnitreeCfg(Config):
        """Unitree SDK configuration"""

        net_if: str = "eth0"
        """network interface to communicate with the robot"""

        robot: Literal["h1", "g1"]
        msg_type: Literal["hg", "go"]
        control_mode: str = "position"
        hand_type: Literal["Dex-3", "Inspire", "NONE"] = "NONE"

        lowcmd_topic: str = "rt/lowcmd"
        lowstate_topic: str = "rt/lowstate"

        enable_odometry: bool = False
        sport_state_topic: str = "rt/odommodestate"

        control_dt: float = 0.02
        """control command dt"""

    env_type: str = "UnitreeEnv"  # For unitree_sdk2py
    # env_type: str = "UnitreeCppEnv" # For unitree_cpp
    """UnitreeEnv for unitree_sdk2py, UnitreeCppEnv for unitree_cpp, check README for more details"""

    unitree: UnitreeCfg

    odometry_type: Literal["NONE", "DUMMY", "UNITREE", "ZED"] = "DUMMY"  # pyright: ignore[reportIncompatibleVariableOverride]

    joint2motor_idx: list[int] | None = None
    """Mapping from env dof to motor index, None for direct mapping"""
    weak_motor: list[int] = []

    hand_retarget: None = None  # TODO

    @model_validator(mode="after")
    def check_joint2motor_idx(self):
        if self.joint2motor_idx is not None and len(self.joint2motor_idx) != self.dof.num_dofs:
            raise ValueError("joint2motor_idx length must match dof.num_dofs")
        return self

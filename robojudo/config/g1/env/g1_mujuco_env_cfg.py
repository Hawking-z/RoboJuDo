import numpy as np

from robojudo.environment.env_cfgs import MujocoEnvCfg

from .g1_env_cfg import G1_12EnvCfg, G1_23EnvCfg, G1EnvCfg

from robojudo.environment.env_cfgs import (
    ExternalPerceptionCfg,
    ExternalPerceptionDebugCfg,
    MujocoCameraPerceptionCfg,
    MujocoTerrainCfg,
    TerrainHeightSamplerCfg,
    TerrainPerceptionCfg,
    TerrainRaycastCfg,
)

class G1MujocoEnvCfg(G1EnvCfg, MujocoEnvCfg):
    env_type: str = MujocoEnvCfg.model_fields["env_type"].default
    is_sim: bool = MujocoEnvCfg.model_fields["is_sim"].default
    # ====== ENV CONFIGURATION ======

    update_with_fk: bool = True


class G1_23MujocoEnvCfg(G1_23EnvCfg, MujocoEnvCfg):
    env_type: str = MujocoEnvCfg.model_fields["env_type"].default
    is_sim: bool = MujocoEnvCfg.model_fields["is_sim"].default
    # ====== ENV CONFIGURATION ======
    update_with_fk: bool = True


class G1_12MujocoEnvCfg(G1_12EnvCfg, MujocoEnvCfg):
    env_type: str = MujocoEnvCfg.model_fields["env_type"].default
    is_sim: bool = MujocoEnvCfg.model_fields["is_sim"].default
    # ====== ENV CONFIGURATION ======
    update_with_fk: bool = False

class G1PerceptionMujocoEnvCfg(G1MujocoEnvCfg):
    terrain: MujocoTerrainCfg = MujocoTerrainCfg(type="complex")
    external_perception: ExternalPerceptionCfg = ExternalPerceptionCfg(
        enabled=True,
        cameras={
            "d435i": MujocoCameraPerceptionCfg(
                link_name="torso_link",
                resolution=[480, 270],
                hfov=89.51,
                pos=[
                    0.04764571478 + 0.0039635,
                    -0.01,
                    0.46268178553 - 0.044 + 0.016,],
                rot=[np.radians(0.5), np.radians(48),0],
                near=0.1,
                far=2.5,
                render_mode="depth",
            ),
        },
        # terrain=TerrainPerceptionCfg(
        #     raycast=TerrainRaycastCfg(
        #         origin_z_offset=10.0,
        #         geom_groups=[3],
        #         miss_value=100.0,
        #     ),
        #     height_samplers={
        #         "height_scan": TerrainHeightSamplerCfg(
        #             link="pelvis",
        #             follow="yaw",
        #             offset=[0.10, 0.0, 0.0],
        #             points={
        #                 "type": "grid",
        #                 "x": [ -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        #                 "y": [-0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4],
        #             },
        #         ),
        #     },
        # ),
        debug=ExternalPerceptionDebugCfg(
            show_camera_windows=True,
            draw_height_points=True,
            draw_camera_frustum=True,
        ),
    )

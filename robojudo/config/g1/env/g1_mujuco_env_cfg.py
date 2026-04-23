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
            "front": MujocoCameraPerceptionCfg(
                link_name="torso_link",
                resolution=[320, 240],
                hfov=75.0,
                pos=[0.22, 0.0, 0.10],
                rot=[0.0, 0.30, 0.0],
                near=0.2,
                far=4.0,
                render_mode="both",
            ),
        },
        terrain=TerrainPerceptionCfg(
            raycast=TerrainRaycastCfg(
                origin_z_offset=5.0,
                geom_groups=[3],
                miss_value=100.0,
            ),
            height_samplers={
                "height_scan": TerrainHeightSamplerCfg(
                    link="torso_link",
                    follow="yaw",
                    offset=[0.10, 0.0, 0.0],
                    points={
                        "type": "grid",
                        "size": [0.8, 0.6],
                        "resolution": [0.1, 0.1],
                    },
                ),
            },
        ),
        debug=ExternalPerceptionDebugCfg(
            show_camera_windows=True,
            draw_height_points=True,
        ),
    )

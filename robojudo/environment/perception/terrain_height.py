from __future__ import annotations

import re

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R

from robojudo.environment.env_cfgs import ExternalPerceptionDebugCfg, TerrainHeightSamplerCfg, TerrainPerceptionCfg
from robojudo.environment.perception.base import PerceptionProvider

TERRAIN_GEOM_GROUP = 3
RAYCAST_HIT_RGBA = np.array([0.2, 0.7, 1.0, 1.0])
RAYCAST_MISS_RGBA = np.array([1.0, 0.2, 0.2, 1.0])


def _match_name(name: str, pattern: str) -> bool:
    if name == pattern:
        return True
    return re.fullmatch(pattern, name) is not None


class TerrainHeightProvider(PerceptionProvider):
    def __init__(self, terrain_cfg: TerrainPerceptionCfg, debug_cfg: ExternalPerceptionDebugCfg):
        self.terrain_cfg = terrain_cfg
        self.debug_cfg = debug_cfg
        self.model = None
        self.data = None
        self.body_names: list[str] = []
        self.body_name_to_id: dict[str, int] = {}
        self.samplers: dict[str, dict[str, object]] = {}
        self._terrain_ray_geomgroup = np.zeros(6, dtype=np.uint8)
        self._terrain_ray_geomgroup[self.terrain_cfg.raycast.geom_groups] = 1
        self._last_debug_points: list[tuple[np.ndarray, bool]] = []

    def bind(self, model, data, viewer=None) -> None:
        self.model = model
        self.data = data
        self._refresh_body_metadata()
        self._prepare_terrain_geom_groups()
        self._initialize_samplers()

    def refresh(self) -> dict[str, object]:
        outputs: dict[str, object] = {}
        if self.model is None or self.data is None or not self.samplers:
            return outputs

        mujoco.mj_forward(self.model, self.data)
        body_pos = np.asarray(self.data.xpos, dtype=np.float64)
        body_quat = np.asarray(self.data.xquat, dtype=np.float64)[:, [1, 2, 3, 0]]
        debug_points: list[tuple[np.ndarray, bool]] = []

        for name, sampler in self.samplers.items():
            link_ids = sampler["link_ids"]
            local_points = sampler["local_points"]
            world_points = self._transform_local_points(
                local_points=local_points,
                link_pos=body_pos[link_ids],
                link_quat=body_quat[link_ids],
                follow=sampler["cfg"].follow,
                offset=sampler["cfg"].offset,
            )
            heights, hit_mask = self._query_points(world_points)
            sampler["world_points"] = world_points
            sampler["heights"] = heights
            sampler["hit_mask"] = hit_mask

            output = heights[0] if heights.shape[0] == 1 else heights
            outputs[sampler["cfg"].resolved_output_key(name)] = output.astype(np.float32, copy=False)

            flat_points = world_points.reshape(-1, 3)
            flat_heights = heights.reshape(-1)
            flat_hits = hit_mask.reshape(-1)
            for point, height, hit in zip(flat_points, flat_heights, flat_hits, strict=False):
                debug_pos = np.array([point[0], point[1], height if hit else point[2]], dtype=np.float64)
                debug_points.append((debug_pos, bool(hit)))

        self._last_debug_points = debug_points
        return outputs

    def render_debug(self, viewer=None) -> None:
        if viewer is None or not self.debug_cfg.draw_height_points:
            return
        for idx, (point, hit) in enumerate(self._last_debug_points):
            radius = 0.01 if hit else 0.008
            rgba = RAYCAST_HIT_RGBA if hit else RAYCAST_MISS_RGBA
            viewer.add_marker(
                pos=point,
                size=np.array([radius, radius, radius]),
                rgba=rgba,
                type=mujoco.mjtGeom.mjGEOM_SPHERE,
                label="",
                id=7000 + idx,
            )

    def _refresh_body_metadata(self) -> None:
        self.body_names = []
        self.body_name_to_id = {}
        for body_id in range(self.model.nbody):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body_id)
            if name is None:
                continue
            self.body_names.append(name)
            self.body_name_to_id[name] = body_id

    def _prepare_terrain_geom_groups(self) -> None:
        terrain_group = self.terrain_cfg.raycast.geom_groups[0] if self.terrain_cfg.raycast.geom_groups else TERRAIN_GEOM_GROUP
        world_geom_ids = np.flatnonzero(np.asarray(self.model.geom_bodyid) == 0).astype(np.int32)
        if world_geom_ids.size == 0:
            return
        for geom_id in world_geom_ids:
            if self.model.geom_group[geom_id] not in self.terrain_cfg.raycast.geom_groups:
                self.model.geom_group[geom_id] = terrain_group

    def _initialize_samplers(self) -> None:
        self.samplers = {}
        for name, cfg in self.terrain_cfg.height_samplers.items():
            if not cfg.enabled:
                continue

            local_points = self._build_points(cfg.points)
            matched_names = [body_name for body_name in self.body_names if _match_name(body_name, cfg.link)]
            if not matched_names:
                raise ValueError(f"No body matched height sampler link pattern '{cfg.link}'")

            link_ids = np.asarray([self.body_name_to_id[body_name] for body_name in matched_names], dtype=np.int64)
            local_points = np.repeat(local_points[None, :, :], link_ids.size, axis=0)
            self.samplers[name] = {
                "cfg": cfg,
                "link_ids": link_ids,
                "local_points": local_points,
                "world_points": np.zeros_like(local_points, dtype=np.float64),
                "heights": np.zeros(local_points.shape[:2], dtype=np.float64),
                "hit_mask": np.zeros(local_points.shape[:2], dtype=bool),
            }

    def _build_points(self, points_cfg) -> np.ndarray:
        if isinstance(points_cfg, dict):
            if points_cfg.get("type") != "grid":
                raise ValueError(f"Unsupported height sampler points type: {points_cfg.get('type')}")

            if "x" in points_cfg and "y" in points_cfg:
                xs = np.asarray(points_cfg["x"], dtype=np.float64)
                ys = np.asarray(points_cfg["y"], dtype=np.float64)
            else:
                size = np.asarray(points_cfg["size"], dtype=np.float64)
                resolution = np.asarray(points_cfg["resolution"], dtype=np.float64)
                xs = np.arange(-size[0] / 2.0, size[0] / 2.0 + 1e-6, resolution[0], dtype=np.float64)
                ys = np.arange(-size[1] / 2.0, size[1] / 2.0 + 1e-6, resolution[1], dtype=np.float64)
            gx, gy = np.meshgrid(xs, ys, indexing="ij")
            points = np.zeros((gx.size, 3), dtype=np.float64)
            points[:, 0] = gx.reshape(-1)
            points[:, 1] = gy.reshape(-1)
            return points

        points = np.asarray(points_cfg, dtype=np.float64)
        if points.ndim == 1:
            if points.size % 3 != 0:
                raise ValueError("Explicit height sampler points must be divisible into xyz triplets")
            points = points.reshape(-1, 3)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("Height sampler points must have shape [P, 3]")
        return points

    def _transform_local_points(
        self,
        local_points: np.ndarray,
        link_pos: np.ndarray,
        link_quat: np.ndarray,
        follow: str,
        offset: list[float],
    ) -> np.ndarray:
        points = local_points + np.asarray(offset, dtype=np.float64).reshape(1, 1, 3)
        if follow == "none":
            return points + link_pos[:, None, :]

        if follow == "yaw":
            yaw = self._yaw_from_quat_xyzw(link_quat).reshape(-1, 1)
            cos_yaw = np.cos(yaw)
            sin_yaw = np.sin(yaw)
            rotated = points.copy()
            rotated[..., 0] = cos_yaw * points[..., 0] - sin_yaw * points[..., 1]
            rotated[..., 1] = sin_yaw * points[..., 0] + cos_yaw * points[..., 1]
        elif follow == "full":
            point_count = points.shape[1]
            repeated_quat = np.repeat(link_quat[:, None, :], point_count, axis=1).reshape(-1, 4)
            rotated = R.from_quat(repeated_quat).apply(points.reshape(-1, 3))
            rotated = rotated.reshape(points.shape)
        else:
            raise ValueError(f"Unsupported follow mode '{follow}'")
        return rotated + link_pos[:, None, :]

    def _query_points(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        flat_points = points.reshape(-1, 3)
        flat_heights = np.empty(flat_points.shape[0], dtype=np.float64)
        flat_hits = np.zeros(flat_points.shape[0], dtype=bool)
        for idx, point in enumerate(flat_points):
            origin = point.copy()
            origin[2] += self.terrain_cfg.raycast.origin_z_offset
            geomid = np.array([-1], dtype=np.int32)
            distance = mujoco.mj_ray(
                self.model,
                self.data,
                origin,
                np.array([0.0, 0.0, -1.0], dtype=np.float64),
                self._terrain_ray_geomgroup,
                1,
                -1,
                geomid,
            )
            if distance < 0.0:
                flat_heights[idx] = self.terrain_cfg.raycast.miss_value
            else:
                flat_heights[idx] = origin[2] - distance
                flat_hits[idx] = True
        return flat_heights.reshape(points.shape[:2]), flat_hits.reshape(points.shape[:2])

    @staticmethod
    def _yaw_from_quat_xyzw(quat_xyzw: np.ndarray) -> np.ndarray:
        x = quat_xyzw[..., 0]
        y = quat_xyzw[..., 1]
        z = quat_xyzw[..., 2]
        w = quat_xyzw[..., 3]
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        return np.arctan2(siny_cosp, cosy_cosp)

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R

from robojudo.environment.env_cfgs import ExternalPerceptionDebugCfg, MujocoCameraPerceptionCfg
from robojudo.environment.perception.base import PerceptionProvider

MUJOCO_CAMERA_FRAME_QUAT_WXYZ = np.array([0.5, 0.5, -0.5, -0.5], dtype=np.float64)
MUJOCO_CAMERA_FRAME_ROTATION = R.from_quat(MUJOCO_CAMERA_FRAME_QUAT_WXYZ[[1, 2, 3, 0]])


@dataclass
class CameraRuntime:
    name: str
    cfg: MujocoCameraPerceptionCfg
    renderer: mujoco.Renderer | None = None


def _iter_bodies(root_body):
    body = root_body.first_body()
    while body is not None:
        yield body
        yield from _iter_bodies(body)
        body = root_body.next_body(body)


def _find_body(spec, body_name: str):
    if hasattr(spec, "find_body"):
        body = spec.find_body(body_name)
        if body is not None:
            return body

    if hasattr(spec, "body"):
        body = spec.body(body_name)
        if body is not None and getattr(body, "name", None) == body_name:
            return body

    if spec.worldbody.name == body_name:
        return spec.worldbody
    for body in _iter_bodies(spec.worldbody):
        if body.name == body_name:
            return body
    return None


class MujocoCameraProvider(PerceptionProvider):
    def __init__(
        self,
        cameras: dict[str, MujocoCameraPerceptionCfg],
        debug_cfg: ExternalPerceptionDebugCfg,
        visible_geom_groups: list[int] | None = None,
    ):
        self.cameras = {
            name: CameraRuntime(name=name, cfg=cfg)
            for name, cfg in cameras.items()
            if cfg.enabled
        }
        self.debug_cfg = debug_cfg
        self.model = None
        self.data = None
        self.last_outputs: dict[str, object] = {}
        self.scene_option = self._build_scene_option(visible_geom_groups or [0, 1, 2])

    def attach_to_spec(self, spec) -> None:
        for runtime in self.cameras.values():
            body = _find_body(spec, runtime.cfg.link_name)
            if body is None:
                raise ValueError(f"Body '{runtime.cfg.link_name}' not found in MJCF for camera '{runtime.name}'")
            camera = body.add_camera()
            camera.name = self._camera_model_name(runtime.name)
            camera.pos = np.asarray(runtime.cfg.pos, dtype=np.float64)
            camera.quat = self._euler_xyz_to_mujoco_camera_quat(runtime.cfg.rot)
            camera.fovy = self._horizontal_to_vertical_fov(
                runtime.cfg.hfov,
                runtime.cfg.resolution[0],
                runtime.cfg.resolution[1],
            )

    def bind(self, model, data, viewer=None) -> None:
        self.model = model
        self.data = data
        for runtime in self.cameras.values():
            try:
                runtime.renderer = mujoco.Renderer(
                    model,
                    height=int(runtime.cfg.resolution[1]),
                    width=int(runtime.cfg.resolution[0]),
                )
            except Exception as exc:
                raise RuntimeError(
                    "Failed to initialize MuJoCo camera renderer. "
                    "Set a working OpenGL backend such as MUJOCO_GL=egl for headless rendering."
                ) from exc

    def refresh(self) -> dict[str, object]:
        outputs: dict[str, object] = {}
        if self.model is None or self.data is None:
            return outputs

        for runtime in self.cameras.values():
            renderer = runtime.renderer
            if renderer is None:
                continue

            if runtime.cfg.render_mode in {"color", "both"}:
                color = self._render(renderer, runtime, depth=False)
                outputs[runtime.cfg.color_output_key(runtime.name)] = color

            if runtime.cfg.render_mode in {"depth", "both"}:
                depth = self._render(renderer, runtime, depth=True)
                outputs[runtime.cfg.depth_output_key(runtime.name)] = np.clip(depth, runtime.cfg.near, runtime.cfg.far)

        self.last_outputs = outputs
        return outputs

    def render_debug(self, viewer=None) -> None:
        if not self.debug_cfg.show_camera_windows or not self.last_outputs:
            return

        for runtime in self.cameras.values():
            color_key = runtime.cfg.color_output_key(runtime.name)
            depth_key = runtime.cfg.depth_output_key(runtime.name)
            if color_key in self.last_outputs:
                cv2.imshow(color_key, cv2.cvtColor(self.last_outputs[color_key], cv2.COLOR_RGB2BGR))
            if depth_key in self.last_outputs:
                cv2.imshow(depth_key, self._depth_to_display(self.last_outputs[depth_key], runtime.cfg.near, runtime.cfg.far))
        cv2.waitKey(1)

    def close(self) -> None:
        for runtime in self.cameras.values():
            if runtime.renderer is not None:
                runtime.renderer.close()
        if self.debug_cfg.show_camera_windows:
            cv2.destroyAllWindows()

    def _render(self, renderer: mujoco.Renderer, runtime: CameraRuntime, *, depth: bool):
        if depth:
            renderer.enable_depth_rendering()
        else:
            renderer.disable_depth_rendering()
        renderer.update_scene(
            self.data,
            camera=self._camera_model_name(runtime.name),
            scene_option=self.scene_option,
        )
        frame = renderer.render().copy()
        if depth:
            renderer.disable_depth_rendering()
        return frame

    @staticmethod
    def _camera_model_name(name: str) -> str:
        return f"external_camera_{name}"

    @staticmethod
    def _horizontal_to_vertical_fov(horizontal_fov_deg: float, width: int, height: int) -> float:
        horizontal_fov_rad = math.radians(horizontal_fov_deg)
        return math.degrees(2.0 * math.atan(math.tan(horizontal_fov_rad / 2.0) * (height / width)))

    @staticmethod
    def _euler_xyz_to_mujoco_camera_quat(rot_xyz: list[float]) -> np.ndarray:
        local_rotation = R.from_euler("xyz", rot_xyz, degrees=False)
        corrected_rotation = local_rotation * MUJOCO_CAMERA_FRAME_ROTATION
        quat_xyzw = corrected_rotation.as_quat()
        return quat_xyzw[[3, 0, 1, 2]]

    @staticmethod
    def _depth_to_display(depth_frame: np.ndarray, near: float, far: float) -> np.ndarray:
        depth_vis = (depth_frame - near) / (far - near + 1e-8)
        depth_vis = (255.0 * (1.0 - depth_vis)).astype(np.uint8)
        return cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)

    @staticmethod
    def _build_scene_option(geom_groups: list[int]) -> mujoco.MjvOption:
        scene_option = mujoco.MjvOption()
        scene_option.geomgroup[:] = 0
        for group in geom_groups:
            scene_option.geomgroup[group] = 1
        return scene_option

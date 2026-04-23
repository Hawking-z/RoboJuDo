from __future__ import annotations

import time
from dataclasses import dataclass

import cv2
import numpy as np

try:
    import pyrealsense2 as rs
except ImportError as e:
    raise ImportError(
        "pyrealsense2 未安装。请先安装 Intel RealSense SDK 对应的 Python 包。"
    ) from e


@dataclass
class CameraConfig:
    width: int = 640
    height: int = 480
    fps: int = 30
    enable_color: bool = False
    vfov_deg: float = 58.0  # 这里只保留作记录，当前预览逻辑未直接使用


class RealSenseDepthCamera:
    """纯 Python 的 RealSense 深度相机封装，不依赖 ROS。"""

    def __init__(self, config: CameraConfig | None = None):
        self.config = config or CameraConfig()
        self.pipeline = rs.pipeline()
        self.rs_config = rs.config()
        self.align = None
        self.profile = None
        self.depth_scale = None
        self.intrinsics = None
        self.started = False

    def start(self) -> None:
        self.rs_config.enable_stream(
            rs.stream.depth,
            self.config.width,
            self.config.height,
            rs.format.z16,
            self.config.fps,
        )
        if self.config.enable_color:
            self.rs_config.enable_stream(
                rs.stream.color,
                self.config.width,
                self.config.height,
                rs.format.bgr8,
                self.config.fps,
            )
            self.align = rs.align(rs.stream.color)

        self.profile = self.pipeline.start(self.rs_config)
        depth_sensor = self.profile.get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()

        # 取一次内参，后面做点云/投影时会很有用
        depth_stream_profile = self.profile.get_stream(rs.stream.depth).as_video_stream_profile()
        self.intrinsics = depth_stream_profile.get_intrinsics()

        # 给设备一点时间稳定
        self.pipeline.wait_for_frames(1000)
        self.started = True

    def stop(self) -> None:
        if self.started:
            self.pipeline.stop()
            self.started = False

    def get_frames(self, timeout_ms: int | None = None) -> dict[str, np.ndarray | float | None]:
        if not self.started:
            raise RuntimeError("相机尚未启动，请先调用 start()")

        timeout_ms = timeout_ms or int(2000 / max(self.config.fps, 1))
        frames = self.pipeline.wait_for_frames(timeout_ms)
        if self.align is not None:
            frames = self.align.process(frames)

        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame() if self.config.enable_color else None

        if not depth_frame:
            raise RuntimeError("未获取到 depth frame")

        depth_m = np.asanyarray(depth_frame.get_data(), dtype=np.float32) * self.depth_scale
        color_bgr = None
        if color_frame:
            color_bgr = np.asanyarray(color_frame.get_data())

        return {
            "timestamp": time.time(),
            "depth_m": depth_m,
            "color_bgr": color_bgr,
        }

    @staticmethod
    def depth_to_colormap(
        depth_m: np.ndarray,
        max_distance_m: float = 3.0,
        min_distance_m: float = 0.0,
    ) -> np.ndarray:
        """把米制深度图转成便于观察的伪彩图。"""
        clipped = np.clip(depth_m, min_distance_m, max_distance_m)
        denom = max(max_distance_m - min_distance_m, 1e-6)
        normalized = ((clipped - min_distance_m) / denom * 255.0).astype(np.uint8)
        colored = cv2.applyColorMap(255 - normalized, cv2.COLORMAP_JET)

        invalid_mask = (depth_m <= 0.0) | ~np.isfinite(depth_m)
        colored[invalid_mask] = 0
        return colored

    @staticmethod
    def add_overlay(
        image: np.ndarray,
        fps: float,
        depth_m: np.ndarray,
        center_text: bool = True,
    ) -> np.ndarray:
        vis = image.copy()

        valid = depth_m[np.isfinite(depth_m) & (depth_m > 0)]
        min_depth = float(valid.min()) if valid.size else float("nan")
        max_depth = float(valid.max()) if valid.size else float("nan")

        cv2.putText(
            vis,
            f"FPS: {fps:.1f}",
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            vis,
            f"Depth range: {min_depth:.2f}m ~ {max_depth:.2f}m",
            (12, 58),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        if center_text:
            h, w = depth_m.shape[:2]
            cx, cy = w // 2, h // 2
            center_depth = depth_m[cy, cx]
            cv2.drawMarker(vis, (cx, cy), (255, 255, 255), cv2.MARKER_CROSS, 18, 2)
            text = f"Center: {center_depth:.2f}m" if np.isfinite(center_depth) and center_depth > 0 else "Center: invalid"
            cv2.putText(
                vis,
                text,
                (cx + 12, cy - 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
        return vis

    def preview(
        self,
        max_distance_m: float = 3.0,
        min_distance_m: float = 0.0,
        window_name: str = "RealSense Depth Preview",
    ) -> None:
        if not self.started:
            self.start()

        last_t = time.time()
        try:
            while True:
                data = self.get_frames()
                depth_m = data["depth_m"]
                color_bgr = data["color_bgr"]

                now = time.time()
                dt = max(now - last_t, 1e-6)
                fps = 1.0 / dt
                last_t = now

                depth_vis = self.depth_to_colormap(
                    depth_m,
                    max_distance_m=max_distance_m,
                    min_distance_m=min_distance_m,
                )
                depth_vis = self.add_overlay(depth_vis, fps=fps, depth_m=depth_m)

                if color_bgr is not None:
                    color_vis = color_bgr.copy()
                    cv2.putText(
                        color_vis,
                        "Color",
                        (12, 28),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )
                    show = np.hstack([color_vis, depth_vis])
                else:
                    show = depth_vis

                cv2.imshow(window_name, show)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):  # ESC / q
                    break
        finally:
            cv2.destroyAllWindows()
            self.stop()


def main() -> None:
    camera = RealSenseDepthCamera(
        CameraConfig(
            width=640,
            height=480,
            fps=30,
            enable_color=False,
        )
    )
    camera.preview(max_distance_m=3.0, min_distance_m=0.15)


if __name__ == "__main__":
    main()
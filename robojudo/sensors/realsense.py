from __future__ import annotations

import ctypes
import multiprocessing as mp
import multiprocessing.shared_memory as mp_shm
import os
import time
from typing import Literal

import numpy as np
import pyrealsense2 as rs

REALSENSE_PROCESS_FREQUENCY_CHECK_INTERVAL = 500


class MpSharedHeader(ctypes.Structure):
    _fields_ = [
        ("timestamp", ctypes.c_double),  # bytes: 8
        ("writer_status", ctypes.c_uint32),  # bytes: 4, 0: idle, 1: writing
        ("writer_termination_signal", ctypes.c_uint32),  # bytes: 4, 0: alive, 1: should terminate
        ("_pad", ctypes.c_uint32 * 4),  # bytes: 16, pad to 32 bytes
    ]


SIZE_OF_MP_SHARED_HEADER = ctypes.sizeof(MpSharedHeader)  # bytes: 32
assert SIZE_OF_MP_SHARED_HEADER == 32


class RealSenseCamera:
    def __init__(self, resolution: tuple[int, int], fps: int):
        self.resolution = resolution  # (width, height)
        self.fps = fps
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(
            rs.stream.depth,
            self.resolution[0],
            self.resolution[1],
            rs.format.z16,
            fps,
        )
        self.profile = self.pipeline.start(self.config)
        self.align = rs.align(rs.stream.depth)
        self.depth_scale = self.profile.get_device().first_depth_sensor().get_depth_scale()
        depth_stream_profile = self.profile.get_stream(rs.stream.depth).as_video_stream_profile()
        intr = depth_stream_profile.get_intrinsics()
        print("Depth intrinsics:")
        print(f"width = {intr.width}")
        print(f"height = {intr.height}")
        print(f"fx = {intr.fx}")
        print(f"fy = {intr.fy}")
        print(f"ppx = {intr.ppx}")
        print(f"ppy = {intr.ppy}")
        # 让设备初始化更稳定一些
        _ = self.pipeline.wait_for_frames(1000)  # 1000 ms

    def get_frame(self) -> rs.depth_frame | None:
        timeout_ms = int(1000 / self.fps)
        frames = self.pipeline.wait_for_frames(timeout_ms * 2)
        depth_frame = frames.get_depth_frame()
        return depth_frame

    def get_camera_data(self) -> np.ndarray | None:
        depth_frame = self.get_frame()
        if depth_frame is None:
            return None
        depth_data = np.asanyarray(depth_frame.get_data(), dtype=np.float32) * self.depth_scale
        return depth_data

    def close(self) -> None:
        try:
            self.pipeline.stop()
        except Exception:
            pass



def camera_process_func(
    resolution: tuple[int, int],
    fps: int,
    shm_name: str,
    camera_process_affinity: set[int] | None,
) -> None:
    if camera_process_affinity is not None:
        os.sched_setaffinity(os.getpid(), camera_process_affinity)

    camera = RealSenseCamera(resolution, fps)
    shared_memory = mp.shared_memory.SharedMemory(name=shm_name)
    header = MpSharedHeader.from_buffer(shared_memory.buf)
    image_buffer = np.ndarray(
        resolution[::-1], dtype=np.float32, buffer=shared_memory.buf, offset=SIZE_OF_MP_SHARED_HEADER
    )

    camera_process_start_time = time.time()
    camera_process_counter = 0

    try:
        while True:
            camera_data = camera.get_camera_data()
            if camera_data is None:
                continue

            # mark in header to start writing
            header.writer_status = 1
            image_buffer[:] = camera_data
            header.timestamp = time.time()
            # mark in header to stop writing
            header.writer_status = 0

            if header.writer_termination_signal == 1:
                print("[RealSense child] Writer termination signal set, exiting camera process.")
                break

            camera_process_counter += 1
            if camera_process_counter % REALSENSE_PROCESS_FREQUENCY_CHECK_INTERVAL == 0:
                hz = camera_process_counter / max(time.time() - camera_process_start_time, 1e-6)
                print(f"[RealSense child] Camera process running at {hz:.4f} Hz.")
                camera_process_counter = 0
                camera_process_start_time = time.time()
    finally:
        header = None
        image_buffer = None
        shared_memory.close()  # unlink in the main process
        camera.close()


class RsCamera:
    """去掉 ROS 依赖后的版本，尽量保留原文件结构。"""

    def __init__(
        self,
        rs_resolution: tuple[int, int] = (480, 270),  # (width, height)
        rs_fps: int = 60,
        rs_vfov_deg: float = 58.0,
        camera_individual_process: bool = False,
        camera_dead_behavior: Literal["restart", "raise_error", "none"] = "restart",
        main_process_affinity: set[int] | None = None,
        camera_process_affinity: set[int] | None = None,
    ):
        self.rs_resolution = rs_resolution
        self.rs_fps = rs_fps
        self.rs_vfov_deg = rs_vfov_deg
        self.camera_individual_process = camera_individual_process
        self.camera_dead_behavior = camera_dead_behavior
        self.main_process_affinity = main_process_affinity
        self.camera_process_affinity = camera_process_affinity

        self.camera: RealSenseCamera | None = None
        self.camera_process: mp.Process | None = None
        self.request_queue = None
        self.result_queue = None

        self.rs_depth_data: np.ndarray | None = None
        self.rs_shared_memory: mp_shm.SharedMemory | None = None
        self.rs_shared_header: MpSharedHeader | None = None
        self.rs_image_buffer: np.ndarray | None = None
        self.rs_data_fresh_counter = 0

        self.initialize_camera()

    def _log_info(self, msg: str) -> None:
        print(f"[RealSense][INFO] {msg}")

    def _log_warn(self, msg: str) -> None:
        print(f"[RealSense][WARN] {msg}")

    def _log_error(self, msg: str) -> None:
        print(f"[RealSense][ERROR] {msg}")

    def initialize_camera(self) -> None:
        """Initialize the RealSense camera with the specified configuration."""
        if self.camera_individual_process:
            self.rs_depth_data = np.zeros(self.rs_resolution[::-1], dtype=np.float32)
            shm_size = (
                SIZE_OF_MP_SHARED_HEADER
                + np.prod(self.rs_resolution[::-1]) * np.dtype(self.rs_depth_data.dtype).itemsize
            )
            self.rs_shared_memory = mp_shm.SharedMemory(create=True, size=shm_size)
            self.rs_shared_header = MpSharedHeader.from_buffer(self.rs_shared_memory.buf)
            self.rs_shared_header.timestamp = 0.0
            self.rs_shared_header.writer_status = 0
            self.rs_shared_header.writer_termination_signal = 0

            self.rs_image_buffer = np.ndarray(
                self.rs_resolution[::-1],
                dtype=np.float32,
                buffer=self.rs_shared_memory.buf,
                offset=SIZE_OF_MP_SHARED_HEADER,
            )
            self.rs_data_fresh_counter = 0
            self.camera_process = mp.Process(
                target=camera_process_func,
                args=(
                    self.rs_resolution,
                    self.rs_fps,
                    self.rs_shared_memory.name,
                    self.camera_process_affinity,
                ),
                daemon=True,
            )
            self.camera_process.start()
            if self.main_process_affinity is not None:
                os.sched_setaffinity(os.getpid(), self.main_process_affinity)

            # Dummy refresh，尽量沿用原逻辑
            self.refresh_rs_data()
        else:
            self.camera = RealSenseCamera(
                resolution=self.rs_resolution,
                fps=self.rs_fps,
            )
            self.rs_depth_data = np.zeros(self.rs_resolution[::-1], dtype=np.float32)

    def restart_camera(self) -> None:
        """Restart the camera (process), but reusing the resources as much as possible."""
        self._log_info("Restarting RealSense camera.")

        if self.camera_individual_process:
            if self.rs_shared_header is not None:
                self.rs_shared_header.writer_termination_signal = 0
            self.camera_process = mp.Process(
                target=camera_process_func,
                args=(
                    self.rs_resolution,
                    self.rs_fps,
                    self.rs_shared_memory.name,
                    self.camera_process_affinity,
                ),
                daemon=True,
            )
            self.camera_process.start()
        else:
            if self.camera is not None:
                self.camera.close()
            self.camera = RealSenseCamera(
                resolution=self.rs_resolution,
                fps=self.rs_fps,
            )

    def handle_camera_dead_behavior(self) -> None:
        if self.camera_dead_behavior == "restart":
            self._log_error("Camera process is not alive. Restarting one.")
            self.restart_camera()
        elif self.camera_dead_behavior == "raise_error":
            raise RuntimeError("Camera process is not alive. Exiting.")
        elif self.camera_dead_behavior == "none":
            self._log_warn("Camera process is not alive. User chose to do nothing.")
        else:
            raise ValueError(f"Invalid camera process dead behavior: {self.camera_dead_behavior}")

    def refresh_rs_data(self, verbose: bool = False) -> bool:
        """Refresh the depth data only."""
        refreshed = False

        if self.camera_individual_process:
            if self.camera_process is None or not self.camera_process.is_alive():
                self.handle_camera_dead_behavior()

            if self.rs_shared_header is not None and self.rs_shared_header.writer_status == 0:
                rs_timestamp = self.rs_shared_header.timestamp
                if self.rs_image_buffer is not None and self.rs_depth_data is not None:
                    self.rs_depth_data[:] = self.rs_image_buffer
                if verbose and rs_timestamp > 0:
                    delay = time.time() - rs_timestamp
                    self._log_info(f"Realsense depth data delayed: {delay:.4f} s.")
                refreshed = True
            self.rs_data_fresh_counter += 1
        else:
            if self.camera is None:
                self.handle_camera_dead_behavior()
            if self.camera is not None:
                camera_data = self.camera.get_camera_data()
                if camera_data is not None:
                    self.rs_depth_data = camera_data
                    refreshed = True

        return refreshed

    def get_depth_data(self, refresh: bool = True) -> np.ndarray | None:
        if refresh:
            self.refresh_rs_data()
        return self.rs_depth_data

    def close(self) -> None:
        if self.camera_individual_process and self.camera_process is not None:
            if self.rs_shared_header is not None:
                self.rs_shared_header.writer_termination_signal = 1
            self.camera_process.join(timeout=1.0)
            if self.camera_process.is_alive():
                self._log_warn("Camera process is still alive after timeout. Terminating and joining.")
                self.camera_process.terminate()
                self.camera_process.join()
            self.camera_process = None

            self.rs_image_buffer = None
            self.rs_shared_header = None
            if self.rs_shared_memory is not None:
                self.rs_shared_memory.close()
                self.rs_shared_memory.unlink()
                self.rs_shared_memory = None
        else:
            if self.camera is not None:
                self.camera.close()
                self.camera = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


if __name__ == "__main__":
    import cv2

    camera = RsCamera(
        rs_resolution=(480, 270),
        rs_fps=60,
        camera_individual_process=True,  # 想测试子进程版本可改成 True
        camera_dead_behavior="restart",
    )

    last_time = time.time()

    try:
        while True:
            ok = camera.refresh_rs_data()
            depth = camera.get_depth_data(refresh=False)
            if not ok or depth is None:
                continue
            print("Got depth data with shape:", depth.shape)
            print("depth:", depth)
            now = time.time()
            fps = 1.0 / max(now - last_time, 1e-6)
            last_time = now

            # 显示时做一个简单的裁剪和伪彩色映射
            depth_vis = np.clip(depth, 0.15, 3.0)
            depth_vis = ((depth_vis - 0.15) / (3.0 - 0.15) * 255.0).astype(np.uint8)
            depth_vis = cv2.applyColorMap(255 - depth_vis, cv2.COLORMAP_JET)

            h, w = depth.shape
            cx, cy = w // 2, h // 2
            center_depth = depth[cy, cx]

            cv2.drawMarker(depth_vis, (cx, cy), (255, 255, 255), cv2.MARKER_CROSS, 18, 2)
            cv2.putText(depth_vis, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(
                depth_vis,
                f"Center depth: {center_depth:.3f} m",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

            cv2.imshow("RealSense Depth", depth_vis)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
    finally:
        camera.close()
        cv2.destroyAllWindows()
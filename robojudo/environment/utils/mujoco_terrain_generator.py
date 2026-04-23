from dataclasses import dataclass
from typing import List, Tuple, Optional
import random
import math


# =========================================================
# geom role presets
# =========================================================

ROLE_PRESETS = {
    "visual": {
        "contype": 0,
        "conaffinity": 0,
        "group": 1,
    },
    "collision": {
        "contype": 1,
        "conaffinity": 1,
        "group": 0,
    },
}


@dataclass
class Geom:
    name: str
    pos: Tuple[float, float, float]
    size: Tuple[float, float, float]
    geom_type: str = "box"
    rgba: Tuple[float, float, float, float] = (0.5, 0.5, 0.5, 1.0)

    contype: int = 1
    conaffinity: int = 1
    group: int = 0

    extra: str = ""

    def to_xml(self, indent: int = 4) -> str:
        sp = " " * indent
        rgba_str = f"{self.rgba[0]:.3f} {self.rgba[1]:.3f} {self.rgba[2]:.3f} {self.rgba[3]:.3f}"

        if self.geom_type == "plane":
            xml = (
                f'{sp}<geom name="{self.name}" type="{self.geom_type}"\n'
                f'{sp}      pos="{self.pos[0]:.6f} {self.pos[1]:.6f} {self.pos[2]:.6f}"\n'
                f'{sp}      size="{self.size[0]:.6f} {self.size[1]:.6f} {self.size[2]:.6f}"\n'
                f'{sp}      rgba="{rgba_str}"\n'
                f'{sp}      contype="{self.contype}" conaffinity="{self.conaffinity}" group="{self.group}"'
            )
        else:
            xml = (
                f'{sp}<geom name="{self.name}" type="{self.geom_type}"\n'
                f'{sp}      pos="{self.pos[0]:.6f} {self.pos[1]:.6f} {self.pos[2]:.6f}"\n'
                f'{sp}      size="{self.size[0]:.6f} {self.size[1]:.6f} {self.size[2]:.6f}"\n'
                f'{sp}      rgba="{rgba_str}"\n'
                f'{sp}      contype="{self.contype}" conaffinity="{self.conaffinity}" group="{self.group}"'
            )

        if self.extra.strip():
            xml += f"\n{sp}      {self.extra.strip()}"

        xml += "/>"
        return xml


class TerrainBuilder:
    def __init__(self):
        self.geoms: List[Geom] = []
        self.counter = 0

    def _new_name(self, prefix: str) -> str:
        self.counter += 1
        return f"{prefix}_{self.counter}"

    def _resolve_role(
        self,
        role: str,
        contype: Optional[int],
        conaffinity: Optional[int],
        group: Optional[int],
    ) -> Tuple[int, int, int]:
        if role not in ROLE_PRESETS:
            raise ValueError(f"Unknown role: {role}, supported: {list(ROLE_PRESETS.keys())}")

        preset = ROLE_PRESETS[role]
        return (
            preset["contype"] if contype is None else contype,
            preset["conaffinity"] if conaffinity is None else conaffinity,
            preset["group"] if group is None else group,
        )

    def add_box(
        self,
        name: Optional[str],
        center_x: float,
        center_y: float,
        center_z: float,
        half_x: float,
        half_y: float,
        half_z: float,
        rgba=(0.5, 0.5, 0.5, 1.0),
        role: str = "collision",
        contype: Optional[int] = None,
        conaffinity: Optional[int] = None,
        group: Optional[int] = None,
        extra: str = "",
    ):
        contype_v, conaffinity_v, group_v = self._resolve_role(role, contype, conaffinity, group)

        self.geoms.append(
            Geom(
                name=name if name is not None else self._new_name("box"),
                pos=(center_x, center_y, center_z),
                size=(half_x, half_y, half_z),
                rgba=rgba,
                contype=contype_v,
                conaffinity=conaffinity_v,
                group=group_v,
                extra=extra,
            )
        )

    def add_plane_floor(
        self,
        name="floor",
        size=(20.0, 20.0, 0.1),
        pos=(0.0, 0.0, 0.0),
        rgba=(0.8, 0.9, 0.8, 1.0),
        role: str = "collision",
        contype: Optional[int] = None,
        conaffinity: Optional[int] = None,
        group: Optional[int] = None,
        extra: str = "",
    ):
        contype_v, conaffinity_v, group_v = self._resolve_role(role, contype, conaffinity, group)

        self.geoms.append(
            Geom(
                name=name,
                pos=pos,
                size=size,
                geom_type="plane",
                rgba=rgba,
                contype=contype_v,
                conaffinity=conaffinity_v,
                group=group_v,
                extra=extra,
            )
        )

    # =========================
    # 1. 上楼梯（实心）
    # =========================
    def add_stairs_up(
        self,
        x_start: float,
        y: float,
        n_steps: int,
        step_depth: float,
        step_height: float,
        step_width_half: float,
        base_z: float = 0.0,
        prefix: str = "stairs_up",
        rgba_start: float = 0.70,
        rgba_end: float = 0.15,
        role: str = "collision",
        contype: Optional[int] = None,
        conaffinity: Optional[int] = None,
        group: Optional[int] = None,
    ):
        for i in range(n_steps):
            level = i + 1
            total_height = level * step_height
            center_x = x_start + i * step_depth + step_depth / 2.0
            center_z = base_z + total_height / 2.0
            gray = rgba_start if n_steps == 1 else rgba_start + (rgba_end - rgba_start) * i / (n_steps - 1)

            self.add_box(
                name=f"{prefix}_{level}",
                center_x=center_x,
                center_y=y,
                center_z=center_z,
                half_x=step_depth / 2.0,
                half_y=step_width_half,
                half_z=total_height / 2.0,
                rgba=(gray, gray, gray, 1.0),
                role=role,
                contype=contype,
                conaffinity=conaffinity,
                group=group,
            )

    # =========================
    # 2. 下楼梯（实心）
    # =========================
    def add_stairs_down(
        self,
        x_start: float,
        y: float,
        n_steps: int,
        step_depth: float,
        step_height: float,
        step_width_half: float,
        top_height: float,
        base_z: float = 0.0,
        prefix: str = "stairs_down",
        rgba_start: float = 0.20,
        rgba_end: float = 0.70,
        role: str = "collision",
        contype: Optional[int] = None,
        conaffinity: Optional[int] = None,
        group: Optional[int] = None,
    ):
        for i in range(n_steps):
            remain_height = max(top_height - i * step_height, 0.0)
            if remain_height <= 0.0:
                break

            center_x = x_start + i * step_depth + step_depth / 2.0
            center_z = base_z + remain_height / 2.0
            gray = rgba_start if n_steps == 1 else rgba_start + (rgba_end - rgba_start) * i / (n_steps - 1)

            self.add_box(
                name=f"{prefix}_{i+1}",
                center_x=center_x,
                center_y=y,
                center_z=center_z,
                half_x=step_depth / 2.0,
                half_y=step_width_half,
                half_z=remain_height / 2.0,
                rgba=(gray, gray, gray, 1.0),
                role=role,
                contype=contype,
                conaffinity=conaffinity,
                group=group,
            )

    # =========================
    # 3. 平台
    # =========================
    def add_platform(
        self,
        x_start: float,
        y: float,
        length: float,
        width_half: float,
        height: float,
        base_z: float = 0.0,
        name: Optional[str] = None,
        rgba=(0.45, 0.45, 0.55, 1.0),
        role: str = "collision",
        contype: Optional[int] = None,
        conaffinity: Optional[int] = None,
        group: Optional[int] = None,
    ):
        self.add_box(
            name=name or self._new_name("platform"),
            center_x=x_start + length / 2.0,
            center_y=y,
            center_z=base_z + height / 2.0,
            half_x=length / 2.0,
            half_y=width_half,
            half_z=height / 2.0,
            rgba=rgba,
            role=role,
            contype=contype,
            conaffinity=conaffinity,
            group=group,
        )

    # =========================
    # 4. 斜坡（box离散近似）
    # =========================
    def add_slope_boxes(
        self,
        x_start: float,
        y: float,
        length: float,
        width_half: float,
        start_height: float,
        end_height: float,
        n_segments: int,
        base_z: float = 0.0,
        prefix: str = "slope",
        rgba=(0.5, 0.4, 0.3, 1.0),
        role: str = "collision",
        contype: Optional[int] = None,
        conaffinity: Optional[int] = None,
        group: Optional[int] = None,
    ):
        seg_len = length / n_segments
        for i in range(n_segments):
            h = start_height + (end_height - start_height) * (i + 1) / n_segments
            cx = x_start + i * seg_len + seg_len / 2.0
            cz = base_z + h / 2.0
            self.add_box(
                name=f"{prefix}_{i+1}",
                center_x=cx,
                center_y=y,
                center_z=cz,
                half_x=seg_len / 2.0,
                half_y=width_half,
                half_z=h / 2.0,
                rgba=rgba,
                role=role,
                contype=contype,
                conaffinity=conaffinity,
                group=group,
            )

    # =========================
    # 5. gap terrain
    # =========================
    def add_gaps(
        self,
        x_start: float,
        y: float,
        n_blocks: int,
        block_length: float,
        gap_length: float,
        width_half: float,
        height: float,
        base_z: float = 0.0,
        prefix: str = "gapblock",
        rgba=(0.35, 0.45, 0.65, 1.0),
        role: str = "collision",
        contype: Optional[int] = None,
        conaffinity: Optional[int] = None,
        group: Optional[int] = None,
    ):
        x = x_start
        for i in range(n_blocks):
            self.add_box(
                name=f"{prefix}_{i+1}",
                center_x=x + block_length / 2.0,
                center_y=y,
                center_z=base_z + height / 2.0,
                half_x=block_length / 2.0,
                half_y=width_half,
                half_z=height / 2.0,
                rgba=rgba,
                role=role,
                contype=contype,
                conaffinity=conaffinity,
                group=group,
            )
            x += block_length + gap_length

    # =========================
    # 6. 离散踏脚石
    # =========================
    def add_stepping_stones(
        self,
        centers: List[Tuple[float, float]],
        stone_half_size: Tuple[float, float],
        height: float,
        base_z: float = 0.0,
        prefix: str = "stone",
        rgba=(0.55, 0.55, 0.45, 1.0),
        role: str = "collision",
        contype: Optional[int] = None,
        conaffinity: Optional[int] = None,
        group: Optional[int] = None,
    ):
        hx, hy = stone_half_size
        for i, (cx, cy) in enumerate(centers):
            self.add_box(
                name=f"{prefix}_{i+1}",
                center_x=cx,
                center_y=cy,
                center_z=base_z + height / 2.0,
                half_x=hx,
                half_y=hy,
                half_z=height / 2.0,
                rgba=rgba,
                role=role,
                contype=contype,
                conaffinity=conaffinity,
                group=group,
            )

    # =========================
    # 7. 波浪地形（box近似）
    # =========================
    def add_wave_blocks(
        self,
        x_start: float,
        y: float,
        n_blocks: int,
        block_length: float,
        width_half: float,
        base_height: float,
        amplitude: float,
        base_z: float = 0.0,
        prefix: str = "wave",
        rgba=(0.45, 0.6, 0.5, 1.0),
        role: str = "collision",
        contype: Optional[int] = None,
        conaffinity: Optional[int] = None,
        group: Optional[int] = None,
    ):
        for i in range(n_blocks):
            h = base_height + amplitude * math.sin(2.0 * math.pi * i / max(n_blocks, 1))
            h = max(h, 0.01)
            cx = x_start + i * block_length + block_length / 2.0
            cz = base_z + h / 2.0
            self.add_box(
                name=f"{prefix}_{i+1}",
                center_x=cx,
                center_y=y,
                center_z=cz,
                half_x=block_length / 2.0,
                half_y=width_half,
                half_z=h / 2.0,
                rgba=rgba,
                role=role,
                contype=contype,
                conaffinity=conaffinity,
                group=group,
            )

    # =========================
    # 8. 随机高低块
    # =========================
    def add_random_blocks(
        self,
        x_start: float,
        y: float,
        n_blocks: int,
        block_length: float,
        width_half: float,
        min_height: float,
        max_height: float,
        seed: int = 0,
        base_z: float = 0.0,
        prefix: str = "randblock",
        rgba=(0.55, 0.35, 0.35, 1.0),
        role: str = "collision",
        contype: Optional[int] = None,
        conaffinity: Optional[int] = None,
        group: Optional[int] = None,
    ):
        rng = random.Random(seed)
        for i in range(n_blocks):
            h = rng.uniform(min_height, max_height)
            cx = x_start + i * block_length + block_length / 2.0
            cz = base_z + h / 2.0
            self.add_box(
                name=f"{prefix}_{i+1}",
                center_x=cx,
                center_y=y,
                center_z=cz,
                half_x=block_length / 2.0,
                half_y=width_half,
                half_z=h / 2.0,
                rgba=rgba,
                role=role,
                contype=contype,
                conaffinity=conaffinity,
                group=group,
            )

    # =========================
    # 9. 左右交替窄台
    # =========================
    def add_alternating_side_blocks(
        self,
        x_start: float,
        y_center: float,
        n_blocks: int,
        block_length: float,
        block_half_width: float,
        lateral_offset: float,
        height: float,
        base_z: float = 0.0,
        prefix: str = "altblock",
        rgba=(0.40, 0.50, 0.70, 1.0),
        role: str = "collision",
        contype: Optional[int] = None,
        conaffinity: Optional[int] = None,
        group: Optional[int] = None,
    ):
        for i in range(n_blocks):
            cy = y_center + (lateral_offset if i % 2 == 0 else -lateral_offset)
            cx = x_start + i * block_length + block_length / 2.0
            cz = base_z + height / 2.0
            self.add_box(
                name=f"{prefix}_{i+1}",
                center_x=cx,
                center_y=cy,
                center_z=cz,
                half_x=block_length / 2.0,
                half_y=block_half_width,
                half_z=height / 2.0,
                rgba=rgba,
                role=role,
                contype=contype,
                conaffinity=conaffinity,
                group=group,
            )

    def export_xml(self, out_file: str, wrap_with_mujoco: bool = True):
        lines = []
        if wrap_with_mujoco:
            lines.append("<mujoco>")
            lines.append("  <worldbody>")

        for geom in self.geoms:
            lines.append(geom.to_xml(indent=4 if wrap_with_mujoco else 0))

        if wrap_with_mujoco:
            lines.append("  </worldbody>")
            lines.append("</mujoco>")

        with open(out_file, "w", encoding="utf-8") as f:
            f.write("\n\n".join(lines))


def make_empty_terrain(out_file="terrain.xml"):
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(
            "<mujoco>\n"
            "  <worldbody>\n"
            "  </worldbody>\n"
            "</mujoco>\n"
        )


def make_default_complex_terrain(out_file="terrain.xml"):
    tb = TerrainBuilder()

    tb.add_stairs_up(
        x_start=1.0,
        y=0.0,
        n_steps=20,
        step_depth=0.35,
        step_height=0.18,
        step_width_half=10.0,
        prefix="stairs_col",
        role="collision",
    )
    tb.add_platform(
        x_start=-3.0,
        y=0.0,
        length=1.0,
        width_half=2.0,
        height=0.1,
        name="platform_col",
        role="collision",
    )

    tb.export_xml(out_file)


if __name__ == "__main__":
    file = "assets/robots/terrain/complex_terrain.xml"
    make_default_complex_terrain(file)

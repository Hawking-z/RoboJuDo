from collections.abc import Mapping
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


class RingBuffer:
    def __init__(self, length: int, data_shape: Tuple[int, ...], dtype=np.float32):
        self.length = int(length)
        self.shape = tuple(data_shape)
        self.buf = np.zeros((self.length, *self.shape), dtype=dtype)
        self.head = 0

    def push(self, x: np.ndarray) -> None:
        self.buf[self.head] = x
        self.head = (self.head + 1) % self.length

    def get_last(self, count: int) -> np.ndarray:
        if not 1 <= count <= self.length:
            raise ValueError(f"count must be in [1, {self.length}], got {count}.")
        start = (self.head - count) % self.length
        if start + count <= self.length:
            return self.buf[start : start + count].copy()
        return np.concatenate((self.buf[start:], self.buf[: start + count - self.length]), axis=0)

    def copy_last_to(self, count: int, out: np.ndarray) -> None:
        out[...] = self.get_last(count)

    def reset(self) -> None:
        self.buf.fill(0)
        self.head = 0


@dataclass(frozen=True)
class _HeadSourceSpec:
    kind: str
    source_name: str
    C: int
    start: int
    end: int
    tail_shape: Tuple[int, ...]
    front_shape: Tuple[int, ...]
    sensor_idx: int = -1
    dep_head: str = ""


class ObsAssembler:
    """Single-env numpy obs assembler aligned with the main ObsAssembler abstraction."""

    NOISE_SUFFIX = "_noise"

    def __init__(
        self,
        sensors: Dict[str, dict],
        obs_heads: Dict[str, dict],
        *,
        dtype=np.float32,
        clip_observations: float = 100.0,
    ):
        self.dtype = np.dtype(dtype)
        self.clip_observations = float(clip_observations)
        self._do_clip = self.clip_observations > 0.0

        self._build_constants(sensors)
        self._build_heads(obs_heads)
        self._alloc_storage()
        self._compile_heads()

    # ------------------------------------------------------------------
    # Schema parsing
    # ------------------------------------------------------------------

    def _scale(self, value, shape: Tuple[int, ...]) -> np.ndarray:
        scale = np.asarray(value, dtype=self.dtype)
        if scale.ndim == 0:
            return scale
        try:
            return np.broadcast_to(scale, shape).astype(self.dtype, copy=False)
        except ValueError as exc:
            raise ValueError(f"[ObsAssemblerNP] scale with shape {scale.shape} cannot broadcast to {shape}.") from exc

    def _build_constants(self, sensors: Dict[str, dict]) -> None:
        self.names: List[str] = list(sensors.keys())
        self._name2idx = {name: i for i, name in enumerate(self.names)}

        self.shape: List[Tuple[int, ...]] = []
        self.frames: List[int] = []
        self.stride: List[int] = []
        self.sensor_numel: List[int] = []
        self.obs_scale: List[np.ndarray] = []

        for name in self.names:
            cfg = sensors[name]
            shape = tuple(int(dim) for dim in cfg["shape"])
            if len(shape) < 1:
                raise ValueError(f"[ObsAssemblerNP] sensor '{name}' shape must have at least 1 dim.")
            if any(dim <= 0 for dim in shape):
                raise ValueError(f"[ObsAssemblerNP] sensor '{name}' shape dims must be positive, got {shape}.")

            frames = int(cfg.get("frames", 1))
            stride = int(cfg.get("stride", 1))
            if frames <= 0:
                raise ValueError(f"[ObsAssemblerNP] sensor '{name}' frames must be positive, got {frames}.")
            if stride <= 0:
                raise ValueError(f"[ObsAssemblerNP] sensor '{name}' stride must be positive, got {stride}.")

            self.shape.append(shape)
            self.frames.append(frames)
            self.stride.append(stride)
            self.sensor_numel.append(int(np.prod(shape, dtype=np.int64)))
            self.obs_scale.append(self._scale(cfg.get("obs_scale", 1.0), shape))

    def _parse_sensor_source(self, token: str) -> Optional[Tuple[int, str]]:
        if token in self._name2idx:
            return self._name2idx[token], token
        return None

    def _build_heads(self, heads: Dict[str, dict]) -> None:
        self._head_names: List[str] = list(heads.keys())
        self._head_topo_order: List[str] = []

        self.head_histlen: Dict[str, int] = {}
        self.head_history_mode: Dict[str, str] = {}
        self.head_tail_ndim: Dict[str, int] = {}
        self.head_tailshape: Dict[str, Tuple[int, ...]] = {}
        self.S_head: Dict[str, int] = {}
        self._head_source_specs: Dict[str, List[_HeadSourceSpec]] = {}
        self._head_spec: Dict[str, dict] = {}
        self._head_dep_names: Dict[str, List[str]] = {}

        raw_head_cfg: Dict[str, dict] = {}
        self._used_idx = set()

        for head_name, cfg in heads.items():
            if "flatten" in cfg or "mode" in cfg:
                raise ValueError(
                    f"[ObsAssemblerNP] Head '{head_name}' uses legacy flatten/mode. "
                    "Use tail_ndim/history_mode instead."
                )

            sources = cfg.get("sources")
            if sources is None:
                raise ValueError(f"[ObsAssemblerNP] Head '{head_name}' must define sources.")
            if isinstance(sources, (str, bytes)) or isinstance(sources, Mapping):
                raise ValueError(f"[ObsAssemblerNP] Head '{head_name}' sources must be a sequence of names.")
            try:
                raw_sources = list(sources)
            except TypeError as exc:
                raise ValueError(f"[ObsAssemblerNP] Head '{head_name}' sources must be a sequence of names.") from exc
            if not raw_sources:
                raise ValueError(f"[ObsAssemblerNP] Head '{head_name}' must have at least one source.")
            if any(not isinstance(token, str) for token in raw_sources):
                raise ValueError(f"[ObsAssemblerNP] Head '{head_name}' sources must contain only string names.")

            history_len = int(cfg.get("history_len", 1))
            if history_len <= 0:
                raise ValueError(
                    f"[ObsAssemblerNP] Head '{head_name}' history_len must be positive, got {history_len}."
                )
            tail_ndim = int(cfg.get("tail_ndim", 0))
            if tail_ndim < 0:
                raise ValueError(f"[ObsAssemblerNP] Head '{head_name}' tail_ndim must be non-negative.")
            history_mode = str(cfg.get("history_mode", "merge"))
            if history_mode not in {"keep", "merge"}:
                raise ValueError(
                    f"[ObsAssemblerNP] Head '{head_name}' history_mode must be 'keep' or 'merge', got {history_mode!r}."
                )

            raw_head_cfg[head_name] = {
                "sources": raw_sources,
                "history_len": history_len,
                "tail_ndim": tail_ndim,
                "history_mode": history_mode,
            }

        visit_state: Dict[str, int] = {}

        def visit(head_name: str) -> None:
            state = visit_state.get(head_name, 0)
            if state == 2:
                return
            if state == 1:
                raise ValueError(f"[ObsAssemblerNP] Head dependency cycle detected at '{head_name}'.")

            visit_state[head_name] = 1
            cfg = raw_head_cfg[head_name]
            source_specs: List[_HeadSourceSpec] = []
            source_order: List[str] = []
            source_ranges: List[Tuple[int, int]] = []
            source_slices: List[dict] = []
            dep_names: List[str] = []
            tail_shape: Optional[Tuple[int, ...]] = None
            c_offset = 0

            for token in cfg["sources"]:
                sensor_parsed = self._parse_sensor_source(token)
                if sensor_parsed is not None:
                    sensor_idx, display_name = sensor_parsed
                    sensor_shape = self.shape[sensor_idx]
                    if cfg["tail_ndim"] > len(sensor_shape):
                        raise ValueError(
                            f"[ObsAssemblerNP] Head '{head_name}' tail_ndim={cfg['tail_ndim']} exceeds rank "
                            f"{len(sensor_shape)} of sensor '{self.names[sensor_idx]}'."
                        )
                    if cfg["tail_ndim"] == 0:
                        source_tail = ()
                        front_shape = sensor_shape
                    else:
                        source_tail = sensor_shape[-cfg["tail_ndim"] :]
                        front_shape = sensor_shape[: -cfg["tail_ndim"]]
                    C_i = self.frames[sensor_idx] * int(np.prod(front_shape or (1,), dtype=np.int64))
                    source_specs.append(
                        _HeadSourceSpec(
                            kind="sensor",
                            source_name=display_name,
                            C=C_i,
                            start=c_offset,
                            end=c_offset + C_i,
                            tail_shape=source_tail,
                            front_shape=front_shape,
                            sensor_idx=sensor_idx,
                        )
                    )
                    self._used_idx.add(sensor_idx)
                    slice_meta = {
                        "kind": "sensor",
                        "source": display_name,
                        "sensor": self.names[sensor_idx],
                        "sensor_shape": sensor_shape,
                        "sensor_frames": self.frames[sensor_idx],
                    }
                else:
                    if token.endswith(self.NOISE_SUFFIX):
                        base = token[: -len(self.NOISE_SUFFIX)]
                        if base in raw_head_cfg:
                            raise ValueError(
                                f"[ObsAssemblerNP] Head '{head_name}' cannot reference head '{token}' with *_noise suffix."
                            )
                        if base in self._name2idx:
                            raise KeyError(f"[ObsAssemblerNP] noise source '{token}' is unsupported in deployment.")

                    if token not in raw_head_cfg:
                        raise KeyError(f"[ObsAssemblerNP] unknown sensor or head '{token}' in head '{head_name}'.")

                    visit(token)
                    dep_spec = self._head_spec[token]
                    source_tail = dep_spec["tail_shape"]
                    C_i = dep_spec["history_len"] * dep_spec["C_total"]
                    source_specs.append(
                        _HeadSourceSpec(
                            kind="head",
                            source_name=token,
                            C=C_i,
                            start=c_offset,
                            end=c_offset + C_i,
                            tail_shape=source_tail,
                            front_shape=(C_i,),
                            dep_head=token,
                        )
                    )
                    dep_names.append(token)
                    slice_meta = {
                        "kind": "head",
                        "source": token,
                        "head": token,
                        "head_history_len": dep_spec["history_len"],
                        "head_history_mode": dep_spec["history_mode"],
                        "head_output_shape": dep_spec["output_shape"],
                        "ref_shape": (C_i, *source_tail),
                    }

                if tail_shape is None:
                    tail_shape = source_tail
                elif tail_shape != source_tail:
                    raise ValueError(
                        f"[ObsAssemblerNP] Head '{head_name}' tail mismatch among sources: "
                        f"expected {tail_shape}, got {source_tail} from '{token}'."
                    )

                source_order.append(token)
                source_ranges.append((c_offset, c_offset + C_i))
                source_slices.append(
                    {
                        **slice_meta,
                        "start": c_offset,
                        "end": c_offset + C_i,
                        "C": C_i,
                    }
                )
                c_offset += C_i

            assert tail_shape is not None

            per_step_shape = (c_offset, *tail_shape) if tail_shape else (c_offset,)
            if cfg["history_mode"] == "keep":
                output_shape = (cfg["history_len"], *per_step_shape)
            elif cfg["history_len"] == 1:
                output_shape = per_step_shape
            else:
                output_shape = (cfg["history_len"] * c_offset, *tail_shape) if tail_shape else (cfg["history_len"] * c_offset,)

            self.head_histlen[head_name] = cfg["history_len"]
            self.head_history_mode[head_name] = cfg["history_mode"]
            self.head_tail_ndim[head_name] = cfg["tail_ndim"]
            self.head_tailshape[head_name] = tail_shape
            self.S_head[head_name] = c_offset
            self._head_source_specs[head_name] = source_specs
            self._head_dep_names[head_name] = dep_names
            self._head_spec[head_name] = {
                "output_shape": tuple(output_shape),
                "history_len": cfg["history_len"],
                "history_mode": cfg["history_mode"],
                "tail_ndim": cfg["tail_ndim"],
                "per_step_shape": tuple(per_step_shape),
                "C_total": c_offset,
                "tail_shape": tail_shape,
                "source_order": source_order,
                "source_ranges": source_ranges,
                "source_slices": source_slices,
                "dtype": self.dtype,
            }
            self._head_topo_order.append(head_name)
            visit_state[head_name] = 2

        for head_name in self._head_names:
            visit(head_name)

        self._used_idx = sorted(self._used_idx)

    # ------------------------------------------------------------------
    # Storage
    # ------------------------------------------------------------------

    def _alloc_storage(self) -> None:
        count = len(self.names)
        self.cache: List[Optional[np.ndarray]] = [None] * count
        self.rings: List[Optional[RingBuffer]] = [None] * count
        self._tick: List[Optional[int]] = [None] * count

        for i in self._used_idx:
            self.cache[i] = np.zeros((self.frames[i], *self.shape[i]), dtype=self.dtype)
            if self.frames[i] > 1:
                self.rings[i] = RingBuffer(self.frames[i], self.shape[i], dtype=self.dtype)
            if self.stride[i] > 1:
                self._tick[i] = 0

    def _compile_heads(self) -> None:
        self._head_step_buf: Dict[str, np.ndarray] = {}
        self._head_history_buf: Dict[str, np.ndarray] = {}
        self._head_output_buf: Dict[str, np.ndarray] = {}
        self._head_ref_buf: Dict[str, np.ndarray] = {}
        self._head_rings: Dict[str, Optional[RingBuffer]] = {}
        self._head_assignments: Dict[str, List[Tuple[_HeadSourceSpec, np.ndarray, Tuple[int, ...]]]] = {}

        for head in self._head_topo_order:
            spec = self._head_spec[head]
            history_len = spec["history_len"]
            history_mode = spec["history_mode"]
            C_total = spec["C_total"]
            tail_shape = spec["tail_shape"]

            step_shape = (C_total, *tail_shape) if tail_shape else (C_total,)
            step_buf = np.zeros(step_shape, dtype=self.dtype)
            self._head_step_buf[head] = step_buf

            if history_len == 1:
                history_buf = step_buf.reshape((1, *step_shape))
                ref_buf = step_buf
                output_buf = history_buf if history_mode == "keep" else step_buf
                ring = None
            else:
                history_shape = (history_len, *step_shape)
                history_buf = np.zeros(history_shape, dtype=self.dtype)
                ref_shape = (history_len * C_total, *tail_shape) if tail_shape else (history_len * C_total,)
                ref_buf = history_buf.reshape(ref_shape)
                output_buf = history_buf if history_mode == "keep" else ref_buf
                ring = RingBuffer(history_len, step_shape, dtype=self.dtype)

            self._head_history_buf[head] = history_buf
            self._head_output_buf[head] = output_buf
            self._head_ref_buf[head] = ref_buf
            self._head_rings[head] = ring

            assignments = []
            for source_spec in self._head_source_specs[head]:
                dst = step_buf[source_spec.start : source_spec.end]
                if source_spec.kind == "head":
                    src = self._head_ref_buf[source_spec.dep_head]
                else:
                    src = self.cache[source_spec.sensor_idx]
                view_shape = (source_spec.C, *source_spec.tail_shape) if source_spec.tail_shape else (source_spec.C,)
                assignments.append((source_spec, src, view_shape))
            self._head_assignments[head] = assignments

    # ------------------------------------------------------------------
    # Runtime
    # ------------------------------------------------------------------

    def _tick_down(self, sensor_idx: int) -> bool:
        tick = self._tick[sensor_idx]
        if tick is None:
            return True
        if tick == 0:
            self._tick[sensor_idx] = self.stride[sensor_idx] - 1
            return True
        self._tick[sensor_idx] = tick - 1
        return False

    def _input_array(self, name: str, value, shape: Tuple[int, ...]) -> np.ndarray:
        arr = np.asarray(value, dtype=self.dtype)
        if arr.shape == shape:
            return arr
        raise ValueError(f"[ObsAssemblerNP] input '{name}' shape must be {shape}, got {arr.shape}.")

    def step(self, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        for i in self._used_idx:
            if not self._tick_down(i):
                continue

            name = self.names[i]
            x = self._input_array(name, inputs[name], self.shape[i]) * self.obs_scale[i]
            if self.frames[i] == 1:
                self.cache[i][0] = x
                continue

            ring = self.rings[i]
            ring.push(x)
            ring.copy_last_to(self.frames[i], self.cache[i])

        out: Dict[str, np.ndarray] = {}
        for head in self._head_topo_order:
            step_buf = self._head_step_buf[head]
            for source_spec, src, view_shape in self._head_assignments[head]:
                step_buf[source_spec.start : source_spec.end] = src.reshape(view_shape)

            if self._do_clip:
                np.clip(step_buf, -self.clip_observations, self.clip_observations, out=step_buf)

            ring = self._head_rings[head]
            if ring is not None:
                ring.push(step_buf)
                ring.copy_last_to(self.head_histlen[head], self._head_history_buf[head])

            out[head] = self._head_output_buf[head]

        return {head: out[head] for head in self._head_names}

    def reset(self) -> None:
        for i in self._used_idx:
            self.cache[i].fill(0)
            if self.rings[i] is not None:
                self.rings[i].reset()
            if self._tick[i] is not None:
                self._tick[i] = 0

        for head in self._head_topo_order:
            self._head_step_buf[head].fill(0)
            self._head_history_buf[head].fill(0)
            if self._head_rings[head] is not None:
                self._head_rings[head].reset()

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def head_output_spec(self) -> Dict[str, dict]:
        return {head: dict(self._head_spec[head]) for head in self._head_names}

    def info_str(self) -> str:
        def fmt_shape(shape) -> str:
            shape = tuple(shape)
            return "(" + ", ".join(str(x) for x in shape) + ("," if len(shape) == 1 else "") + ")"

        lines = [
            "ObsAssemblerNP",
            f"  dtype={self.dtype}",
            f"  clip_observations={self.clip_observations}",
            f"  sensors={len(self.names)}, used_sensors={len(self._used_idx)}, heads={len(self._head_names)}",
            "",
            "Sensors:",
        ]

        used = set(self._used_idx)
        for i, name in enumerate(self.names):
            cache = self.cache[i]
            ring = self.rings[i]
            tick = self._tick[i]
            scale = self.obs_scale[i]
            if np.ndim(scale) == 0:
                scale_desc = str(float(scale))
            else:
                scale_desc = f"array_shape={fmt_shape(scale.shape)}"

            storage = []
            if cache is not None:
                storage.append(f"cache_shape={fmt_shape(cache.shape)}")
            else:
                storage.append("cache_shape=None")
            if ring is not None:
                storage.append(f"ring_shape={fmt_shape(ring.buf.shape)}")
                storage.append(f"ring_head={ring.head}")
            if tick is not None:
                storage.append(f"tick={tick}")

            lines.append(
                f"  sensor {name}: shape={fmt_shape(self.shape[i])}, frames={self.frames[i]}, "
                f"stride={self.stride[i]}, obs_scale={scale_desc}, used={i in used}, " + ", ".join(storage)
            )

        lines.extend(["", "Heads:"])
        spec = self.head_output_spec()
        for head in self._head_names:
            source_names = []
            blocks = []
            for source_slice in spec[head]["source_slices"]:
                source_names.append(source_slice["source"])
                if source_slice["kind"] == "sensor":
                    detail = f"sensor={source_slice['sensor']}"
                else:
                    detail = f"head={source_slice['head']}"
                blocks.append(
                    f"{source_slice['source']}[{source_slice['start']}:{source_slice['end']}, {detail}]"
                )

            ring = self._head_rings[head]
            history_storage = "none" if ring is None else f"ring_shape={fmt_shape(ring.buf.shape)}, ring_head={ring.head}"
            lines.append(
                f"  head {head}: sources=[{', '.join(source_names)}], history_len={self.head_histlen[head]}, "
                f"history_mode={self.head_history_mode[head]}, tail_ndim={self.head_tail_ndim[head]}, "
                f"C_total={self.S_head[head]}, tail={fmt_shape(self.head_tailshape[head])}, "
                f"output_shape={fmt_shape(spec[head]['output_shape'])}, history_storage={history_storage}"
            )
            lines.append(f"    blocks: {', '.join(blocks)}")

        lines.append("")
        lines.append("Head topo order: " + " -> ".join(self._head_topo_order))
        return "\n".join(lines)

    def print_info(self) -> None:
        print(self.info_str())


if __name__ == "__main__":
    sensors = {
        "a": {"shape": (2,), "frames": 2, "stride": 1},
        "b": {"shape": (1,), "frames": 1, "stride": 2},
    }
    heads = {
        "props": {"sources": ["a", "b"], "history_len": 3, "tail_ndim": 0, "history_mode": "keep"},
        "actor": {"sources": ["props"], "history_len": 1, "tail_ndim": 0, "history_mode": "merge"},
    }
    assembler = ObsAssembler(sensors, heads, clip_observations=10.0)
    assembler.print_info()

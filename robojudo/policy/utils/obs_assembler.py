from collections.abc import Mapping
from typing import Dict, List, Optional, Tuple

import numpy as np


class RingBuffer:
    def __init__(self, length: int, data_shape: Tuple[int, ...], dtype=np.float32):
        self.length = int(length)
        self.buf = np.zeros((self.length, *tuple(data_shape)), dtype=dtype)
        self.head = 0

    def push(self, x: np.ndarray) -> None:
        self.buf[self.head] = x
        self.head = (self.head + 1) % self.length

    def get_last(self, count: int) -> np.ndarray:
        if not 1 <= count <= self.length:
            raise ValueError(f"count must be in [1, {self.length}], got {count}.")
        start = (self.head - count) % self.length
        if start + count <= self.length:
            return self.buf[start:start + count].copy()
        return np.concatenate((self.buf[start:], self.buf[:start + count - self.length]), axis=0)

    def copy_last_flat_to(self, count: int, out: np.ndarray) -> None:
        out[...] = self.get_last(count).reshape(out.shape)

    def reset(self) -> None:
        self.buf.fill(0)
        self.head = 0


class ObsAssembler:
    """Numpy observation assembler for one robot environment."""

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
        self._do_clip = self.clip_observations > 0

        self._build_constants(sensors)
        self._build_heads(obs_heads)
        self._alloc_storage()
        self._alloc_head_history()

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
        self.lead: List[int] = []
        self.tail: List[Tuple[int, ...]] = []
        self.frames: List[int] = []
        self.stride: List[int] = []
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
            self.lead.append(shape[0])
            self.tail.append(tuple(shape[1:]))
            self.frames.append(frames)
            self.stride.append(stride)
            self.obs_scale.append(self._scale(cfg.get("obs_scale", 1.0), shape))

    def _build_heads(self, heads: Dict[str, dict]) -> None:
        self._head_names: List[str] = list(heads.keys())
        self.head_sources: Dict[str, List[int]] = {}
        self.head_histlen: Dict[str, int] = {}
        self.head_flatten: Dict[str, bool] = {}
        self.head_tailshape: Dict[str, Tuple[int, ...]] = {}
        self.S_head: Dict[str, int] = {}
        self._head_direct_sources: Dict[str, List[Tuple[str, object]]] = {}
        self._head_topo_order: List[str] = []

        raw_sources: Dict[str, List[str]] = {}
        for head, cfg in heads.items():
            sources = cfg.get("sources")
            if sources is None:
                raise ValueError(f"[ObsAssemblerNP] Head '{head}' must define sources.")
            if isinstance(sources, (str, bytes)) or isinstance(sources, Mapping):
                raise ValueError(f"[ObsAssemblerNP] Head '{head}' sources must be a sequence of names.")
            try:
                raw = list(sources)
            except TypeError as exc:
                raise ValueError(f"[ObsAssemblerNP] Head '{head}' sources must be a sequence of names.") from exc
            if not raw:
                raise ValueError(f"[ObsAssemblerNP] Head '{head}' must have at least one source.")
            if any(not isinstance(token, str) for token in raw):
                raise ValueError(f"[ObsAssemblerNP] Head '{head}' sources must contain only string names.")

            history_len = int(cfg.get("history_len", 1))
            if history_len <= 0:
                raise ValueError(
                    f"[ObsAssemblerNP] Head '{head}' history_len must be positive, got {history_len}."
                )
            flatten = bool(cfg.get("flatten", False))
            if head in self._name2idx and (raw != [head] or history_len != 1 or not flatten):
                raise ValueError(
                    f"[ObsAssemblerNP] Head '{head}' has the same name as a sensor. "
                    "Same-name heads are only allowed as sources=[name], history_len=1, flatten=True."
                )

            raw_sources[head] = raw
            self.head_histlen[head] = history_len
            self.head_flatten[head] = flatten

        visit_state: Dict[str, int] = {}

        def visit(head: str, stack: List[str]) -> List[int]:
            state = visit_state.get(head, 0)
            if state == 2:
                return self.head_sources[head]
            if state == 1:
                cycle = " -> ".join(stack + [head])
                raise ValueError(f"[ObsAssemblerNP] Head dependency cycle detected: {cycle}.")

            visit_state[head] = 1
            expanded: List[int] = []
            direct: List[Tuple[str, object]] = []
            block_dims: List[int] = []
            block_tails: List[Tuple[int, ...]] = []

            for token in raw_sources[head]:
                if token in self._name2idx:
                    sensor_idx = self._name2idx[token]
                    expanded.append(sensor_idx)
                    direct.append(("sensor", sensor_idx))
                    block_dims.append(self.frames[sensor_idx] * self.lead[sensor_idx])
                    block_tails.append(self.tail[sensor_idx])
                    continue

                if token not in raw_sources:
                    raise KeyError(f"[ObsAssemblerNP] unknown sensor or head '{token}' in head '{head}'.")

                visit(token, stack + [head])
                if not self.head_flatten[token]:
                    raise ValueError(
                        f"[ObsAssemblerNP] Head '{head}' references head '{token}', which must have flatten=True."
                    )
                expanded.extend(self.head_sources[token])
                direct.append(("head", token))
                block_dims.append(self.head_histlen[token] * self.S_head[token])
                block_tails.append(self.head_tailshape[token])

            base_tail = block_tails[0]
            for tail in block_tails:
                if tail != base_tail:
                    raise ValueError(f"[ObsAssemblerNP] Head '{head}' tail mismatch among sources.")

            self.head_sources[head] = expanded
            self._head_direct_sources[head] = direct
            self.head_tailshape[head] = base_tail
            self.S_head[head] = sum(block_dims)
            visit_state[head] = 2
            self._head_topo_order.append(head)
            return expanded

        for head in self._head_names:
            visit(head, [])

        used = set()
        for sources in self.head_sources.values():
            used.update(sources)
        self._used_idx = sorted(used)

    # ------------------------------------------------------------------
    # Storage
    # ------------------------------------------------------------------

    def _alloc_storage(self) -> None:
        count = len(self.names)
        self.cache: List[Optional[np.ndarray]] = [None] * count
        self.rings: List[Optional[RingBuffer]] = [None] * count
        self._tick: List[Optional[int]] = [None] * count

        for i in self._used_idx:
            shape = self.shape[i]
            frames = self.frames[i]
            lead = self.lead[i]
            tail = self.tail[i]

            if self.stride[i] > 1:
                self._tick[i] = 0

            if frames == 1:
                self.cache[i] = np.zeros(shape, dtype=self.dtype)
            else:
                self.rings[i] = RingBuffer(frames, (lead, *tail), dtype=self.dtype)
                self.cache[i] = np.zeros((frames * lead, *tail), dtype=self.dtype)

    def _alloc_head_history(self) -> None:
        self._head_rings: Dict[str, Optional[RingBuffer]] = {}
        for head in self._head_topo_order:
            history_len = self.head_histlen[head]
            if history_len == 1:
                self._head_rings[head] = None
                continue
            tail = self.head_tailshape[head]
            self._head_rings[head] = RingBuffer(history_len, (self.S_head[head], *tail), dtype=self.dtype)

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
                self.cache[i][...] = x
                continue

            lead = self.lead[i]
            tail = self.tail[i]
            ring = self.rings[i]
            ring.push(x.reshape(lead, *tail))
            ring.copy_last_flat_to(self.frames[i], self.cache[i])

        out: Dict[str, np.ndarray] = {}
        for head in self._head_topo_order:
            parts = []
            for kind, data in self._head_direct_sources[head]:
                if kind == "sensor":
                    parts.append(self.cache[data])
                else:
                    parts.append(out[data])

            y = parts[0].copy() if len(parts) == 1 else np.concatenate(parts, axis=0)
            if self._do_clip:
                np.clip(y, -self.clip_observations, self.clip_observations, out=y)

            history_len = self.head_histlen[head]
            if history_len == 1:
                out[head] = y if self.head_flatten[head] else y[np.newaxis, ...]
                continue

            ring = self._head_rings[head]
            ring.push(y)
            history = ring.get_last(history_len)
            if self.head_flatten[head]:
                tail = self.head_tailshape[head]
                out[head] = history.reshape(history_len * self.S_head[head], *tail)
            else:
                out[head] = history

        return {head: out[head] for head in self._head_names}

    def reset(self) -> None:
        for i in self._used_idx:
            self.cache[i].fill(0)
            if self.rings[i] is not None:
                self.rings[i].reset()
            if self._tick[i] is not None:
                self._tick[i] = 0

        for ring in self._head_rings.values():
            if ring is not None:
                ring.reset()

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def head_output_spec(self) -> Dict[str, dict]:
        spec = {}
        for head in self._head_names:
            history_len = self.head_histlen[head]
            flatten = self.head_flatten[head]
            tail = self.head_tailshape[head]
            per_step_dim = self.S_head[head]

            if history_len == 1:
                if flatten:
                    shape = (per_step_dim, *tail)
                else:
                    shape = (1, per_step_dim, *tail)
            else:
                if flatten:
                    shape = (history_len * per_step_dim, *tail)
                else:
                    shape = (history_len, per_step_dim, *tail)

            spec[head] = {
                "output_shape": shape,
                "history_len": history_len,
                "flatten": flatten,
                "per_step_dim": per_step_dim,
                "tail": tail,
                "dtype": self.dtype,
            }
        return spec

    def info_str(self) -> str:
        def fmt_shape(shape) -> str:
            return "(" + ", ".join(str(x) for x in tuple(shape)) + ("," if len(tuple(shape)) == 1 else "") + ")"

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
                f"stride={self.stride[i]}, lead={self.lead[i]}, tail={fmt_shape(self.tail[i])}, "
                f"obs_scale={scale_desc}, used={i in used}, " + ", ".join(storage)
            )

        lines.extend(["", "Heads:"])
        spec = self.head_output_spec()
        for head in self._head_names:
            source_names = []
            blocks = []
            offset = 0
            for kind, data in self._head_direct_sources[head]:
                if kind == "sensor":
                    source_name = self.names[data]
                    width = self.frames[data] * self.lead[data]
                    tail = self.tail[data]
                else:
                    source_name = data
                    width = self.head_histlen[data] * self.S_head[data]
                    tail = self.head_tailshape[data]
                source_names.append(source_name)
                blocks.append(
                    f"{source_name}[{offset}:{offset + width}, tail={fmt_shape(tail)}]"
                )
                offset += width

            ring = self._head_rings[head]
            history_storage = "none" if ring is None else f"ring_shape={fmt_shape(ring.buf.shape)}, ring_head={ring.head}"
            lines.append(
                f"  head {head}: sources=[{', '.join(source_names)}], history_len={self.head_histlen[head]}, "
                f"flatten={self.head_flatten[head]}, per_step_dim={self.S_head[head]}, "
                f"tail={fmt_shape(self.head_tailshape[head])}, output_shape={fmt_shape(spec[head]['output_shape'])}, "
                f"history_storage={history_storage}"
            )
            lines.append(f"    blocks: {', '.join(blocks)}")

        lines.append("")
        lines.append("Head topo order: " + " -> ".join(self._head_topo_order))
        return "\n".join(lines)

    def print_info(self) -> None:
        print(self.info_str())
    

if __name__ == "__main__":
    sensors = {
        "a": {"shape": (18,32), "frames": 8, "stride": 5},
        "b": {"shape": (4,), "frames": 2, "stride": 1},
        "c": {"shape": (3,), "frames": 1, "stride": 1},
        
    }
    heads = {
        "h1": {"sources": ["a"], "history_len": 1, "flatten": False},
        
    }
    assembler = ObsAssembler(sensors, heads, clip_observations=10.0)
    assembler.print_info()
    inputs = {
        "a": np.arange(18*32).reshape(18,32),
        "b": [3.0, 4.0, 5.0, 6.0],
        "c": [7.0, 8.0, 9.0],
    }
    for t in range(10):
        print(f"Step {t}:")
        out = assembler.step(inputs)
        for head, value in out.items():
            print(f"  {head}: {value}")
            print(f"  {head} shape: {value.shape}")
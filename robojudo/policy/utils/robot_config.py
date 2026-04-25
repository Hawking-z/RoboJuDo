import re

import numpy as np
import yaml
from typing import Any, Dict, List, Optional, Tuple

def product(v: List[int]) -> int:
    """计算 shape 的乘积，并做合法性检查（完全对应 C++ 的 product 函数）"""
    if not v:
        return 0  # 空 shape 非法
    p = 1
    for x in v:
        x = int(x)
        if x <= 0:
            raise RuntimeError("shape 维度必须为正整数")
        p *= x
        if p > np.iinfo(np.int32).max:
            raise RuntimeError("shape 维度乘积过大")
    return int(p)


def as_vec_i(node: Any) -> List[int]:
    """把 YAML 中的序列解析为 int list，对应 C++ 的 as_vec_i"""
    if not isinstance(node, (list, tuple)):
        raise RuntimeError("期望一个整数数组 (YAML sequence)")
    return [int(x) for x in node]


def as_vec_s(node: Any) -> List[str]:
    """把 YAML 中的序列解析为 str list，对应 C++ 的 as_vec_s"""
    if not isinstance(node, (list, tuple)):
        raise RuntimeError("期望一个字符串数组 (YAML sequence)")
    return [str(x) for x in node]


def node_has(node: Dict[str, Any], key: str) -> bool:
    """判断字典是否有这个 key 且非 None，对应 C++ 的 node_has"""
    return isinstance(node, dict) and (key in node) and (node[key] is not None)


DOF_PARAM_FIELDS = {
    "kp": "kp",
    "kd": "kd",
    "action_scale": "scale",
    "torque_limits": "torque_limits",
    "default_pos": "default_pos",
}


class DofConfig:
    def __init__(self):
        self.isaac_order: List[str] = []
        self.name_to_index: Dict[str, int] = {}
        # 以下全部是长度 = DOF 的 numpy 向量（float32）
        self.kp: np.ndarray = None
        self.kd: np.ndarray = None
        self.scale: np.ndarray = None
        self.torque_limits: np.ndarray = None
        self.default_pos: np.ndarray = None

class SensorSpec:
    def __init__(self, name: str):
        self.name: str = name
        self.shape: List[int] = []   # 多维 shape
        self.elem_dim: int = 0       # 展平长度
        self.frames: int = 1         # 传感器自身时间窗
        self.stride: int = 1         # 传感器采样间隔
        self.scale: float = 1.0      # obs_scale

class ObsSpec:
    class SourceSlice:
        def __init__(
            self,
            kind: str = "sensor",
            source: str = "",
            start: int = 0,
            end: int = 0,
            C: int = 0,
            tail_shape: Optional[List[int]] = None,
        ):
            self.kind = kind
            self.source = source
            self.start = start
            self.end = end
            self.C = C
            self.tail_shape = list(tail_shape or [])

    def __init__(self, name: str):
        self.name: str = name
        self.sources: List[str] = []  # source 名字列表（sensor / sensor_noise / obs head）
        self.history_len: int = 1
        self.tail_ndim: int = 0
        self.history_mode: str = "merge"

        self.per_step_dim: int = 0  # C_total
        self.final_dim: int = 0     # history_len * per_step_dim, used for internal head ref view
        self.tail_shape: List[int] = []
        self.output_shape: List[int] = []

        # 与 sources 对齐，保存每个 source 在 per_step 向量中的切片信息
        self.slices: List[ObsSpec.SourceSlice] = []

class RobotConfig:
    def __init__(self):
        self.num_actions: int = 0
        self.dof: DofConfig = DofConfig()

        self.sensors: Dict[str, SensorSpec] = {}
        self.obs_map: Dict[str, ObsSpec] = {}
        self.deploy_obs_heads: List[str] = []

        self.clip_obs: float = 0.0
        self.clip_actions: float = 0.0
        self.pd_rate: float = 1000.0
        self.infer_rate: float = 100.0
        self.exp_avg_decay: float = 1.0

        # gait 调度参数
        self.gait_config: Dict[str, Any] = {}

    def obs_assembler_schema(self) -> Tuple[Dict[str, dict], Dict[str, dict]]:
        sensors = {
            name: {
                "shape": spec.shape,
                "frames": spec.frames,
                "stride": spec.stride,
                "obs_scale": spec.scale,
            }
            for name, spec in self.sensors.items()
        }
        obs_heads = {
            name: {
                "sources": spec.sources,
                "history_len": spec.history_len,
                "tail_ndim": spec.tail_ndim,
                "history_mode": spec.history_mode,
            }
            for name, spec in self.obs_map.items()
        }
        return sensors, obs_heads

    def make_obs_assembler(self, obs_assembler_cls=None):
        if obs_assembler_cls is None:
            from robojudo.policy.utils.obs_assembler import ObsAssembler

            obs_assembler_cls = ObsAssembler

        sensors, obs_heads = self.obs_assembler_schema()
        return obs_assembler_cls(sensors, obs_heads, clip_observations=self.clip_obs)
    
    # -------- 从 YAML 读取 + 校验 --------
    @staticmethod
    def from_yaml_file(path: str) -> "RobotConfig":
        with open(path, "r", encoding="utf-8") as f:
            root = yaml.safe_load(f)

        if root is None:
            raise RuntimeError(f"无法加载 YAML 文件: {path}")

        cfg = RobotConfig()

        # 基本标量
        cfg.num_actions = int(root["num_actions"])
        cfg.clip_obs = float(root.get("clip_obs", 0.0))
        cfg.clip_actions = float(root.get("clip_actions", 0.0))
        cfg.pd_rate = float(root.get("pd_rate", 1000.0))
        cfg.infer_rate = float(root.get("infer_rate", 100.0))
        cfg.exp_avg_decay = float(root.get("exp_avg_decay", 1.0))
        cfg.gait_config = root.get("gait", {})

        # ---------- dof_config ----------
        if not node_has(root, "dof_config"):
            raise RuntimeError("缺少 dof_config 段")
        dcfg = root["dof_config"]

        if not node_has(dcfg, "isaac_order"):
            raise RuntimeError("dof_config 缺少 isaac_order")
        cfg.dof.isaac_order = as_vec_s(dcfg["isaac_order"])

        for i, jn in enumerate(cfg.dof.isaac_order):
            cfg.dof.name_to_index[jn] = i

        DOF = len(cfg.dof.isaac_order)
        if DOF <= 0:
            raise RuntimeError("isaac_order 为空")

        # 初始化 dof 向量
        def init_vec() -> np.ndarray:
            return np.zeros(DOF, dtype=np.float32)

        cfg.dof.kp = init_vec()
        cfg.dof.kd = init_vec()
        cfg.dof.scale = init_vec()
        cfg.dof.torque_limits = init_vec()
        cfg.dof.default_pos = init_vec()
        assigned = {
            field: np.zeros(DOF, dtype=bool)
            for field in DOF_PARAM_FIELDS
        }

        for rule_name, rule_node in dcfg.items():
            if rule_name == "isaac_order":
                continue
            if not isinstance(rule_node, dict):
                raise RuntimeError(f"dof_config 规则必须是 map: {rule_name}")

            if rule_name in cfg.dof.name_to_index:
                matched_indices = [cfg.dof.name_to_index[rule_name]]
            else:
                try:
                    pattern = re.compile(rule_name)
                except re.error as e:
                    raise RuntimeError(f"dof_config 正则表达式非法: {rule_name}, 错误: {e}") from e
                matched_indices = [
                    i for i, joint_name in enumerate(cfg.dof.isaac_order)
                    if pattern.search(joint_name)
                ]

            if not matched_indices:
                raise RuntimeError(f"dof_config 规则未匹配任何关节: {rule_name}")

            unknown_fields = [field for field in rule_node if field not in DOF_PARAM_FIELDS]
            if unknown_fields:
                raise RuntimeError(
                    f"dof_config 规则 {rule_name} 包含未知字段: {', '.join(unknown_fields)}"
                )

            try:
                for field_name, attr_name in DOF_PARAM_FIELDS.items():
                    if field_name not in rule_node:
                        continue
                    value = float(rule_node[field_name])
                    target = getattr(cfg.dof, attr_name)
                    for idx in matched_indices:
                        target[idx] = value
                        assigned[field_name][idx] = True
            except Exception as e:
                raise RuntimeError(f"解析关节参数失败: {rule_name}, 错误: {e}") from e

        for field_name in DOF_PARAM_FIELDS:
            missing = [
                cfg.dof.isaac_order[i]
                for i, is_set in enumerate(assigned[field_name])
                if not is_set
            ]
            if missing:
                raise RuntimeError(
                    f"dof_config 未为以下关节设置 {field_name}: {', '.join(missing)}"
                )

        # ---------- obs_config ----------
        if not node_has(root, "obs_config"):
            raise RuntimeError("缺少 obs_config 段")
        ocfg = root["obs_config"]

        if not node_has(ocfg, "sensors"):
            raise RuntimeError("obs_config 缺少 sensors 段")
        sc = ocfg["sensors"]

        if not isinstance(sc, dict):
            raise RuntimeError("sensors 必须是一个 map")

        # 读取各个 sensor 规格
        for name, sn in sc.items():
            ss = SensorSpec(name)
            if not node_has(sn, "shape"):
                raise RuntimeError(f"sensor {name} 缺少 shape")
            ss.shape = as_vec_i(sn["shape"])
            ss.elem_dim = product(ss.shape)
            ss.frames = int(sn.get("frames", 1))
            ss.stride = int(sn.get("stride", 1))
            ss.scale = float(sn.get("obs_scale", 1.0))
            if ss.frames <= 0:
                raise RuntimeError(f"sensor {name} 的 frames 必须 >=1")
            if ss.stride <= 0:
                raise RuntimeError(f"sensor {name} 的 stride 必须 >=1")
            cfg.sensors[name] = ss

        # 读取各个 obs 头（prop_obs / encoder_obs 等）
        for key, on in ocfg.items():
            if key == "sensors":
                continue
            if not isinstance(on, dict):
                raise RuntimeError(f"obs {key} 必须是一个 map")

            if node_has(on, "flatten") or node_has(on, "mode"):
                raise RuntimeError(
                    f"obs {key} 使用了旧字段 flatten/mode，请改用 tail_ndim/history_mode"
                )

            ospec = ObsSpec(key)
            if not node_has(on, "sources"):
                raise RuntimeError(f"obs {key} 缺少 sources")
            ospec.sources = as_vec_s(on["sources"])
            ospec.history_len = int(on.get("history_len", 1))
            if ospec.history_len <= 0:
                raise RuntimeError(f"obs {key} 的 history_len 必须 >=1")
            ospec.tail_ndim = int(on.get("tail_ndim", 0))
            if ospec.tail_ndim < 0:
                raise RuntimeError(f"obs {key} 的 tail_ndim 必须 >=0")
            ospec.history_mode = str(on.get("history_mode", "merge"))
            if ospec.history_mode not in ("keep", "merge"):
                raise RuntimeError(f"obs {key} 的 history_mode 必须是 keep 或 merge")

            cfg.obs_map[ospec.name] = ospec

        raw_obs_sources = {
            name: list(spec.sources)
            for name, spec in cfg.obs_map.items()
        }
        visit_state: Dict[str, int] = {}

        def normalize_sensor_source(src: str) -> Optional[str]:
            if src in cfg.sensors:
                return src
            if src.endswith("_noise"):
                base = src[:-len("_noise")]
                if base in cfg.sensors:
                    return base
            return None

        def resolve_obs_head(head_name: str, stack: List[str]) -> ObsSpec:
            state = visit_state.get(head_name, 0)
            if state == 2:
                return cfg.obs_map[head_name]
            if state == 1:
                cycle = " -> ".join(stack + [head_name])
                raise RuntimeError(f"obs head 依赖存在环: {cycle}")

            visit_state[head_name] = 1
            ospec = cfg.obs_map[head_name]
            offset = 0
            clean_sources: List[str] = []
            slices: List[ObsSpec.SourceSlice] = []
            dep_tail_shape: Optional[List[int]] = None

            for raw_src in raw_obs_sources[head_name]:
                sensor_name = normalize_sensor_source(raw_src)
                if sensor_name is not None:
                    clean_sources.append(sensor_name)
                    ss = cfg.sensors[sensor_name]
                    if ospec.tail_ndim > len(ss.shape):
                        raise RuntimeError(
                            f"obs {head_name} 的 tail_ndim={ospec.tail_ndim} 超过了 source {sensor_name} 的 rank={len(ss.shape)}"
                        )
                    if ospec.tail_ndim == 0:
                        tail_shape = []
                        front_shape = list(ss.shape)
                    else:
                        tail_shape = list(ss.shape[-ospec.tail_ndim:])
                        front_shape = list(ss.shape[:-ospec.tail_ndim])
                    C_i = ss.frames * (product(front_shape) if front_shape else 1)
                    if dep_tail_shape is None:
                        dep_tail_shape = list(tail_shape)
                    elif dep_tail_shape != list(tail_shape):
                        raise RuntimeError(
                            f"obs {head_name} 的 source tail 不一致，期望 {dep_tail_shape}，实际 {tail_shape}"
                        )
                    slices.append(
                        ObsSpec.SourceSlice(
                            kind="sensor",
                            source=sensor_name,
                            start=offset,
                            end=offset + C_i,
                            C=C_i,
                            tail_shape=tail_shape,
                        )
                    )
                    offset += C_i
                    continue

                if raw_src.endswith("_noise"):
                    base = raw_src[:-len("_noise")]
                    if base in cfg.obs_map:
                        raise RuntimeError(f"obs {head_name} 不能引用带 _noise 后缀的 obs head: {raw_src}")

                if raw_src not in cfg.obs_map:
                    raise RuntimeError(f"obs {head_name} 引用了未知 sensor 或 obs head: {raw_src}")

                dep_spec = resolve_obs_head(raw_src, stack + [head_name])
                clean_sources.append(raw_src)
                tail_shape = list(dep_spec.tail_shape)
                if dep_tail_shape is None:
                    dep_tail_shape = list(tail_shape)
                elif dep_tail_shape != list(tail_shape):
                    raise RuntimeError(
                        f"obs {head_name} 的 source tail 不一致，期望 {dep_tail_shape}，实际 {tail_shape}"
                    )
                C_i = dep_spec.final_dim
                slices.append(
                    ObsSpec.SourceSlice(
                        kind="head",
                        source=raw_src,
                        start=offset,
                        end=offset + C_i,
                        C=C_i,
                        tail_shape=tail_shape,
                    )
                )
                offset += C_i

            ospec.sources = clean_sources
            ospec.slices = slices
            ospec.per_step_dim = offset
            ospec.final_dim = ospec.history_len * ospec.per_step_dim
            ospec.tail_shape = dep_tail_shape or []
            if ospec.history_mode == "keep":
                ospec.output_shape = [ospec.history_len, ospec.per_step_dim, *ospec.tail_shape]
            elif ospec.history_len == 1:
                ospec.output_shape = [ospec.per_step_dim, *ospec.tail_shape]
            else:
                ospec.output_shape = [ospec.final_dim, *ospec.tail_shape]
            visit_state[head_name] = 2
            return ospec

        for head_name in raw_obs_sources:
            resolve_obs_head(head_name, [])

        if node_has(root, "deploy_obs_heads"):
            cfg.deploy_obs_heads = as_vec_s(root["deploy_obs_heads"])
        elif node_has(root, "deploy_heads"):
            cfg.deploy_obs_heads = as_vec_s(root["deploy_heads"])
        else:
            cfg.deploy_obs_heads = list(cfg.obs_map.keys())

        if not cfg.deploy_obs_heads:
            raise RuntimeError("deploy_obs_heads 不能为空")

        missing_deploy_obs = [name for name in cfg.deploy_obs_heads if name not in cfg.obs_map]
        if missing_deploy_obs:
            raise RuntimeError(
                f"deploy_obs_heads 引用了未知 obs head: {', '.join(missing_deploy_obs)}"
            )
        return cfg

if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        raise SystemExit("用法: python robot_config.py /path/to/robot_config.yaml")

    cfg = RobotConfig.from_yaml_file(sys.argv[1])
    print("Num actions:", cfg.num_actions)
    print("DOF names:", cfg.dof.isaac_order)
    print("DOF Kp:", cfg.dof.kp)
    print("DOF Kd:", cfg.dof.kd)
    print("DOF torque limits:", cfg.dof.torque_limits)
    print("DOF default pos:", cfg.dof.default_pos)
    print("gait config:", cfg.gait_config)
    print("Number of sensors:", len(cfg.sensors))
    print("Sensors:")
    for sname, sspec in cfg.sensors.items():
        print(f"Sensor {sname}: shape = {sspec.shape}, frames = {sspec.frames}, elem_dim = {sspec.elem_dim}, scale = {sspec.scale}")
    print("Obs specs:")
    for oname, ospec in cfg.obs_map.items():
        print(f"Obs {oname}: final_dim = {ospec.final_dim}")

    assembler = cfg.make_obs_assembler()
    print("ObsAssembler output spec:")
    for name, spec in assembler.head_output_spec().items():
        print(f"Obs {name}: output_shape = {spec['output_shape']}")

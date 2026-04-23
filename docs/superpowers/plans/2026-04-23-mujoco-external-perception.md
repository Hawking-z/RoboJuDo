# MuJoCo External Perception Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add configurable MuJoCo camera and terrain-height perception outputs to `MujocoEnv` and expose them through `env.get_data()` in a way that can later be reused by real environments.

**Architecture:** Extend the base environment with a generic extra-data cache, introduce a small perception package with MuJoCo camera and terrain providers plus a manager, then wire `MujocoEnv` to publish provider outputs as flat environment-data keys. Keep policy changes minimal and focused on robust env-data lookup.

**Tech Stack:** Python 3.11, Pydantic config models, MuJoCo Python bindings, NumPy, OpenCV, unittest

---

### Task 1: Define the shared config and env-data contract

**Files:**
- Modify: `robojudo/environment/env_cfgs.py`
- Modify: `robojudo/environment/base_env.py`
- Test: `tests/test_custom_policy.py`

- [ ] **Step 1: Write the failing test**

```python
def test_environment_get_data_merges_external_sensor_outputs(self):
    env = _DummyEnvironmentForExternalData()
    env.set_extra_env_data({
        "camera_front_depth": np.ones((4, 4), dtype=np.float32),
        "height_scan": np.arange(6, dtype=np.float32),
    })

    data = env.get_data()

    np.testing.assert_allclose(data["camera_front_depth"], np.ones((4, 4), dtype=np.float32))
    np.testing.assert_allclose(data["height_scan"], np.arange(6, dtype=np.float32))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_custom_policy.py -k external_sensor_outputs -v`
Expected: FAIL because `Environment` has no external-data cache API yet.

- [ ] **Step 3: Write minimal implementation**

```python
class Environment(ABC):
    def __init__(...):
        self._extra_env_data: dict[str, object] = {}

    def set_extra_env_data(self, data: dict[str, object] | None):
        self._extra_env_data = {} if data is None else dict(data)

    def get_data(self):
        env_data = {...}
        env_data.update(self._extra_env_data)
        return Box(env_data)
```

Add typed config models in `env_cfgs.py` for:
- external perception root
- camera config
- terrain raycast config
- terrain height sampler config
- debug config

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_custom_policy.py -k external_sensor_outputs -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_custom_policy.py robojudo/environment/base_env.py robojudo/environment/env_cfgs.py
git commit -m "feat: add external perception env contract"
```

### Task 2: Add the MuJoCo perception package

**Files:**
- Create: `robojudo/environment/perception/__init__.py`
- Create: `robojudo/environment/perception/base.py`
- Create: `robojudo/environment/perception/manager.py`
- Create: `robojudo/environment/perception/mujoco_camera.py`
- Create: `robojudo/environment/perception/terrain_height.py`
- Test: `tests/test_custom_policy.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_custom_policy_reads_external_sensor_from_mapping_env_data(self):
    cfg = _write_robot_config_with_sensor("camera_front_depth", [4, 4])
    policy = CustomPolicy(cfg_policy=cfg, device="cpu")
    env_data = {
        "base_quat": np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        "base_ang_vel": np.zeros(3, dtype=np.float32),
        "dof_pos": np.asarray(cfg.obs_dof.default_pos, dtype=np.float32),
        "dof_vel": np.zeros(cfg.obs_dof.num_dofs, dtype=np.float32),
        "camera_front_depth": np.ones((4, 4), dtype=np.float32),
    }

    obs, extras = policy.get_observation(env_data, {})

    self.assertEqual(obs.dtype, np.float32)
    self.assertIn("camera_front_depth", extras["obs_outputs"])
```

```python
def test_mujoco_camera_config_normalizes_output_keys(self):
    cfg = MujocoCameraPerceptionCfg(link_name="head", output_name="front")
    self.assertEqual(cfg.depth_output_key(), "camera_front_depth")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_custom_policy.py -k "external_sensor or normalizes_output_keys" -v`
Expected: FAIL because config helpers and robust sensor access do not exist yet.

- [ ] **Step 3: Write minimal implementation**

Define:

```python
class PerceptionProvider(ABC):
    def outputs(self) -> dict[str, np.ndarray]:
        ...
```

```python
class PerceptionManager:
    def refresh(self) -> dict[str, np.ndarray]:
        merged = {}
        for provider in self.providers:
            merged.update(provider.refresh())
        return merged
```

Add MuJoCo-specific providers for:
- camera config parsing and rendering
- terrain raycast and height sampling

Keep provider outputs flat and shape-stable.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_custom_policy.py -k "external_sensor or normalizes_output_keys" -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_custom_policy.py robojudo/environment/perception robojudo/policy/custom_policy.py
git commit -m "feat: add mujoco perception providers"
```

### Task 3: Wire MuJoCo perception into MujocoEnv

**Files:**
- Modify: `robojudo/environment/mujoco_env.py`
- Modify: `robojudo/environment/env_cfgs.py`
- Test: `tests/test_custom_policy.py`

- [ ] **Step 1: Write the failing test**

```python
def test_mujoco_env_cfg_accepts_external_perception_camera_and_height_sampler(self):
    cfg = MujocoEnvCfg(
        xml="robot.xml",
        dof=_minimal_dof_cfg(),
        external_perception={
            "enabled": True,
            "cameras": {
                "front": {
                    "link_name": "head",
                    "resolution": [64, 48],
                    "render_mode": "depth",
                }
            },
            "terrain": {
                "height_samplers": {
                    "height_scan": {
                        "link": "torso_link",
                        "points": {"type": "grid", "size": [0.4, 0.2], "resolution": [0.2, 0.2]},
                    }
                }
            },
        },
    )

    self.assertTrue(cfg.external_perception.enabled)
    self.assertIn("front", cfg.external_perception.cameras)
    self.assertIn("height_scan", cfg.external_perception.terrain.height_samplers)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_custom_policy.py -k accepts_external_perception_camera_and_height_sampler -v`
Expected: FAIL because `MujocoEnvCfg` does not accept the nested config yet.

- [ ] **Step 3: Write minimal implementation**

Update `MujocoEnv` to:
- compile a model with configured cameras attached
- initialize a perception manager
- refresh provider outputs during `update()`
- merge outputs into `env.get_data()`
- optionally display camera frames and height debug points

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_custom_policy.py -k accepts_external_perception_camera_and_height_sampler -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_custom_policy.py robojudo/environment/mujoco_env.py robojudo/environment/env_cfgs.py
git commit -m "feat: wire external perception into mujoco env"
```

### Task 4: Verify regressions and finish

**Files:**
- Modify: `tests/test_custom_policy.py`
- Verify: `robojudo/environment/base_env.py`
- Verify: `robojudo/environment/mujoco_env.py`
- Verify: `robojudo/environment/perception/*.py`

- [ ] **Step 1: Run focused tests**

Run: `pytest tests/test_custom_policy.py -v`
Expected: PASS

- [ ] **Step 2: Run style checks on touched files**

Run: `ruff check robojudo/environment/base_env.py robojudo/environment/env_cfgs.py robojudo/environment/mujoco_env.py robojudo/environment/perception tests/test_custom_policy.py`
Expected: All checks passed

- [ ] **Step 3: Review requirements against spec**

Check:
- `env.get_data()` publishes external perception outputs
- camera and height sampler configs are present in `MujocoEnvCfg`
- `MujocoEnv` remains MuJoCo-only while keeping future real extension seams clear
- policy code can consume external sensor keys

- [ ] **Step 4: Commit**

```bash
git add docs/superpowers/specs/2026-04-23-mujoco-external-perception-design.md docs/superpowers/plans/2026-04-23-mujoco-external-perception.md
git commit -m "docs: capture mujoco external perception design and plan"
```

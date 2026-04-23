# MuJoCo External Perception Design

**Date:** 2026-04-23

## Goal

Add configurable external perception to `MujocoEnv` so MuJoCo sim can produce camera and terrain-height observations as environment data, while keeping the design easy to extend to real sensors later.

## Scope

This change only implements the MuJoCo side.

Included:
- Configurable body-mounted cameras in MuJoCo
- Configurable terrain raycast and height samplers
- Environment-level external perception outputs exposed through `env.get_data()`
- Debug display hooks for camera frames and height points
- A structure that can later host real-sensor providers without changing policy-facing APIs

Not included:
- Real sensor implementation
- Async camera pipelines
- Image compression / recording
- Automatic policy wiring beyond exposing environment data

## Existing Constraints

- Policies already consume `env_data` through `Environment.get_data()`.
- `CustomPolicy` already supports arbitrary sensor names as long as values exist in `env_data`.
- `MujocoEnv` is currently responsible for simulation state update and viewer rendering, but not external perception.
- Future real support should reuse the same environment-data contract instead of branching policy code per backend.

## Design

### 1. Environment Data Contract

Add a generic external-data cache to `Environment`. `get_data()` will merge the usual proprioceptive state with externally produced sensor data.

Implications:
- Policies can reference external sensors by name in robot config without knowing whether the source is MuJoCo or real hardware.
- Non-perception environments keep working because the cache defaults to empty.

### 2. Config-Driven Perception

Extend `MujocoEnvCfg` with an `external_perception` section containing:
- `enabled`
- `cameras`
- `terrain.raycast`
- `terrain.height_samplers`
- `debug`

The config owns:
- which sensors are active
- how they are parameterized
- which output keys they publish into environment data

The config does not own:
- policy sensor schemas
- observation assembly

### 3. Perception Package

Create `robojudo/environment/perception/` with clear boundaries:

- `base.py`
  - common dataclasses and provider base interface
- `manager.py`
  - owns provider lifecycle and output aggregation
- `mujoco_camera.py`
  - MuJoCo camera config parsing, camera creation, renderer setup, frame rendering
- `terrain_height.py`
  - terrain raycast setup, point generation, transforms, height sampling

This is intentionally MuJoCo-first but manager/provider boundaries are generic so future real providers can plug in without changing `MujocoEnv` call sites.

### 4. MujocoEnv Integration

`MujocoEnv` will:
- build the MuJoCo model with configured cameras attached before `MjData` creation
- initialize a perception manager after model/data creation
- refresh external perception after state updates
- expose provider outputs through `get_data()`
- keep debug rendering optional

`MujocoEnv` should not own sensor-specific math once providers exist.

### 5. Output Naming

Outputs are published as flat environment-data keys, for example:
- `camera_front_depth`
- `camera_front_color`
- `height_scan`

Flat names are preferred over nested structures or slash-delimited keys because existing env-data usage mixes attribute and mapping access, and flat names avoid compatibility surprises.

Each provider config can override output names explicitly.

### 6. Future Real Extension

Future real support should add providers that implement the same output contract:
- a RealSense depth provider can publish `camera_front_depth`
- a terrain estimator can publish `height_scan`

That means:
- policy config stays unchanged
- observation assembly stays unchanged
- only environment/provider wiring differs

The shared extension seam is:
- environment-level extra data cache
- provider interface returning `dict[str, np.ndarray]`
- config objects describing sensor outputs

## Risks

### Camera Attachment Compatibility

MuJoCo Python APIs differ by version. Camera creation should support both newer `MjSpec` helpers and older fallback creation paths.

### Viewer / Renderer Coupling

Rendering logic must work even when debug visualization is disabled. Perception rendering should not require the interactive viewer.

### Shape Stability

Policy-side sensor schemas require fixed shapes. Config validation must reject malformed resolutions, point grids, and sampler definitions early.

### Terrain Group Assumptions

Raycast needs stable terrain geom selection. Group handling should be explicit and configurable.

## Testing Strategy

- Unit-test config and policy-facing integration without needing a full viewer session
- Unit-test extra env data merging in `Environment.get_data()`
- Unit-test provider-independent behavior such as output naming and policy lookup
- Add targeted tests for `CustomPolicy` consuming external sensor keys from environment data

## Acceptance Criteria

- `MujocoEnvCfg` can describe cameras and height samplers
- `MujocoEnv` publishes configured camera and height outputs in `env.get_data()`
- No existing policy path regresses when no external perception is configured
- `CustomPolicy` can read external sensor values from environment data using configured sensor names
- The new structure does not bake MuJoCo-specific logic into policy code

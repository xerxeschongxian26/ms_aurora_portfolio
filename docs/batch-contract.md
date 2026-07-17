# Aurora Batch input contract

Executable source of truth: [`src/aurora_inference/contract.py`](../src/aurora_inference/contract.py).

Aurora can fail **silently** on malformed input — plausible, physically wrong forecasts with no exception. This document is the human-readable mirror of the validators that gate every inference call.

## What a `Batch` is

A `Batch` is not one tensor. It has three variable groups plus metadata:

| Field | Type | Shape | Notes |
|---|---|---|---|
| `surf_vars` | `dict[str, Tensor]` | `(B, T, H, W)` | Surface fields — no level axis |
| `atmos_vars` | `dict[str, Tensor]` | `(B, T, L, H, W)` | Pressure-level fields |
| `static_vars` | `dict[str, Tensor]` | `(H, W)` | **No** batch or time dim |
| `metadata` | `Metadata` | — | `lat`, `lon`, `time`, `atmos_levels` |

Dimensions, in order: **Batch × Time × Pressure Levels × Latitude × Longitude**.

Why the shapes differ:

- `surf_vars` — surface fields; no atmospheric level dimension
- `atmos_vars` — span batch, time, pressure levels, and spatial grid
- `static_vars` — do not vary across time or level

### Required variables (`AuroraPretrained` / `AuroraSmallPretrained`)

Held in `AURORA_PRETRAINED_SPEC` (`ModelSpec`):

| Group | Keys |
|---|---|
| Surface | `2t`, `10u`, `10v`, `msl` |
| Static | `lsm`, `slt`, `z` |
| Atmospheric | `t`, `u`, `v`, `q`, `z` |
| Pressure levels (hPa), **in this order** | `50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000` (13 levels) |

`ModelSpec` parameterises keys, levels, and timestep so multi-checkpoint support later is a config change, not a rewrite. Extra keys are allowed; **missing required keys fail**.

> **Hardening (deferred):** reject *unknown* keys (keys present in the batch but not in `ModelSpec`) as well. WP2 only requires that every spec key is present — surplus fields pass. Tightening to an exact key set is a reasonable later change; not in Stage 0 acceptance.

Other Aurora variants (`AuroraAirPollution`, `AuroraWave`) have different contracts — deferred beyond Stage 0.

## Common traps

### Two different `z` keys

| Key | Physical field |
|---|---|
| `static_vars["z"]` | Orography (surface geopotential) |
| `atmos_vars["z"]` | Geopotential at pressure levels |

Same name, different fields. Mixing them up is a classic silent-failure source.

### Time semantics

- Time index `0` = earlier input step (**t0**); index `1` = current input step (**t1**)
- Input `Batch` has **T == 2** exactly
- Model output `Batch` has **T == 1** (single predicted step)
- `metadata.time` has length `B`; **each element is the datetime at index 1 (t1), not t0**
- **t0 and t1 must be exactly 6 hours apart** — this cannot be recovered from `Batch` alone

CDS/ERA5 long names differ from Aurora short names (e.g. `2m_temperature` → `2t`). Upstream Aurora may silently omit missing keys; **our** `validate_batch` rejects them. Wrong keys can still `KeyError` or mis-embed inside the model — another reason not to rely on Aurora as the gate.

### Lat / lon orientation

| Coord | Rule |
|---|---|
| `lat` | Strictly **decreasing** from +90 → −90 |
| `lon` | Strictly **increasing** in **`[0, 360)`** — must **not** include 360 (ERA5 often uses `[-180, 180]`) |
| dtypes | `float32` or `float64` for coords; weather fields must be `float32` |

**Flip coords and fields together.** ERA5 often arrives with ascending latitude. Reversing only `metadata.lat` desynchronizes coordinates from data: `lat[i]` must still label spatial row `i` of every field. When correcting orientation, flip the **H** axis of all `surf_vars`, `atmos_vars`, and `static_vars` in lockstep with `lat` (and reorder the **W** axis the same way if you remap `lon`). Never mutate coordinates alone after the fact.

Stage 0 fixtures / `SyntheticSource` must **build** lat decreasing with fields already consistent — there is nothing to flip. The lockstep rule applies when constructing batches from real ERA5 (Stage 1 `ArcoEra5Source`).

### 1-D lat / lon only (Stage 0)

Stage 0 assumes a **global regular ERA5 grid**: `metadata.lat` and `metadata.lon` are **1-D** vectors with `len(lat) == H` and `len(lon) == W`. 2-D curvilinear coordinate arrays are out of scope.

### Data quality

- All weather tensors (`surf`, `atmos`, `static`) must be **`float32`** (ERA5/xarray often loads `float64` — cast explicitly)
- No NaN or Inf in any weather tensor

### Pressure levels

`metadata.atmos_levels` must match the spec **exactly, including order**. Misordered levels produce silently wrong forecasts. The `L` axis of every atmos tensor must match this ordering.

## Global grid note (§2c)

The native global 0.25° grid is **`(H, W) = (721, 1440)`**.

`721` is **not** divisible by 16 — that is fine. Aurora was trained on this native global grid. The divisible-by-16 patch constraint matters when you **crop to a region**. **Stage 0 does not crop.** Do not add a regional-cropping helper “for later”; regional inference is not in this project.

Tests and synthetic fixtures may use a smaller regular grid (e.g. `H=32, W=64`) with the same 1-D lat/lon contract.

## Validation boundary

Two gates — both raise `BatchContractError` with actionable messages:

| Gate | When | What |
|---|---|---|
| `validate_input_times(t0, t1)` | Every `BatchSource.load()` — **source seam** | Exact 6-hour spacing (default) |
| `validate_batch(batch, spec)` | Immediately before `model.forward()` — **inference boundary** | Keys, shapes, levels, coords, dtype/finite |

Do **not** rely on Aurora’s internal asserts as the production gate.

`spec.input_timestep_hours` exists on `ModelSpec` but is not yet wired into `validate_input_times` (WP3 will call the source-seam gate from every backend).

## Shape grammar (validators)

Checked by `validate_batch`:

1. All required surf / static / atmos keys present
2. Surface tensors rank 4; `T == 2`
3. Atmospheric tensors rank 5; `T == 2`; `L == len(atmos_levels)`
4. Static tensors rank 2, shape `(H, W)`
5. `H`, `W` consistent across every surf, atmos, and static tensor
6. `len(lat) == H`, `len(lon) == W`; lat/lon are 1-D
7. `lat` strictly decreasing; values in `[-90, 90]`
8. `lon` strictly increasing; values in `[0, 360)`
9. `atmos_levels` matches the expected 13 levels **in order**
10. `len(metadata.time) == B`
11. Weather tensors are `float32` and finite throughout
12. Input spacing enforced separately via `validate_input_times` (see above)

Spatial shape is anchored to `spec.surf_vars[0]` (not a hardcoded `"2t"`), so the same grammar works across checkpoints.

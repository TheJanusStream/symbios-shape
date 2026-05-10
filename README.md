# symbios-shape

A pure-Rust derivation engine for **CGA Shape Grammars**, as popularised by
Esri CityEngine. Define procedural building rules as text or Rust code; derive
a flat list of oriented mesh instances ready for any renderer.

```text
Lot  --> Extrude(15) Split(Y) { 3: Ground | ~1: Floor | 2: Roof }
Floor --> Repeat(X, 2.5) { Window }
Roof  --> Taper(0.75) I("RoofMesh")
```

## Features

- **All core CGA ops** — `Extrude`, `Split`, `Repeat`, `Comp(Faces)`, `Taper`,
  `Scale`, `Translate`, `Rotate`, `Align`, `Offset`, `Roof`, `Attach`, `I`, `Mat`
- **15 roof types** — Pyramid, Shed, Gable, Hip, Flat, OpenGable, BoxGable,
  PyramidHip, Butterfly, MShaped, Gambrel, Mansard, Saltbox, Jerkinhead, DutchGable
- **Three split modes** — absolute (`3:`), relative (`'0.3:`), floating (`~1:`)
- **Stochastic rules** — `70% BrickFacade | 30% GlassFacade` with reproducible
  seeded RNG
- **Implicit terminals** — unknown rule names resolve to `I(rule_name)` without
  an explicit `I(...)` op
- **Material propagation** — `Mat("Brick")` stamps a material that flows through
  all branching ops to the final `Terminal`
- **Rich face profiles** — `FaceProfile` describes each terminal's cross-section
  (Rectangle, Taper, Triangle, Trapezoid, Polygon) for accurate mesh generation
- **Genetic evolution** — `ShapeGenotype` wraps the rule table for use with
  `symbios-genetics` algorithms (mutation, BLX-α crossover)
- **Spatial queries** — `ShapeModel::query()` returns a `TerminalQuery` for
  OBB-vs-OBB overlap tests; `obb_overlap(&a, &b)` is also exposed directly
- **Lightweight** — depends on `glam` (math), `nom` (parsing), `rand`
  (stochastic rules), `thiserror` (errors), `serde` (round-trip), and
  `symbios-genetics` (evolution); no engine or runtime required
- **DoS-hardened** — bounded queue, depth, terminal, and identifier limits

## Installation

```toml
[dependencies]
symbios-shape = "0.2"
```

## Quick Start

```rust
use symbios_shape::{Interpreter, Scope, Vec3, Quat};
use symbios_shape::grammar::parse_ops;

let mut interp = Interpreter::new();

interp.add_rule("Lot",    parse_ops("Extrude(12) Split(Y) { 3: Ground | ~1: Floor | 2: Roof }").unwrap());
interp.add_rule("Ground", parse_ops(r#"I("GroundFloor")"#).unwrap());
interp.add_rule("Floor",  parse_ops(r#"I("UpperFloor")"#).unwrap());
interp.add_rule("Roof",   parse_ops(r#"Taper(0.8) I("Roof")"#).unwrap());

// XZ footprint — Y is set by Extrude
let footprint = Scope::new(Vec3::ZERO, Quat::IDENTITY, Vec3::new(10.0, 0.0, 10.0));
let model = interp.derive(footprint, "Lot").unwrap();

println!("{} terminals", model.len()); // 3
for t in &model.terminals {
    println!("  {} @ {:?}", t.mesh_id, t.scope.position);
}
```

## Grammar Syntax

### Operations

| Op        | Syntax                                                          | Description                                                                  |
|-----------|-----------------------------------------------------------------|------------------------------------------------------------------------------|
| Extrude   | `Extrude(h)`                                                    | Set Y size to `h` (h > 0)                                                    |
| Taper     | `Taper(t)`                                                      | Pyramidal taper, t ∈ [0, 1]                                                  |
| Scale     | `Scale(x, y, z)`                                                | Multiply scope size (all > 0)                                                |
| Translate | `Translate(x, y, z)`                                            | Shift origin in local space                                                  |
| Rotate    | `Rotate(w, x, y, z)`                                            | Apply quaternion rotation                                                    |
| Align     | `Align(Y, Up)`                                                  | Rotate so local axis points at world direction                               |
| Split     | `Split(Y) { 3: A \| ~1: B \| '0.2: C }`                         | Divide along axis (see Snap-aware Split below)                               |
| Repeat    | `Repeat(X, 2.5) { Window }` or `Repeat(X, [2, 1.5, 3]) { Bay }` | Tile along axis, stretch to fill                                             |
| Comp      | `Comp(Faces) { Top: R \| Side: R \| Bottom: R }`                | Decompose into face scopes                                                   |
| Offset    | `Offset(-0.2) { Inside: R \| Border: R }`                       | Inset a 2D face scope                                                        |
| Roof      | `Roof(Gable, 30) { Slope: R \| GableEnd: R }`                   | Generate roof geometry                                                       |
| Attach    | `Attach(Up) { Surface: R }`                                     | Project a scope onto a sloped face                                           |
| I         | `I("MeshId")` or `I(MeshId)`                                    | Emit terminal                                                                |
| Mat       | `Mat("Brick")` or `Mat("Brick", 1800)`                          | Set material (id, optional density kg/m³)                                    |
| Polygon   | `Polygon((0,0), (4,0), (4,2), (2,2), (2,4), (0,4))`             | Stamp a polygon FaceProfile on next terminal                                 |
| RegSnap   | `RegSnap("bays")`                                               | Register the 6 face planes of the current scope as snap-planes under a label |
| IfClear   | `IfClear { Window }`                                            | Invoke rule only if no terminal occludes the scope (OBB test)                |
| IfOccluded| `IfOccluded { Patch }`                                          | Invoke rule only if a terminal occludes the scope                            |
| Rule ref  | `RuleName`                                                      | Delegate to another rule                                                     |

### Split Sizes

| Prefix   | Mode     | Meaning                               |
|----------|----------|---------------------------------------|
| *(none)* | Absolute | Fixed world-unit size                 |
| `'`      | Relative | Fraction of total scope dimension     |
| `~`      | Floating | Proportional share of remaining space |

Floating slots divide the space left after all absolute and relative slots are
placed. Example: `Split(Y) { 3: Base | ~1: Mid | ~2: Top }` on a 9m scope →
Base=3m, Mid=2m, Top=4m.

### Repeat

Tiles a scope along an axis. The tile-size argument can be either a single
positive float (uniform tile) or a bracketed list (cycled pattern):

```text
Repeat(X, 2.5) { Window }            # uniform tiles
Repeat(X, [2, 1.5, 3]) { Bay }       # cycle pattern: 2, 1.5, 3, 2, 1.5, 3, …
```

The pattern is appended greedily — the next tile from the cycle is added while
it still fits — then **all** placed tiles are scaled by the same factor
`total / Σ(placed)` so the cycle fills the scope exactly with no gap and no
overshoot. A 1-element list `[t]` is identical to the legacy uniform form.

### Named Rules

```text
# Deterministic
Lot --> Extrude(10) Split(Y) { ~1: Floor | 2: Roof }

# Stochastic (weights need not sum to 100)
Facade --> 70% BrickWall | 30% GlassCurtain
```

### Comp Face Selectors

`Top`, `Bottom`, `Front`, `Back`, `Left`, `Right`, `Side` (all four sides),
`All` or `_` (catch-all). Each face scope is oriented so local **Z** points
along the outward normal.

### Align

Rotates the scope so that a local axis points in a world direction (shortest-arc
rotation). Named targets: `Up`, `Down`, `Right`, `Left`, `Forward`, `Back`.

```text
Align(Y, Up)         # local Y → world up
Align(Z, Forward)    # local Z → world forward (0,0,-1)
```

### Offset

Insets a 2D face scope, producing an `Inside` rectangle and four `Border`
strips. The distance must be negative (inset). Rejects if the inset exceeds
half the scope dimension.

```text
Offset(-0.2) { Inside: Glass | Border: Frame }
```

Selectors: `Inside`, `Border`, `All`.

### Roof

Generates roof geometry above a volume scope. Supports 15 roof types with
configurable pitch, overhang, secondary pitch, ridge offset, fascia depth, and
tier height.

```text
Roof(Gable, 30) { Slope: Tiles | GableEnd: Bricks }
Roof(Hip, 30, 0.5) { Slope: Tiles }
Roof(Gambrel, 45, 20) { LowerSlope: Shingles | UpperSlope: Tiles }
Roof(Saltbox, 45, offset=0.3) { Slope: Tiles | GableEnd: Bricks }
Roof(DutchGable, 45, tier=0.7) { Slope: Tiles | GableEnd: Bricks }
Roof(Hip, 30, fascia=0.3) { Slope: Tiles | Fascia: Board }
Roof(Mansard, 60, secondary=20, tier=0.4) { LowerSlope: Plaster | UpperSlope: Tiles }
```

Positional args:

- `Roof(<type>, <pitch>)` — pitch in degrees, open interval `(0°, 90°)`.
- `Roof(<type>, <pitch>, <overhang>)` — for all types except Gambrel/Mansard.
- `Roof(<Gambrel|Mansard>, <pitch>, <secondary_pitch>)` — Gambrel/Mansard
  consume the third positional as the secondary (shallower) pitch.

Named args (any order, comma-separated, override positionals):

- `overhang=N` — eave overhang in world units (default `0`).
- `offset=N` — ridge offset fraction for `Saltbox` (default `0.5`).
- `tier=N` — pitch-break height fraction for `Gambrel`, `Mansard`,
  `Jerkinhead`, `DutchGable` (each type uses its own default if unset).
- `fascia=N` — vertical fascia band depth below each perimeter eave.
- `secondary=N` — secondary pitch in degrees, for `Gambrel` / `Mansard`.

Roof types: `Pyramid`, `Shed`, `Gable`, `Hip`, `Flat`, `OpenGable`, `BoxGable`,
`PyramidHip`, `Butterfly`, `MShaped`, `Gambrel`, `Mansard`, `Saltbox`,
`Jerkinhead`, `DutchGable`.

Face selectors: `Slope`, `GableEnd`, `LowerSlope`, `UpperSlope`, `HipEnd`,
`ValleySlope`, `OuterSlope`, `InnerSlope`, `Fascia`, `All`.

#### Fascia

When `fascia=N` (or `RoofConfig::fascia_depth = N`) is set with `N > 0`, a
vertical fascia band of depth `N` is generated below each perimeter eave.
One fascia panel per slope panel whose lower edge sits at the perimeter eave;
the fascia faces horizontally outward. Supported on `Gable`, `Hip`, `Pyramid`,
`PyramidHip`, `Shed`, `BoxGable`, `OpenGable`, `Saltbox`, `Jerkinhead`,
`DutchGable`, `Gambrel`, `Mansard`, and the outer slopes of `MShaped`.
`Flat` and `Butterfly` produce no fascia panels (no horizontal perimeter eave).

### Polygon

Stamps an explicit polygonal `FaceProfile` on the next terminal in the rule —
the same override mechanism that `Taper(t)` uses. Vertices are 2-D `(x, y)`
points in the scope's local floor plane (XZ), measured in world units; the
renderer triangulates and extrudes along the local Y axis.

```text
Polygon((0,0), (4,0), (4,2), (2,2), (2,4), (0,4)) I("Mass")
```

Useful for L-shaped, T-shaped, or otherwise non-rectangular footprints. Capped
at 256 vertices; minimum 3.

### Snap-lines (`RegSnap` + `Split(snap=...)`)

`RegSnap("label")` records all six face planes of the current scope into the
[`ShapeModel::snap_planes`](src/model.rs) registry under the given label. A
snap-aware `Split` reads that registry and shifts its interior boundaries to
the nearest matching plane along its split axis when within tolerance.

```text
GroundLeft --> RegSnap("bays") I("LeftFloor")           # registers planes
Upper      --> Split(X, snap="bays") { 4.5: Bay | 5.5: Bay }    # snaps to bays
Upper      --> Split(X, snap="bays", tol=0.5) { … }     # explicit tolerance
```

Default `tol` is **5%** of the split axis length. The snap match requires the
plane normal to be parallel (within ~25°) to the split axis, and the plane to
fall within `tol` world units of the resolved boundary; on a hit, the
boundary moves to the plane and the two adjacent slot widths absorb the
offset (slot total is preserved).

**BFS-order caveat**: a snap-aware `Split` only sees planes registered by
rules already drained from the queue. If you want Upper bays to snap to
Ground bay edges, structure the grammar so Upper (or its ancestors) reach
their snap-aware Split *after* Ground's bay rules execute their `RegSnap`
— e.g. add one rule of indirection on Upper.

### Occlusion query (`IfClear` / `IfOccluded`)

Both ops invoke a sub-rule conditionally based on whether the current scope
overlaps any already-emitted terminal. The overlap test is a true OBB-vs-OBB
check via the Separating Axis Theorem (handles rotated and thin face scopes).

```text
Window --> IfClear { WindowMesh }    # only if no terminal occludes the scope
Patch  --> IfOccluded { PatchMesh }  # only if a terminal does occlude
```

The query is also exposed on `ShapeModel` for post-derivation use:

```rust
let q = model.query();
if q.overlaps(&candidate_scope) { /* ... */ }
for hit in q.overlapping(&candidate_scope) { /* ... */ }
```

The `IfClear` / `IfOccluded` ops consult the model **at the moment they are
processed** — same BFS-order caveat as snap-lines: structural elements must
be derived before the conditional ops that depend on them.

### Mat

Sets the material on the next terminal. The material carries an asset
identifier and an optional mass density in kg/m³. When density is supplied,
the interpreter computes [`MassProperties`](#mass-properties) for every
terminal stamped with this material.

```text
Mat("Brick")          # id only — no density, no mass properties
Mat("Brick", 1800)    # id + density (kg/m³)
```

### Mass properties

Each `Terminal` has an optional `mass_properties: Option<MassProperties>`
populated when its material has a density. `MassProperties` carries:

- `mass` (kg) — `density × volume` of the terminal's solid form.
- `centroid` (`Vec3`) — world-frame centre of mass.
- `inertia` (`Option<DMat3>`) — moment-of-inertia tensor about the centroid
  in the world frame, in kg·m². Closed-form is implemented for `Rectangle`,
  `Triangle`, `Trapezoid`, and `Polygon` profiles. `Taper(t)` populates mass
  and centroid (frustum formulas) but leaves inertia `None`.

Mass propagates correctly through `Split`, `Comp`, and `Repeat` because
`Mat(...)` is recorded on the work item before branching, then re-applied to
each child terminal. Downstream consumers (e.g. `bevy_symbios_shape`) can
read `mass_properties` directly for physics, LOD, and IK without
recomputing volumes.

### Attach

Projects a new horizontal scope onto a sloped face for attaching dormers or
surface details. The resulting scope sits on the face with its Y axis aligned to
the specified world direction.

```text
Attach(Up) { Surface: DormerMass }
```

Selectors: `Surface`, `All`.

## Rust API

### Deterministic rules

```rust
interp.add_rule("Floor", parse_ops("Repeat(X, 2.5) { Window }").unwrap());
```

### Stochastic rules

```rust
interp.add_weighted_rules("Facade", vec![
    (70.0, parse_ops("BrickWall").unwrap()),
    (30.0, parse_ops("GlassWall").unwrap()),
]).unwrap();

interp.seed = 42; // reproducible derivation
```

### Output

`derive` returns a `ShapeModel` containing a `Vec<Terminal>` and any
`SnapPlane`s registered along the way:

```rust
pub struct ShapeModel {
    pub terminals: Vec<Terminal>,
    pub snap_planes: Vec<SnapPlane>,  // populated by RegSnap("…")
}

pub struct Terminal {
    pub scope: Scope,                            // OBB: position, rotation, size
    pub mesh_id: String,                         // asset to spawn
    pub face_profile: FaceProfile,               // Rectangle, Taper, Triangle, Trapezoid, Polygon
    pub material: Option<Material>,              // id + optional density
    pub mass_properties: Option<MassProperties>, // populated when material has density
}

pub struct Material {
    pub id: String,
    pub density: Option<f64>,  // kg/m³ — drives MassProperties when Some
}

pub struct MassProperties {
    pub mass: f64,                       // kg (world-frame)
    pub centroid: Vec3,                  // world-frame centre of mass
    pub inertia: Option<glam::DMat3>,    // about centroid, kg·m² (None for Taper profile)
}
```

`FaceProfile` describes each terminal's 2D cross-section for mesh generation:

| Variant                             | Description                                  |
|-------------------------------------|----------------------------------------------|
| `Rectangle`                         | Full rectangular face (default)              |
| `Taper(t)`                          | Legacy tapered prism (0 = box, 1 = pyramid)  |
| `Triangle { peak_offset }`          | Triangular face (0.5 = symmetric gable)      |
| `Trapezoid { top_width, offset_x }` | Trapezoidal face (hip roof slopes)           |
| `Polygon(Vec<DVec2>)`               | Arbitrary polygon (straight skeleton output) |

## Safety Limits

The engine enforces hard caps to prevent runaway grammars:

| Limit                          | Default  |
|--------------------------------|----------|
| Max derivation depth           | 64       |
| Max work queue size            | 100 000  |
| Max terminals                  | 100 000  |
| Max ops per rule               | 1 024    |
| Max split slots                | 256      |
| Max comp cases                 | 32       |
| Max rule variants (stochastic) | 64       |
| Max identifier length          | 64 chars |

Exceeding a limit returns `Err(ShapeError::CapacityOverflow)` or
`Err(ShapeError::DepthLimitExceeded)` rather than panicking.

## Genetic Evolution

The `genetics` module provides `ShapeGenotype`, a wrapper around the rule table
that implements `symbios_genetics::Genotype`. Plug it into any
`symbios-genetics` algorithm (SimpleGA, NSGA-II, MAP-Elites) to evolve
procedural building grammars.

```rust
use symbios_shape::genetics::ShapeGenotype;
use symbios_genetics::Genotype;
use rand::SeedableRng;
use rand_pcg::Pcg64;

let dna = ShapeGenotype::from_interpreter(&interp);
let mut rng = Pcg64::seed_from_u64(42);

let mut child = dna.clone();
child.mutate(&mut rng, 0.3);           // Gaussian jitter on parametric floats
let offspring = dna.crossover(&child, &mut rng);  // BLX-α blending
```

## Architecture

- **OBB scopes** — all geometry is Oriented Bounding Boxes; no mesh boolean ops
- **BFS queue** — breadth-first expansion; depth is tracked per work item.
  Both `RegSnap` / `Split(snap=...)` and `IfClear` / `IfOccluded` consult
  the derivation state at the moment they fire, so structural rules must
  drain from the queue before any rule that depends on them
- **Pure derivation** — the engine produces a `ShapeModel` (a flat list of
  terminals plus the snap planes recorded during the run); rendering is the
  caller's responsibility
- **Spatial query view** — `model.query()` returns a `TerminalQuery` for OBB
  overlap tests against the emitted terminals; `obb_overlap(&a, &b)` is also
  exposed as a free function for ad-hoc checks
- **Serde support** — `Scope`, `ShapeOp`, `ShapeModel`, and all sub-types
  implement `Serialize`/`Deserialize`

## License

MIT

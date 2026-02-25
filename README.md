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
  `Scale`, `Translate`, `Rotate`, `I`, `Mat`
- **Three split modes** — absolute (`3:`), relative (`'0.3:`), floating (`~1:`)
- **Stochastic rules** — `70% BrickFacade | 30% GlassFacade` with reproducible
  seeded RNG
- **Implicit terminals** — unknown rule names resolve to `I(rule_name)` without
  an explicit `I(...)` op
- **Material propagation** — `Mat("Brick")` stamps a material that flows through
  all branching ops to the final `Terminal`
- **Lightweight** — depends only on `glam` (math), `nom` (parsing), and `rand`
  (stochastic rules); no engine or runtime required
- **DoS-hardened** — bounded queue, depth, terminal, and identifier limits

## Installation

```toml
[dependencies]
symbios-shape = "0.1"
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

| Op | Syntax | Description |
|----|--------|-------------|
| Extrude | `Extrude(h)` | Set Y size to `h` (h > 0) |
| Taper | `Taper(t)` | Pyramidal taper, t ∈ [0, 1] |
| Scale | `Scale(x, y, z)` | Multiply scope size (all > 0) |
| Translate | `Translate(x, y, z)` | Shift origin in local space |
| Rotate | `Rotate(w, x, y, z)` | Apply quaternion rotation |
| Split | `Split(Y) { 3: A \| ~1: B \| '0.2: C }` | Divide along axis |
| Repeat | `Repeat(X, 2.5) { Window }` | Tile along axis, stretch to fill |
| Comp | `Comp(Faces) { Top: R \| Side: R \| Bottom: R }` | Decompose into face scopes |
| I | `I("MeshId")` or `I(MeshId)` | Emit terminal |
| Mat | `Mat("Brick")` or `Mat(Brick)` | Set material (propagates to terminal) |
| Rule ref | `RuleName` | Delegate to another rule |

### Split Sizes

| Prefix | Mode | Meaning |
|--------|------|---------|
| *(none)* | Absolute | Fixed world-unit size |
| `'` | Relative | Fraction of total scope dimension |
| `~` | Floating | Proportional share of remaining space |

Floating slots divide the space left after all absolute and relative slots are
placed. Example: `Split(Y) { 3: Base | ~1: Mid | ~2: Top }` on a 9m scope →
Base=3m, Mid=2m, Top=4m.

### Named Rules

```text
# Deterministic
Lot --> Extrude(10) Split(Y) { ~1: Floor | 2: Roof }

# Stochastic (weights need not sum to 100)
Facade --> 70% BrickWall | 30% GlassCurtain
```

### Comp Face Selectors

`Top`, `Bottom`, `Front`, `Back`, `Left`, `Right`, `Side` (all four sides),
`All` (catch-all). Each face scope is oriented so local **Z** points along the
outward normal.

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

`derive` returns a `ShapeModel` containing a `Vec<Terminal>`:

```rust
pub struct Terminal {
    pub scope: Scope,       // position, rotation, size (OBB)
    pub mesh_id: String,    // asset to spawn
    pub taper: f64,         // 0 = box, 1 = pyramid
    pub material: Option<String>,
}
```

## Safety Limits

The engine enforces hard caps to prevent runaway grammars:

| Limit | Default |
|-------|---------|
| Max derivation depth | 64 |
| Max work queue size | 100 000 |
| Max terminals | 100 000 |
| Max ops per rule | 1 024 |
| Max split slots | 256 |
| Max comp cases | 32 |
| Max rule variants (stochastic) | 64 |
| Max identifier length | 64 chars |

Exceeding a limit returns `Err(ShapeError::CapacityOverflow)` or
`Err(ShapeError::DepthLimitExceeded)` rather than panicking.

## Architecture

- **OBB scopes** — all geometry is Oriented Bounding Boxes; no mesh boolean ops
- **BFS queue** — breadth-first expansion; depth is tracked per work item
- **Pure derivation** — the engine produces a `ShapeModel` (a flat list of
  terminals); rendering is the caller's responsibility
- **Serde support** — `Scope`, `ShapeOp`, `ShapeModel`, and all sub-types
  implement `Serialize`/`Deserialize`

## License

MIT

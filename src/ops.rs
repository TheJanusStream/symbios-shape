use serde::{Deserialize, Serialize};

use crate::scope::{Quat, Vec3};

/// The axis along which a `Split` or `Repeat` operation acts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Axis {
    X,
    Y,
    Z,
}

/// Sizing mode for a single slot within a `Split` operation.
///
/// Mirrors CityEngine CGA syntax:
/// - `Absolute(n)`: a fixed world-unit size.
/// - `Relative(t)`: a fraction `t` of the scope's total dimension (prefix `'` in CGA text).
/// - `Floating(n)`: a weight that shares the remaining space after absolutes are consumed
///   (prefix `~` in CGA text). Multiple floating slots divide the remainder proportionally.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum SplitSize {
    Absolute(f64),
    Relative(f64),
    Floating(f64),
}

impl SplitSize {
    pub fn is_valid(&self) -> bool {
        match self {
            SplitSize::Absolute(v) | SplitSize::Relative(v) | SplitSize::Floating(v) => {
                v.is_finite() && *v > 0.0
            }
        }
    }
}

/// A single slot in a `Split` operation: a size mode paired with a successor rule name.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SplitSlot {
    pub size: SplitSize,
    /// The name of the shape rule to invoke on the resulting child scope.
    pub rule: String,
}

/// Face selectors for the `Comp(Faces)` decomposition.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum FaceSelector {
    Top,
    Bottom,
    Front,
    Back,
    Left,
    Right,
    /// Matches all non-top, non-bottom faces (shorthand for the four sides).
    Side,
    /// Matches all faces not otherwise mapped.
    All,
}

impl FaceSelector {
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "top" | "Top" => Some(Self::Top),
            "bottom" | "Bottom" => Some(Self::Bottom),
            "front" | "Front" => Some(Self::Front),
            "back" | "Back" => Some(Self::Back),
            "left" | "Left" => Some(Self::Left),
            "right" | "Right" => Some(Self::Right),
            "side" | "Side" => Some(Self::Side),
            "all" | "All" | "_" => Some(Self::All),
            _ => None,
        }
    }
}

/// A single mapping in a `Comp(Faces)` block: a face selector → rule name.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CompFaceCase {
    pub selector: FaceSelector,
    pub rule: String,
}

/// The decomposition target for a `Comp` operation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum CompTarget {
    /// Decomposes the volume into its six axis-aligned face scopes.
    Faces(Vec<CompFaceCase>),
}

/// Face selectors for the `Offset` operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum OffsetSelector {
    /// The inset/outset region (the area inside the border).
    Inside,
    /// The surrounding border strips.
    Border,
    /// Matches any selector not otherwise mapped.
    All,
}

impl OffsetSelector {
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "inside" | "Inside" => Some(Self::Inside),
            "border" | "Border" => Some(Self::Border),
            "all" | "All" | "_" => Some(Self::All),
            _ => None,
        }
    }
}

/// A single mapping in an `Offset` block: a selector → rule name.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OffsetCase {
    pub selector: OffsetSelector,
    pub rule: String,
}

/// Roof shape types for the `Roof` operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RoofType {
    // ── Original types ────────────────────────────────────────────────────────
    /// Four triangular slope panels meeting at a single apex.
    Pyramid,
    /// Single slope panel from front eave to back eave.
    Shed,
    /// Two slope panels meeting at a horizontal ridge; two triangular gable ends.
    Gable,
    /// Four trapezoidal slope panels meeting at a horizontal ridge.
    Hip,

    // ── New types ─────────────────────────────────────────────────────────────
    /// Flat horizontal roof — a single horizontal panel covering the scope top.
    Flat,
    /// Two rectangular slope panels only (Gable without the triangular end panels).
    OpenGable,
    /// Two slope panels + two rectangular (non-tapered) gable-end wall panels.
    BoxGable,
    /// Four panels from a rectangular base meeting at a single apex point (no ridge).
    /// Equivalent to `Pyramid` for square footprints; left/right end panels are triangular.
    PyramidHip,
    /// Two inward-tilting slopes forming a central valley (inverted Gable).
    Butterfly,
    /// Four panels forming two parallel ridges with a central valley between them (M profile).
    MShaped,
    /// Two pitches per slope: steeper lower zone + shallower upper zone (barn roof).
    /// Requires `secondary_pitch` in `RoofConfig`.
    Gambrel,
    /// Gambrel applied to all four sides: 4 steep lower panels + 4 shallow upper panels.
    /// Requires `secondary_pitch` in `RoofConfig`.
    Mansard,
    /// Asymmetric Gable: the ridge is offset toward one end (`ridge_offset` in `RoofConfig`).
    /// Front slope is steeper; back slope is shallower. Gable ends are asymmetric triangles.
    Saltbox,
    /// Gable with clipped hip ends: the upper corners of each gable end are replaced by
    /// small triangular hip panels. Controlled by `tier_height` in `RoofConfig`.
    Jerkinhead,
    /// Hip roof with a small gable rising from the ridge centre.
    /// Controlled by `tier_height` (fraction of slope from base where the gable starts).
    DutchGable,
}

impl RoofType {
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "pyramid" | "Pyramid" => Some(Self::Pyramid),
            "shed" | "Shed" => Some(Self::Shed),
            "gable" | "Gable" => Some(Self::Gable),
            "hip" | "Hip" => Some(Self::Hip),
            "flat" | "Flat" => Some(Self::Flat),
            "openGable" | "OpenGable" => Some(Self::OpenGable),
            "boxGable" | "BoxGable" => Some(Self::BoxGable),
            "pyramidHip" | "PyramidHip" => Some(Self::PyramidHip),
            "butterfly" | "Butterfly" => Some(Self::Butterfly),
            "mShaped" | "MShaped" => Some(Self::MShaped),
            "gambrel" | "Gambrel" => Some(Self::Gambrel),
            "mansard" | "Mansard" => Some(Self::Mansard),
            "saltbox" | "Saltbox" => Some(Self::Saltbox),
            "jerkinhead" | "Jerkinhead" => Some(Self::Jerkinhead),
            "dutchGable" | "DutchGable" => Some(Self::DutchGable),
            _ => None,
        }
    }
}

/// Face selectors for the `Roof` operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RoofFaceSelector {
    /// The main sloped panel(s) — front/back in most roof types.
    Slope,
    /// The triangular vertical end panels of a Gable or Saltbox roof.
    GableEnd,
    /// The steeper, lower zone of a Gambrel or Mansard roof.
    LowerSlope,
    /// The shallower, upper zone of a Gambrel or Mansard roof.
    UpperSlope,
    /// The small triangular hip panels at the clipped ends of a Jerkinhead roof.
    HipEnd,
    /// The inward-facing slopes of a Butterfly or MShaped valley.
    ValleySlope,
    /// The outer slopes of an MShaped roof (facing away from the valley).
    OuterSlope,
    /// The inner slopes of an MShaped roof (facing toward the valley).
    InnerSlope,
    /// Matches any selector not otherwise mapped.
    All,
}

impl RoofFaceSelector {
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "slope" | "Slope" => Some(Self::Slope),
            "gable" | "GableEnd" | "gableEnd" => Some(Self::GableEnd),
            "lowerSlope" | "LowerSlope" => Some(Self::LowerSlope),
            "upperSlope" | "UpperSlope" => Some(Self::UpperSlope),
            "hipEnd" | "HipEnd" => Some(Self::HipEnd),
            "valleySlope" | "ValleySlope" => Some(Self::ValleySlope),
            "outerSlope" | "OuterSlope" => Some(Self::OuterSlope),
            "innerSlope" | "InnerSlope" => Some(Self::InnerSlope),
            "all" | "All" | "_" => Some(Self::All),
            _ => None,
        }
    }
}

/// A single mapping in a `Roof` block: a face selector → rule name.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RoofCase {
    pub selector: RoofFaceSelector,
    pub rule: String,
}

/// Rich parametric configuration for the `Roof` operation.
///
/// All angular values are in degrees. Lengths are in world units.
/// Optional fields default as described; see each field doc.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RoofConfig {
    pub roof_type: RoofType,
    /// Primary pitch angle in degrees from horizontal. Must be in (0°, 90°).
    pub pitch: f64,
    /// Secondary pitch angle in degrees. Used by `Gambrel` (upper zone) and `Mansard`.
    /// If `None` when required, defaults to `pitch / 2`.
    pub secondary_pitch: Option<f64>,
    /// Extra overhang beyond the scope footprint on each side. Default `0.0`.
    pub overhang: f64,
    /// Ridge offset for `Saltbox`: fraction [0, 1] of the scope depth (Z) where the
    /// ridge is positioned from the front. Default `0.5` (symmetric / centred ridge).
    pub ridge_offset: f64,
    /// Thickness of the roof fascia edge in world units. Default `0.0` (flat panels).
    pub fascia_depth: f64,
    /// Normalised height at which the pitch break occurs for `Gambrel`, `Mansard`,
    /// `Jerkinhead`, and `DutchGable`. `0.5` means the break is at half the eave-to-ridge
    /// distance. `None` uses a type-specific default.
    pub tier_height: Option<f64>,
}

impl RoofConfig {
    /// Creates a minimal config for deterministic types (Pyramid, Shed, Gable, Hip, Flat, …).
    pub fn new(roof_type: RoofType, pitch: f64) -> Self {
        Self {
            roof_type,
            pitch,
            secondary_pitch: None,
            overhang: 0.0,
            ridge_offset: 0.5,
            fascia_depth: 0.0,
            tier_height: None,
        }
    }

    /// Returns the secondary pitch, defaulting to `pitch / 2` if unset.
    pub fn secondary_pitch_or_default(&self) -> f64 {
        self.secondary_pitch.unwrap_or(self.pitch / 2.0)
    }

    /// Returns the tier height, defaulting to `default` if unset.
    pub fn tier_height_or(&self, default: f64) -> f64 {
        self.tier_height.unwrap_or(default)
    }
}

/// Selector for the `Attach` operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AttachSelector {
    /// The projected scope that sits on (or comes out of) the surface.
    Surface,
    /// Matches any selector not otherwise mapped.
    All,
}

impl AttachSelector {
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "surface" | "Surface" => Some(Self::Surface),
            "all" | "All" | "_" => Some(Self::All),
            _ => None,
        }
    }
}

/// A single mapping in an `Attach` block: a selector → rule name.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AttachCase {
    pub selector: AttachSelector,
    pub rule: String,
}

/// The atomic CGA operations that the interpreter executes.
///
/// Every operation transforms the current `Scope` into zero or more child scopes,
/// each tagged with a rule name that will be recursively evaluated.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ShapeOp {
    /// Lifts a 2-D footprint (XZ plane) into a 3-D volume by setting the Y size.
    Extrude(f64),

    /// Pyramidal taper: scales the top face toward the centroid.
    /// `amount` ∈ `[0, 1]`: 0 = no taper, 1 = full pyramid (top collapses to a point).
    Taper(f64),

    /// Applies an additional rotation to the scope (cumulative with existing rotation).
    Rotate(Quat),

    /// Translates the scope origin in local space.
    Translate(Vec3),

    /// Scales the scope size along each axis (multiplicative).
    Scale(Vec3),

    /// Divides the scope along `axis` into ordered slots.
    Split { axis: Axis, slots: Vec<SplitSlot> },

    /// Tiles the scope along `axis` with child scopes of approximate size `tile_size`.
    /// The tile count is `floor(total / tile_size)`; the actual tile size is then
    /// stretched to `total / count` so the tiles fill the scope exactly with no gap.
    Repeat {
        axis: Axis,
        tile_size: f64,
        rule: String,
    },

    /// Decomposes the scope into its geometric components (faces, edges, vertices).
    Comp(CompTarget),

    /// Terminal: replace the scope with the named mesh asset.
    /// This is the "terminal symbol" — produces a `Terminal` node in the output model.
    I(String),

    /// Sets the material identifier on the current work item.
    /// The material is propagated to the final `Terminal`, allowing downstream
    /// renderers to apply textures/shaders without changing the scope.
    /// Syntax: `Mat("Brick")` or `Mat(Brick)`.
    Mat(String),

    /// Calls a named sub-rule on the current scope unchanged.
    /// Used for grammar rule references that don't transform the scope themselves.
    Rule(String),

    /// Rotates the scope so that the specified local axis points in the given world direction.
    ///
    /// Applies the shortest-arc rotation from the current world direction of `local_axis`
    /// to `target`. Useful for recovering from accumulated rotations.
    /// Syntax: `Align(Y, Up)`, `Align(Z, Forward)`, etc.
    /// Named targets: `Up`=(0,1,0), `Down`=(0,-1,0), `Right`=(1,0,0), `Left`=(-1,0,0),
    /// `Forward`=(0,0,-1), `Back`=(0,0,1).
    Align { local_axis: Axis, target: Vec3 },

    /// Creates an inset (`distance < 0`) frame on a 2D face scope.
    ///
    /// Produces up to two kinds of child scopes:
    /// - `Inside`: the inset rectangle.
    /// - `Border`: four surrounding strips (bottom, top, left, right), each invoking the same rule.
    ///
    /// Syntax: `Offset(-0.2) { Inside: Glass | Border: Frame }`
    Offset {
        distance: f64,
        cases: Vec<OffsetCase>,
    },

    /// Generates a roof structure above the current scope using rich parametric configuration.
    ///
    /// Operates on a volume scope. The `config` contains the roof type, primary pitch angle,
    /// optional secondary pitch, overhang, ridge offset, fascia depth, and tier height.
    ///
    /// Syntax examples:
    /// - `Roof(Gable, 30) { Slope: Tiles | GableEnd: Bricks }` — basic Gable
    /// - `Roof(Hip, 30, 0.5) { Slope: Tiles }` — Hip with overhang
    /// - `Roof(Gambrel, 45, 20) { LowerSlope: Shingles | UpperSlope: Tiles }` — Gambrel
    /// - `Roof(Saltbox, 45, offset=0.3) { Slope: Tiles | GableEnd: Bricks }` — Saltbox
    /// - `Roof(DutchGable, 45, tier=0.7) { Slope: Tiles | GableEnd: Bricks }` — Dutch Gable
    Roof {
        config: RoofConfig,
        cases: Vec<RoofCase>,
    },

    /// Projects a new horizontal scope out of a sloped face for attaching dormers or details.
    ///
    /// `world_axis` defines the "up" direction for the attached scope (usually world Y).
    /// The resulting scope sits on the face's surface with its Y axis aligned to `world_axis`,
    /// inheriting the face's width and height but with depth = 0.
    ///
    /// Syntax: `Attach(Up) { Surface: DormerMass }`
    Attach {
        world_axis: Vec3,
        cases: Vec<AttachCase>,
    },
}

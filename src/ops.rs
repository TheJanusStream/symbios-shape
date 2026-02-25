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
}

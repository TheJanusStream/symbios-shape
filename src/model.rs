use serde::{Deserialize, Serialize};

use crate::scope::Scope;

/// The 2D profile shape of a terminal face, used by renderers to construct geometry.
///
/// Replaces the old `taper: f64` field. Each variant describes the cross-sectional
/// outline of the face panel within the scope's local XY plane:
/// - X runs from 0 (left edge) to `scope.size.x` (right edge).
/// - Y runs from 0 (bottom edge) to `scope.size.y` (top edge).
///
/// The renderer extrudes this 2D outline along the local Z axis by `scope.size.z`
/// (which is 0 for flat face panels produced by `Roof` and `Comp(Faces)`).
///
/// Coordinate convention for `Trapezoid` and `Triangle`: values are **normalized**
/// (0.0 = left/bottom, 1.0 = right/top) so they are independent of the actual scope size.
/// The renderer scales them by `scope.size.x` before vertex generation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum FaceProfile {
    /// Full rectangular face — the default for walls, floors, and generic volumes.
    Rectangle,

    /// Legacy tapered prism, equivalent to the old `taper` field.
    ///
    /// `t` ∈ `[0, 1]`: 0 = box, 1 = full pyramid. The renderer maps this
    /// to the existing `build_tapered_cuboid` path for backward compatibility.
    Taper(f64),

    /// Triangular face: base at Y = 0 (full scope width), apex at Y = scope.size.y.
    ///
    /// `peak_offset` ∈ `[0, 1]`: horizontal offset of the apex from the left edge,
    /// normalized to the scope width. `0.5` = symmetric triangle (standard gable).
    /// `0.3` = apex shifted left (asymmetric, used by Saltbox gable ends).
    Triangle { peak_offset: f64 },

    /// Trapezoidal face: rectangular base at Y = 0, narrower top edge at Y = scope.size.y.
    ///
    /// Both values are normalized to the scope width (`scope.size.x = 1.0`):
    /// - `top_width` ∈ `[0, 1]`: width of the top edge as a fraction of the base width.
    /// - `offset_x` ∈ `[0, 1]`: left indent of the top edge from the scope left.
    ///
    /// Invariant: `offset_x + top_width ≤ 1.0`.
    /// A symmetric trapezoid has `offset_x = (1 - top_width) / 2`.
    Trapezoid { top_width: f64, offset_x: f64 },

    /// Arbitrary convex or concave polygon, produced by the straight skeleton algorithm
    /// for complex (L-shaped, T-shaped, etc.) building footprints.
    ///
    /// Vertices are in the scope's local XZ (floor) plane, measured in world units
    /// from the scope origin. The renderer triangulates this polygon and extrudes it
    /// along the local Y axis by the roof pitch height.
    Polygon(Vec<glam::DVec2>),
}

impl FaceProfile {
    /// Returns `true` if the profile is the default `Rectangle` shape.
    pub fn is_rectangle(&self) -> bool {
        matches!(self, Self::Rectangle)
    }

    /// Returns the legacy taper coefficient if this profile was set by `ShapeOp::Taper`.
    pub fn taper_coeff(&self) -> Option<f64> {
        match self {
            Self::Taper(t) => Some(*t),
            Self::Rectangle => Some(0.0),
            _ => None,
        }
    }
}

/// A fully-resolved terminal node in the shape model.
///
/// Represents a concrete mesh instance placed at the given `scope`.
/// The `mesh_id` identifies which asset to spawn (e.g. `"Window"`, `"Door"`, `"Pillar"`).
/// `face_profile` describes the 2D cross-section shape of this terminal (replaces the old `taper`).
/// `material`: optional material/texture identifier set by `Mat(...)` operations.
/// This is the "DOM" that Bevy (or any renderer) reads to spawn entities.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Terminal {
    pub scope: Scope,
    pub mesh_id: String,
    /// 2D face profile describing the cross-sectional shape of this terminal.
    /// Replaces the old `taper: f64` field. Use [`FaceProfile::taper_coeff`] to
    /// obtain a legacy taper coefficient for backward-compatible renderers.
    pub face_profile: FaceProfile,
    /// Optional material identifier stamped by a `Mat("...")` operation.
    pub material: Option<String>,
}

impl Terminal {
    pub fn new(scope: Scope, mesh_id: impl Into<String>) -> Self {
        Self {
            scope,
            mesh_id: mesh_id.into(),
            face_profile: FaceProfile::Rectangle,
            material: None,
        }
    }

    /// Creates a terminal with a legacy taper factor (0 = box, 1 = pyramid).
    /// Converts the taper into a `FaceProfile` automatically.
    pub fn new_with_taper(scope: Scope, mesh_id: impl Into<String>, taper: f64) -> Self {
        let face_profile = taper_to_profile(taper);
        Self {
            scope,
            mesh_id: mesh_id.into(),
            face_profile,
            material: None,
        }
    }

    pub fn new_full(
        scope: Scope,
        mesh_id: impl Into<String>,
        taper: f64,
        material: Option<String>,
    ) -> Self {
        Self {
            scope,
            mesh_id: mesh_id.into(),
            face_profile: taper_to_profile(taper),
            material,
        }
    }

    /// Creates a terminal with an explicit `FaceProfile`.
    pub fn new_profiled(
        scope: Scope,
        mesh_id: impl Into<String>,
        face_profile: FaceProfile,
        material: Option<String>,
    ) -> Self {
        Self {
            scope,
            mesh_id: mesh_id.into(),
            face_profile,
            material,
        }
    }
}

/// Converts a legacy taper coefficient to the closest `FaceProfile`.
pub fn taper_to_profile(taper: f64) -> FaceProfile {
    if taper <= 0.0 {
        FaceProfile::Rectangle
    } else if (taper - 1.0).abs() < 1e-9 {
        FaceProfile::Triangle { peak_offset: 0.5 }
    } else {
        FaceProfile::Taper(taper.clamp(0.0, 1.0))
    }
}

/// The output of a shape grammar derivation.
///
/// Contains all terminal nodes produced by the grammar, ready for rendering.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ShapeModel {
    pub terminals: Vec<Terminal>,
}

impl ShapeModel {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn push(&mut self, terminal: Terminal) {
        self.terminals.push(terminal);
    }

    pub fn len(&self) -> usize {
        self.terminals.len()
    }

    pub fn is_empty(&self) -> bool {
        self.terminals.is_empty()
    }
}

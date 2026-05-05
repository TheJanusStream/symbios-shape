use serde::{Deserialize, Serialize};

use crate::scope::{Scope, Vec3};

/// Material stamped on a [`Terminal`] by a `Mat("...")` grammar op.
///
/// Carries an asset identifier (used by downstream renderers to look up
/// textures / shaders) and an optional mass density in kg/m³ used to
/// derive [`MassProperties`] for physics, LOD, and IK pipelines.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Material {
    /// Identifier the renderer uses to resolve textures / shaders.
    pub id: String,
    /// Mass density in kg/m³. When `Some`, the interpreter computes
    /// [`MassProperties`] for every terminal stamped with this material.
    /// When `None`, the terminal's [`Terminal::mass_properties`] stays `None`.
    pub density: Option<f64>,
}

impl Material {
    /// Creates a material with no density (purely a renderer-side identifier).
    pub fn new(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            density: None,
        }
    }

    /// Creates a material with a mass density in kg/m³.
    pub fn with_density(id: impl Into<String>, density: f64) -> Self {
        Self {
            id: id.into(),
            density: Some(density),
        }
    }
}

/// A snap-plane recorded during derivation.
///
/// Snap planes are emitted by the `RegSnap("label")` op (which records all six
/// face planes of the current scope) and consumed by the snap-aware variant
/// of `Split` (`Split(axis, snap="label") { … }`). They give downstream
/// elements a way to align to landmarks established earlier in the derivation
/// — e.g. ground-floor bay edges become snap planes that upper-floor bays
/// align to.
///
/// The plane is defined in world-space by `point` (a point on the plane) and
/// `normal` (a unit-length plane normal). `label` groups planes for selective
/// querying.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SnapPlane {
    pub point: Vec3,
    pub normal: Vec3,
    pub label: String,
}

/// Volumetric mass properties of a terminal.
///
/// Populated by the interpreter when the terminal's [`Material`] carries a
/// density. Downstream consumers (Bevy, IK solvers, LOD selectors) read these
/// directly rather than recomputing from `scope` + `face_profile`.
///
/// All quantities are in **world-frame** SI units: `mass` in kg, `centroid` in
/// metres from the world origin, `inertia` is the moment-of-inertia tensor
/// about the centroid expressed as a 3×3 matrix in kg·m².
///
/// `inertia` is `None` for face profiles whose closed-form tensor is not
/// implemented (currently only [`FaceProfile::Taper`]); `mass` and `centroid`
/// are always populated whenever a density is available.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MassProperties {
    pub mass: f64,
    pub centroid: Vec3,
    pub inertia: Option<glam::DMat3>,
}

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
/// `face_profile` describes the 2D cross-section shape of this terminal.
/// `material`: optional [`Material`] (id + density) set by `Mat(...)` operations.
/// `mass_properties`: derived volumetric properties when the material carries a density.
/// This is the "DOM" that Bevy (or any renderer) reads to spawn entities.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Terminal {
    pub scope: Scope,
    pub mesh_id: String,
    /// 2D face profile describing the cross-sectional shape of this terminal.
    /// Replaces the old `taper: f64` field. Use [`FaceProfile::taper_coeff`] to
    /// obtain a legacy taper coefficient for backward-compatible renderers.
    pub face_profile: FaceProfile,
    /// Optional material stamped by a `Mat("...", density?)` operation.
    pub material: Option<Material>,
    /// Volumetric mass properties (mass / centroid / inertia tensor), populated
    /// when `material` carries a `density`. Always `None` otherwise.
    pub mass_properties: Option<MassProperties>,
}

impl Terminal {
    pub fn new(scope: Scope, mesh_id: impl Into<String>) -> Self {
        Self {
            scope,
            mesh_id: mesh_id.into(),
            face_profile: FaceProfile::Rectangle,
            material: None,
            mass_properties: None,
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
            mass_properties: None,
        }
    }

    /// Creates a terminal with an explicit [`FaceProfile`] and optional material.
    /// `mass_properties` is automatically computed when the material has a density.
    pub fn new_profiled(
        scope: Scope,
        mesh_id: impl Into<String>,
        face_profile: FaceProfile,
        material: Option<Material>,
    ) -> Self {
        let mass_properties = material
            .as_ref()
            .and_then(|m| m.density)
            .and_then(|rho| compute_mass_properties(&scope, &face_profile, rho));
        Self {
            scope,
            mesh_id: mesh_id.into(),
            face_profile,
            material,
            mass_properties,
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
/// Contains all terminal nodes produced by the grammar plus any snap planes
/// recorded along the way (via `RegSnap`).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ShapeModel {
    pub terminals: Vec<Terminal>,
    /// Snap planes accumulated during derivation. See [`SnapPlane`] and the
    /// `RegSnap` / `Split(snap=...)` grammar ops.
    pub snap_planes: Vec<SnapPlane>,
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

// ── Mass properties ──────────────────────────────────────────────────────────
//
// Closed-form mass / centroid / inertia computations for each [`FaceProfile`].
// All output quantities are in the world frame (post-transform by `scope`).
//
// Profile interpretations (matching the renderer convention):
//   - `Rectangle`         — solid box of size (sx, sy, sz).
//   - `Taper(t)`          — frustum tapering along local Y; top scaled by (1−t).
//                           Mass + centroid only; inertia = `None`.
//   - `Triangle`          — vertical XY face panel extruded along Z.
//   - `Trapezoid`         — vertical XY face panel extruded along Z.
//   - `Polygon(verts)`    — horizontal XZ footprint extruded along Y.
//
// Returns `None` when the geometry is degenerate (zero volume / collinear poly /
// non-finite values) so the caller can leave [`Terminal::mass_properties`]
// empty rather than producing NaN-laden tensors.

/// Computes [`MassProperties`] for a terminal given its scope, face profile,
/// and material density. Returns `None` if the geometry is degenerate.
pub fn compute_mass_properties(
    scope: &Scope,
    profile: &FaceProfile,
    density: f64,
) -> Option<MassProperties> {
    if !density.is_finite() || density <= 0.0 {
        return None;
    }
    if !scope.size.is_finite() {
        return None;
    }
    match profile {
        FaceProfile::Rectangle => box_mass_properties(scope, density),
        FaceProfile::Taper(t) => taper_mass_properties(scope, *t, density),
        FaceProfile::Triangle { peak_offset } => {
            let sx = scope.size.x;
            let sy = scope.size.y;
            let verts = vec![
                glam::DVec2::new(0.0, 0.0),
                glam::DVec2::new(sx, 0.0),
                glam::DVec2::new(peak_offset.clamp(0.0, 1.0) * sx, sy),
            ];
            prism_mass_properties_xy(&verts, scope, density)
        }
        FaceProfile::Trapezoid {
            top_width,
            offset_x,
        } => {
            let sx = scope.size.x;
            let sy = scope.size.y;
            let tw = top_width.clamp(0.0, 1.0);
            let ox = offset_x.clamp(0.0, 1.0);
            let verts = vec![
                glam::DVec2::new(0.0, 0.0),
                glam::DVec2::new(sx, 0.0),
                glam::DVec2::new((ox + tw) * sx, sy),
                glam::DVec2::new(ox * sx, sy),
            ];
            prism_mass_properties_xy(&verts, scope, density)
        }
        FaceProfile::Polygon(verts) => prism_mass_properties_xz(verts, scope, density),
    }
}

/// Solid uniform-density box. Inertia tensor is the standard cuboid formula
/// transformed into world space via R · I_local · Rᵀ.
fn box_mass_properties(scope: &Scope, density: f64) -> Option<MassProperties> {
    let sx = scope.size.x;
    let sy = scope.size.y;
    let sz = scope.size.z;
    if sx <= 0.0 || sy <= 0.0 || sz <= 0.0 {
        return None;
    }
    let volume = sx * sy * sz;
    if !volume.is_finite() {
        return None;
    }
    let mass = density * volume;
    if !mass.is_finite() {
        return None;
    }
    let local_centroid = Vec3::new(sx * 0.5, sy * 0.5, sz * 0.5);
    let centroid = scope.position + scope.rotation * local_centroid;
    if !centroid.is_finite() {
        return None;
    }
    let i_xx = mass * (sy * sy + sz * sz) / 12.0;
    let i_yy = mass * (sx * sx + sz * sz) / 12.0;
    let i_zz = mass * (sx * sx + sy * sy) / 12.0;
    let i_local = glam::DMat3::from_diagonal(glam::DVec3::new(i_xx, i_yy, i_zz));
    let inertia = rotate_inertia(i_local, scope.rotation);
    Some(MassProperties {
        mass,
        centroid,
        inertia: Some(inertia),
    })
}

/// Tapered cuboid (frustum) along local Y. Top face scaled by `(1 − t)`.
/// Mass + centroid only; inertia tensor is left `None`.
fn taper_mass_properties(scope: &Scope, t: f64, density: f64) -> Option<MassProperties> {
    let sx = scope.size.x;
    let sy = scope.size.y;
    let sz = scope.size.z;
    if sx <= 0.0 || sy <= 0.0 || sz <= 0.0 {
        return None;
    }
    let r = (1.0 - t.clamp(0.0, 1.0)).max(0.0);
    // Frustum volume V = sy/3 · (A_bottom + A_top + √(A_bottom·A_top))
    //                  = sx·sy·sz/3 · (1 + r² + r)
    let volume = sx * sy * sz * (1.0 + r * r + r) / 3.0;
    if !volume.is_finite() || volume <= 0.0 {
        return None;
    }
    let mass = density * volume;
    if !mass.is_finite() {
        return None;
    }
    // Frustum centroid Y above base = h/4 · (1 + 2r + 3r²) / (1 + r + r²).
    // For r = 1 (box) → sy/2; for r = 0 (full pyramid) → sy/4.
    let denom = 1.0 + r + r * r;
    let cy = if denom > 1e-12 {
        sy * 0.25 * (1.0 + 2.0 * r + 3.0 * r * r) / denom
    } else {
        sy * 0.25
    };
    let local_centroid = Vec3::new(sx * 0.5, cy, sz * 0.5);
    let centroid = scope.position + scope.rotation * local_centroid;
    if !centroid.is_finite() {
        return None;
    }
    Some(MassProperties {
        mass,
        centroid,
        inertia: None,
    })
}

/// Prism whose 2-D cross-section is `verts` in the local XY plane, extruded
/// along local Z by `scope.size.z`. Used by `Triangle` and `Trapezoid`.
fn prism_mass_properties_xy(
    verts: &[glam::DVec2],
    scope: &Scope,
    density: f64,
) -> Option<MassProperties> {
    let depth = scope.size.z;
    if depth <= 0.0 {
        return None;
    }
    let m = polygon_moments(verts)?;
    let volume = m.area * depth;
    if !volume.is_finite() || volume <= 0.0 {
        return None;
    }
    let mass = density * volume;
    if !mass.is_finite() {
        return None;
    }
    // Centred polygon second moments: J = ∫(x − Cx)² dA, K = ∫(y − Cy)² dA,
    // P = ∫(x − Cx)(y − Cy) dA. Per-unit-area normalised: divide by area.
    let j = (m.ixx_origin - m.area * m.cx * m.cx) / m.area;
    let k = (m.iyy_origin - m.area * m.cy * m.cy) / m.area;
    let p = (m.ixy_origin - m.area * m.cx * m.cy) / m.area;
    let local_centroid = Vec3::new(m.cx, m.cy, depth * 0.5);
    let centroid = scope.position + scope.rotation * local_centroid;
    if !centroid.is_finite() {
        return None;
    }
    // Prism inertia about its centroid, axes aligned with scope local frame:
    //   I_xx = m·(d²/12 + ⟨(y−Cy)²⟩) = m·d²/12 + m·k
    //   I_yy = m·d²/12 + m·j
    //   I_zz = m·(j + k)
    //   I_xy = −m·p, I_xz = I_yz = 0  (z is the extrusion axis through centroid)
    let d2_12 = depth * depth / 12.0;
    let i_xx = mass * (d2_12 + k);
    let i_yy = mass * (d2_12 + j);
    let i_zz = mass * (j + k);
    let i_xy = -mass * p;
    let i_local = glam::DMat3::from_cols_array(&[
        i_xx, i_xy, 0.0, // col 0
        i_xy, i_yy, 0.0, // col 1
        0.0, 0.0, i_zz, // col 2
    ]);
    let inertia = rotate_inertia(i_local, scope.rotation);
    Some(MassProperties {
        mass,
        centroid,
        inertia: Some(inertia),
    })
}

/// Prism whose 2-D cross-section is `verts` in the local XZ floor plane,
/// extruded along local Y by `scope.size.y`. Used by `Polygon`.
fn prism_mass_properties_xz(
    verts: &[glam::DVec2],
    scope: &Scope,
    density: f64,
) -> Option<MassProperties> {
    let height = scope.size.y;
    if height <= 0.0 {
        return None;
    }
    // Polygon vertices interpreted as (X, Z). Re-use the XY moments routine
    // by reading the second component as "z" and remapping at the end.
    let m = polygon_moments(verts)?;
    let volume = m.area * height;
    if !volume.is_finite() || volume <= 0.0 {
        return None;
    }
    let mass = density * volume;
    if !mass.is_finite() {
        return None;
    }
    // ⟨(x − Cx)²⟩, ⟨(z − Cz)²⟩, ⟨(x − Cx)(z − Cz)⟩
    let j = (m.ixx_origin - m.area * m.cx * m.cx) / m.area;
    let k = (m.iyy_origin - m.area * m.cy * m.cy) / m.area;
    let p = (m.ixy_origin - m.area * m.cx * m.cy) / m.area;
    let local_centroid = Vec3::new(m.cx, height * 0.5, m.cy);
    let centroid = scope.position + scope.rotation * local_centroid;
    if !centroid.is_finite() {
        return None;
    }
    // Prism along Y through centroid:
    //   I_xx = m·(h²/12 + ⟨(z−Cz)²⟩)
    //   I_yy = m·(⟨(x−Cx)²⟩ + ⟨(z−Cz)²⟩)
    //   I_zz = m·(h²/12 + ⟨(x−Cx)²⟩)
    //   I_xz = −m·p, I_xy = I_yz = 0
    let h2_12 = height * height / 12.0;
    let i_xx = mass * (h2_12 + k);
    let i_yy = mass * (j + k);
    let i_zz = mass * (h2_12 + j);
    let i_xz = -mass * p;
    let i_local = glam::DMat3::from_cols_array(&[
        i_xx, 0.0, i_xz, // col 0
        0.0, i_yy, 0.0, // col 1
        i_xz, 0.0, i_zz, // col 2
    ]);
    let inertia = rotate_inertia(i_local, scope.rotation);
    Some(MassProperties {
        mass,
        centroid,
        inertia: Some(inertia),
    })
}

/// 2-D polygon moments via the shoelace formula and its higher-moment variants.
struct PolygonMoments {
    area: f64,
    cx: f64,
    cy: f64,
    /// ∫∫ x² dA, about the original axes (not centroidal).
    ixx_origin: f64,
    /// ∫∫ y² dA, about the original axes.
    iyy_origin: f64,
    /// ∫∫ x·y dA, about the original axes.
    ixy_origin: f64,
}

fn polygon_moments(verts: &[glam::DVec2]) -> Option<PolygonMoments> {
    if verts.len() < 3 {
        return None;
    }
    let n = verts.len();
    let mut a2 = 0.0_f64; // 2·signed-area
    let mut cx6 = 0.0_f64;
    let mut cy6 = 0.0_f64;
    let mut ixx12 = 0.0_f64;
    let mut iyy12 = 0.0_f64;
    let mut ixy24 = 0.0_f64;
    for i in 0..n {
        let p = verts[i];
        let q = verts[(i + 1) % n];
        let cross = p.x * q.y - q.x * p.y;
        a2 += cross;
        // Centroid sums (× 6).
        cx6 += (p.x + q.x) * cross;
        cy6 += (p.y + q.y) * cross;
        // Note: standard polygon-moment formulas yield ∫y² dA and ∫x² dA;
        // the variable names here track the integrand, not the axis label.
        ixx12 += (p.y * p.y + p.y * q.y + q.y * q.y) * cross;
        iyy12 += (p.x * p.x + p.x * q.x + q.x * q.x) * cross;
        ixy24 += (p.x * q.y + 2.0 * p.x * p.y + 2.0 * q.x * q.y + q.x * p.y) * cross;
    }
    let area = a2 * 0.5;
    if !area.is_finite() || area.abs() < 1e-15 {
        return None;
    }
    let cx = cx6 / (6.0 * area);
    let cy = cy6 / (6.0 * area);
    // |area| handles CW vs CCW input — moments must be positive.
    let abs_area = area.abs();
    let area_sign = if area >= 0.0 { 1.0 } else { -1.0 };
    Some(PolygonMoments {
        area: abs_area,
        cx,
        cy,
        ixx_origin: (ixx12 / 12.0) * area_sign,
        iyy_origin: (iyy12 / 12.0) * area_sign,
        ixy_origin: (ixy24 / 24.0) * area_sign,
    })
}

/// Transforms a body-frame inertia tensor to world frame: R · I · Rᵀ.
fn rotate_inertia(i_local: glam::DMat3, rotation: crate::scope::Quat) -> glam::DMat3 {
    let r = glam::DMat3::from_quat(rotation);
    r * i_local * r.transpose()
}

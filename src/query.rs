//! Spatial query primitives over an in-progress or completed [`ShapeModel`].
//!
//! Currently provides oriented-bounding-box (OBB) overlap tests against the
//! emitted terminals. Used internally by the `IfClear` / `IfOccluded` grammar
//! ops and exposed publicly via [`ShapeModel::query`] so downstream tools can
//! filter, count, or audit terminals after derivation.
//!
//! The overlap test is a true OBB-vs-OBB separating-axis-theorem (SAT) check
//! treating each scope as a non-empty box, with thin face scopes (`size.z = 0`,
//! `size.y = 0`, etc.) given a tiny epsilon thickness so they still produce
//! finite separating-axis projections.

use crate::model::{ShapeModel, Terminal};
use crate::scope::{Quat, Scope, Vec3};

/// Read-only spatial query view over a [`ShapeModel`].
pub struct TerminalQuery<'a> {
    terminals: &'a [Terminal],
}

impl<'a> TerminalQuery<'a> {
    pub(crate) fn new(terminals: &'a [Terminal]) -> Self {
        Self { terminals }
    }

    /// Returns `true` if any terminal's OBB overlaps the given query `scope`.
    pub fn overlaps(&self, scope: &Scope) -> bool {
        self.terminals
            .iter()
            .any(|t| scope_obb_overlaps_terminal(scope, t))
    }

    /// Returns an iterator over every terminal whose OBB overlaps `scope`.
    pub fn overlapping(&self, scope: &'a Scope) -> impl Iterator<Item = &'a Terminal> {
        self.terminals
            .iter()
            .filter(move |t| scope_obb_overlaps_terminal(scope, t))
    }

    /// Returns the underlying terminal slice (for callers that want to iterate
    /// directly without an overlap predicate).
    pub fn terminals(&self) -> &'a [Terminal] {
        self.terminals
    }
}

impl ShapeModel {
    /// Returns a read-only query view over this model's terminals.
    pub fn query(&self) -> TerminalQuery<'_> {
        TerminalQuery::new(&self.terminals)
    }
}

// ── OBB overlap (Separating Axis Theorem) ────────────────────────────────────

/// Minimum half-extent assumed for any axis whose actual size is zero (face
/// scopes, footprint scopes). Without this the SAT projections would collapse
/// to a single point and the test would be unstable; with it, infinitely thin
/// scopes are treated as having a 1 µm thickness for overlap purposes.
const THIN_HALF_EXTENT: f64 = 0.5e-6;

/// Pre-computed OBB representation: world-space centre, half-extents along
/// each local axis, and the three local axes expressed in world space.
struct Obb {
    centre: Vec3,
    half: Vec3,
    axes: [Vec3; 3],
}

fn obb_from_scope(scope: &Scope) -> Obb {
    // Half-extent guard: zero-size axes get a tiny finite thickness so the
    // SAT projection stays well-defined.
    let hx = (scope.size.x * 0.5).max(THIN_HALF_EXTENT);
    let hy = (scope.size.y * 0.5).max(THIN_HALF_EXTENT);
    let hz = (scope.size.z * 0.5).max(THIN_HALF_EXTENT);
    let local_centre = Vec3::new(scope.size.x * 0.5, scope.size.y * 0.5, scope.size.z * 0.5);
    let centre = scope.position + scope.rotation * local_centre;
    let axes = [
        scope.rotation * Vec3::X,
        scope.rotation * Vec3::Y,
        scope.rotation * Vec3::Z,
    ];
    Obb {
        centre,
        half: Vec3::new(hx, hy, hz),
        axes,
    }
}

/// Tests whether two OBBs overlap using the Separating Axis Theorem.
///
/// 15 axes are tested: 3 from each box (6 total) + 9 cross-products. If any
/// axis separates the projections, the boxes do not overlap.
///
/// References: Christer Ericson, "Real-Time Collision Detection", §4.4.1.
pub fn obb_overlap(a: &Scope, b: &Scope) -> bool {
    let a = obb_from_scope(a);
    let b = obb_from_scope(b);
    obb_overlap_impl(&a, &b)
}

fn obb_overlap_impl(a: &Obb, b: &Obb) -> bool {
    // 3×3 rotation matrix expressing b's axes in a's local frame, plus
    // |abs| version for cross-axis tests. EPS guards against parallel-edge
    // numerical jitter (per Ericson §4.4.1).
    const EPS: f64 = 1e-9;
    let mut r = [[0.0_f64; 3]; 3];
    let mut abs_r = [[0.0_f64; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            r[i][j] = a.axes[i].dot(b.axes[j]);
            abs_r[i][j] = r[i][j].abs() + EPS;
        }
    }
    // Translation, expressed in a's local frame.
    let t_world = b.centre - a.centre;
    let t = [
        t_world.dot(a.axes[0]),
        t_world.dot(a.axes[1]),
        t_world.dot(a.axes[2]),
    ];
    let a_h = [a.half.x, a.half.y, a.half.z];
    let b_h = [b.half.x, b.half.y, b.half.z];

    // Test the 3 axes of A.
    for i in 0..3 {
        let ra = a_h[i];
        let rb = b_h[0] * abs_r[i][0] + b_h[1] * abs_r[i][1] + b_h[2] * abs_r[i][2];
        if t[i].abs() > ra + rb {
            return false;
        }
    }
    // Test the 3 axes of B.
    for j in 0..3 {
        let ra = a_h[0] * abs_r[0][j] + a_h[1] * abs_r[1][j] + a_h[2] * abs_r[2][j];
        let rb = b_h[j];
        let proj = t[0] * r[0][j] + t[1] * r[1][j] + t[2] * r[2][j];
        if proj.abs() > ra + rb {
            return false;
        }
    }
    // 9 cross-product axes (A_i × B_j). Each test follows the same template;
    // unrolled for clarity.
    macro_rules! cross_test {
        ($i:expr, $j:expr, $ra:expr, $rb:expr, $tt:expr) => {
            if ($tt).abs() > $ra + $rb {
                return false;
            }
        };
    }
    // i = 0
    cross_test!(
        0,
        0,
        a_h[1] * abs_r[2][0] + a_h[2] * abs_r[1][0],
        b_h[1] * abs_r[0][2] + b_h[2] * abs_r[0][1],
        t[2] * r[1][0] - t[1] * r[2][0]
    );
    cross_test!(
        0,
        1,
        a_h[1] * abs_r[2][1] + a_h[2] * abs_r[1][1],
        b_h[0] * abs_r[0][2] + b_h[2] * abs_r[0][0],
        t[2] * r[1][1] - t[1] * r[2][1]
    );
    cross_test!(
        0,
        2,
        a_h[1] * abs_r[2][2] + a_h[2] * abs_r[1][2],
        b_h[0] * abs_r[0][1] + b_h[1] * abs_r[0][0],
        t[2] * r[1][2] - t[1] * r[2][2]
    );
    // i = 1
    cross_test!(
        1,
        0,
        a_h[0] * abs_r[2][0] + a_h[2] * abs_r[0][0],
        b_h[1] * abs_r[1][2] + b_h[2] * abs_r[1][1],
        t[0] * r[2][0] - t[2] * r[0][0]
    );
    cross_test!(
        1,
        1,
        a_h[0] * abs_r[2][1] + a_h[2] * abs_r[0][1],
        b_h[0] * abs_r[1][2] + b_h[2] * abs_r[1][0],
        t[0] * r[2][1] - t[2] * r[0][1]
    );
    cross_test!(
        1,
        2,
        a_h[0] * abs_r[2][2] + a_h[2] * abs_r[0][2],
        b_h[0] * abs_r[1][1] + b_h[1] * abs_r[1][0],
        t[0] * r[2][2] - t[2] * r[0][2]
    );
    // i = 2
    cross_test!(
        2,
        0,
        a_h[0] * abs_r[1][0] + a_h[1] * abs_r[0][0],
        b_h[1] * abs_r[2][2] + b_h[2] * abs_r[2][1],
        t[1] * r[0][0] - t[0] * r[1][0]
    );
    cross_test!(
        2,
        1,
        a_h[0] * abs_r[1][1] + a_h[1] * abs_r[0][1],
        b_h[0] * abs_r[2][2] + b_h[2] * abs_r[2][0],
        t[1] * r[0][1] - t[0] * r[1][1]
    );
    cross_test!(
        2,
        2,
        a_h[0] * abs_r[1][2] + a_h[1] * abs_r[0][2],
        b_h[0] * abs_r[2][1] + b_h[1] * abs_r[2][0],
        t[1] * r[0][2] - t[0] * r[1][2]
    );
    true
}

/// Convenience wrapper: tests overlap between a scope and a terminal's scope.
pub(crate) fn scope_obb_overlaps_terminal(scope: &Scope, terminal: &Terminal) -> bool {
    obb_overlap(scope, &terminal.scope)
}

// ── Snap-plane helpers ────────────────────────────────────────────────────────

/// Registers the six face planes of `scope` under `label` into `out`.
pub(crate) fn register_scope_snap_planes(
    scope: &Scope,
    label: &str,
    out: &mut Vec<crate::model::SnapPlane>,
) {
    // For each pair of opposite faces, emit a plane at the face centre with
    // the outward normal in world space.
    let cx = scope.size.x * 0.5;
    let cy = scope.size.y * 0.5;
    let cz = scope.size.z * 0.5;
    let local_face_centres = [
        (Vec3::new(0.0, cy, cz), -Vec3::X),         // -X face
        (Vec3::new(scope.size.x, cy, cz), Vec3::X), // +X face
        (Vec3::new(cx, 0.0, cz), -Vec3::Y),         // -Y face
        (Vec3::new(cx, scope.size.y, cz), Vec3::Y), // +Y face
        (Vec3::new(cx, cy, 0.0), -Vec3::Z),         // -Z face
        (Vec3::new(cx, cy, scope.size.z), Vec3::Z), // +Z face
    ];
    for (local_pt, local_normal) in local_face_centres {
        let world_pt = scope.position + scope.rotation * local_pt;
        let world_normal = (scope.rotation * local_normal).normalize();
        out.push(crate::model::SnapPlane {
            point: world_pt,
            normal: world_normal,
            label: label.to_string(),
        });
    }
}

/// Snaps interior `Split` boundaries to the nearest registered snap-plane
/// along `axis` (under `label`) within `tolerance`.
///
/// `sizes` is the resolved per-slot length array; on return the interior
/// positions are shifted to align with snap-planes where possible. Slot total
/// is preserved (a snapped boundary takes width from one neighbour and gives
/// it to the other).
pub(crate) fn snap_split_boundaries(
    scope: &Scope,
    axis: crate::ops::Axis,
    sizes: &mut [f64],
    label: &str,
    tolerance: f64,
    snap_planes: &[crate::model::SnapPlane],
) {
    if sizes.len() < 2 || tolerance <= 0.0 || snap_planes.is_empty() {
        return;
    }
    // Local axis vector & total length.
    let (local_axis_vec, total) = match axis {
        crate::ops::Axis::X => (Vec3::X, scope.size.x),
        crate::ops::Axis::Y => (Vec3::Y, scope.size.y),
        crate::ops::Axis::Z => (Vec3::Z, scope.size.z),
    };
    if total <= 0.0 {
        return;
    }
    // World axis derived from scope.rotation.
    let world_axis = scope.rotation * local_axis_vec;

    // Collect snap-plane projections onto the split axis, in scope-local
    // coordinates, retaining only planes whose normal is parallel-enough to
    // the split axis (so it represents an actual perpendicular cut).
    let mut planes_local: Vec<f64> = Vec::new();
    let scope_local_origin = scope.position;
    for plane in snap_planes {
        if plane.label != label {
            continue;
        }
        let parallel = plane.normal.dot(world_axis).abs();
        if parallel < 0.9 {
            // Plane normal not aligned with split axis — skip.
            continue;
        }
        // Scope-local position along axis = (plane.point - scope.position) · world_axis.
        let local_pos = (plane.point - scope_local_origin).dot(world_axis);
        if local_pos < -tolerance || local_pos > total + tolerance {
            continue;
        }
        planes_local.push(local_pos);
    }
    if planes_local.is_empty() {
        return;
    }

    // For each interior boundary, find nearest snap-plane and adjust.
    let n = sizes.len();
    let mut cumulative: Vec<f64> = Vec::with_capacity(n);
    let mut acc = 0.0;
    for s in sizes.iter() {
        acc += *s;
        cumulative.push(acc);
    }
    // Interior boundary indices: 0..n-1 (last cumulative = total, fixed).
    for boundary in 0..(n - 1) {
        let pos = cumulative[boundary];
        // Find closest snap plane within tolerance.
        let mut best: Option<f64> = None;
        let mut best_dist = tolerance;
        for &p in &planes_local {
            let d = (p - pos).abs();
            if d <= best_dist {
                best = Some(p);
                best_dist = d;
            }
        }
        let Some(target) = best else { continue };
        // Don't move past adjacent boundaries (preserve ordering).
        let lower = if boundary == 0 {
            0.0
        } else {
            cumulative[boundary - 1]
        };
        let upper = cumulative[boundary + 1];
        if target <= lower + 1e-9 || target >= upper - 1e-9 {
            continue;
        }
        cumulative[boundary] = target;
    }

    // Re-derive sizes from cumulative.
    let mut prev = 0.0;
    for (i, c) in cumulative.iter().enumerate() {
        sizes[i] = c - prev;
        prev = *c;
    }
}

// Suppress unused-warning when `Quat` isn't directly referenced.
const _: fn() = || {
    let _: Quat = Quat::IDENTITY;
};

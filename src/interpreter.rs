/// Queue-based CGA Shape Grammar interpreter.
///
/// The interpreter owns a set of named rules. Each rule has one or more
/// weighted variants — for deterministic rules there is exactly one variant.
/// Derivation starts from a root `Scope` and root rule name, expanding rules
/// breadth-first until every branch terminates. Branches terminate either via
/// an explicit `I(mesh_id)` op or by referencing an unknown rule name, which
/// is treated as an implicit `I(rule_name)` terminal ("leaf shorthand").
use std::collections::{HashMap, VecDeque};
use std::f64::consts::{FRAC_PI_2, PI};

use rand::SeedableRng;
use rand_pcg::Pcg64;
use serde::{Deserialize, Serialize};

use crate::error::ShapeError;
use crate::model::{FaceProfile, ShapeModel, Terminal, taper_to_profile};
use crate::ops::{
    AttachCase, AttachSelector, Axis, CompTarget, FaceSelector, OffsetCase, OffsetSelector,
    RoofCase, RoofConfig, RoofFaceSelector, RoofType, ShapeOp, SplitSize, SplitSlot,
};
use crate::scope::{Quat, Scope, Vec3};

/// Safety caps (DoS protection).
const MAX_DEPTH: usize = 64;
const MAX_QUEUE: usize = 100_000;
const MAX_TERMINALS: usize = 100_000;

// ── Weighted rule variant ─────────────────────────────────────────────────────

/// One alternative in a stochastic or deterministic rule.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct WeightedVariant {
    /// Relative weight (need not sum to 1.0 across variants).
    pub weight: f64,
    pub ops: Vec<ShapeOp>,
}

// ── Work queue item ───────────────────────────────────────────────────────────

struct WorkItem {
    scope: Scope,
    rule: String,
    depth: usize,
    /// Taper value set by `ShapeOp::Taper` within this rule invocation; propagated
    /// to the terminal. Branching ops (Split/Comp/Repeat) reset it to 0.0 for
    /// children — taper is not accumulated across rule boundaries.
    taper: f64,
    /// Explicit face profile set by `Roof` panel generation; overrides `taper`
    /// when computing the terminal's `face_profile`.
    face_profile_override: Option<FaceProfile>,
    /// Material identifier set by `Mat("...")` ops; propagates to child scopes.
    material: Option<String>,
}

// ── Split size resolution ─────────────────────────────────────────────────────

/// Resolves `SplitSlot` sizes against `total_dim`, returning absolute sizes.
fn resolve_split_sizes(slots: &[SplitSlot], total_dim: f64) -> Result<Vec<f64>, ShapeError> {
    if slots.is_empty() {
        return Err(ShapeError::EmptySplit);
    }
    for slot in slots {
        if !slot.size.is_valid() {
            return match &slot.size {
                SplitSize::Floating(v) => Err(ShapeError::InvalidFloatingSize(*v)),
                _ => Err(ShapeError::InvalidNumericValue),
            };
        }
    }

    let mut fixed: Vec<Option<f64>> = Vec::with_capacity(slots.len());
    let mut used = 0.0_f64;
    let mut float_weight_total = 0.0_f64;

    for slot in slots {
        match slot.size {
            SplitSize::Absolute(v) => {
                fixed.push(Some(v));
                used += v;
            }
            SplitSize::Relative(t) => {
                let s = total_dim * t;
                fixed.push(Some(s));
                used += s;
            }
            SplitSize::Floating(w) => {
                fixed.push(None);
                float_weight_total += w;
            }
        }
    }

    // Guard against absolute-size sum overflow (e.g. 256 slots each with 1e307).
    if !used.is_finite() {
        return Err(ShapeError::InvalidNumericValue);
    }

    if used > total_dim + 1e-9 {
        return Err(ShapeError::SplitOverflow(total_dim));
    }

    let remaining = (total_dim - used).max(0.0);

    // Guard against weight sum overflow (e.g. 256 slots each with weight 1e307).
    if !float_weight_total.is_finite() {
        return Err(ShapeError::InvalidNumericValue);
    }

    let mut result = Vec::with_capacity(slots.len());
    for (i, slot) in slots.iter().enumerate() {
        match fixed[i] {
            Some(v) => result.push(v),
            None => {
                let w = match slot.size {
                    SplitSize::Floating(w) => w,
                    _ => unreachable!(),
                };
                if float_weight_total <= 0.0 {
                    return Err(ShapeError::NoFloatingSlots);
                }
                // Compute the ratio first (≤ 1.0) to avoid an intermediate
                // product overflow when both `remaining` and `w` are large.
                result.push(remaining * (w / float_weight_total));
            }
        }
    }

    Ok(result)
}

// ── Scope slicing helpers ─────────────────────────────────────────────────────

/// Creates a child scope that is a sub-interval `[offset, offset+size]` of the
/// parent scope along `axis`. All measurements are in local (scope) units.
fn slice_scope(parent: &Scope, axis: Axis, offset: f64, size: f64) -> Scope {
    let offset_vec = match axis {
        Axis::X => Vec3::new(offset, 0.0, 0.0),
        Axis::Y => Vec3::new(0.0, offset, 0.0),
        Axis::Z => Vec3::new(0.0, 0.0, offset),
    };

    let child_position = parent.position + parent.rotation * offset_vec;

    let child_size = match axis {
        Axis::X => Vec3::new(size, parent.size.y, parent.size.z),
        Axis::Y => Vec3::new(parent.size.x, size, parent.size.z),
        Axis::Z => Vec3::new(parent.size.x, parent.size.y, size),
    };

    Scope::new(child_position, parent.rotation, child_size)
}

// ── Face decomposition ────────────────────────────────────────────────────────

/// All six canonical faces of an OBB, each with a proper outward-facing orientation.
///
/// **Convention:** Local **Z** points along the outward normal.  Local **X** is
/// world-horizontal along the face; Local **Y** is world-up for vertical faces.
///
/// This means `Split(X)` tiles a wall horizontally and `Split(Y)` divides it
/// into floors — the same grammar rule works on any vertical face without manual
/// rotation hacks.
///
/// Each entry: `(selector, local_offset, face_size, rotation_delta)`.
/// `face_size` uses Z=0 (flattened 2-D canvas).
fn face_descs(scope_size: Vec3) -> [(FaceSelector, Vec3, Vec3, Quat); 6] {
    let sx = scope_size.x;
    let sy = scope_size.y;
    let sz = scope_size.z;

    [
        // Bottom: outward = -Y.  Local X=+X, Local Y=+Z, Local Z=-Y.
        // Rotation: from_axis_angle(X, +π/2) → X→X, Y→+Z, Z→-Y.
        (
            FaceSelector::Bottom,
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(sx, sz, 0.0),
            Quat::from_axis_angle(Vec3::X, FRAC_PI_2),
        ),
        // Top: outward = +Y.  Local X=+X, Local Y=-Z, Local Z=+Y.
        // Rotation: from_axis_angle(X, -π/2) → X→X, Y→-Z, Z→+Y.
        // Origin at (0, sy, sz) so local-Y tiles from back to front.
        (
            FaceSelector::Top,
            Vec3::new(0.0, sy, sz),
            Vec3::new(sx, sz, 0.0),
            Quat::from_axis_angle(Vec3::X, -FRAC_PI_2),
        ),
        // Front: outward = -Z.  Local X=-X, Local Y=+Y, Local Z=-Z.
        // Rotation: from_axis_angle(Y, π) → X→-X, Y→Y, Z→-Z.
        // Origin at (sx, 0, 0) so local-X tiles from right to left in world.
        (
            FaceSelector::Front,
            Vec3::new(sx, 0.0, 0.0),
            Vec3::new(sx, sy, 0.0),
            Quat::from_axis_angle(Vec3::Y, PI),
        ),
        // Back: outward = +Z.  Local X=+X, Local Y=+Y, Local Z=+Z.
        // Rotation: identity.
        // Origin at (0, 0, sz).
        (
            FaceSelector::Back,
            Vec3::new(0.0, 0.0, sz),
            Vec3::new(sx, sy, 0.0),
            Quat::IDENTITY,
        ),
        // Left: outward = -X.  Local X=+Z, Local Y=+Y, Local Z=-X.
        // Rotation: from_axis_angle(Y, -π/2) → X→+Z, Y→Y, Z→-X.
        // Origin at (0, 0, 0).
        (
            FaceSelector::Left,
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(sz, sy, 0.0),
            Quat::from_axis_angle(Vec3::Y, -FRAC_PI_2),
        ),
        // Right: outward = +X.  Local X=-Z, Local Y=+Y, Local Z=+X.
        // Rotation: from_axis_angle(Y, +π/2) → X→-Z, Y→Y, Z→+X.
        // Origin at (sx, 0, sz).
        (
            FaceSelector::Right,
            Vec3::new(sx, 0.0, sz),
            Vec3::new(sz, sy, 0.0),
            Quat::from_axis_angle(Vec3::Y, FRAC_PI_2),
        ),
    ]
}

/// Finds the rule to apply to a face given a list of `CompFaceCase`s.
fn find_face_rule(selector: FaceSelector, cases: &[crate::ops::CompFaceCase]) -> Option<&str> {
    for case in cases {
        if case.selector == selector {
            return Some(&case.rule);
        }
    }
    let is_side = matches!(
        selector,
        FaceSelector::Front | FaceSelector::Back | FaceSelector::Left | FaceSelector::Right
    );
    if is_side {
        for case in cases {
            if case.selector == FaceSelector::Side {
                return Some(&case.rule);
            }
        }
    }
    for case in cases {
        if case.selector == FaceSelector::All {
            return Some(&case.rule);
        }
    }
    None
}

/// Returns the local-space unit vector for the given axis.
fn axis_vec(axis: Axis) -> Vec3 {
    match axis {
        Axis::X => Vec3::X,
        Axis::Y => Vec3::Y,
        Axis::Z => Vec3::Z,
    }
}

/// Rule look-up for `Offset` cases.
fn find_offset_rule(selector: OffsetSelector, cases: &[OffsetCase]) -> Option<&str> {
    for c in cases {
        if c.selector == selector {
            return Some(&c.rule);
        }
    }
    for c in cases {
        if c.selector == OffsetSelector::All {
            return Some(&c.rule);
        }
    }
    None
}

/// Rule look-up for `Roof` cases.
fn find_roof_rule(selector: RoofFaceSelector, cases: &[RoofCase]) -> Option<&str> {
    for c in cases {
        if c.selector == selector {
            return Some(&c.rule);
        }
    }
    for c in cases {
        if c.selector == RoofFaceSelector::All {
            return Some(&c.rule);
        }
    }
    None
}

/// Rule look-up for `Attach` cases.
fn find_attach_rule(selector: AttachSelector, cases: &[AttachCase]) -> Option<&str> {
    for c in cases {
        if c.selector == selector {
            return Some(&c.rule);
        }
    }
    for c in cases {
        if c.selector == AttachSelector::All {
            return Some(&c.rule);
        }
    }
    None
}

/// Rotations for the four cardinal slope directions used by `Roof`.
///
/// All rotations are expressed as deltas in the **parent scope's local frame**
/// (composed as `scope.rotation * delta` to obtain the world-space rotation).
///
/// Convention: Local Z = outward normal (away from building), Local Y = up the slope,
/// matching the `Comp(Faces)` face convention extended to tilted surfaces.
///
/// - `front_rot`: outward normal = (0, cos α, −sin α) — up & forward (−Z world)
/// - `back_rot`:  outward normal = (0, cos α, +sin α) — up & backward (+Z world)
/// - `left_rot`:  outward normal = (−sin α, cos α, 0) — up & left (−X world)
/// - `right_rot`: outward normal = (+sin α, cos α, 0) — up & right (+X world)
fn roof_slope_rotations(alpha: f64) -> (Quat, Quat, Quat, Quat) {
    // front: mirror the Comp-Front face (flip X via Y-π rotation), then tilt the slope.
    //   Local X = (−1, 0, 0), Local Z = (0, cos α, −sin α) — outward up & forward.
    let front_rot =
        Quat::from_axis_angle(Vec3::X, FRAC_PI_2 - alpha) * Quat::from_axis_angle(Vec3::Y, PI);
    // back: identity orientation tilted by (α − π/2).
    //   Local X = (+1, 0, 0), Local Z = (0, cos α, +sin α) — outward up & backward.
    let back_rot = Quat::from_axis_angle(Vec3::X, alpha - FRAC_PI_2);
    // left: Comp-Left face orientation (Y, −π/2) then tilt.
    //   Local X = (0, 0, +1), Local Z = (−sin α, cos α, 0) — outward up & left.
    let left_rot = Quat::from_axis_angle(Vec3::Z, alpha - FRAC_PI_2)
        * Quat::from_axis_angle(Vec3::Y, -FRAC_PI_2);
    // right: Comp-Right face orientation (Y, +π/2) then tilt.
    //   Local X = (0, 0, −1), Local Z = (+sin α, cos α, 0) — outward up & right.
    let right_rot = Quat::from_axis_angle(Vec3::Z, FRAC_PI_2 - alpha)
        * Quat::from_axis_angle(Vec3::Y, FRAC_PI_2);
    (front_rot, back_rot, left_rot, right_rot)
}

// ── Roof geometry ─────────────────────────────────────────────────────────────

/// Generates roof panel scopes for all supported `RoofType` variants.
///
/// Each panel is a flat scope (size.z = 0) with an orientation that places
/// local Z along the outward normal and local Y up the slope, consistent with
/// the `Comp(Faces)` convention. The `face_profile_override` field on each
/// child `WorkItem` carries the exact 2D cross-section shape.
#[allow(clippy::too_many_arguments)]
fn apply_roof(
    config: &RoofConfig,
    cases: &[RoofCase],
    scope: &Scope,
    depth: usize,
    material: &Option<String>,
    queue: &mut VecDeque<WorkItem>,
    model: &mut ShapeModel,
    max_terminals: usize,
) -> Result<(), ShapeError> {
    if !config.pitch.is_finite() || config.pitch <= 0.0 || config.pitch >= 90.0 {
        return Err(ShapeError::InvalidRoofAngle(config.pitch));
    }
    if !config.overhang.is_finite() || config.overhang < 0.0 {
        return Err(ShapeError::InvalidNumericValue);
    }

    let sx = scope.size.x;
    let sz = scope.size.z;
    let o = config.overhang;
    let alpha = config.pitch.to_radians();
    let cos_a = alpha.cos();
    let tan_a = alpha.tan();
    let (front_rot, back_rot, left_rot, right_rot) = roof_slope_rotations(alpha);
    // y_anchor: Y offset below scope.position due to eave overhang projection.
    let y_anchor = -o * tan_a;
    // Slope lengths from eave to ridge centre (including overhang).
    let fb_len = (sz / 2.0 + o) / cos_a;
    let lr_len = (sx / 2.0 + o) / cos_a;
    // Ridge height above eave (driven by depth, no overhang contribution to height).
    let h = (sz / 2.0) * tan_a;

    if !fb_len.is_finite() || !lr_len.is_finite() || !h.is_finite() {
        return Err(ShapeError::InvalidNumericValue);
    }

    // Panel tuple: (local_offset, face_size, rot_delta, selector, face_profile)
    type Panel = (Vec3, Vec3, Quat, RoofFaceSelector, FaceProfile);

    let panels: Vec<Panel> = match config.roof_type {
        // ── Flat ─────────────────────────────────────────────────────────────
        // One horizontal panel covering the scope top (same geometry as Comp Top).
        RoofType::Flat => vec![(
            Vec3::new(0.0, 0.0, sz),
            Vec3::new(sx, sz, 0.0),
            Quat::from_axis_angle(Vec3::X, -FRAC_PI_2),
            RoofFaceSelector::Slope,
            FaceProfile::Rectangle,
        )],

        // ── Shed ─────────────────────────────────────────────────────────────
        // One slope from front eave to back eave (front_rot convention).
        RoofType::Shed => {
            let shed_h = sz * tan_a;
            vec![
                (
                    Vec3::new(sx + o, y_anchor, -o),
                    Vec3::new(sx + 2.0 * o, (sz + 2.0 * o) / cos_a, 0.0),
                    front_rot,
                    RoofFaceSelector::Slope,
                    FaceProfile::Rectangle,
                ),
                (
                    Vec3::new(0.0, 0.0, 0.0),
                    Vec3::new(sz, shed_h, 0.0),
                    Quat::from_axis_angle(Vec3::Y, -FRAC_PI_2),
                    RoofFaceSelector::GableEnd,
                    FaceProfile::Triangle { peak_offset: 1.0 },
                ),
                (
                    Vec3::new(sx, 0.0, sz),
                    Vec3::new(sz, shed_h, 0.0),
                    Quat::from_axis_angle(Vec3::Y, FRAC_PI_2),
                    RoofFaceSelector::GableEnd,
                    FaceProfile::Triangle { peak_offset: 0.0 },
                ),
            ]
        }

        // ── Gable / OpenGable / BoxGable ──────────────────────────────────────
        RoofType::Gable | RoofType::OpenGable | RoofType::BoxGable => {
            let (slope_len, eave_len, ridge_h) = if sx >= sz {
                ((sz / 2.0 + o) / cos_a, sx + 2.0 * o, (sz / 2.0) * tan_a)
            } else {
                ((sx / 2.0 + o) / cos_a, sz + 2.0 * o, (sx / 2.0) * tan_a)
            };

            let mut panels = if sx >= sz {
                vec![
                    (
                        Vec3::new(sx + o, y_anchor, -o),
                        Vec3::new(eave_len, slope_len, 0.0),
                        front_rot,
                        RoofFaceSelector::Slope,
                        FaceProfile::Rectangle,
                    ),
                    (
                        Vec3::new(-o, y_anchor, sz + o),
                        Vec3::new(eave_len, slope_len, 0.0),
                        back_rot,
                        RoofFaceSelector::Slope,
                        FaceProfile::Rectangle,
                    ),
                ]
            } else {
                vec![
                    (
                        Vec3::new(-o, y_anchor, -o),
                        Vec3::new(eave_len, slope_len, 0.0),
                        left_rot,
                        RoofFaceSelector::Slope,
                        FaceProfile::Rectangle,
                    ),
                    (
                        Vec3::new(sx + o, y_anchor, sz + o),
                        Vec3::new(eave_len, slope_len, 0.0),
                        right_rot,
                        RoofFaceSelector::Slope,
                        FaceProfile::Rectangle,
                    ),
                ]
            };

            if config.roof_type != RoofType::OpenGable {
                let profile = if config.roof_type == RoofType::BoxGable {
                    FaceProfile::Rectangle
                } else {
                    FaceProfile::Triangle { peak_offset: 0.5 }
                };

                if sx >= sz {
                    panels.extend(vec![
                        (
                            Vec3::new(0.0, 0.0, 0.0),
                            Vec3::new(sz, ridge_h, 0.0),
                            Quat::from_axis_angle(Vec3::Y, -FRAC_PI_2),
                            RoofFaceSelector::GableEnd,
                            profile.clone(),
                        ),
                        (
                            Vec3::new(sx, 0.0, sz),
                            Vec3::new(sz, ridge_h, 0.0),
                            Quat::from_axis_angle(Vec3::Y, FRAC_PI_2),
                            RoofFaceSelector::GableEnd,
                            profile,
                        ),
                    ]);
                } else {
                    panels.extend(vec![
                        (
                            Vec3::new(sx, 0.0, 0.0),
                            Vec3::new(sx, ridge_h, 0.0),
                            Quat::from_axis_angle(Vec3::Y, PI),
                            RoofFaceSelector::GableEnd,
                            profile.clone(),
                        ),
                        (
                            Vec3::new(0.0, 0.0, sz),
                            Vec3::new(sx, ridge_h, 0.0),
                            Quat::IDENTITY,
                            RoofFaceSelector::GableEnd,
                            profile,
                        ),
                    ]);
                }
            }
            panels
        }

        // ── Pyramid / PyramidHip / Hip ─────────────────────────────────────────
        // For non-square bases, true pyramids with equal pitch are mathematically
        // impossible; they correctly degenerate into a Hip roof with a ridge.
        RoofType::Pyramid | RoofType::PyramidHip | RoofType::Hip => {
            let eave_w = sx + 2.0 * o;
            let eave_d = sz + 2.0 * o;
            let max_run = sx.min(sz) / 2.0 + o;
            let slope_len = max_run / cos_a;

            let (fb_profile, lr_profile) = if sx > sz + 1e-5 {
                let top_w = (sx - sz) / eave_w;
                let off_x = (sz / 2.0 + o) / eave_w;
                (
                    FaceProfile::Trapezoid {
                        top_width: top_w,
                        offset_x: off_x,
                    },
                    FaceProfile::Triangle { peak_offset: 0.5 },
                )
            } else if sz > sx + 1e-5 {
                let top_w = (sz - sx) / eave_d;
                let off_x = (sx / 2.0 + o) / eave_d;
                (
                    FaceProfile::Triangle { peak_offset: 0.5 },
                    FaceProfile::Trapezoid {
                        top_width: top_w,
                        offset_x: off_x,
                    },
                )
            } else {
                (
                    FaceProfile::Triangle { peak_offset: 0.5 },
                    FaceProfile::Triangle { peak_offset: 0.5 },
                )
            };

            vec![
                (
                    Vec3::new(sx + o, y_anchor, -o),
                    Vec3::new(eave_w, slope_len, 0.0),
                    front_rot,
                    RoofFaceSelector::Slope,
                    fb_profile.clone(),
                ),
                (
                    Vec3::new(-o, y_anchor, sz + o),
                    Vec3::new(eave_w, slope_len, 0.0),
                    back_rot,
                    RoofFaceSelector::Slope,
                    fb_profile,
                ),
                (
                    Vec3::new(-o, y_anchor, -o),
                    Vec3::new(eave_d, slope_len, 0.0),
                    left_rot,
                    RoofFaceSelector::Slope,
                    lr_profile.clone(),
                ),
                (
                    Vec3::new(sx + o, y_anchor, sz + o),
                    Vec3::new(eave_d, slope_len, 0.0),
                    right_rot,
                    RoofFaceSelector::Slope,
                    lr_profile,
                ),
            ]
        }

        // ── Butterfly ─────────────────────────────────────────────────────────
        // Two inward-tilting slopes with a valley at centre (z = sz/2).
        // Panels run FROM the valley TOWARD each eave using back_rot/front_rot.
        RoofType::Butterfly => {
            // The valley sits (sz/2 + o)*tan_a below eave level.
            let y_valley = y_anchor - (sz / 2.0 + o) * tan_a;
            if !y_valley.is_finite() {
                return Err(ShapeError::InvalidNumericValue);
            }
            vec![
                // Front valley slope: valley → front eave (back_rot points toward -Z = front)
                (
                    Vec3::new(-o, y_valley, sz / 2.0),
                    Vec3::new(sx + 2.0 * o, fb_len, 0.0),
                    back_rot,
                    RoofFaceSelector::ValleySlope,
                    FaceProfile::Rectangle,
                ),
                // Back valley slope: valley → back eave (front_rot points toward +Z = back)
                (
                    Vec3::new(sx + o, y_valley, sz / 2.0),
                    Vec3::new(sx + 2.0 * o, fb_len, 0.0),
                    front_rot,
                    RoofFaceSelector::ValleySlope,
                    FaceProfile::Rectangle,
                ),
            ]
        }

        // ── MShaped ───────────────────────────────────────────────────────────
        // Two ridges (at z = sz/4 and z = 3*sz/4) with a valley at z = sz/2.
        // Four slopes: outer-front, inner-front (valley), inner-back (valley), outer-back.
        RoofType::MShaped => {
            let quarter = sz / 4.0;
            let h_m = quarter * tan_a;
            if !h_m.is_finite() {
                return Err(ShapeError::InvalidNumericValue);
            }
            let slope_m = quarter / cos_a;
            let y_valley = y_anchor - h_m; // valley is h_m below the outer ridges
            vec![
                // Outer front: eave (z=-o) → front ridge (z=sz/4)
                (
                    Vec3::new(sx + o, y_anchor, -o),
                    Vec3::new(sx + 2.0 * o, slope_m, 0.0),
                    front_rot,
                    RoofFaceSelector::OuterSlope,
                    FaceProfile::Rectangle,
                ),
                // Inner front: valley (z=sz/2) → front ridge (z=sz/4), using back_rot
                (
                    Vec3::new(-o, y_valley, sz / 2.0),
                    Vec3::new(sx + 2.0 * o, slope_m, 0.0),
                    back_rot,
                    RoofFaceSelector::InnerSlope,
                    FaceProfile::Rectangle,
                ),
                // Inner back: valley (z=sz/2) → back ridge (z=3*sz/4), using front_rot
                (
                    Vec3::new(sx + o, y_valley, sz / 2.0),
                    Vec3::new(sx + 2.0 * o, slope_m, 0.0),
                    front_rot,
                    RoofFaceSelector::InnerSlope,
                    FaceProfile::Rectangle,
                ),
                // Outer back: eave (z=sz+o) → back ridge (z=3*sz/4)
                (
                    Vec3::new(-o, y_anchor, sz + o),
                    Vec3::new(sx + 2.0 * o, slope_m, 0.0),
                    back_rot,
                    RoofFaceSelector::OuterSlope,
                    FaceProfile::Rectangle,
                ),
            ]
        }

        // ── Gambrel ───────────────────────────────────────────────────────────
        // Two-pitch front/back barn roof: steep lower zone + shallow upper zone.
        RoofType::Gambrel => {
            let alpha2 = config.secondary_pitch_or_default().to_radians();
            if !alpha2.is_finite() || alpha2 <= 0.0 || alpha2 >= FRAC_PI_2 {
                return Err(ShapeError::InvalidNumericValue);
            }
            let cos_a2 = alpha2.cos();
            let tan_a2 = alpha2.tan();
            let tier = config.tier_height_or(0.5).clamp(0.01, 0.99);
            let (ufr, ubr, ulr, urr) = roof_slope_rotations(alpha2);

            if sx >= sz {
                let run_z = sz / 2.0 + o;
                let break_run = (tier * run_z).clamp(o, run_z - 1e-3);
                let h_break = break_run * tan_a;
                if !h_break.is_finite() {
                    return Err(ShapeError::InvalidNumericValue);
                }
                let lower_slope = break_run / cos_a;
                let upper_run = run_z - break_run;
                let upper_slope = upper_run / cos_a2;
                let upper_h = upper_run * tan_a2;
                let y_break = y_anchor + h_break;
                let eave_w = sx + 2.0 * o;
                let wall_y_break = h_break - o * tan_a;
                let wall_break_run = break_run - o;
                let mid_w = (sz - 2.0 * wall_break_run).max(0.0);

                let lower_gable_profile = if mid_w > 1e-9 {
                    FaceProfile::Trapezoid {
                        top_width: mid_w / sz,
                        offset_x: wall_break_run / sz,
                    }
                } else {
                    FaceProfile::Triangle { peak_offset: 0.5 }
                };

                vec![
                    (
                        Vec3::new(sx + o, y_anchor, -o),
                        Vec3::new(eave_w, lower_slope, 0.0),
                        front_rot,
                        RoofFaceSelector::LowerSlope,
                        FaceProfile::Rectangle,
                    ),
                    (
                        Vec3::new(-o, y_anchor, sz + o),
                        Vec3::new(eave_w, lower_slope, 0.0),
                        back_rot,
                        RoofFaceSelector::LowerSlope,
                        FaceProfile::Rectangle,
                    ),
                    (
                        Vec3::new(sx + o, y_break, -o + break_run),
                        Vec3::new(eave_w, upper_slope, 0.0),
                        ufr,
                        RoofFaceSelector::UpperSlope,
                        FaceProfile::Rectangle,
                    ),
                    (
                        Vec3::new(-o, y_break, sz + o - break_run),
                        Vec3::new(eave_w, upper_slope, 0.0),
                        ubr,
                        RoofFaceSelector::UpperSlope,
                        FaceProfile::Rectangle,
                    ),
                    (
                        Vec3::new(0.0, 0.0, 0.0),
                        Vec3::new(sz, wall_y_break, 0.0),
                        Quat::from_axis_angle(Vec3::Y, -FRAC_PI_2),
                        RoofFaceSelector::GableEnd,
                        lower_gable_profile.clone(),
                    ),
                    (
                        Vec3::new(sx, 0.0, sz),
                        Vec3::new(sz, wall_y_break, 0.0),
                        Quat::from_axis_angle(Vec3::Y, FRAC_PI_2),
                        RoofFaceSelector::GableEnd,
                        lower_gable_profile,
                    ),
                    (
                        Vec3::new(0.0, wall_y_break, wall_break_run),
                        Vec3::new(mid_w, upper_h, 0.0),
                        Quat::from_axis_angle(Vec3::Y, -FRAC_PI_2),
                        RoofFaceSelector::GableEnd,
                        FaceProfile::Triangle { peak_offset: 0.5 },
                    ),
                    (
                        Vec3::new(sx, wall_y_break, sz - wall_break_run),
                        Vec3::new(mid_w, upper_h, 0.0),
                        Quat::from_axis_angle(Vec3::Y, FRAC_PI_2),
                        RoofFaceSelector::GableEnd,
                        FaceProfile::Triangle { peak_offset: 0.5 },
                    ),
                ]
            } else {
                let run_x = sx / 2.0 + o;
                let break_run = (tier * run_x).clamp(o, run_x - 1e-3);
                let h_break = break_run * tan_a;
                if !h_break.is_finite() {
                    return Err(ShapeError::InvalidNumericValue);
                }
                let lower_slope = break_run / cos_a;
                let upper_run = run_x - break_run;
                let upper_slope = upper_run / cos_a2;
                let upper_h = upper_run * tan_a2;
                let y_break = y_anchor + h_break;
                let eave_d = sz + 2.0 * o;
                let wall_y_break = h_break - o * tan_a;
                let wall_break_run = break_run - o;
                let mid_d = (sx - 2.0 * wall_break_run).max(0.0);

                let lower_gable_profile = if mid_d > 1e-9 {
                    FaceProfile::Trapezoid {
                        top_width: mid_d / sx,
                        offset_x: wall_break_run / sx,
                    }
                } else {
                    FaceProfile::Triangle { peak_offset: 0.5 }
                };

                vec![
                    (
                        Vec3::new(-o, y_anchor, -o),
                        Vec3::new(eave_d, lower_slope, 0.0),
                        left_rot,
                        RoofFaceSelector::LowerSlope,
                        FaceProfile::Rectangle,
                    ),
                    (
                        Vec3::new(sx + o, y_anchor, sz + o),
                        Vec3::new(eave_d, lower_slope, 0.0),
                        right_rot,
                        RoofFaceSelector::LowerSlope,
                        FaceProfile::Rectangle,
                    ),
                    (
                        Vec3::new(-o + break_run, y_break, -o),
                        Vec3::new(eave_d, upper_slope, 0.0),
                        ulr,
                        RoofFaceSelector::UpperSlope,
                        FaceProfile::Rectangle,
                    ),
                    (
                        Vec3::new(sx + o - break_run, y_break, sz + o),
                        Vec3::new(eave_d, upper_slope, 0.0),
                        urr,
                        RoofFaceSelector::UpperSlope,
                        FaceProfile::Rectangle,
                    ),
                    (
                        Vec3::new(sx, 0.0, 0.0),
                        Vec3::new(sx, wall_y_break, 0.0),
                        Quat::from_axis_angle(Vec3::Y, PI),
                        RoofFaceSelector::GableEnd,
                        lower_gable_profile.clone(),
                    ),
                    (
                        Vec3::new(0.0, 0.0, sz),
                        Vec3::new(sx, wall_y_break, 0.0),
                        Quat::IDENTITY,
                        RoofFaceSelector::GableEnd,
                        lower_gable_profile,
                    ),
                    (
                        Vec3::new(sx - wall_break_run, wall_y_break, 0.0),
                        Vec3::new(mid_d, upper_h, 0.0),
                        Quat::from_axis_angle(Vec3::Y, PI),
                        RoofFaceSelector::GableEnd,
                        FaceProfile::Triangle { peak_offset: 0.5 },
                    ),
                    (
                        Vec3::new(wall_break_run, wall_y_break, sz),
                        Vec3::new(mid_d, upper_h, 0.0),
                        Quat::IDENTITY,
                        RoofFaceSelector::GableEnd,
                        FaceProfile::Triangle { peak_offset: 0.5 },
                    ),
                ]
            }
        }

        // ── Mansard ───────────────────────────────────────────────────────────
        // Gambrel applied to all four sides: 4 steep lower + 4 shallow upper panels.
        RoofType::Mansard => {
            let alpha2 = config.secondary_pitch_or_default().to_radians();
            if !alpha2.is_finite() || alpha2 <= 0.0 || alpha2 >= FRAC_PI_2 {
                return Err(ShapeError::InvalidNumericValue);
            }
            let cos_a2 = alpha2.cos();
            let tier = config.tier_height_or(0.5).clamp(0.01, 0.99);

            let max_run = sx.min(sz) / 2.0 + o;
            let break_run = (tier * max_run).clamp(o, max_run - 1e-3);
            let h_break = break_run * tan_a;

            if !h_break.is_finite() {
                return Err(ShapeError::InvalidNumericValue);
            }

            let lower_slope = break_run / cos_a;
            let y_break = y_anchor + h_break;

            let eave_w = sx + 2.0 * o;
            let eave_d = sz + 2.0 * o;

            let mid_w = (eave_w - 2.0 * break_run).max(0.0);
            let mid_d = (eave_d - 2.0 * break_run).max(0.0);

            let lower_fb_profile = if mid_w > 1e-9 {
                FaceProfile::Trapezoid {
                    top_width: mid_w / eave_w,
                    offset_x: break_run / eave_w,
                }
            } else {
                FaceProfile::Triangle { peak_offset: 0.5 }
            };

            let lower_lr_profile = if mid_d > 1e-9 {
                FaceProfile::Trapezoid {
                    top_width: mid_d / eave_d,
                    offset_x: break_run / eave_d,
                }
            } else {
                FaceProfile::Triangle { peak_offset: 0.5 }
            };

            let (ufr, ubr, ulr, urr) = roof_slope_rotations(alpha2);

            let upper_run = mid_w.min(mid_d) / 2.0;
            let upper_slope = upper_run / cos_a2;

            let top_w = (mid_w - 2.0 * upper_run).max(0.0);
            let top_d = (mid_d - 2.0 * upper_run).max(0.0);

            let upper_fb_profile = if top_w > 1e-9 {
                FaceProfile::Trapezoid {
                    top_width: top_w / mid_w,
                    offset_x: upper_run / mid_w,
                }
            } else {
                FaceProfile::Triangle { peak_offset: 0.5 }
            };

            let upper_lr_profile = if top_d > 1e-9 {
                FaceProfile::Trapezoid {
                    top_width: top_d / mid_d,
                    offset_x: upper_run / mid_d,
                }
            } else {
                FaceProfile::Triangle { peak_offset: 0.5 }
            };

            vec![
                // Lower steep slopes
                (
                    Vec3::new(sx + o, y_anchor, -o),
                    Vec3::new(eave_w, lower_slope, 0.0),
                    front_rot,
                    RoofFaceSelector::LowerSlope,
                    lower_fb_profile.clone(),
                ),
                (
                    Vec3::new(-o, y_anchor, sz + o),
                    Vec3::new(eave_w, lower_slope, 0.0),
                    back_rot,
                    RoofFaceSelector::LowerSlope,
                    lower_fb_profile,
                ),
                (
                    Vec3::new(-o, y_anchor, -o),
                    Vec3::new(eave_d, lower_slope, 0.0),
                    left_rot,
                    RoofFaceSelector::LowerSlope,
                    lower_lr_profile.clone(),
                ),
                (
                    Vec3::new(sx + o, y_anchor, sz + o),
                    Vec3::new(eave_d, lower_slope, 0.0),
                    right_rot,
                    RoofFaceSelector::LowerSlope,
                    lower_lr_profile,
                ),
                // Upper shallow slopes
                (
                    Vec3::new(sx + o - break_run, y_break, -o + break_run),
                    Vec3::new(mid_w, upper_slope, 0.0),
                    ufr,
                    RoofFaceSelector::UpperSlope,
                    upper_fb_profile.clone(),
                ),
                (
                    Vec3::new(-o + break_run, y_break, sz + o - break_run),
                    Vec3::new(mid_w, upper_slope, 0.0),
                    ubr,
                    RoofFaceSelector::UpperSlope,
                    upper_fb_profile,
                ),
                (
                    Vec3::new(-o + break_run, y_break, -o + break_run),
                    Vec3::new(mid_d, upper_slope, 0.0),
                    ulr,
                    RoofFaceSelector::UpperSlope,
                    upper_lr_profile.clone(),
                ),
                (
                    Vec3::new(sx + o - break_run, y_break, sz + o - break_run),
                    Vec3::new(mid_d, upper_slope, 0.0),
                    urr,
                    RoofFaceSelector::UpperSlope,
                    upper_lr_profile,
                ),
            ]
        }

        // ── Saltbox ───────────────────────────────────────────────────────────
        // Asymmetric Gable: ridge offset from front by `ridge_offset` fraction of depth.
        // Front slope is steeper (pitch = alpha); back slope angle derived from h and depth.
        RoofType::Saltbox => {
            let (orient_z, _width, depth) = if sx >= sz {
                (true, sx, sz)
            } else {
                (false, sz, sx)
            };
            let ridge_d = depth * config.ridge_offset;
            if !ridge_d.is_finite() || ridge_d <= 0.0 || ridge_d >= depth {
                return Err(ShapeError::InvalidNumericValue);
            }
            let h_s = ridge_d * tan_a;
            let back_depth = depth - ridge_d;
            let alpha_back = ((h_s) / (back_depth + o)).atan();
            let cos_ab = alpha_back.cos();
            if !h_s.is_finite() || !alpha_back.is_finite() || cos_ab < 1e-9 {
                return Err(ShapeError::InvalidNumericValue);
            }
            let front_len = (ridge_d + o) / cos_a;
            let back_len = (back_depth + o) / cos_ab;
            if !front_len.is_finite() || !back_len.is_finite() {
                return Err(ShapeError::InvalidNumericValue);
            }
            let (_, back_rot_s, _, right_rot_s) = roof_slope_rotations(alpha_back);
            let peak_fwd = ridge_d / depth;

            if orient_z {
                vec![
                    (
                        Vec3::new(sx + o, y_anchor, -o),
                        Vec3::new(sx + 2.0 * o, front_len, 0.0),
                        front_rot,
                        RoofFaceSelector::Slope,
                        FaceProfile::Rectangle,
                    ),
                    (
                        Vec3::new(-o, y_anchor, sz + o),
                        Vec3::new(sx + 2.0 * o, back_len, 0.0),
                        back_rot_s,
                        RoofFaceSelector::Slope,
                        FaceProfile::Rectangle,
                    ),
                    (
                        Vec3::new(0.0, 0.0, 0.0),
                        Vec3::new(sz, h_s, 0.0),
                        Quat::from_axis_angle(Vec3::Y, -FRAC_PI_2),
                        RoofFaceSelector::GableEnd,
                        FaceProfile::Triangle {
                            peak_offset: peak_fwd,
                        },
                    ),
                    (
                        Vec3::new(sx, 0.0, sz),
                        Vec3::new(sz, h_s, 0.0),
                        Quat::from_axis_angle(Vec3::Y, FRAC_PI_2),
                        RoofFaceSelector::GableEnd,
                        FaceProfile::Triangle {
                            peak_offset: 1.0 - peak_fwd,
                        },
                    ),
                ]
            } else {
                vec![
                    (
                        Vec3::new(-o, y_anchor, -o),
                        Vec3::new(sz + 2.0 * o, front_len, 0.0),
                        left_rot,
                        RoofFaceSelector::Slope,
                        FaceProfile::Rectangle,
                    ),
                    (
                        Vec3::new(sx + o, y_anchor, sz + o),
                        Vec3::new(sz + 2.0 * o, back_len, 0.0),
                        right_rot_s,
                        RoofFaceSelector::Slope,
                        FaceProfile::Rectangle,
                    ),
                    (
                        Vec3::new(sx, 0.0, 0.0),
                        Vec3::new(sx, h_s, 0.0),
                        Quat::from_axis_angle(Vec3::Y, PI),
                        RoofFaceSelector::GableEnd,
                        FaceProfile::Triangle {
                            peak_offset: 1.0 - peak_fwd,
                        },
                    ),
                    (
                        Vec3::new(0.0, 0.0, sz),
                        Vec3::new(sx, h_s, 0.0),
                        Quat::IDENTITY,
                        RoofFaceSelector::GableEnd,
                        FaceProfile::Triangle {
                            peak_offset: peak_fwd,
                        },
                    ),
                ]
            }
        }

        // ── Jerkinhead ────────────────────────────────────────────────────────
        // Gable with clipped-hip corners: main slopes are Trapezoid; small HipEnd triangles
        // fill the clipped gable-end corners.
        RoofType::Jerkinhead => {
            let tier = config.tier_height_or(0.25).clamp(0.01, 0.99);
            let orient_z = sx >= sz;
            let (width, depth) = if orient_z { (sx, sz) } else { (sz, sx) };

            let max_clip = (width / 2.0 + o).min(depth / 2.0);
            let clip_run = (tier * depth / 2.0).clamp(0.0, max_clip - 1e-3);

            let eave_w = width + 2.0 * o;
            let top_w = (eave_w - 2.0 * clip_run).max(0.0);

            let slope_len = (depth / 2.0 + o) / cos_a;

            let slope_profile = if top_w > 1e-9 {
                FaceProfile::Trapezoid {
                    top_width: top_w / eave_w,
                    offset_x: clip_run / eave_w,
                }
            } else {
                FaceProfile::Triangle { peak_offset: 0.5 }
            };

            let true_h = (depth / 2.0) * tan_a;
            let wall_h = (true_h - clip_run * tan_a).max(0.0);
            let wall_profile = if clip_run > 1e-9 {
                FaceProfile::Trapezoid {
                    top_width: (2.0 * clip_run) / depth,
                    offset_x: (depth / 2.0 - clip_run) / depth,
                }
            } else {
                FaceProfile::Triangle { peak_offset: 0.5 }
            };

            let hip_base_w = 2.0 * clip_run + 2.0 * o;
            let hip_slope_len = (clip_run + o) / cos_a;
            let hip_profile = FaceProfile::Triangle { peak_offset: 0.5 };

            if orient_z {
                let left_hip_origin = Vec3::new(-o, wall_h - o * tan_a, sz / 2.0 - clip_run - o);
                let right_hip_origin =
                    Vec3::new(sx + o, wall_h - o * tan_a, sz / 2.0 + clip_run + o);
                vec![
                    (
                        Vec3::new(sx + o, y_anchor, -o),
                        Vec3::new(eave_w, slope_len, 0.0),
                        front_rot,
                        RoofFaceSelector::Slope,
                        slope_profile.clone(),
                    ),
                    (
                        Vec3::new(-o, y_anchor, sz + o),
                        Vec3::new(eave_w, slope_len, 0.0),
                        back_rot,
                        RoofFaceSelector::Slope,
                        slope_profile,
                    ),
                    (
                        Vec3::new(0.0, 0.0, 0.0),
                        Vec3::new(sz, wall_h, 0.0),
                        Quat::from_axis_angle(Vec3::Y, -FRAC_PI_2),
                        RoofFaceSelector::GableEnd,
                        wall_profile.clone(),
                    ),
                    (
                        Vec3::new(sx, 0.0, sz),
                        Vec3::new(sz, wall_h, 0.0),
                        Quat::from_axis_angle(Vec3::Y, FRAC_PI_2),
                        RoofFaceSelector::GableEnd,
                        wall_profile,
                    ),
                    (
                        left_hip_origin,
                        Vec3::new(hip_base_w, hip_slope_len, 0.0),
                        left_rot,
                        RoofFaceSelector::HipEnd,
                        hip_profile.clone(),
                    ),
                    (
                        right_hip_origin,
                        Vec3::new(hip_base_w, hip_slope_len, 0.0),
                        right_rot,
                        RoofFaceSelector::HipEnd,
                        hip_profile,
                    ),
                ]
            } else {
                let front_hip_origin = Vec3::new(sx / 2.0 + clip_run + o, wall_h - o * tan_a, -o);
                let back_hip_origin =
                    Vec3::new(sx / 2.0 - clip_run - o, wall_h - o * tan_a, sz + o);
                vec![
                    (
                        Vec3::new(-o, y_anchor, -o),
                        Vec3::new(eave_w, slope_len, 0.0),
                        left_rot,
                        RoofFaceSelector::Slope,
                        slope_profile.clone(),
                    ),
                    (
                        Vec3::new(sx + o, y_anchor, sz + o),
                        Vec3::new(eave_w, slope_len, 0.0),
                        right_rot,
                        RoofFaceSelector::Slope,
                        slope_profile,
                    ),
                    (
                        Vec3::new(sx, 0.0, 0.0),
                        Vec3::new(sx, wall_h, 0.0),
                        Quat::from_axis_angle(Vec3::Y, PI),
                        RoofFaceSelector::GableEnd,
                        wall_profile.clone(),
                    ),
                    (
                        Vec3::new(0.0, 0.0, sz),
                        Vec3::new(sx, wall_h, 0.0),
                        Quat::IDENTITY,
                        RoofFaceSelector::GableEnd,
                        wall_profile,
                    ),
                    (
                        front_hip_origin,
                        Vec3::new(hip_base_w, hip_slope_len, 0.0),
                        front_rot,
                        RoofFaceSelector::HipEnd,
                        hip_profile.clone(),
                    ),
                    (
                        back_hip_origin,
                        Vec3::new(hip_base_w, hip_slope_len, 0.0),
                        back_rot,
                        RoofFaceSelector::HipEnd,
                        hip_profile,
                    ),
                ]
            }
        }

        // ── DutchGable ────────────────────────────────────────────────────────
        // Hip roof with a small gable rising from the ridge centre.
        // `tier_height` controls the fraction of the horizontal run used for the lower Hip portion.
        RoofType::DutchGable => {
            let tier = config.tier_height_or(0.7).clamp(0.01, 0.99);
            let orient_z = sx >= sz;
            let (width, depth) = if orient_z { (sx, sz) } else { (sz, sx) };

            let max_run = width.min(depth) / 2.0 + o;
            let break_run = (tier * max_run).clamp(o, max_run - 1e-3);

            if !break_run.is_finite() {
                return Err(ShapeError::InvalidNumericValue);
            }

            let y_break = y_anchor + break_run * tan_a;
            let eave_w = width + 2.0 * o;
            let eave_d = depth + 2.0 * o;

            let top_w = (eave_w - 2.0 * break_run).max(0.0);
            let top_d = (eave_d - 2.0 * break_run).max(0.0);

            let lower_slope_len = break_run / cos_a;

            let fb_profile = if top_w > 1e-9 {
                FaceProfile::Trapezoid {
                    top_width: top_w / eave_w,
                    offset_x: break_run / eave_w,
                }
            } else {
                FaceProfile::Triangle { peak_offset: 0.5 }
            };

            let lr_profile = if top_d > 1e-9 {
                FaceProfile::Trapezoid {
                    top_width: top_d / eave_d,
                    offset_x: break_run / eave_d,
                }
            } else {
                FaceProfile::Triangle { peak_offset: 0.5 }
            };

            let upper_run = (depth / 2.0 + o) - break_run;
            let upper_slope_len = upper_run / cos_a;
            let upper_h = upper_run * tan_a;

            if orient_z {
                vec![
                    // Lower Hip front/back
                    (
                        Vec3::new(sx + o, y_anchor, -o),
                        Vec3::new(eave_w, lower_slope_len, 0.0),
                        front_rot,
                        RoofFaceSelector::Slope,
                        fb_profile.clone(),
                    ),
                    (
                        Vec3::new(-o, y_anchor, sz + o),
                        Vec3::new(eave_w, lower_slope_len, 0.0),
                        back_rot,
                        RoofFaceSelector::Slope,
                        fb_profile,
                    ),
                    // Lower Hip left/right
                    (
                        Vec3::new(-o, y_anchor, -o),
                        Vec3::new(eave_d, lower_slope_len, 0.0),
                        left_rot,
                        RoofFaceSelector::Slope,
                        lr_profile.clone(),
                    ),
                    (
                        Vec3::new(sx + o, y_anchor, sz + o),
                        Vec3::new(eave_d, lower_slope_len, 0.0),
                        right_rot,
                        RoofFaceSelector::Slope,
                        lr_profile,
                    ),
                    // Upper Gable front/back
                    (
                        Vec3::new(sx + o - break_run, y_break, -o + break_run),
                        Vec3::new(top_w, upper_slope_len, 0.0),
                        front_rot,
                        RoofFaceSelector::Slope,
                        FaceProfile::Rectangle,
                    ),
                    (
                        Vec3::new(-o + break_run, y_break, sz + o - break_run),
                        Vec3::new(top_w, upper_slope_len, 0.0),
                        back_rot,
                        RoofFaceSelector::Slope,
                        FaceProfile::Rectangle,
                    ),
                    // Small gable ends (Left/Right)
                    (
                        Vec3::new(-o + break_run, y_break, -o + break_run),
                        Vec3::new(top_d, upper_h, 0.0),
                        Quat::from_axis_angle(Vec3::Y, -FRAC_PI_2),
                        RoofFaceSelector::GableEnd,
                        FaceProfile::Triangle { peak_offset: 0.5 },
                    ),
                    (
                        Vec3::new(sx + o - break_run, y_break, sz + o - break_run),
                        Vec3::new(top_d, upper_h, 0.0),
                        Quat::from_axis_angle(Vec3::Y, FRAC_PI_2),
                        RoofFaceSelector::GableEnd,
                        FaceProfile::Triangle { peak_offset: 0.5 },
                    ),
                ]
            } else {
                vec![
                    // Lower Hip left/right (which are the main slopes now)
                    (
                        Vec3::new(-o, y_anchor, -o),
                        Vec3::new(eave_w, lower_slope_len, 0.0),
                        left_rot,
                        RoofFaceSelector::Slope,
                        fb_profile.clone(),
                    ),
                    (
                        Vec3::new(sx + o, y_anchor, sz + o),
                        Vec3::new(eave_w, lower_slope_len, 0.0),
                        right_rot,
                        RoofFaceSelector::Slope,
                        fb_profile,
                    ),
                    // Lower Hip front/back (which are the gable ends now)
                    (
                        Vec3::new(sx + o, y_anchor, -o),
                        Vec3::new(eave_d, lower_slope_len, 0.0),
                        front_rot,
                        RoofFaceSelector::Slope,
                        lr_profile.clone(),
                    ),
                    (
                        Vec3::new(-o, y_anchor, sz + o),
                        Vec3::new(eave_d, lower_slope_len, 0.0),
                        back_rot,
                        RoofFaceSelector::Slope,
                        lr_profile,
                    ),
                    // Upper Gable left/right
                    (
                        Vec3::new(-o + break_run, y_break, -o + break_run),
                        Vec3::new(top_w, upper_slope_len, 0.0),
                        left_rot,
                        RoofFaceSelector::Slope,
                        FaceProfile::Rectangle,
                    ),
                    (
                        Vec3::new(sx + o - break_run, y_break, sz + o - break_run),
                        Vec3::new(top_w, upper_slope_len, 0.0),
                        right_rot,
                        RoofFaceSelector::Slope,
                        FaceProfile::Rectangle,
                    ),
                    // Small gable ends (Front/Back)
                    (
                        Vec3::new(sx + o - break_run, y_break, -o + break_run),
                        Vec3::new(top_d, upper_h, 0.0),
                        Quat::from_axis_angle(Vec3::Y, PI),
                        RoofFaceSelector::GableEnd,
                        FaceProfile::Triangle { peak_offset: 0.5 },
                    ),
                    (
                        Vec3::new(-o + break_run, y_break, sz + o - break_run),
                        Vec3::new(top_d, upper_h, 0.0),
                        Quat::IDENTITY,
                        RoofFaceSelector::GableEnd,
                        FaceProfile::Triangle { peak_offset: 0.5 },
                    ),
                ]
            }
        }
    };

    if queue.len() + panels.len() > MAX_QUEUE {
        return Err(ShapeError::CapacityOverflow);
    }
    for (local_off, face_size, rot_delta, selector, profile) in panels {
        let Some(rule) = find_roof_rule(selector, cases) else {
            continue;
        };
        // Skip degenerate panels (zero-area).
        if face_size.x < 1e-9 || face_size.y < 1e-9 {
            continue;
        }
        let face_pos = scope.position + scope.rotation * local_off;
        let face_rot = (scope.rotation * rot_delta).normalize();
        let face_scope = Scope::new(face_pos, face_rot, face_size);
        face_scope.validate()?;
        if model.len() + queue.len() >= max_terminals {
            return Err(ShapeError::CapacityOverflow);
        }
        queue.push_back(WorkItem {
            scope: face_scope,
            rule: rule.to_string(),
            depth: depth + 1,
            taper: 0.0,
            face_profile_override: Some(profile),
            material: material.clone(),
        });
    }

    Ok(())
}

// ── Stochastic selection ──────────────────────────────────────────────────────

fn select_variant<'a>(variants: &'a [WeightedVariant], rng: &mut Pcg64) -> &'a [ShapeOp] {
    if variants.is_empty() {
        return &[];
    }
    if variants.len() == 1 {
        return &variants[0].ops;
    }
    let total: f64 = variants.iter().map(|v| v.weight).sum();
    use rand::Rng;
    let r: f64 = rng.random::<f64>() * total;
    let mut acc = 0.0;
    for v in variants {
        acc += v.weight;
        if r < acc {
            return &v.ops;
        }
    }
    &variants.last().unwrap().ops
}

// ── Interpreter ───────────────────────────────────────────────────────────────

/// The CGA Shape Grammar derivation engine.
///
/// Rules are registered by name, then `derive` is called with a root scope and
/// root rule name. The engine expands rules breadth-first until every branch
/// terminates with an `I(mesh)` terminal.
///
/// Stochastic rules with multiple weighted variants use the engine's `seed` for
/// reproducible randomness — the same seed always yields the same building.
pub struct Interpreter {
    rules: HashMap<String, Vec<WeightedVariant>>,
    pub max_depth: usize,
    pub max_terminals: usize,
    /// Seed for stochastic rule selection. Default 0.
    pub seed: u64,
}

impl Default for Interpreter {
    fn default() -> Self {
        Self::new()
    }
}

impl Interpreter {
    pub fn new() -> Self {
        Self {
            rules: HashMap::new(),
            max_depth: MAX_DEPTH,
            max_terminals: MAX_TERMINALS,
            seed: 0,
        }
    }

    /// Returns a reference to the full rule table (rule name → weighted variants).
    pub fn rules(&self) -> &HashMap<String, Vec<WeightedVariant>> {
        &self.rules
    }

    /// Directly inserts a pre-built variant list for `name`, bypassing weight validation.
    ///
    /// Intended for restoring snapshots produced by [`ShapeGenotype::to_interpreter`].
    pub fn set_variants(&mut self, name: impl Into<String>, variants: Vec<WeightedVariant>) {
        self.rules.insert(name.into(), variants);
    }

    /// Registers a deterministic production rule.
    pub fn add_rule(&mut self, name: impl Into<String>, ops: Vec<ShapeOp>) {
        self.rules
            .insert(name.into(), vec![WeightedVariant { weight: 1.0, ops }]);
    }

    /// Registers a stochastic rule with multiple weighted alternatives.
    ///
    /// `variants` is a list of `(relative_weight, ops)` pairs. Weights need not
    /// sum to 1.0 — they are normalised internally during selection.
    ///
    /// Returns `Err(InvalidNumericValue)` if any weight is non-finite or negative.
    pub fn add_weighted_rules(
        &mut self,
        name: impl Into<String>,
        variants: Vec<(f64, Vec<ShapeOp>)>,
    ) -> Result<(), ShapeError> {
        for (weight, _) in &variants {
            if !weight.is_finite() || *weight < 0.0 {
                return Err(ShapeError::InvalidNumericValue);
            }
        }
        let wvs = variants
            .into_iter()
            .map(|(weight, ops)| WeightedVariant { weight, ops })
            .collect();
        self.rules.insert(name.into(), wvs);
        Ok(())
    }

    /// Returns true if a rule with `name` is registered.
    pub fn has_rule(&self, name: &str) -> bool {
        self.rules.contains_key(name)
    }

    /// Derives the shape model starting from `root_scope` and `root_rule`.
    ///
    /// Uses a breadth-first work queue to expand rules until all branches
    /// terminate via `I(mesh_id)` or an unknown rule name (implicit terminal).
    /// A fresh RNG seeded from `self.seed` is created for each call, making
    /// derivations reproducible for the same `seed` value.
    pub fn derive(
        &self,
        root_scope: Scope,
        root_rule: impl Into<String>,
    ) -> Result<ShapeModel, ShapeError> {
        root_scope.validate()?;

        let mut model = ShapeModel::new();
        let mut queue: VecDeque<WorkItem> = VecDeque::new();
        let mut rng = Pcg64::seed_from_u64(self.seed);

        queue.push_back(WorkItem {
            scope: root_scope,
            rule: root_rule.into(),
            depth: 0,
            taper: 0.0,
            face_profile_override: None,
            material: None,
        });

        while let Some(item) = queue.pop_front() {
            if queue.len() > MAX_QUEUE {
                return Err(ShapeError::CapacityOverflow);
            }
            if item.depth > self.max_depth {
                return Err(ShapeError::DepthLimitExceeded(self.max_depth));
            }

            let ops = match self.rules.get(&item.rule) {
                Some(variants) => select_variant(variants, &mut rng),
                None => {
                    // Unknown rule → implicit I(rule_name) terminal.
                    if model.len() >= self.max_terminals {
                        return Err(ShapeError::CapacityOverflow);
                    }
                    let profile = item
                        .face_profile_override
                        .unwrap_or_else(|| taper_to_profile(item.taper));
                    model.push(Terminal::new_profiled(
                        item.scope,
                        &item.rule,
                        profile,
                        item.material,
                    ));
                    continue;
                }
            };

            self.apply_ops(
                item.scope,
                item.taper,
                item.face_profile_override,
                item.material,
                ops,
                item.depth,
                &mut queue,
                &mut model,
            )?;
        }

        Ok(model)
    }

    /// Processes the ops sequence for a single rule invocation.
    ///
    /// Transformation ops (`Extrude`, `Scale`, etc.) mutate `scope` in place.
    /// The first branching op (`Split`, `Comp`, `Repeat`) or terminal op
    /// (`I`, `Rule`) ends the sequence by pushing new work items.
    #[allow(clippy::too_many_arguments)]
    fn apply_ops(
        &self,
        initial_scope: Scope,
        initial_taper: f64,
        initial_face_profile: Option<FaceProfile>,
        initial_material: Option<String>,
        ops: &[ShapeOp],
        depth: usize,
        queue: &mut VecDeque<WorkItem>,
        model: &mut ShapeModel,
    ) -> Result<(), ShapeError> {
        let mut scope = initial_scope;
        let mut taper = initial_taper;
        let mut face_profile = initial_face_profile;
        let mut material = initial_material;

        for op in ops {
            match op {
                // ── Transformations ───────────────────────────────────────
                ShapeOp::Extrude(h) => {
                    if !h.is_finite() || *h <= 0.0 {
                        return Err(ShapeError::InvalidNumericValue);
                    }
                    // Face scopes from Comp(Faces) have size.z == 0 (the outward-normal
                    // direction) and a non-zero size.y (the face height).  Extruding a
                    // face scope should push it outward along the normal (local Z), not
                    // collapse the height by overwriting size.y.
                    // Footprint scopes have size.y == 0; Extrude gives them their height.
                    if scope.size.z.abs() < 1e-9 && scope.size.y.abs() > 1e-9 {
                        scope.size.z = *h;
                    } else {
                        scope.size.y = *h;
                    }
                }

                ShapeOp::Taper(amount) => {
                    if !amount.is_finite() {
                        return Err(ShapeError::InvalidNumericValue);
                    }
                    taper = amount.clamp(0.0, 1.0);
                }

                ShapeOp::Rotate(q) => {
                    if !q.is_finite() {
                        return Err(ShapeError::InvalidNumericValue);
                    }
                    // Reject degenerate (near-zero) quaternions that cannot represent a
                    // rotation. Normalize non-unit inputs so that glam's fast-path
                    // `rotation * vec` (which assumes a unit quaternion) is correct.
                    let len_sq = q.length_squared();
                    if !len_sq.is_finite() || len_sq < 1e-12 {
                        return Err(ShapeError::InvalidNumericValue);
                    }
                    scope.rotation = (scope.rotation * q.normalize()).normalize();
                }

                ShapeOp::Translate(v) => {
                    if !v.is_finite() {
                        return Err(ShapeError::InvalidNumericValue);
                    }
                    scope.position += scope.rotation * *v;
                    // Two individually-finite values can add to INFINITY
                    // (e.g. f64::MAX/2 + f64::MAX/2). Catch the overflow here.
                    if !scope.position.is_finite() {
                        return Err(ShapeError::InvalidNumericValue);
                    }
                }

                ShapeOp::Scale(v) => {
                    if !v.is_finite() || v.x <= 0.0 || v.y <= 0.0 || v.z <= 0.0 {
                        return Err(ShapeError::InvalidNumericValue);
                    }
                    scope.size *= *v;
                    // Two individually-finite scale values can multiply to INFINITY
                    // (e.g. 1e200 * 1e200). Catch the overflow here before it
                    // propagates into Split/Repeat and causes NaN via ∞ − ∞.
                    if !scope.size.is_finite() {
                        return Err(ShapeError::InvalidNumericValue);
                    }
                }

                ShapeOp::Mat(mat_id) => {
                    material = Some(mat_id.clone());
                }

                // ── Transform: Align ─────────────────────────────────────
                ShapeOp::Align { local_axis, target } => {
                    // length_squared() can overflow to INFINITY for large-but-finite
                    // vectors (e.g. (1e200, 1e200, 1e200)); INFINITY > 1e-12 so the
                    // naive check would pass, then normalize() divides by INFINITY
                    // yielding a zero vector and a silent no-op rotation.
                    let len_sq = target.length_squared();
                    if !target.is_finite() || !len_sq.is_finite() || len_sq < 1e-12 {
                        return Err(ShapeError::InvalidAlignTarget);
                    }
                    let target_norm = target.normalize();
                    let current = scope.rotation * axis_vec(*local_axis);
                    // from_rotation_arc gives the shortest-arc rotation; it degenerates
                    // when vectors are antiparallel — handle that with a fallback 180°.
                    let dot = current.dot(target_norm);
                    let q = if (dot + 1.0).abs() < 1e-9 {
                        // Choose the cardinal axis least parallel to `current` (smallest
                        // absolute component) to form the cross product. This avoids the
                        // discontinuous snap caused by a hard threshold: the selection
                        // only changes when two components are exactly equal, which is
                        // rare and well-conditioned.
                        let perp = if current.x.abs() <= current.y.abs()
                            && current.x.abs() <= current.z.abs()
                        {
                            current.cross(Vec3::X).normalize()
                        } else if current.y.abs() <= current.z.abs() {
                            current.cross(Vec3::Y).normalize()
                        } else {
                            current.cross(Vec3::Z).normalize()
                        };
                        Quat::from_axis_angle(perp, PI)
                    } else {
                        Quat::from_rotation_arc(current, target_norm)
                    };
                    scope.rotation = (q * scope.rotation).normalize();
                }

                // ── Branching: Offset ─────────────────────────────────────
                ShapeOp::Offset { distance, cases } => {
                    if !distance.is_finite() {
                        return Err(ShapeError::InvalidNumericValue);
                    }
                    // Negative distance = inset.
                    let inset = -*distance;
                    if inset <= 0.0 {
                        return Err(ShapeError::InvalidNumericValue);
                    }
                    let sx = scope.size.x;
                    let sy = scope.size.y;
                    let inside_w = sx - 2.0 * inset;
                    let inside_h = sy - 2.0 * inset;
                    // Explicit NaN guard: if sx/sy are non-finite (e.g. leaked
                    // Infinity from an upstream op), the subtraction produces NaN,
                    // which compares false for `< 0.0` and would bypass the check.
                    if !inside_w.is_finite()
                        || !inside_h.is_finite()
                        || inside_w < 0.0
                        || inside_h < 0.0
                    {
                        return Err(ShapeError::OffsetTooLarge);
                    }
                    if let Some(rule) = find_offset_rule(OffsetSelector::Inside, cases) {
                        if queue.len() >= MAX_QUEUE {
                            return Err(ShapeError::CapacityOverflow);
                        }
                        let pos = scope.position + scope.rotation * Vec3::new(inset, inset, 0.0);
                        let child_scope =
                            Scope::new(pos, scope.rotation, Vec3::new(inside_w, inside_h, 0.0));
                        child_scope.validate()?;
                        queue.push_back(WorkItem {
                            scope: child_scope,
                            rule: rule.to_string(),
                            depth: depth + 1,
                            taper: 0.0,
                            face_profile_override: None,
                            material: material.clone(),
                        });
                    }
                    if let Some(rule) = find_offset_rule(OffsetSelector::Border, cases) {
                        // 4 surrounding strips: bottom, top, left, right.
                        let strips = [
                            (Vec3::new(0.0, 0.0, 0.0), Vec3::new(sx, inset, 0.0)),
                            (Vec3::new(0.0, sy - inset, 0.0), Vec3::new(sx, inset, 0.0)),
                            (
                                Vec3::new(0.0, inset, 0.0),
                                Vec3::new(inset, sy - 2.0 * inset, 0.0),
                            ),
                            (
                                Vec3::new(sx - inset, inset, 0.0),
                                Vec3::new(inset, sy - 2.0 * inset, 0.0),
                            ),
                        ];
                        if queue.len() + strips.len() > MAX_QUEUE {
                            return Err(ShapeError::CapacityOverflow);
                        }
                        for (local_off, strip_size) in strips {
                            let pos = scope.position + scope.rotation * local_off;
                            let child_scope = Scope::new(pos, scope.rotation, strip_size);
                            child_scope.validate()?;
                            queue.push_back(WorkItem {
                                scope: child_scope,
                                rule: rule.to_string(),
                                depth: depth + 1,
                                taper: 0.0,
                                face_profile_override: None,
                                material: material.clone(),
                            });
                        }
                    }
                    return Ok(());
                }

                // ── Branching: Roof ───────────────────────────────────────
                ShapeOp::Roof { config, cases } => {
                    apply_roof(
                        config,
                        cases,
                        &scope,
                        depth,
                        &material,
                        queue,
                        model,
                        self.max_terminals,
                    )?;
                    return Ok(());
                }

                // ── Branching: Attach ─────────────────────────────────────
                ShapeOp::Attach { world_axis, cases } => {
                    let len_sq = world_axis.length_squared();
                    if !world_axis.is_finite() || !len_sq.is_finite() || len_sq < 1e-12 {
                        return Err(ShapeError::InvalidAlignTarget);
                    }
                    let axis_norm = world_axis.normalize();
                    // Build a new scope whose Y axis = world_axis.
                    // The new scope sits at the same corner as the current scope,
                    // has X = scope.size.x, Y = scope.size.y, Z = 0 (flat surface).
                    let rot = Quat::from_rotation_arc(Vec3::Y, axis_norm);
                    let attach_scope = Scope::new(
                        scope.position,
                        rot.normalize(),
                        Vec3::new(scope.size.x, scope.size.y, 0.0),
                    );
                    attach_scope.validate()?;
                    if let Some(rule) = find_attach_rule(crate::ops::AttachSelector::Surface, cases)
                    {
                        if queue.len() >= MAX_QUEUE {
                            return Err(ShapeError::CapacityOverflow);
                        }
                        queue.push_back(WorkItem {
                            scope: attach_scope,
                            rule: rule.to_string(),
                            depth: depth + 1,
                            taper: 0.0,
                            face_profile_override: None,
                            material: material.clone(),
                        });
                    }
                    return Ok(());
                }

                // ── Branching: Split ──────────────────────────────────────
                ShapeOp::Split { axis, slots } => {
                    let total = match axis {
                        Axis::X => scope.size.x,
                        Axis::Y => scope.size.y,
                        Axis::Z => scope.size.z,
                    };
                    let sizes = resolve_split_sizes(slots, total)?;
                    if queue.len() + slots.len() > MAX_QUEUE {
                        return Err(ShapeError::CapacityOverflow);
                    }
                    let mut offset = 0.0;
                    for (slot, size) in slots.iter().zip(sizes.iter()) {
                        let child = slice_scope(&scope, *axis, offset, *size);
                        child.validate()?;
                        queue.push_back(WorkItem {
                            scope: child,
                            rule: slot.rule.clone(),
                            depth: depth + 1,
                            taper: 0.0,
                            face_profile_override: None,
                            material: material.clone(),
                        });
                        offset += size;
                    }
                    return Ok(());
                }

                // ── Branching: Repeat ─────────────────────────────────────
                //
                // Uses `floor()` for tile count (never fewer tiles than fit),
                // then stretches actual tile size to fill the scope with no gaps.
                // Example: 10.5m scope / 2m target → 5 tiles × 2.1m each.
                ShapeOp::Repeat {
                    axis,
                    tile_size,
                    rule,
                } => {
                    if !tile_size.is_finite() || *tile_size <= 0.0 {
                        return Err(ShapeError::InvalidNumericValue);
                    }
                    let total = match axis {
                        Axis::X => scope.size.x,
                        Axis::Y => scope.size.y,
                        Axis::Z => scope.size.z,
                    };
                    // Defensive: scope.size should always be finite after earlier
                    // checks, but if an Infinity scope size ever sneaks through
                    // (e.g. from a Roof child), `0.0 * Infinity = NaN` at i=0.
                    if !total.is_finite() || total <= 0.0 {
                        return Err(ShapeError::InvalidNumericValue);
                    }
                    // `total / tile_size` can overflow to INFINITY for very small
                    // tile_size values (e.g. f64::MIN_POSITIVE). The subsequent
                    // `as usize` cast would saturate to usize::MAX, and
                    // `queue.len() + usize::MAX` wraps to 0 in release mode,
                    // bypassing the capacity guard and looping usize::MAX times.
                    let n_tiles_f = (total / tile_size).floor();
                    if !n_tiles_f.is_finite() {
                        return Err(ShapeError::CapacityOverflow);
                    }
                    let n_tiles = n_tiles_f as usize;
                    if queue.len().saturating_add(n_tiles) > MAX_QUEUE {
                        return Err(ShapeError::CapacityOverflow);
                    }
                    if n_tiles > 0 {
                        let actual_size = total / n_tiles as f64;
                        for i in 0..n_tiles {
                            let offset = i as f64 * actual_size;
                            let child = slice_scope(&scope, *axis, offset, actual_size);
                            child.validate()?;
                            queue.push_back(WorkItem {
                                scope: child,
                                rule: rule.clone(),
                                depth: depth + 1,
                                taper: 0.0,
                                face_profile_override: None,
                                material: material.clone(),
                            });
                        }
                    }
                    return Ok(());
                }

                // ── Branching: Comp ───────────────────────────────────────
                //
                // Each face scope is properly oriented so that local Z points
                // along the outward face normal. Rules can then use Split(X/Y)
                // or Repeat(X) naturally on any face of the parent volume.
                ShapeOp::Comp(CompTarget::Faces(cases)) => {
                    // face_descs always returns exactly 6 faces; guard before any push.
                    if queue.len() + 6 > MAX_QUEUE {
                        return Err(ShapeError::CapacityOverflow);
                    }
                    for (selector, offset_local, face_size, rot_delta) in face_descs(scope.size) {
                        let rule = match find_face_rule(selector, cases) {
                            Some(r) => r,
                            None => continue,
                        };
                        let face_pos = scope.position + scope.rotation * offset_local;
                        let face_rotation = scope.rotation * rot_delta;
                        let face_scope = Scope::new(face_pos, face_rotation, face_size);
                        face_scope.validate()?;
                        queue.push_back(WorkItem {
                            scope: face_scope,
                            rule: rule.to_string(),
                            depth: depth + 1,
                            taper: 0.0,
                            face_profile_override: None,
                            material: material.clone(),
                        });
                    }
                    return Ok(());
                }

                // ── Terminal: mesh instance ───────────────────────────────
                ShapeOp::I(mesh_id) => {
                    if model.len() >= self.max_terminals {
                        return Err(ShapeError::CapacityOverflow);
                    }
                    let profile = face_profile
                        .take()
                        .unwrap_or_else(|| taper_to_profile(taper));
                    model.push(Terminal::new_profiled(scope, mesh_id, profile, material));
                    return Ok(());
                }

                // ── Delegate: named sub-rule ──────────────────────────────
                ShapeOp::Rule(name) => {
                    queue.push_back(WorkItem {
                        scope,
                        rule: name.clone(),
                        depth: depth + 1,
                        taper,
                        face_profile_override: face_profile,
                        material,
                    });
                    return Ok(());
                }
            }
        }

        // Ops exhausted without a terminal — scope is silently discarded
        // (matches CGA "delete this shape" semantics for empty successors).
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::{Axis, SplitSize, SplitSlot};
    use crate::scope::{Quat, Vec3};

    fn slot(size: SplitSize, rule: &str) -> SplitSlot {
        SplitSlot {
            size,
            rule: rule.to_string(),
        }
    }

    #[test]
    fn test_resolve_split_absolute() {
        let slots = vec![
            slot(SplitSize::Absolute(3.0), "A"),
            slot(SplitSize::Absolute(7.0), "B"),
        ];
        let sizes = resolve_split_sizes(&slots, 10.0).unwrap();
        assert!((sizes[0] - 3.0).abs() < 1e-9);
        assert!((sizes[1] - 7.0).abs() < 1e-9);
    }

    #[test]
    fn test_resolve_split_floating_equal() {
        let slots = vec![
            slot(SplitSize::Floating(1.0), "A"),
            slot(SplitSize::Floating(1.0), "B"),
        ];
        let sizes = resolve_split_sizes(&slots, 10.0).unwrap();
        assert!((sizes[0] - 5.0).abs() < 1e-9);
        assert!((sizes[1] - 5.0).abs() < 1e-9);
    }

    #[test]
    fn test_resolve_split_mixed() {
        let slots = vec![
            slot(SplitSize::Absolute(2.0), "Base"),
            slot(SplitSize::Floating(1.0), "A"),
            slot(SplitSize::Floating(1.0), "B"),
        ];
        let sizes = resolve_split_sizes(&slots, 10.0).unwrap();
        assert!((sizes[0] - 2.0).abs() < 1e-9);
        assert!((sizes[1] - 4.0).abs() < 1e-9);
        assert!((sizes[2] - 4.0).abs() < 1e-9);
    }

    #[test]
    fn test_resolve_split_overflow_rejected() {
        let slots = vec![
            slot(SplitSize::Absolute(6.0), "A"),
            slot(SplitSize::Absolute(6.0), "B"),
        ];
        assert!(matches!(
            resolve_split_sizes(&slots, 10.0),
            Err(ShapeError::SplitOverflow(_))
        ));
    }

    #[test]
    fn test_derive_extrude_then_terminal() {
        let mut interp = Interpreter::new();
        interp.add_rule(
            "Lot",
            vec![ShapeOp::Extrude(10.0), ShapeOp::I("Building".to_string())],
        );
        let scope = Scope::unit();
        let model = interp.derive(scope, "Lot").unwrap();
        assert_eq!(model.len(), 1);
        assert_eq!(model.terminals[0].mesh_id, "Building");
        assert!((model.terminals[0].scope.size.y - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_derive_split_y_three_floors() {
        let mut interp = Interpreter::new();
        interp.add_rule(
            "Building",
            vec![ShapeOp::Split {
                axis: Axis::Y,
                slots: vec![
                    slot(SplitSize::Absolute(2.0), "Ground"),
                    slot(SplitSize::Floating(1.0), "Upper"),
                    slot(SplitSize::Absolute(1.5), "Roof"),
                ],
            }],
        );
        let scope = Scope::new(Vec3::ZERO, Quat::IDENTITY, Vec3::new(10.0, 10.0, 10.0));
        let model = interp.derive(scope, "Building").unwrap();
        assert_eq!(model.len(), 3);
        assert!((model.terminals[0].scope.size.y - 2.0).abs() < 1e-9);
        assert!((model.terminals[1].scope.size.y - 6.5).abs() < 1e-9);
        assert!((model.terminals[2].scope.size.y - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_derive_depth_limit() {
        let mut interp = Interpreter::new();
        interp.add_rule("A", vec![ShapeOp::Rule("A".to_string())]);
        interp.max_depth = 5;
        let model = interp.derive(Scope::unit(), "A");
        assert!(matches!(model, Err(ShapeError::DepthLimitExceeded(_))));
    }

    #[test]
    fn test_derive_comp_faces() {
        let mut interp = Interpreter::new();
        interp.add_rule(
            "Box",
            vec![ShapeOp::Comp(CompTarget::Faces(vec![
                crate::ops::CompFaceCase {
                    selector: FaceSelector::Top,
                    rule: "Roof".to_string(),
                },
                crate::ops::CompFaceCase {
                    selector: FaceSelector::Side,
                    rule: "Wall".to_string(),
                },
                crate::ops::CompFaceCase {
                    selector: FaceSelector::Bottom,
                    rule: "Base".to_string(),
                },
            ]))],
        );
        let scope = Scope::new(Vec3::ZERO, Quat::IDENTITY, Vec3::new(5.0, 3.0, 5.0));
        let model = interp.derive(scope, "Box").unwrap();
        assert_eq!(model.len(), 6);
    }

    #[test]
    fn test_derive_repeat() {
        let mut interp = Interpreter::new();
        interp.add_rule(
            "Facade",
            vec![ShapeOp::Repeat {
                axis: Axis::X,
                tile_size: 2.0,
                rule: "Window".to_string(),
            }],
        );
        let scope = Scope::new(Vec3::ZERO, Quat::IDENTITY, Vec3::new(10.0, 4.0, 0.0));
        let model = interp.derive(scope, "Facade").unwrap();
        // 10 / 2 = 5 tiles, each stretched to exactly 2.0m (no remainder here)
        assert_eq!(model.len(), 5);
    }

    #[test]
    fn test_derive_mat_propagates() {
        let mut interp = Interpreter::new();
        interp.add_rule(
            "R",
            vec![
                ShapeOp::Mat("Brick".to_string()),
                ShapeOp::I("Wall".to_string()),
            ],
        );
        let model = interp.derive(Scope::unit(), "R").unwrap();
        assert_eq!(model.terminals[0].material, Some("Brick".to_string()));
    }

    // ── Issue 1: empty-variants panic ─────────────────────────────────────────

    #[test]
    fn test_empty_variants_discards_shape() {
        let mut interp = Interpreter::new();
        // add_weighted_rules with an empty vec must not panic; the scope is
        // silently discarded (consistent with CGA "delete shape" semantics).
        interp.add_weighted_rules("Empty", vec![]).unwrap();
        let model = interp.derive(Scope::unit(), "Empty").unwrap();
        assert_eq!(model.len(), 0);
    }

    // ── Issue 1 (review #10): n_tiles INFINITY cast ───────────────────────────

    #[test]
    fn test_repeat_tiny_tile_size_rejected() {
        // tile_size = f64::MIN_POSITIVE is finite and > 0, passes validation.
        // But total / f64::MIN_POSITIVE overflows to INFINITY, and
        // INFINITY as usize saturates to usize::MAX, causing overflow in the
        // queue length arithmetic. Must be caught as CapacityOverflow.
        let mut interp = Interpreter::new();
        interp.add_rule(
            "R",
            vec![ShapeOp::Repeat {
                axis: Axis::X,
                tile_size: f64::MIN_POSITIVE,
                rule: "Tile".to_string(),
            }],
        );
        let scope = Scope::new(Vec3::ZERO, Quat::IDENTITY, Vec3::new(1.0, 1.0, 1.0));
        assert!(matches!(
            interp.derive(scope, "R"),
            Err(ShapeError::CapacityOverflow)
        ));
    }

    // ── Issue 3 (review #10): Scale multiplication overflow ───────────────────

    #[test]
    fn test_scale_multiply_overflow_to_infinity_rejected() {
        // Each Scale value is individually finite and positive, but scope.size *= v
        // can overflow to INFINITY. Must be caught after the multiplication.
        let mut interp = Interpreter::new();
        interp.add_rule(
            "R",
            vec![
                ShapeOp::Scale(Vec3::new(1e200, 1.0, 1.0)),
                ShapeOp::Scale(Vec3::new(1e200, 1.0, 1.0)), // 1e200*1e200=INFINITY
                ShapeOp::I("Mesh".to_string()),
            ],
        );
        let scope = Scope::new(Vec3::ZERO, Quat::IDENTITY, Vec3::new(1.0, 1.0, 1.0));
        assert!(matches!(
            interp.derive(scope, "R"),
            Err(ShapeError::InvalidNumericValue)
        ));
    }

    #[test]
    fn test_split_absolute_sum_overflow_rejected() {
        // Absolute slot sizes whose sum overflows to INFINITY must be rejected.
        let slots = vec![
            slot(SplitSize::Absolute(f64::MAX), "A"),
            slot(SplitSize::Absolute(f64::MAX), "B"),
        ];
        assert!(matches!(
            resolve_split_sizes(&slots, f64::MAX),
            Err(ShapeError::InvalidNumericValue)
        ));
    }

    // ── Issue 3: queue capacity accounting ────────────────────────────────────

    #[test]
    fn test_repeat_respects_combined_queue_limit() {
        // A Repeat whose tile count alone is fine (< MAX_QUEUE) but combined with
        // the existing queue would exceed MAX_QUEUE should be rejected.
        // We can't easily fill the queue to 99_999 in a unit test, so we use
        // the public max_depth / max_terminals to drive overflow indirectly.
        // Instead, verify the guard fires for a very large n_tiles (> MAX_QUEUE).
        let mut interp = Interpreter::new();
        // tile_size so small that n_tiles >> MAX_QUEUE (scope is 1e10, tile = 1e-1 → 1e11 tiles)
        interp.add_rule(
            "Big",
            vec![ShapeOp::Repeat {
                axis: Axis::X,
                tile_size: 1e-1,
                rule: "Tile".to_string(),
            }],
        );
        let scope = Scope::new(Vec3::ZERO, Quat::IDENTITY, Vec3::new(1e10, 1.0, 1.0));
        assert!(matches!(
            interp.derive(scope, "Big"),
            Err(ShapeError::CapacityOverflow)
        ));
    }

    // ── Issue 4: negative scale via API ──────────────────────────────────────

    #[test]
    fn test_api_negative_scale_rejected() {
        let mut interp = Interpreter::new();
        interp.add_rule(
            "R",
            vec![
                ShapeOp::Scale(Vec3::new(-1.0, 1.0, 1.0)),
                ShapeOp::I("Mesh".to_string()),
            ],
        );
        assert!(matches!(
            interp.derive(Scope::unit(), "R"),
            Err(ShapeError::InvalidNumericValue)
        ));
    }

    #[test]
    fn test_api_zero_scale_rejected() {
        let mut interp = Interpreter::new();
        interp.add_rule(
            "R",
            vec![
                ShapeOp::Scale(Vec3::new(0.0, 1.0, 1.0)),
                ShapeOp::I("Mesh".to_string()),
            ],
        );
        assert!(matches!(
            interp.derive(Scope::unit(), "R"),
            Err(ShapeError::InvalidNumericValue)
        ));
    }

    // ── Issue 1 (review #14): intermediate product overflow in floating split ──

    #[test]
    fn test_split_floating_large_remaining_no_overflow() {
        // remaining ≈ 1e308, w = 2.0, float_weight_total = 3.0.
        // Old code: (1e308 * 2.0) / 3.0 = INFINITY / 3.0 = INFINITY.
        // Fixed:    1e308 * (2.0 / 3.0) = finite.
        let slots = vec![
            slot(SplitSize::Floating(2.0), "A"),
            slot(SplitSize::Floating(1.0), "B"),
        ];
        let sizes = resolve_split_sizes(&slots, 1e308).unwrap();
        assert!(sizes[0].is_finite(), "size[0] overflowed to {}", sizes[0]);
        assert!(sizes[1].is_finite(), "size[1] overflowed to {}", sizes[1]);
        // Proportions must be 2/3 and 1/3.
        assert!((sizes[0] / sizes[1] - 2.0).abs() < 1e-6);
    }

    // ── Issue 5: float_weight_total overflow ──────────────────────────────────

    #[test]
    fn test_split_floating_weight_overflow_rejected() {
        // Two floating slots each with weight near f64::MAX; their sum overflows
        // to INFINITY in float_weight_total, which should be caught and rejected.
        let slots = vec![
            slot(SplitSize::Floating(f64::MAX), "A"),
            slot(SplitSize::Floating(f64::MAX), "B"),
        ];
        assert!(matches!(
            resolve_split_sizes(&slots, 10.0),
            Err(ShapeError::InvalidNumericValue)
        ));
    }

    #[test]
    fn test_stochastic_rule_deterministic_with_seed() {
        let mut interp = Interpreter::new();
        interp
            .add_weighted_rules(
                "Facade",
                vec![
                    (70.0, vec![ShapeOp::I("Brick".to_string())]),
                    (30.0, vec![ShapeOp::I("Glass".to_string())]),
                ],
            )
            .unwrap();
        interp.seed = 42;
        // Same seed → same result
        let m1 = interp.derive(Scope::unit(), "Facade").unwrap();
        let m2 = interp.derive(Scope::unit(), "Facade").unwrap();
        assert_eq!(m1.terminals[0].mesh_id, m2.terminals[0].mesh_id);
    }

    #[test]
    fn test_face_comp_orientations() {
        // After Comp, each face scope should have local Z pointing along its outward normal.
        // We verify by checking the rotation: applying the face rotation to (0,0,1) should
        // give the expected world-space normal direction.
        let mut interp = Interpreter::new();
        interp.add_rule(
            "Box",
            vec![ShapeOp::Comp(CompTarget::Faces(vec![
                crate::ops::CompFaceCase {
                    selector: FaceSelector::All,
                    rule: "Face".to_string(),
                },
            ]))],
        );
        let scope = Scope::new(Vec3::ZERO, Quat::IDENTITY, Vec3::new(4.0, 3.0, 2.0));
        let model = interp.derive(scope, "Box").unwrap();
        assert_eq!(model.len(), 6);

        // Collect the outward normals by rotating (0,0,1) with each face's rotation
        let normals: Vec<Vec3> = model
            .terminals
            .iter()
            .map(|t| t.scope.rotation * Vec3::Z)
            .collect();

        // We expect exactly one terminal pointing in each of the 6 cardinal directions
        let expected = [
            Vec3::NEG_Y, // Bottom
            Vec3::Y,     // Top
            Vec3::NEG_Z, // Front
            Vec3::Z,     // Back
            Vec3::NEG_X, // Left
            Vec3::X,     // Right
        ];
        for exp in &expected {
            assert!(
                normals.iter().any(|n| (*n - *exp).length() < 1e-6),
                "missing normal {:?}, got {:?}",
                exp,
                normals
            );
        }

        // face_descs order is deterministic: Bottom, Top, Front, Back, Left, Right.
        // Verify that the face origin positions lie on the correct parent faces.
        // scope: position=(0,0,0), size sx=4, sy=3, sz=2.
        let pos = |i: usize| model.terminals[i].scope.position;
        assert!(
            (pos(0) - Vec3::new(0.0, 0.0, 0.0)).length() < 1e-6,
            "Bottom pos"
        ); // at y=0
        assert!(
            (pos(1) - Vec3::new(0.0, 3.0, 2.0)).length() < 1e-6,
            "Top pos"
        ); // at y=sy, origin shifted to (0,sy,sz)
        assert!(
            (pos(2) - Vec3::new(4.0, 0.0, 0.0)).length() < 1e-6,
            "Front pos"
        ); // at z=0, origin shifted to (sx,0,0)
        assert!(
            (pos(3) - Vec3::new(0.0, 0.0, 2.0)).length() < 1e-6,
            "Back pos"
        ); // at z=sz
        assert!(
            (pos(4) - Vec3::new(0.0, 0.0, 0.0)).length() < 1e-6,
            "Left pos"
        ); // at x=0
        assert!(
            (pos(5) - Vec3::new(4.0, 0.0, 2.0)).length() < 1e-6,
            "Right pos"
        ); // at x=sx, origin shifted to (sx,0,sz)
    }

    // ── Issue 4 (review #14): negative scope size rejected by validate() ────────

    #[test]
    fn test_negative_scope_size_rejected() {
        let scope = Scope::new(Vec3::ZERO, Quat::IDENTITY, Vec3::new(-1.0, 1.0, 1.0));
        let interp = Interpreter::new();
        assert!(matches!(
            interp.derive(scope, "Anything"),
            Err(ShapeError::InvalidNumericValue)
        ));
    }

    #[test]
    fn test_zero_scope_size_accepted() {
        // Y=0 is a valid 2D footprint; derive should succeed (rule unknown → implicit terminal).
        let scope = Scope::new(Vec3::ZERO, Quat::IDENTITY, Vec3::new(10.0, 0.0, 10.0));
        let interp = Interpreter::new();
        let model = interp.derive(scope, "Footprint").unwrap();
        assert_eq!(model.len(), 1);
    }

    // ── Issue 1 (review #11): unnormalized quaternion in root scope ───────────

    #[test]
    fn test_unnormalized_root_quat_rejected() {
        // DQuat::from_xyzw(2,0,0,0) is finite but has length 2 — not a unit quat.
        let bad_q = Quat::from_xyzw(0.0, 0.0, 0.0, 2.0);
        let scope = Scope::new(Vec3::ZERO, bad_q, Vec3::ONE);
        let interp = Interpreter::new();
        assert!(matches!(
            interp.derive(scope, "Anything"),
            Err(ShapeError::InvalidNumericValue)
        ));
    }

    #[test]
    fn test_degenerate_rotate_op_rejected() {
        // A zero quaternion (len_sq < 1e-12) cannot represent a rotation — must reject.
        let zero_q = Quat::from_xyzw(0.0, 0.0, 0.0, 0.0);
        let mut interp = Interpreter::new();
        interp.add_rule(
            "R",
            vec![ShapeOp::Rotate(zero_q), ShapeOp::I("M".to_string())],
        );
        assert!(matches!(
            interp.derive(Scope::unit(), "R"),
            Err(ShapeError::InvalidNumericValue)
        ));
    }

    #[test]
    fn test_scaled_rotate_op_normalized() {
        // A quaternion with magnitude 2 (e.g. IDENTITY * 2) is non-unit but valid;
        // it must be normalised to IDENTITY rather than rejected.
        let scaled_q = Quat::from_xyzw(0.0, 0.0, 0.0, 2.0); // IDENTITY * 2
        let mut interp = Interpreter::new();
        interp.add_rule(
            "R",
            vec![ShapeOp::Rotate(scaled_q), ShapeOp::I("M".to_string())],
        );
        // Should succeed; the terminal scope rotation should be IDENTITY.
        let model = interp.derive(Scope::unit(), "R").unwrap();
        assert_eq!(model.len(), 1);
        let r = model.terminals[0].scope.rotation;
        assert!(
            (r.length_squared() - 1.0).abs() < 1e-9,
            "rotation should be unit"
        );
    }

    // ── Issue 2 (review #12): invalid weights in add_weighted_rules ──────────

    #[test]
    fn test_nan_weight_rejected() {
        let mut interp = Interpreter::new();
        assert!(matches!(
            interp.add_weighted_rules("R", vec![(f64::NAN, vec![ShapeOp::I("M".to_string())])]),
            Err(ShapeError::InvalidNumericValue)
        ));
    }

    #[test]
    fn test_infinite_weight_rejected() {
        let mut interp = Interpreter::new();
        assert!(matches!(
            interp.add_weighted_rules(
                "R",
                vec![(f64::INFINITY, vec![ShapeOp::I("M".to_string())])]
            ),
            Err(ShapeError::InvalidNumericValue)
        ));
    }

    #[test]
    fn test_negative_weight_rejected() {
        let mut interp = Interpreter::new();
        assert!(matches!(
            interp.add_weighted_rules("R", vec![(-1.0, vec![ShapeOp::I("M".to_string())])]),
            Err(ShapeError::InvalidNumericValue)
        ));
    }

    // ── Feature: Align ───────────────────────────────────────────────────────

    #[test]
    fn test_align_y_to_world_up_when_rotated() {
        // Rotate 90° around Z (Y → -X), then Align(Y, Up) should restore Y = +Y.
        let mut interp = Interpreter::new();
        let ninety_z = Quat::from_axis_angle(Vec3::Z, std::f64::consts::FRAC_PI_2);
        interp.add_rule(
            "R",
            vec![
                ShapeOp::Rotate(ninety_z),
                ShapeOp::Align {
                    local_axis: Axis::Y,
                    target: Vec3::Y,
                },
                ShapeOp::I("M".to_string()),
            ],
        );
        let model = interp.derive(Scope::unit(), "R").unwrap();
        assert_eq!(model.len(), 1);
        let world_y = model.terminals[0].scope.rotation * Vec3::Y;
        assert!(
            (world_y - Vec3::Y).length() < 1e-6,
            "expected Y=(0,1,0), got {:?}",
            world_y
        );
    }

    #[test]
    fn test_align_already_aligned_is_noop() {
        let mut interp = Interpreter::new();
        interp.add_rule(
            "R",
            vec![
                ShapeOp::Align {
                    local_axis: Axis::Y,
                    target: Vec3::Y,
                },
                ShapeOp::I("M".to_string()),
            ],
        );
        let model = interp.derive(Scope::unit(), "R").unwrap();
        let rot = model.terminals[0].scope.rotation;
        assert!((rot.length_squared() - 1.0).abs() < 1e-9);
        // Rotation should still be unit (identity-like for already-aligned)
        let world_y = rot * Vec3::Y;
        assert!((world_y - Vec3::Y).length() < 1e-6);
    }

    #[test]
    fn test_align_zero_target_rejected() {
        let mut interp = Interpreter::new();
        interp.add_rule(
            "R",
            vec![
                ShapeOp::Align {
                    local_axis: Axis::Y,
                    target: Vec3::ZERO,
                },
                ShapeOp::I("M".to_string()),
            ],
        );
        assert!(matches!(
            interp.derive(Scope::unit(), "R"),
            Err(ShapeError::InvalidAlignTarget)
        ));
    }

    // ── Feature: Offset ──────────────────────────────────────────────────────

    #[test]
    fn test_offset_inset_produces_inside_and_border() {
        let mut interp = Interpreter::new();
        interp.add_rule(
            "R",
            vec![ShapeOp::Offset {
                distance: -0.5,
                cases: vec![
                    crate::ops::OffsetCase {
                        selector: crate::ops::OffsetSelector::Inside,
                        rule: "Glass".to_string(),
                    },
                    crate::ops::OffsetCase {
                        selector: crate::ops::OffsetSelector::Border,
                        rule: "Frame".to_string(),
                    },
                ],
            }],
        );
        // 4×3 face scope (z=0)
        let scope = Scope::new(Vec3::ZERO, Quat::IDENTITY, Vec3::new(4.0, 3.0, 0.0));
        let model = interp.derive(scope, "R").unwrap();
        // 1 Inside + 4 Border strips = 5 terminals
        assert_eq!(model.len(), 5);
        // Inside scope: size = (3.0, 2.0, 0.0), positioned at (0.5, 0.5, 0.0)
        let inside = model
            .terminals
            .iter()
            .find(|t| t.mesh_id == "Glass")
            .unwrap();
        assert!((inside.scope.size.x - 3.0).abs() < 1e-9);
        assert!((inside.scope.size.y - 2.0).abs() < 1e-9);
        assert!((inside.scope.position - Vec3::new(0.5, 0.5, 0.0)).length() < 1e-9);
    }

    #[test]
    fn test_offset_too_large_rejected() {
        let mut interp = Interpreter::new();
        interp.add_rule(
            "R",
            vec![ShapeOp::Offset {
                distance: -2.0, // 2*2.0 = 4 > 3 (sy)
                cases: vec![crate::ops::OffsetCase {
                    selector: crate::ops::OffsetSelector::Inside,
                    rule: "A".to_string(),
                }],
            }],
        );
        let scope = Scope::new(Vec3::ZERO, Quat::IDENTITY, Vec3::new(4.0, 3.0, 0.0));
        assert!(matches!(
            interp.derive(scope, "R"),
            Err(ShapeError::OffsetTooLarge)
        ));
    }

    #[test]
    fn test_offset_positive_distance_rejected() {
        let mut interp = Interpreter::new();
        interp.add_rule(
            "R",
            vec![ShapeOp::Offset {
                distance: 0.2,
                cases: vec![crate::ops::OffsetCase {
                    selector: crate::ops::OffsetSelector::Inside,
                    rule: "A".to_string(),
                }],
            }],
        );
        assert!(matches!(
            interp.derive(Scope::unit(), "R"),
            Err(ShapeError::InvalidNumericValue)
        ));
    }

    // ── Feature: Roof ────────────────────────────────────────────────────────

    #[test]
    fn test_roof_shed_produces_one_slope() {
        let mut interp = Interpreter::new();
        interp.add_rule(
            "R",
            vec![ShapeOp::Roof {
                config: RoofConfig::new(RoofType::Shed, 30.0),
                cases: vec![crate::ops::RoofCase {
                    selector: RoofFaceSelector::Slope,
                    rule: "Tiles".to_string(),
                }],
            }],
        );
        let scope = Scope::new(Vec3::ZERO, Quat::IDENTITY, Vec3::new(10.0, 5.0, 8.0));
        let model = interp.derive(scope, "R").unwrap();
        assert_eq!(model.len(), 1);
        assert_eq!(model.terminals[0].mesh_id, "Tiles");
        // Panel positioned at the base of the roof scope (local Y = 0.0)
        assert!((model.terminals[0].scope.position.y - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_roof_gable_produces_four_panels() {
        let mut interp = Interpreter::new();
        interp.add_rule(
            "R",
            vec![ShapeOp::Roof {
                config: RoofConfig::new(RoofType::Gable, 30.0),
                cases: vec![
                    crate::ops::RoofCase {
                        selector: RoofFaceSelector::Slope,
                        rule: "Tiles".to_string(),
                    },
                    crate::ops::RoofCase {
                        selector: RoofFaceSelector::GableEnd,
                        rule: "Bricks".to_string(),
                    },
                ],
            }],
        );
        let scope = Scope::new(Vec3::ZERO, Quat::IDENTITY, Vec3::new(10.0, 5.0, 8.0));
        let model = interp.derive(scope, "R").unwrap();
        // 2 slope + 2 gable-end panels
        assert_eq!(model.len(), 4);
        let tiles: Vec<_> = model
            .terminals
            .iter()
            .filter(|t| t.mesh_id == "Tiles")
            .collect();
        let bricks: Vec<_> = model
            .terminals
            .iter()
            .filter(|t| t.mesh_id == "Bricks")
            .collect();
        assert_eq!(tiles.len(), 2);
        assert_eq!(bricks.len(), 2);
    }

    #[test]
    fn test_roof_hip_produces_four_slopes() {
        let mut interp = Interpreter::new();
        interp.add_rule(
            "R",
            vec![ShapeOp::Roof {
                config: {
                    let mut c = RoofConfig::new(RoofType::Hip, 45.0);
                    c.overhang = 0.3;
                    c
                },
                cases: vec![crate::ops::RoofCase {
                    selector: RoofFaceSelector::Slope,
                    rule: "Tiles".to_string(),
                }],
            }],
        );
        let scope = Scope::new(Vec3::ZERO, Quat::IDENTITY, Vec3::new(10.0, 4.0, 8.0));
        let model = interp.derive(scope, "R").unwrap();
        assert_eq!(model.len(), 4);
    }

    #[test]
    fn test_roof_pyramid_produces_four_tapered_slopes() {
        let mut interp = Interpreter::new();
        interp.add_rule(
            "R",
            vec![ShapeOp::Roof {
                config: RoofConfig::new(RoofType::Pyramid, 40.0),
                cases: vec![crate::ops::RoofCase {
                    selector: RoofFaceSelector::Slope,
                    rule: "Tiles".to_string(),
                }],
            }],
        );
        let scope = Scope::new(Vec3::ZERO, Quat::IDENTITY, Vec3::new(6.0, 3.0, 6.0));
        let model = interp.derive(scope, "R").unwrap();
        assert_eq!(model.len(), 4);
        // All pyramid panels carry Triangle face profile
        for t in &model.terminals {
            assert!(
                matches!(t.face_profile, FaceProfile::Triangle { peak_offset } if (peak_offset - 0.5).abs() < 1e-9),
                "expected Triangle{{peak_offset=0.5}}, got {:?}",
                t.face_profile
            );
        }
    }

    #[test]
    fn test_roof_slope_normals_outward() {
        // All four Hip slopes must have Local Z (= scope.rotation * Z) pointing
        // AWAY from the building:
        //   front  → (0,  cos α, −sin α)   back  → (0, cos α, +sin α)
        //   left   → (−sin α, cos α,  0)   right → (+sin α, cos α,  0)
        let alpha: f64 = 30_f64.to_radians();
        let cos_a = alpha.cos();
        let sin_a = alpha.sin();
        let mut interp = Interpreter::new();
        interp.add_rule(
            "R",
            vec![ShapeOp::Roof {
                config: RoofConfig::new(RoofType::Hip, 30.0),
                cases: vec![crate::ops::RoofCase {
                    selector: RoofFaceSelector::Slope,
                    rule: "S".to_string(),
                }],
            }],
        );
        let scope = Scope::new(Vec3::ZERO, Quat::IDENTITY, Vec3::new(10.0, 4.0, 8.0));
        let model = interp.derive(scope, "R").unwrap();
        assert_eq!(model.len(), 4);
        let normals: Vec<Vec3> = model
            .terminals
            .iter()
            .map(|t| t.scope.rotation * Vec3::Z)
            .collect();
        let expected = [
            Vec3::new(0.0, cos_a, -sin_a), // front: up & forward
            Vec3::new(0.0, cos_a, sin_a),  // back:  up & backward
            Vec3::new(-sin_a, cos_a, 0.0), // left:  up & left
            Vec3::new(sin_a, cos_a, 0.0),  // right: up & right
        ];
        for exp in &expected {
            assert!(
                normals.iter().any(|n| (*n - *exp).length() < 1e-6),
                "missing outward normal {:?}; got {:?}",
                exp,
                normals
            );
        }
        // All normals must have a positive Y component (point upward).
        for n in &normals {
            assert!(n.y > 0.0, "normal pointing downward: {:?}", n);
        }
    }

    #[test]
    fn test_align_antiparallel_fallback_no_nan() {
        // When the local axis is exactly anti-parallel to the target, the fallback
        // 180° rotation must produce a unit quaternion, not NaN.
        // Rotate scope so local Y = −Y (anti-parallel to world Up), then Align(Y, Up).
        let flip_y = Quat::from_axis_angle(Vec3::Z, PI);
        let mut interp = Interpreter::new();
        interp.add_rule(
            "R",
            vec![
                ShapeOp::Rotate(flip_y),
                ShapeOp::Align {
                    local_axis: Axis::Y,
                    target: Vec3::Y,
                },
                ShapeOp::I("M".to_string()),
            ],
        );
        let model = interp.derive(Scope::unit(), "R").unwrap();
        let world_y = model.terminals[0].scope.rotation * Vec3::Y;
        assert!(
            (world_y - Vec3::Y).length() < 1e-6,
            "anti-parallel Align should point Y to world up, got {:?}",
            world_y
        );
        // Quaternion must remain unit.
        let r = model.terminals[0].scope.rotation;
        assert!(
            (r.length_squared() - 1.0).abs() < 1e-9,
            "rotation not unit: length_sq={}",
            r.length_squared()
        );
    }

    #[test]
    fn test_roof_invalid_angle_rejected() {
        let mut interp = Interpreter::new();
        interp.add_rule(
            "R",
            vec![ShapeOp::Roof {
                config: RoofConfig::new(RoofType::Shed, 0.0),
                cases: vec![],
            }],
        );
        assert!(matches!(
            interp.derive(Scope::unit(), "R"),
            Err(ShapeError::InvalidRoofAngle(_))
        ));
    }
}

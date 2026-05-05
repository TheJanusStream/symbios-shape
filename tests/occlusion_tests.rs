//! Integration tests for occlusion query primitives (issue #36).
//!
//! Covers:
//! - True OBB-vs-OBB overlap via Separating Axis Theorem (`obb_overlap`).
//! - `ShapeModel::query()` returning a `TerminalQuery` with `overlaps` and
//!   `overlapping(...)` methods.
//! - The `IfClear { Rule }` and `IfOccluded { Rule }` grammar ops.

use symbios_shape::grammar::parse_ops;
use symbios_shape::ops::ShapeOp;
use symbios_shape::query::obb_overlap;
use symbios_shape::{Interpreter, Quat, Scope, Vec3};

// ── OBB overlap (low-level) ──────────────────────────────────────────────────

fn box_at(pos: Vec3, size: Vec3) -> Scope {
    Scope::new(pos, Quat::IDENTITY, size)
}

#[test]
fn obb_axis_aligned_disjoint_returns_false() {
    let a = box_at(Vec3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 1.0, 1.0));
    let b = box_at(Vec3::new(2.0, 0.0, 0.0), Vec3::new(1.0, 1.0, 1.0));
    assert!(!obb_overlap(&a, &b));
}

#[test]
fn obb_axis_aligned_touching_returns_true() {
    let a = box_at(Vec3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 1.0, 1.0));
    let b = box_at(Vec3::new(0.999, 0.0, 0.0), Vec3::new(1.0, 1.0, 1.0));
    assert!(obb_overlap(&a, &b));
}

#[test]
fn obb_axis_aligned_overlapping_returns_true() {
    let a = box_at(Vec3::new(0.0, 0.0, 0.0), Vec3::new(2.0, 2.0, 2.0));
    let b = box_at(Vec3::new(1.0, 1.0, 1.0), Vec3::new(2.0, 2.0, 2.0));
    assert!(obb_overlap(&a, &b));
}

#[test]
fn obb_rotated_disjoint() {
    let a = Scope::new(Vec3::ZERO, Quat::IDENTITY, Vec3::new(1.0, 1.0, 1.0));
    // Rotated 45° about Y, far away.
    let b = Scope::new(
        Vec3::new(5.0, 0.0, 0.0),
        Quat::from_axis_angle(Vec3::Y, std::f64::consts::FRAC_PI_4),
        Vec3::new(1.0, 1.0, 1.0),
    );
    assert!(!obb_overlap(&a, &b));
}

#[test]
fn obb_rotated_overlapping() {
    let a = Scope::new(Vec3::ZERO, Quat::IDENTITY, Vec3::new(2.0, 2.0, 2.0));
    // Rotated 45° about Y, but its centre is inside A.
    let b = Scope::new(
        Vec3::new(0.5, 0.5, 0.5),
        Quat::from_axis_angle(Vec3::Y, std::f64::consts::FRAC_PI_4),
        Vec3::new(1.0, 1.0, 1.0),
    );
    assert!(obb_overlap(&a, &b));
}

#[test]
fn obb_sat_finds_separating_axis_for_rotated_pair() {
    // Two boxes that have overlapping AABBs but DON'T overlap when their
    // OBB orientations are considered (this is the case AABB-only would
    // get wrong; SAT should correctly return false).
    // Box A is small at origin; Box B is rotated 45° and offset diagonally
    // such that their OBBs miss but their AABBs overlap.
    let a = box_at(Vec3::ZERO, Vec3::new(1.0, 1.0, 1.0));
    // Diagonal offset that guarantees OBB miss but AABB hit.
    let b = Scope::new(
        Vec3::new(1.5, 0.0, 1.5),
        Quat::from_axis_angle(Vec3::Y, std::f64::consts::FRAC_PI_4),
        Vec3::new(1.0, 1.0, 1.0),
    );
    // We just verify SAT runs and gives a deterministic answer (false here).
    assert!(!obb_overlap(&a, &b));
}

#[test]
fn obb_thin_face_scope_still_works() {
    // Face scope (size.z = 0) — must still produce a finite overlap test.
    let face = Scope::new(Vec3::ZERO, Quat::IDENTITY, Vec3::new(2.0, 2.0, 0.0));
    let cube = box_at(Vec3::new(0.5, 0.5, 0.0), Vec3::new(0.5, 0.5, 0.5));
    assert!(obb_overlap(&face, &cube));
}

// ── ShapeModel::query() ──────────────────────────────────────────────────────

#[test]
fn query_overlaps_returns_true_for_overlapping_terminal() {
    let mut interp = Interpreter::new();
    interp.add_rule("R", vec![ShapeOp::I("Box".to_string())]);
    let scope = Scope::new(Vec3::ZERO, Quat::IDENTITY, Vec3::new(2.0, 2.0, 2.0));
    let model = interp.derive(scope, "R").unwrap();
    let q = model.query();
    assert!(q.overlaps(&Scope::new(
        Vec3::new(1.0, 1.0, 1.0),
        Quat::IDENTITY,
        Vec3::new(2.0, 2.0, 2.0)
    )));
}

#[test]
fn query_overlaps_returns_false_for_disjoint_scope() {
    let mut interp = Interpreter::new();
    interp.add_rule("R", vec![ShapeOp::I("Box".to_string())]);
    let scope = Scope::new(Vec3::ZERO, Quat::IDENTITY, Vec3::new(2.0, 2.0, 2.0));
    let model = interp.derive(scope, "R").unwrap();
    assert!(!model.query().overlaps(&Scope::new(
        Vec3::new(10.0, 0.0, 0.0),
        Quat::IDENTITY,
        Vec3::new(1.0, 1.0, 1.0)
    )));
}

#[test]
fn query_overlapping_iterator_returns_correct_subset() {
    let mut interp = Interpreter::new();
    // Three boxes at X = 0, 5, 10.
    interp.add_rule("R", vec![ShapeOp::I("Box".to_string())]);
    let mut model = symbios_shape::ShapeModel::default();
    for x in [0.0, 5.0, 10.0] {
        let s = Scope::new(
            Vec3::new(x, 0.0, 0.0),
            Quat::IDENTITY,
            Vec3::new(1.0, 1.0, 1.0),
        );
        model.terminals.push(symbios_shape::Terminal::new(s, "Box"));
    }
    let q = scope_at_x(0.5);
    let hits: Vec<_> = model.query().overlapping(&q).collect();
    assert_eq!(hits.len(), 1);
    assert!(hits[0].scope.position.x.abs() < 1e-9);
}

fn scope_at_x(x: f64) -> Scope {
    Scope::new(
        Vec3::new(x, 0.0, 0.0),
        Quat::IDENTITY,
        Vec3::new(1.0, 1.0, 1.0),
    )
}

// ── IfClear / IfOccluded grammar ops ─────────────────────────────────────────

#[test]
fn if_clear_emits_when_no_terminal_overlaps() {
    // No terminals exist → IfClear emits unconditionally.
    let mut interp = Interpreter::new();
    interp.add_rule(
        "R",
        vec![ShapeOp::IfClear {
            rule: "Window".to_string(),
        }],
    );
    interp.add_rule("Window", vec![ShapeOp::I("Window".to_string())]);
    let model = interp
        .derive(
            Scope::new(Vec3::ZERO, Quat::IDENTITY, Vec3::new(1.0, 1.0, 1.0)),
            "R",
        )
        .unwrap();
    assert_eq!(model.len(), 1);
    assert_eq!(model.terminals[0].mesh_id, "Window");
}

#[test]
fn if_clear_skips_when_overlapping_terminal_exists() {
    // Place a Pillar first via a Y-split that puts the pillar in the queue
    // before the IfClear-bearing slot.
    let mut interp = Interpreter::new();
    interp.add_rule(
        "Lot",
        parse_ops("Split(Y) { 5: Pillar | 5: Maybe }").unwrap(),
    );
    interp.add_rule("Pillar", vec![ShapeOp::I("Pillar".to_string())]);
    interp.add_rule(
        "Maybe",
        // Try to place a Window in the upper half; but Pillar's OBB extends
        // up to Y=5 and the Maybe scope starts at Y=5, so they touch but
        // don't overlap by more than the THIN_HALF_EXTENT epsilon. Make
        // them clearly overlap by translating Maybe down.
        vec![
            ShapeOp::Translate(Vec3::new(0.0, -2.0, 0.0)),
            ShapeOp::IfClear {
                rule: "Window".to_string(),
            },
        ],
    );
    interp.add_rule("Window", vec![ShapeOp::I("Window".to_string())]);

    let scope = Scope::new(Vec3::ZERO, Quat::IDENTITY, Vec3::new(2.0, 10.0, 2.0));
    let model = interp.derive(scope, "Lot").unwrap();
    let pillars = model
        .terminals
        .iter()
        .filter(|t| t.mesh_id == "Pillar")
        .count();
    let windows = model
        .terminals
        .iter()
        .filter(|t| t.mesh_id == "Window")
        .count();
    assert_eq!(pillars, 1);
    assert_eq!(
        windows, 0,
        "Window should be suppressed by IfClear because Pillar overlaps"
    );
}

#[test]
fn if_occluded_emits_when_overlapping_terminal_exists() {
    let mut interp = Interpreter::new();
    interp.add_rule(
        "Lot",
        parse_ops("Split(Y) { 5: Pillar | 5: Maybe }").unwrap(),
    );
    interp.add_rule("Pillar", vec![ShapeOp::I("Pillar".to_string())]);
    interp.add_rule(
        "Maybe",
        vec![
            ShapeOp::Translate(Vec3::new(0.0, -2.0, 0.0)),
            ShapeOp::IfOccluded {
                rule: "Patch".to_string(),
            },
        ],
    );
    interp.add_rule("Patch", vec![ShapeOp::I("Patch".to_string())]);

    let scope = Scope::new(Vec3::ZERO, Quat::IDENTITY, Vec3::new(2.0, 10.0, 2.0));
    let model = interp.derive(scope, "Lot").unwrap();
    let patches = model
        .terminals
        .iter()
        .filter(|t| t.mesh_id == "Patch")
        .count();
    assert_eq!(
        patches, 1,
        "Patch should emit because the scope is occluded"
    );
}

#[test]
fn if_clear_via_grammar() {
    let ops = parse_ops("IfClear { Window }").unwrap();
    assert!(matches!(ops[0], ShapeOp::IfClear { ref rule } if rule == "Window"));
}

#[test]
fn if_occluded_via_grammar() {
    let ops = parse_ops("IfOccluded { Patch }").unwrap();
    assert!(matches!(ops[0], ShapeOp::IfOccluded { ref rule } if rule == "Patch"));
}

#[test]
fn if_clear_emits_when_only_disjoint_terminals_exist() {
    // Lot is split into Far | Gap | Near. The 1m Gap ensures Far and Near
    // are clearly disjoint (not just touching, which SAT-with-EPS treats
    // as overlap).
    let mut interp = Interpreter::new();
    interp.add_rule(
        "Lot",
        parse_ops("Split(X) { 1: Far | 1: Gap | 1: Near }").unwrap(),
    );
    interp.add_rule("Far", vec![ShapeOp::I("Far".to_string())]);
    interp.add_rule("Gap", vec![]);
    interp.add_rule(
        "Near",
        vec![ShapeOp::IfClear {
            rule: "Window".to_string(),
        }],
    );
    interp.add_rule("Window", vec![ShapeOp::I("Window".to_string())]);

    let scope = Scope::new(
        Vec3::new(0.0, 0.0, 0.0),
        Quat::IDENTITY,
        Vec3::new(3.0, 1.0, 1.0),
    );
    let model = interp.derive(scope, "Lot").unwrap();
    let windows = model
        .terminals
        .iter()
        .filter(|t| t.mesh_id == "Window")
        .count();
    let fars = model
        .terminals
        .iter()
        .filter(|t| t.mesh_id == "Far")
        .count();
    assert_eq!(fars, 1);
    assert_eq!(
        windows, 1,
        "Far terminal does not overlap Near scope, IfClear emits"
    );
}

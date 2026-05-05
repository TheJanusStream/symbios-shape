//! Integration tests for the `Attach` operator (issue #31).
//!
//! `Attach(world_axis) { Surface: Rule }` projects a flat scope out of the
//! current scope, with the new scope's local Y axis pointing along `world_axis`.
//! Tests cover all six cardinal directions plus arbitrary axes and error paths.

use symbios_shape::grammar::parse_ops;
use symbios_shape::ops::{AttachCase, AttachSelector, ShapeOp};
use symbios_shape::{Interpreter, Quat, Scope, ShapeError, Vec3};

fn parent_scope() -> Scope {
    Scope::new(Vec3::ZERO, Quat::IDENTITY, Vec3::new(2.0, 4.0, 6.0))
}

fn run_attach(world_axis: Vec3) -> Vec<symbios_shape::model::Terminal> {
    let mut interp = Interpreter::new();
    interp.add_rule(
        "R",
        vec![ShapeOp::Attach {
            world_axis,
            cases: vec![AttachCase {
                selector: AttachSelector::Surface,
                rule: "Dormer".to_string(),
            }],
        }],
    );
    interp.derive(parent_scope(), "R").unwrap().terminals
}

// ── Surface scope orientation ─────────────────────────────────────────────────

#[test]
fn attach_up_aligns_local_y_with_world_up() {
    let t = run_attach(Vec3::Y);
    assert_eq!(t.len(), 1);
    let new_y = t[0].scope.rotation * Vec3::Y;
    assert!((new_y - Vec3::Y).length() < 1e-9);
}

#[test]
fn attach_down_aligns_local_y_with_world_down() {
    let t = run_attach(Vec3::NEG_Y);
    assert_eq!(t.len(), 1);
    let new_y = t[0].scope.rotation * Vec3::Y;
    assert!((new_y - Vec3::NEG_Y).length() < 1e-9);
}

#[test]
fn attach_forward_aligns_local_y_with_negative_z() {
    let t = run_attach(Vec3::NEG_Z);
    let new_y = t[0].scope.rotation * Vec3::Y;
    assert!((new_y - Vec3::NEG_Z).length() < 1e-9);
}

#[test]
fn attach_back_aligns_local_y_with_positive_z() {
    let t = run_attach(Vec3::Z);
    let new_y = t[0].scope.rotation * Vec3::Y;
    assert!((new_y - Vec3::Z).length() < 1e-9);
}

#[test]
fn attach_right_aligns_local_y_with_positive_x() {
    let t = run_attach(Vec3::X);
    let new_y = t[0].scope.rotation * Vec3::Y;
    assert!((new_y - Vec3::X).length() < 1e-9);
}

#[test]
fn attach_left_aligns_local_y_with_negative_x() {
    let t = run_attach(Vec3::NEG_X);
    let new_y = t[0].scope.rotation * Vec3::Y;
    assert!((new_y - Vec3::NEG_X).length() < 1e-9);
}

#[test]
fn attach_arbitrary_axis_normalizes_internally() {
    // A non-unit world axis is normalized inside Attach.
    let t = run_attach(Vec3::new(2.0, 2.0, 0.0));
    let new_y = t[0].scope.rotation * Vec3::Y;
    let expected = Vec3::new(2.0, 2.0, 0.0).normalize();
    assert!((new_y - expected).length() < 1e-9);
}

// ── Surface scope dimensions ─────────────────────────────────────────────────

#[test]
fn attach_surface_inherits_xy_size_and_zero_z() {
    let t = run_attach(Vec3::Y);
    let s = t[0].scope.size;
    // Parent was (2, 4, 6); the projected surface keeps X and Y, drops Z to 0.
    assert!((s.x - 2.0).abs() < 1e-9);
    assert!((s.y - 4.0).abs() < 1e-9);
    assert!(s.z.abs() < 1e-9);
}

#[test]
fn attach_surface_origin_matches_parent() {
    let t = run_attach(Vec3::Y);
    assert!((t[0].scope.position - Vec3::ZERO).length() < 1e-9);
}

// ── Selector dispatch ────────────────────────────────────────────────────────

#[test]
fn attach_with_no_matching_case_emits_nothing() {
    let mut interp = Interpreter::new();
    interp.add_rule(
        "R",
        vec![ShapeOp::Attach {
            world_axis: Vec3::Y,
            // No Surface and no All → nothing matches.
            cases: vec![],
        }],
    );
    let model = interp.derive(parent_scope(), "R").unwrap();
    assert!(model.is_empty());
}

#[test]
fn attach_all_selector_falls_through_to_surface() {
    let mut interp = Interpreter::new();
    interp.add_rule(
        "R",
        vec![ShapeOp::Attach {
            world_axis: Vec3::Y,
            cases: vec![AttachCase {
                selector: AttachSelector::All,
                rule: "Dormer".to_string(),
            }],
        }],
    );
    let model = interp.derive(parent_scope(), "R").unwrap();
    assert_eq!(model.len(), 1);
    assert_eq!(model.terminals[0].mesh_id, "Dormer");
}

// ── Error paths ──────────────────────────────────────────────────────────────

#[test]
fn attach_zero_axis_rejected() {
    let mut interp = Interpreter::new();
    interp.add_rule(
        "R",
        vec![ShapeOp::Attach {
            world_axis: Vec3::ZERO,
            cases: vec![AttachCase {
                selector: AttachSelector::Surface,
                rule: "X".to_string(),
            }],
        }],
    );
    assert!(matches!(
        interp.derive(parent_scope(), "R"),
        Err(ShapeError::InvalidAlignTarget)
    ));
}

#[test]
fn attach_nan_axis_rejected() {
    let mut interp = Interpreter::new();
    interp.add_rule(
        "R",
        vec![ShapeOp::Attach {
            world_axis: Vec3::new(f64::NAN, 1.0, 0.0),
            cases: vec![AttachCase {
                selector: AttachSelector::Surface,
                rule: "X".to_string(),
            }],
        }],
    );
    assert!(matches!(
        interp.derive(parent_scope(), "R"),
        Err(ShapeError::InvalidAlignTarget)
    ));
}

// ── Grammar surface ──────────────────────────────────────────────────────────

#[test]
fn attach_via_grammar_with_named_axis() {
    let ops = parse_ops("Attach(Up) { Surface: Dormer }").unwrap();
    let mut interp = Interpreter::new();
    interp.add_rule("R", ops);
    let model = interp.derive(parent_scope(), "R").unwrap();
    assert_eq!(model.len(), 1);
    assert_eq!(model.terminals[0].mesh_id, "Dormer");
}

//! Integration tests for the `Align` operator (issue #31).
//!
//! Verifies shortest-arc alignment for parallel, perpendicular, antiparallel,
//! and arbitrary direction targets across all three local axes.

use symbios_shape::ops::{Axis, ShapeOp};
use symbios_shape::{Interpreter, Quat, Scope, ShapeError, Vec3};

fn unit_scope_with_rot(rot: Quat) -> Scope {
    Scope::new(Vec3::ZERO, rot, Vec3::new(2.0, 3.0, 4.0))
}

fn run_align(local_axis: Axis, target: Vec3, start_rot: Quat) -> Quat {
    let mut interp = Interpreter::new();
    interp.add_rule(
        "R",
        vec![
            ShapeOp::Align { local_axis, target },
            ShapeOp::I("Mesh".to_string()),
        ],
    );
    let model = interp.derive(unit_scope_with_rot(start_rot), "R").unwrap();
    model.terminals[0].scope.rotation
}

fn local_axis_in_world(rot: Quat, axis: Axis) -> Vec3 {
    match axis {
        Axis::X => rot * Vec3::X,
        Axis::Y => rot * Vec3::Y,
        Axis::Z => rot * Vec3::Z,
    }
}

// ── Each cardinal direction × each local axis ────────────────────────────────

#[test]
fn align_y_to_up_keeps_identity() {
    let rot = run_align(Axis::Y, Vec3::Y, Quat::IDENTITY);
    let up = local_axis_in_world(rot, Axis::Y);
    assert!((up - Vec3::Y).length() < 1e-9);
}

#[test]
fn align_y_to_world_x_rotates_90_degrees() {
    let rot = run_align(Axis::Y, Vec3::X, Quat::IDENTITY);
    let new_y = local_axis_in_world(rot, Axis::Y);
    assert!((new_y - Vec3::X).length() < 1e-9);
}

#[test]
fn align_z_to_forward_rotates_into_negative_z_world() {
    let rot = run_align(Axis::Z, Vec3::NEG_Z, Quat::IDENTITY);
    let new_z = local_axis_in_world(rot, Axis::Z);
    assert!((new_z - Vec3::NEG_Z).length() < 1e-9);
}

#[test]
fn align_x_to_world_y() {
    let rot = run_align(Axis::X, Vec3::Y, Quat::IDENTITY);
    let new_x = local_axis_in_world(rot, Axis::X);
    assert!((new_x - Vec3::Y).length() < 1e-9);
}

// ── Antiparallel: shortest arc would be undefined; expect a 180° rotation ───

#[test]
fn align_y_to_down_does_full_180() {
    let rot = run_align(Axis::Y, Vec3::NEG_Y, Quat::IDENTITY);
    let new_y = local_axis_in_world(rot, Axis::Y);
    assert!(
        (new_y - Vec3::NEG_Y).length() < 1e-9,
        "Y should land on -Y, got {new_y:?}"
    );
}

#[test]
fn align_z_to_back_does_full_180() {
    let rot = run_align(Axis::Z, Vec3::Z, Quat::IDENTITY);
    let new_z = local_axis_in_world(rot, Axis::Z);
    assert!((new_z - Vec3::Z).length() < 1e-9);
}

// ── Arbitrary direction ───────────────────────────────────────────────────────

#[test]
fn align_y_to_diagonal_target() {
    // Target need not be normalized — Align normalizes internally.
    let target = Vec3::new(1.0, 1.0, 1.0);
    let rot = run_align(Axis::Y, target, Quat::IDENTITY);
    let new_y = local_axis_in_world(rot, Axis::Y);
    let expected = target.normalize();
    assert!(
        (new_y - expected).length() < 1e-9,
        "expected {expected:?}, got {new_y:?}"
    );
}

// ── Composition with prior rotation ──────────────────────────────────────────

#[test]
fn align_re_aligns_after_arbitrary_rotation() {
    // A scope already rotated by some quaternion is realigned so its local Y
    // points to world Up regardless of the starting rotation.
    let prior = Quat::from_axis_angle(Vec3::X, 0.7) * Quat::from_axis_angle(Vec3::Z, 0.4);
    let rot = run_align(Axis::Y, Vec3::Y, prior);
    let new_y = local_axis_in_world(rot, Axis::Y);
    assert!((new_y - Vec3::Y).length() < 1e-9);
}

// ── Error paths ──────────────────────────────────────────────────────────────

#[test]
fn align_with_zero_target_rejected() {
    let mut interp = Interpreter::new();
    interp.add_rule(
        "R",
        vec![ShapeOp::Align {
            local_axis: Axis::Y,
            target: Vec3::ZERO,
        }],
    );
    assert!(matches!(
        interp.derive(unit_scope_with_rot(Quat::IDENTITY), "R"),
        Err(ShapeError::InvalidAlignTarget)
    ));
}

#[test]
fn align_with_nan_target_rejected() {
    let mut interp = Interpreter::new();
    interp.add_rule(
        "R",
        vec![ShapeOp::Align {
            local_axis: Axis::Y,
            target: Vec3::new(f64::NAN, 0.0, 0.0),
        }],
    );
    assert!(matches!(
        interp.derive(unit_scope_with_rot(Quat::IDENTITY), "R"),
        Err(ShapeError::InvalidAlignTarget)
    ));
}

#[test]
fn align_with_infinite_target_rejected() {
    let mut interp = Interpreter::new();
    interp.add_rule(
        "R",
        vec![ShapeOp::Align {
            local_axis: Axis::Y,
            target: Vec3::new(f64::INFINITY, 0.0, 0.0),
        }],
    );
    assert!(matches!(
        interp.derive(unit_scope_with_rot(Quat::IDENTITY), "R"),
        Err(ShapeError::InvalidAlignTarget)
    ));
}

// ── Grammar surface ──────────────────────────────────────────────────────────

#[test]
fn align_via_grammar_named_directions() {
    use symbios_shape::grammar::parse_ops;
    let mut interp = Interpreter::new();
    interp.add_rule("R", parse_ops(r#"Align(Y, Up) I("Mesh")"#).unwrap());
    let model = interp
        .derive(
            Scope::new(
                Vec3::ZERO,
                Quat::from_axis_angle(Vec3::Z, 0.5),
                Vec3::new(2.0, 3.0, 4.0),
            ),
            "R",
        )
        .unwrap();
    let new_y = local_axis_in_world(model.terminals[0].scope.rotation, Axis::Y);
    assert!((new_y - Vec3::Y).length() < 1e-9);
}

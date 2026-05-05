//! Integration tests for the `Offset` operator (issue #31).
//!
//! Verifies inset behavior — `Offset` requires a negative distance, produces
//! 1 Inside scope + 4 Border strips, and rejects out-of-range distances and
//! non-finite inputs.

use symbios_shape::grammar::parse_ops;
use symbios_shape::ops::{OffsetCase, OffsetSelector, ShapeOp};
use symbios_shape::{Interpreter, Quat, Scope, ShapeError, Vec3};

fn face_scope() -> Scope {
    // 2D face scope (size.z = 0) — Offset operates on faces.
    Scope::new(Vec3::ZERO, Quat::IDENTITY, Vec3::new(4.0, 3.0, 0.0))
}

fn case(selector: OffsetSelector, rule: &str) -> OffsetCase {
    OffsetCase {
        selector,
        rule: rule.to_string(),
    }
}

fn run_offset(
    distance: f64,
    cases: Vec<OffsetCase>,
) -> Result<Vec<symbios_shape::model::Terminal>, ShapeError> {
    let mut interp = Interpreter::new();
    interp.add_rule("R", vec![ShapeOp::Offset { distance, cases }]);
    interp.derive(face_scope(), "R").map(|m| m.terminals)
}

// ── Inset (negative distance) ─────────────────────────────────────────────────

#[test]
fn offset_inset_produces_inside_and_four_border_strips() {
    let t = run_offset(
        -0.5,
        vec![
            case(OffsetSelector::Inside, "Glass"),
            case(OffsetSelector::Border, "Frame"),
        ],
    )
    .unwrap();
    let inside = t.iter().filter(|x| x.mesh_id == "Glass").count();
    let border = t.iter().filter(|x| x.mesh_id == "Frame").count();
    assert_eq!(inside, 1);
    assert_eq!(border, 4);
}

#[test]
fn offset_inside_dimensions_match_inset() {
    let t = run_offset(-0.5, vec![case(OffsetSelector::Inside, "Glass")]).unwrap();
    assert_eq!(t.len(), 1);
    let inside = &t[0];
    // Original face was 4 × 3, inset by 0.5 on every side → 3 × 2.
    assert!((inside.scope.size.x - 3.0).abs() < 1e-9);
    assert!((inside.scope.size.y - 2.0).abs() < 1e-9);
    // Origin shifts by inset on each axis.
    assert!((inside.scope.position.x - 0.5).abs() < 1e-9);
    assert!((inside.scope.position.y - 0.5).abs() < 1e-9);
}

#[test]
fn offset_border_strips_cover_perimeter() {
    let t = run_offset(-0.5, vec![case(OffsetSelector::Border, "Frame")]).unwrap();
    assert_eq!(t.len(), 4);
    let inset = 0.5_f64;
    let sx = 4.0_f64;
    let sy = 3.0_f64;
    let mut have_bottom = false;
    let mut have_top = false;
    let mut have_left = false;
    let mut have_right = false;
    for term in &t {
        let p = term.scope.position;
        let s = term.scope.size;
        if (p.y - 0.0).abs() < 1e-9 && (s.x - sx).abs() < 1e-9 && (s.y - inset).abs() < 1e-9 {
            have_bottom = true;
        } else if (p.y - (sy - inset)).abs() < 1e-9
            && (s.x - sx).abs() < 1e-9
            && (s.y - inset).abs() < 1e-9
        {
            have_top = true;
        } else if (p.x - 0.0).abs() < 1e-9 && (s.x - inset).abs() < 1e-9 {
            have_left = true;
        } else if (p.x - (sx - inset)).abs() < 1e-9 && (s.x - inset).abs() < 1e-9 {
            have_right = true;
        }
    }
    assert!(have_bottom && have_top && have_left && have_right);
}

#[test]
fn offset_only_inside_drops_borders() {
    let t = run_offset(-0.5, vec![case(OffsetSelector::Inside, "Glass")]).unwrap();
    assert_eq!(t.len(), 1);
}

#[test]
fn offset_only_border_drops_inside() {
    let t = run_offset(-0.5, vec![case(OffsetSelector::Border, "Frame")]).unwrap();
    assert_eq!(t.len(), 4);
}

#[test]
fn offset_all_selector_routes_both_kinds() {
    let t = run_offset(-0.5, vec![case(OffsetSelector::All, "Both")]).unwrap();
    // 1 inside + 4 border, all named "Both" via the All fallback.
    assert_eq!(t.iter().filter(|x| x.mesh_id == "Both").count(), 5);
}

// ── Edge cases ───────────────────────────────────────────────────────────────

#[test]
fn offset_zero_distance_rejected() {
    assert!(matches!(
        run_offset(0.0, vec![case(OffsetSelector::Inside, "G")]),
        Err(ShapeError::InvalidNumericValue)
    ));
}

#[test]
fn offset_positive_distance_rejected() {
    // Positive distance = outset; only inset (negative) is supported in v0.1.
    assert!(matches!(
        run_offset(0.2, vec![case(OffsetSelector::Inside, "G")]),
        Err(ShapeError::InvalidNumericValue)
    ));
}

#[test]
fn offset_inset_too_large_rejected() {
    // Inset of 2.0 on a 4 × 3 face: width remainder = 4 - 4 = 0,
    // height remainder = 3 - 4 = -1 → OffsetTooLarge.
    assert!(matches!(
        run_offset(-2.0, vec![case(OffsetSelector::Inside, "G")]),
        Err(ShapeError::OffsetTooLarge)
    ));
}

#[test]
fn offset_nan_distance_rejected() {
    assert!(matches!(
        run_offset(f64::NAN, vec![case(OffsetSelector::Inside, "G")]),
        Err(ShapeError::InvalidNumericValue)
    ));
}

#[test]
fn offset_infinite_distance_rejected() {
    assert!(matches!(
        run_offset(f64::INFINITY, vec![case(OffsetSelector::Inside, "G")]),
        Err(ShapeError::InvalidNumericValue)
    ));
}

#[test]
fn offset_inset_exactly_half_dimension_accepted_as_zero_inside() {
    // Inset of exactly half height: width remainder positive, height remainder 0
    // → still accepted (zero-area scope is tolerated by `validate`).
    let t = run_offset(-1.5, vec![case(OffsetSelector::Border, "Frame")]).unwrap();
    assert_eq!(t.len(), 4);
}

// ── Grammar surface ──────────────────────────────────────────────────────────

#[test]
fn offset_via_grammar() {
    let ops = parse_ops("Offset(-0.5) { Inside: Glass | Border: Frame }").unwrap();
    let mut interp = Interpreter::new();
    interp.add_rule("R", ops);
    let model = interp.derive(face_scope(), "R").unwrap();
    assert_eq!(
        model
            .terminals
            .iter()
            .filter(|t| t.mesh_id == "Glass")
            .count(),
        1
    );
    assert_eq!(
        model
            .terminals
            .iter()
            .filter(|t| t.mesh_id == "Frame")
            .count(),
        4
    );
}

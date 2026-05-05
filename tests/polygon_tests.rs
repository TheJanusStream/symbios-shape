//! Integration tests for the `Polygon` operator (issue #30).
//!
//! `Polygon((x1,y1), (x2,y2), …)` stamps a [`FaceProfile::Polygon`] override
//! onto the next terminal in the rule, mirroring how `Taper` overrides the
//! profile. The genetic-mutation path that already produced `Polygon` profiles
//! still works because `FaceProfile::Polygon` is unchanged.

use symbios_shape::grammar::parse_ops;
use symbios_shape::ops::ShapeOp;
use symbios_shape::{FaceProfile, Interpreter, Quat, Scope, Vec3};

fn footprint() -> Scope {
    Scope::new(Vec3::ZERO, Quat::IDENTITY, Vec3::new(4.0, 1.0, 4.0))
}

// ── Parser surface ───────────────────────────────────────────────────────────

#[test]
fn parser_accepts_l_shape_polygon() {
    let ops = parse_ops("Polygon((0,0), (4,0), (4,2), (2,2), (2,4), (0,4)) I(\"Mass\")").unwrap();
    assert_eq!(ops.len(), 2);
    match &ops[0] {
        ShapeOp::Polygon(verts) => {
            assert_eq!(verts.len(), 6);
            assert!((verts[0] - glam::DVec2::new(0.0, 0.0)).length() < 1e-9);
            assert!((verts[3] - glam::DVec2::new(2.0, 2.0)).length() < 1e-9);
        }
        _ => panic!("expected Polygon op, got {:?}", ops[0]),
    }
}

#[test]
fn parser_accepts_three_vertex_minimum() {
    let ops = parse_ops("Polygon((0,0), (1,0), (0,1)) I(\"Tri\")").unwrap();
    assert!(matches!(&ops[0], ShapeOp::Polygon(verts) if verts.len() == 3));
}

#[test]
fn parser_rejects_two_vertex_polygon() {
    assert!(parse_ops("Polygon((0,0), (1,0)) I(\"X\")").is_err());
}

#[test]
fn parser_rejects_single_vertex_polygon() {
    assert!(parse_ops("Polygon((0,0)) I(\"X\")").is_err());
}

#[test]
fn parser_rejects_empty_polygon() {
    assert!(parse_ops("Polygon() I(\"X\")").is_err());
}

#[test]
fn parser_rejects_non_finite_vertex() {
    // `finite_float` rejects inf/NaN at parse time.
    assert!(parse_ops("Polygon((0,0), (1,0), (inf,1)) I(\"X\")").is_err());
}

// ── Interpreter wiring ───────────────────────────────────────────────────────

#[test]
fn polygon_op_stamps_face_profile_on_terminal() {
    let mut interp = Interpreter::new();
    interp.add_rule(
        "R",
        parse_ops("Polygon((0,0), (4,0), (4,2), (2,2), (2,4), (0,4)) I(\"Mass\")").unwrap(),
    );
    let model = interp.derive(footprint(), "R").unwrap();
    assert_eq!(model.len(), 1);
    match &model.terminals[0].face_profile {
        FaceProfile::Polygon(verts) => {
            assert_eq!(verts.len(), 6);
        }
        other => panic!("expected Polygon profile, got {other:?}"),
    }
}

#[test]
fn polygon_op_overrides_taper() {
    // Taper is set first, then Polygon — Polygon must win since both write
    // face_profile_override and Polygon comes later in the op sequence.
    let mut interp = Interpreter::new();
    interp.add_rule(
        "R",
        parse_ops("Taper(0.5) Polygon((0,0), (1,0), (0,1)) I(\"X\")").unwrap(),
    );
    let model = interp.derive(footprint(), "R").unwrap();
    assert!(matches!(
        model.terminals[0].face_profile,
        FaceProfile::Polygon(_)
    ));
}

#[test]
fn polygon_op_propagates_through_implicit_terminal() {
    // Reference an unknown rule name — the implicit terminal still carries the
    // Polygon override stamped earlier in the sequence.
    let mut interp = Interpreter::new();
    interp.add_rule(
        "R",
        parse_ops("Polygon((0,0), (2,0), (2,2), (0,2)) Floor").unwrap(),
    );
    let model = interp.derive(footprint(), "R").unwrap();
    assert_eq!(model.len(), 1);
    assert_eq!(model.terminals[0].mesh_id, "Floor");
    assert!(matches!(
        model.terminals[0].face_profile,
        FaceProfile::Polygon(_)
    ));
}

// ── Polygon profile yields correct mass properties (cross-check with #29) ────

#[test]
fn polygon_op_then_density_yields_mass_properties() {
    let mut interp = Interpreter::new();
    interp.add_rule(
        "R",
        parse_ops(r#"Mat("Concrete", 2400) Polygon((0,0), (4,0), (4,4), (0,4)) I("Slab")"#)
            .unwrap(),
    );
    // Footprint has Y=1m so the mass = area · y · density = 16 · 1 · 2400 = 38_400 kg.
    let model = interp.derive(footprint(), "R").unwrap();
    let mp = model.terminals[0]
        .mass_properties
        .as_ref()
        .expect("polygon + density → mass_properties present");
    assert!(
        (mp.mass - 38_400.0).abs() < 1e-6,
        "expected 38400 kg, got {}",
        mp.mass
    );
}

// ── Direct construction (genetic mutation path) still works ──────────────────

#[test]
fn direct_shape_op_polygon_construction() {
    let verts = vec![
        glam::DVec2::new(0.0, 0.0),
        glam::DVec2::new(2.0, 0.0),
        glam::DVec2::new(2.0, 2.0),
        glam::DVec2::new(0.0, 2.0),
    ];
    let mut interp = Interpreter::new();
    interp.add_rule(
        "R",
        vec![ShapeOp::Polygon(verts), ShapeOp::I("Mass".to_string())],
    );
    let model = interp.derive(footprint(), "R").unwrap();
    assert!(matches!(
        model.terminals[0].face_profile,
        FaceProfile::Polygon(ref v) if v.len() == 4
    ));
}

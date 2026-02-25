use symbios_shape::grammar::{parse_ops, parse_rule};
use symbios_shape::ops::{Axis, CompTarget, FaceSelector, ShapeOp, SplitSize};

// ── Extrude ──────────────────────────────────────────────────────────────────

#[test]
fn parse_extrude_integer() {
    let ops = parse_ops("Extrude(5)").unwrap();
    assert_eq!(ops, vec![ShapeOp::Extrude(5.0)]);
}

#[test]
fn parse_extrude_float() {
    let ops = parse_ops("Extrude(3.75)").unwrap();
    assert_eq!(ops, vec![ShapeOp::Extrude(3.75)]);
}

#[test]
fn parse_extrude_zero_rejected() {
    assert!(parse_ops("Extrude(0)").is_err());
}

#[test]
fn parse_extrude_negative_rejected() {
    assert!(parse_ops("Extrude(-1)").is_err());
}

// ── Taper ─────────────────────────────────────────────────────────────────────

#[test]
fn parse_taper_boundary_values() {
    assert!(parse_ops("Taper(0.0)").is_ok());
    assert!(parse_ops("Taper(1.0)").is_ok());
}

#[test]
fn parse_taper_out_of_range() {
    assert!(parse_ops("Taper(1.1)").is_err());
    assert!(parse_ops("Taper(-0.1)").is_err());
}

// ── Split ─────────────────────────────────────────────────────────────────────

#[test]
fn parse_split_x_absolute() {
    let ops = parse_ops("Split(X) { 3.0: Left | 7.0: Right }").unwrap();
    let ShapeOp::Split { axis, slots } = &ops[0] else {
        panic!()
    };
    assert_eq!(*axis, Axis::X);
    assert_eq!(slots[0].size, SplitSize::Absolute(3.0));
    assert_eq!(slots[0].rule, "Left");
    assert_eq!(slots[1].size, SplitSize::Absolute(7.0));
    assert_eq!(slots[1].rule, "Right");
}

#[test]
fn parse_split_z_floating() {
    let ops = parse_ops("Split(Z) { ~2.0: Front | ~1.0: Back }").unwrap();
    let ShapeOp::Split { axis, slots } = &ops[0] else {
        panic!()
    };
    assert_eq!(*axis, Axis::Z);
    assert_eq!(slots[0].size, SplitSize::Floating(2.0));
    assert_eq!(slots[1].size, SplitSize::Floating(1.0));
}

#[test]
fn parse_split_relative_sizes() {
    let ops = parse_ops("Split(Y) { '0.25: A | '0.75: B }").unwrap();
    let ShapeOp::Split { slots, .. } = &ops[0] else {
        panic!()
    };
    assert_eq!(slots[0].size, SplitSize::Relative(0.25));
    assert_eq!(slots[1].size, SplitSize::Relative(0.75));
}

#[test]
fn parse_split_trailing_pipe_allowed() {
    // Trailing `|` should not cause a parse error.
    assert!(parse_ops("Split(Y) { ~1: A | ~1: B | }").is_ok());
}

#[test]
fn parse_split_quoted_rule_name() {
    let ops = parse_ops(r#"Split(Y) { 3.0: "Ground Floor" | ~1.0: Upper }"#).unwrap();
    let ShapeOp::Split { slots, .. } = &ops[0] else {
        panic!()
    };
    assert_eq!(slots[0].rule, "Ground Floor");
}

// ── Repeat ────────────────────────────────────────────────────────────────────

#[test]
fn parse_repeat_x() {
    let ops = parse_ops("Repeat(X, 2.5) { Window }").unwrap();
    assert_eq!(
        ops[0],
        ShapeOp::Repeat {
            axis: Axis::X,
            tile_size: 2.5,
            rule: "Window".to_string()
        }
    );
}

#[test]
fn parse_repeat_zero_tile_rejected() {
    assert!(parse_ops("Repeat(Y, 0) { Floor }").is_err());
}

// ── Comp ──────────────────────────────────────────────────────────────────────

#[test]
fn parse_comp_faces_all_six() {
    let ops = parse_ops(
        "Comp(Faces) { Top: Roof | Bottom: Base | Front: Wall | Back: Wall | Left: Wall | Right: Wall }",
    )
    .unwrap();
    let ShapeOp::Comp(CompTarget::Faces(cases)) = &ops[0] else {
        panic!()
    };
    assert_eq!(cases.len(), 6);
    assert_eq!(cases[0].selector, FaceSelector::Top);
    assert_eq!(cases[0].rule, "Roof");
}

#[test]
fn parse_comp_faces_side_shorthand() {
    let ops = parse_ops("Comp(Faces) { Top: Roof | Side: Facade }").unwrap();
    let ShapeOp::Comp(CompTarget::Faces(cases)) = &ops[0] else {
        panic!()
    };
    assert_eq!(cases[1].selector, FaceSelector::Side);
    assert_eq!(cases[1].rule, "Facade");
}

#[test]
fn parse_comp_unknown_selector_rejected() {
    assert!(parse_ops("Comp(Faces) { Banana: Roof }").is_err());
}

// ── I (instance) ──────────────────────────────────────────────────────────────

#[test]
fn parse_instance_quoted() {
    let ops = parse_ops(r#"I("PillarMesh")"#).unwrap();
    assert_eq!(ops[0], ShapeOp::I("PillarMesh".to_string()));
}

#[test]
fn parse_instance_unquoted() {
    let ops = parse_ops("I(Window)").unwrap();
    assert_eq!(ops[0], ShapeOp::I("Window".to_string()));
}

// ── Rule reference ────────────────────────────────────────────────────────────

#[test]
fn parse_rule_ref_simple() {
    let ops = parse_ops("Facade").unwrap();
    assert_eq!(ops[0], ShapeOp::Rule("Facade".to_string()));
}

// ── Named grammar rule ────────────────────────────────────────────────────────

#[test]
fn parse_named_rule_round_trip() {
    let rule = parse_rule("Lot --> Extrude(10) Split(Y) { ~1: Floor | 2: Roof }").unwrap();
    assert_eq!(rule.name, "Lot");
    assert_eq!(rule.ops().len(), 2);
    assert_eq!(rule.ops()[0], ShapeOp::Extrude(10.0));
}

// ── Multi-op sequences ────────────────────────────────────────────────────────

#[test]
fn parse_transform_chain() {
    let ops = parse_ops("Extrude(5) Scale(1.0, 0.9, 1.0) Taper(0.5)").unwrap();
    assert_eq!(ops.len(), 3);
}

#[test]
fn parse_empty_input() {
    let ops = parse_ops("").unwrap();
    assert!(ops.is_empty());
}

#[test]
fn parse_whitespace_only() {
    let ops = parse_ops("   \n\t  ").unwrap();
    assert!(ops.is_empty());
}

#[test]
fn parse_block_comment_between_ops() {
    let ops = parse_ops("Extrude(8) /* sets height */ I(\"Mass\")").unwrap();
    assert_eq!(ops.len(), 2);
}

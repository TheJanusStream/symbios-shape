use symbios_shape::grammar::parse_ops;
use symbios_shape::ops::{
    Axis, CompFaceCase, CompTarget, FaceSelector, OffsetCase, OffsetSelector, RoofCase,
    RoofFaceSelector, RoofType, ShapeOp, SplitSize, SplitSlot,
};
use symbios_shape::{Interpreter, Quat, Scope, ShapeError, Vec3};

fn slot(size: SplitSize, rule: &str) -> SplitSlot {
    SplitSlot {
        size,
        rule: rule.to_string(),
    }
}

fn unit_scope() -> Scope {
    Scope::new(Vec3::ZERO, Quat::IDENTITY, Vec3::new(10.0, 10.0, 10.0))
}

// ── Terminal output ───────────────────────────────────────────────────────────

#[test]
fn single_terminal_direct() {
    let mut interp = Interpreter::new();
    interp.add_rule("Root", vec![ShapeOp::I("Cube".to_string())]);
    let model = interp.derive(unit_scope(), "Root").unwrap();
    assert_eq!(model.len(), 1);
    assert_eq!(model.terminals[0].mesh_id, "Cube");
}

#[test]
fn unknown_rule_becomes_implicit_terminal() {
    let interp = Interpreter::new();
    // No rules registered — "Foo" should resolve to a Terminal with mesh_id "Foo"
    let model = interp.derive(unit_scope(), "Foo").unwrap();
    assert_eq!(model.len(), 1);
    assert_eq!(model.terminals[0].mesh_id, "Foo");
}

#[test]
fn empty_ops_produces_no_terminals() {
    let mut interp = Interpreter::new();
    interp.add_rule("Nil", vec![]);
    let model = interp.derive(unit_scope(), "Nil").unwrap();
    assert!(model.is_empty());
}

// ── Extrude ───────────────────────────────────────────────────────────────────

#[test]
fn extrude_sets_y_size() {
    let mut interp = Interpreter::new();
    interp.add_rule(
        "R",
        vec![ShapeOp::Extrude(7.5), ShapeOp::I("Box".to_string())],
    );
    let scope = Scope::new(Vec3::ZERO, Quat::IDENTITY, Vec3::new(10.0, 0.0, 10.0));
    let model = interp.derive(scope, "R").unwrap();
    assert!((model.terminals[0].scope.size.y - 7.5).abs() < 1e-9);
}

// ── Taper ─────────────────────────────────────────────────────────────────────

#[test]
fn taper_propagates_to_terminal() {
    let mut interp = Interpreter::new();
    interp.add_rule(
        "R",
        vec![ShapeOp::Taper(0.6), ShapeOp::I("Cone".to_string())],
    );
    let model = interp.derive(unit_scope(), "R").unwrap();
    assert!((model.terminals[0].taper - 0.6).abs() < 1e-9);
}

#[test]
fn no_taper_defaults_to_zero() {
    let mut interp = Interpreter::new();
    interp.add_rule("R", vec![ShapeOp::I("Box".to_string())]);
    let model = interp.derive(unit_scope(), "R").unwrap();
    assert_eq!(model.terminals[0].taper, 0.0);
}

// ── Scale ─────────────────────────────────────────────────────────────────────

#[test]
fn scale_multiplies_size() {
    let mut interp = Interpreter::new();
    interp.add_rule(
        "R",
        vec![
            ShapeOp::Scale(Vec3::new(0.5, 2.0, 1.0)),
            ShapeOp::I("S".to_string()),
        ],
    );
    let scope = Scope::new(Vec3::ZERO, Quat::IDENTITY, Vec3::new(4.0, 3.0, 2.0));
    let model = interp.derive(scope, "R").unwrap();
    let s = model.terminals[0].scope.size;
    assert!((s.x - 2.0).abs() < 1e-9);
    assert!((s.y - 6.0).abs() < 1e-9);
    assert!((s.z - 2.0).abs() < 1e-9);
}

// ── Translate ─────────────────────────────────────────────────────────────────

#[test]
fn translate_moves_position() {
    let mut interp = Interpreter::new();
    interp.add_rule(
        "R",
        vec![
            ShapeOp::Translate(Vec3::new(1.0, 2.0, 3.0)),
            ShapeOp::I("T".to_string()),
        ],
    );
    let scope = Scope::new(Vec3::ZERO, Quat::IDENTITY, Vec3::ONE);
    let model = interp.derive(scope, "R").unwrap();
    let p = model.terminals[0].scope.position;
    assert!((p.x - 1.0).abs() < 1e-9);
    assert!((p.y - 2.0).abs() < 1e-9);
    assert!((p.z - 3.0).abs() < 1e-9);
}

// ── Split ─────────────────────────────────────────────────────────────────────

#[test]
fn split_y_positions_are_sequential() {
    let mut interp = Interpreter::new();
    interp.add_rule(
        "R",
        vec![ShapeOp::Split {
            axis: Axis::Y,
            slots: vec![
                slot(SplitSize::Absolute(3.0), "A"),
                slot(SplitSize::Absolute(4.0), "B"),
                slot(SplitSize::Absolute(3.0), "C"),
            ],
        }],
    );
    let scope = Scope::new(Vec3::ZERO, Quat::IDENTITY, Vec3::new(10.0, 10.0, 10.0));
    let model = interp.derive(scope, "R").unwrap();
    assert_eq!(model.len(), 3);
    assert!((model.terminals[0].scope.position.y - 0.0).abs() < 1e-9);
    assert!((model.terminals[1].scope.position.y - 3.0).abs() < 1e-9);
    assert!((model.terminals[2].scope.position.y - 7.0).abs() < 1e-9);
    assert!((model.terminals[0].scope.size.y - 3.0).abs() < 1e-9);
    assert!((model.terminals[1].scope.size.y - 4.0).abs() < 1e-9);
    assert!((model.terminals[2].scope.size.y - 3.0).abs() < 1e-9);
}

#[test]
fn split_floating_fills_remainder() {
    let mut interp = Interpreter::new();
    interp.add_rule(
        "R",
        vec![ShapeOp::Split {
            axis: Axis::Y,
            slots: vec![
                slot(SplitSize::Absolute(2.0), "Base"),
                slot(SplitSize::Floating(3.0), "Mid"),
                slot(SplitSize::Floating(1.0), "Top"),
            ],
        }],
    );
    let scope = Scope::new(Vec3::ZERO, Quat::IDENTITY, Vec3::new(10.0, 10.0, 10.0));
    let model = interp.derive(scope, "R").unwrap();
    // Remaining after Absolute(2): 8.0. Ratios 3:1 → 6.0 and 2.0
    assert!((model.terminals[0].scope.size.y - 2.0).abs() < 1e-9);
    assert!((model.terminals[1].scope.size.y - 6.0).abs() < 1e-9);
    assert!((model.terminals[2].scope.size.y - 2.0).abs() < 1e-9);
}

#[test]
fn split_x_widths_correct() {
    let mut interp = Interpreter::new();
    // Rule name "Facade" avoids collision with slot names "L" and "R"
    interp.add_rule(
        "Facade",
        vec![ShapeOp::Split {
            axis: Axis::X,
            slots: vec![
                slot(SplitSize::Floating(1.0), "L"),
                slot(SplitSize::Floating(1.0), "R"),
            ],
        }],
    );
    let scope = Scope::new(Vec3::ZERO, Quat::IDENTITY, Vec3::new(6.0, 4.0, 4.0));
    let model = interp.derive(scope, "Facade").unwrap();
    assert!((model.terminals[0].scope.size.x - 3.0).abs() < 1e-9);
    assert!((model.terminals[1].scope.size.x - 3.0).abs() < 1e-9);
    assert!((model.terminals[1].scope.position.x - 3.0).abs() < 1e-9);
}

// ── Repeat ────────────────────────────────────────────────────────────────────

#[test]
fn repeat_tile_count_and_sizes() {
    let mut interp = Interpreter::new();
    interp.add_rule(
        "R",
        vec![ShapeOp::Repeat {
            axis: Axis::X,
            tile_size: 2.0,
            rule: "W".to_string(),
        }],
    );
    let scope = Scope::new(Vec3::ZERO, Quat::IDENTITY, Vec3::new(7.0, 3.0, 1.0));
    let model = interp.derive(scope, "R").unwrap();
    // floor(7/2) = 3 tiles; actual size stretched to 7.0/3 to fill with no gap
    assert_eq!(model.len(), 3);
    let actual = 7.0_f64 / 3.0;
    assert!((model.terminals[0].scope.size.x - actual).abs() < 1e-9);
    assert!((model.terminals[2].scope.position.x - 2.0 * actual).abs() < 1e-9);
}

#[test]
fn repeat_zero_tiles_when_scope_too_small() {
    let mut interp = Interpreter::new();
    interp.add_rule(
        "R",
        vec![ShapeOp::Repeat {
            axis: Axis::X,
            tile_size: 5.0,
            rule: "W".to_string(),
        }],
    );
    let scope = Scope::new(Vec3::ZERO, Quat::IDENTITY, Vec3::new(4.0, 3.0, 1.0));
    let model = interp.derive(scope, "R").unwrap();
    assert!(model.is_empty());
}

// ── Comp ──────────────────────────────────────────────────────────────────────

#[test]
fn comp_faces_emits_six_terminals() {
    let mut interp = Interpreter::new();
    interp.add_rule(
        "R",
        vec![ShapeOp::Comp(CompTarget::Faces(vec![CompFaceCase {
            selector: FaceSelector::All,
            rule: "Face".to_string(),
        }]))],
    );
    let model = interp.derive(unit_scope(), "R").unwrap();
    assert_eq!(model.len(), 6);
}

#[test]
fn comp_faces_top_rule_selected() {
    let mut interp = Interpreter::new();
    interp.add_rule(
        "R",
        vec![ShapeOp::Comp(CompTarget::Faces(vec![
            CompFaceCase {
                selector: FaceSelector::Top,
                rule: "Roof".to_string(),
            },
            CompFaceCase {
                selector: FaceSelector::All,
                rule: "Wall".to_string(),
            },
        ]))],
    );
    let model = interp.derive(unit_scope(), "R").unwrap();
    // Top → "Roof", all others → "Wall"
    let roofs: Vec<_> = model
        .terminals
        .iter()
        .filter(|t| t.mesh_id == "Roof")
        .collect();
    let walls: Vec<_> = model
        .terminals
        .iter()
        .filter(|t| t.mesh_id == "Wall")
        .collect();
    assert_eq!(roofs.len(), 1);
    assert_eq!(walls.len(), 5);
}

#[test]
fn comp_faces_side_matches_four_walls() {
    let mut interp = Interpreter::new();
    interp.add_rule(
        "R",
        vec![ShapeOp::Comp(CompTarget::Faces(vec![
            CompFaceCase {
                selector: FaceSelector::Top,
                rule: "Top".to_string(),
            },
            CompFaceCase {
                selector: FaceSelector::Bottom,
                rule: "Bot".to_string(),
            },
            CompFaceCase {
                selector: FaceSelector::Side,
                rule: "Side".to_string(),
            },
        ]))],
    );
    let model = interp.derive(unit_scope(), "R").unwrap();
    let sides: Vec<_> = model
        .terminals
        .iter()
        .filter(|t| t.mesh_id == "Side")
        .collect();
    assert_eq!(sides.len(), 4);
}

#[test]
fn comp_faces_unmapped_face_skipped() {
    let mut interp = Interpreter::new();
    // Only map Top — bottom and sides have no rule
    interp.add_rule(
        "R",
        vec![ShapeOp::Comp(CompTarget::Faces(vec![CompFaceCase {
            selector: FaceSelector::Top,
            rule: "Roof".to_string(),
        }]))],
    );
    let model = interp.derive(unit_scope(), "R").unwrap();
    assert_eq!(model.len(), 1);
}

// ── Rule delegation ───────────────────────────────────────────────────────────

#[test]
fn rule_delegates_to_subrule() {
    let mut interp = Interpreter::new();
    interp.add_rule("A", vec![ShapeOp::Rule("B".to_string())]);
    interp.add_rule("B", vec![ShapeOp::I("Leaf".to_string())]);
    let model = interp.derive(unit_scope(), "A").unwrap();
    assert_eq!(model.len(), 1);
    assert_eq!(model.terminals[0].mesh_id, "Leaf");
}

// ── Safety limits ─────────────────────────────────────────────────────────────

#[test]
fn depth_limit_enforced() {
    let mut interp = Interpreter::new();
    interp.add_rule("A", vec![ShapeOp::Rule("A".to_string())]);
    interp.max_depth = 10;
    assert!(matches!(
        interp.derive(unit_scope(), "A"),
        Err(ShapeError::DepthLimitExceeded(_))
    ));
}

#[test]
fn terminal_limit_enforced() {
    let mut interp = Interpreter::new();
    // Repeat 100 tiles, but cap at 5 terminals
    interp.add_rule(
        "R",
        vec![ShapeOp::Repeat {
            axis: Axis::X,
            tile_size: 1.0,
            rule: "T".to_string(),
        }],
    );
    interp.max_terminals = 5;
    let scope = Scope::new(Vec3::ZERO, Quat::IDENTITY, Vec3::new(100.0, 1.0, 1.0));
    assert!(matches!(
        interp.derive(scope, "R"),
        Err(ShapeError::CapacityOverflow)
    ));
}

#[test]
fn invalid_scope_rejected() {
    let interp = Interpreter::new();
    let bad = Scope::new(Vec3::new(f64::NAN, 0.0, 0.0), Quat::IDENTITY, Vec3::ONE);
    assert!(matches!(
        interp.derive(bad, "R"),
        Err(ShapeError::InvalidNumericValue)
    ));
}

// ── Parser + interpreter round-trip ──────────────────────────────────────────

#[test]
fn parse_and_derive_building() {
    let mut interp = Interpreter::new();
    interp.add_rule(
        "Lot",
        parse_ops("Extrude(12) Split(Y) { 3: Ground | ~1: Upper | 2: Roof }").unwrap(),
    );
    interp.add_rule("Ground", parse_ops(r#"I("GroundFloor")"#).unwrap());
    interp.add_rule("Upper", parse_ops(r#"I("Floor")"#).unwrap());
    interp.add_rule("Roof", parse_ops(r#"Taper(0.8) I("Roof")"#).unwrap());

    let footprint = Scope::new(Vec3::ZERO, Quat::IDENTITY, Vec3::new(10.0, 0.0, 10.0));
    let model = interp.derive(footprint, "Lot").unwrap();

    assert_eq!(model.len(), 3);
    assert_eq!(model.terminals[0].mesh_id, "GroundFloor");
    assert!((model.terminals[0].scope.size.y - 3.0).abs() < 1e-9);
    assert_eq!(model.terminals[2].mesh_id, "Roof");
    assert!((model.terminals[2].taper - 0.8).abs() < 1e-9);
}

// ── Infinity / NaN propagation guards ────────────────────────────────────────

/// Issue 1: large-but-finite translate vectors that overflow position to Infinity.
/// MAX/2 + MAX/2 = MAX (still finite); MAX + MAX = Infinity (caught).
#[test]
fn translate_overflow_to_infinity_rejected() {
    let mut interp = Interpreter::new();
    interp.add_rule(
        "R",
        vec![
            ShapeOp::Translate(Vec3::new(f64::MAX, 0.0, 0.0)),
            ShapeOp::Translate(Vec3::new(f64::MAX, 0.0, 0.0)),
            ShapeOp::I("Leaf".to_string()),
        ],
    );
    assert!(matches!(
        interp.derive(unit_scope(), "R"),
        Err(ShapeError::InvalidNumericValue)
    ));
}

/// Issue 2: Align target whose length_squared overflows to Infinity must be rejected.
#[test]
fn align_large_target_length_sq_overflow_rejected() {
    let mut interp = Interpreter::new();
    // (1e200, 1e200, 1e200) is finite component-wise but length_squared = 3e400 = Infinity.
    interp.add_rule(
        "R",
        vec![
            ShapeOp::Align {
                local_axis: Axis::Y,
                target: Vec3::new(1e200, 1e200, 1e200),
            },
            ShapeOp::I("Leaf".to_string()),
        ],
    );
    assert!(matches!(
        interp.derive(unit_scope(), "R"),
        Err(ShapeError::InvalidAlignTarget)
    ));
}

/// Issue 3: A steep-but-valid angle (89°) combined with a very large scope depth
/// causes `fb_len = (sz/2) / cos(89°)` to overflow to Infinity.
/// With sz=1e307: 5e306 / 0.01745 ≈ 2.87e308 > f64::MAX → Infinity.
#[test]
fn roof_trig_overflow_large_scope_rejected() {
    let mut interp = Interpreter::new();
    interp.add_rule(
        "R",
        vec![ShapeOp::Roof {
            roof_type: RoofType::Gable,
            angle: 89.0,
            overhang: 0.0,
            cases: vec![RoofCase {
                selector: RoofFaceSelector::Slope,
                rule: "Tiles".to_string(),
            }],
        }],
    );
    let big_scope = Scope::new(Vec3::ZERO, Quat::IDENTITY, Vec3::new(10.0, 0.0, 1e307));
    assert!(matches!(
        interp.derive(big_scope, "R"),
        Err(ShapeError::InvalidNumericValue)
    ));
}

/// Issue 3 (variant): a large-but-finite overhang overflows slope length.
/// overhang = MAX*0.9 → (MAX*0.9) / cos(45°) ≈ MAX*1.27 → Infinity.
#[test]
fn roof_large_overhang_overflow_rejected() {
    let mut interp = Interpreter::new();
    interp.add_rule(
        "R",
        vec![ShapeOp::Roof {
            roof_type: RoofType::Pyramid,
            angle: 45.0,
            overhang: f64::MAX * 0.9,
            cases: vec![RoofCase {
                selector: RoofFaceSelector::Slope,
                rule: "Tiles".to_string(),
            }],
        }],
    );
    assert!(matches!(
        interp.derive(unit_scope(), "R"),
        Err(ShapeError::InvalidNumericValue)
    ));
}

/// Issue 4: Repeat with a non-finite scope size (defensive guard).
#[test]
fn repeat_non_finite_total_rejected() {
    let mut interp = Interpreter::new();
    interp.add_rule(
        "R",
        vec![ShapeOp::Repeat {
            axis: Axis::X,
            tile_size: 1.0,
            rule: "Leaf".to_string(),
        }],
    );
    // Construct a scope with an Infinity X size, bypassing the normal API.
    let bad_scope = Scope::new(Vec3::ZERO, Quat::IDENTITY, Vec3::new(f64::INFINITY, 1.0, 1.0));
    // The scope fails validate() first (Infinity size), which is fine — the guard fires.
    assert!(interp.derive(bad_scope, "R").is_err());
}

/// Issue 5: Offset with NaN scope dimensions must not bypass the boundary check.
#[test]
fn offset_nan_scope_size_rejected() {
    let mut interp = Interpreter::new();
    interp.add_rule(
        "R",
        vec![ShapeOp::Offset {
            distance: -0.2,
            cases: vec![OffsetCase {
                selector: OffsetSelector::Inside,
                rule: "Leaf".to_string(),
            }],
        }],
    );
    let nan_scope = Scope::new(Vec3::ZERO, Quat::IDENTITY, Vec3::new(f64::NAN, 1.0, 1.0));
    // validate() catches NaN scope size before we even reach Offset — either way, an error.
    assert!(interp.derive(nan_scope, "R").is_err());
}

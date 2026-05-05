//! Integration tests for snap-line / alignment plane primitives (issue #35).
//!
//! Covers `RegSnap("label")` registration and the snap-aware variant of
//! `Split(axis, snap="label", tol=N) { … }`.

use symbios_shape::grammar::parse_ops;
use symbios_shape::ops::{Axis, ShapeOp, SnapBinding, SplitSize, SplitSlot};
use symbios_shape::{Interpreter, Quat, Scope, Vec3};

fn unit_scope() -> Scope {
    Scope::new(Vec3::ZERO, Quat::IDENTITY, Vec3::new(10.0, 10.0, 10.0))
}

// ── RegSnap registers six face planes ────────────────────────────────────────

#[test]
fn reg_snap_registers_six_face_planes() {
    let mut interp = Interpreter::new();
    interp.add_rule(
        "R",
        vec![
            ShapeOp::RegSnap("bays".to_string()),
            ShapeOp::I("Mass".to_string()),
        ],
    );
    let model = interp.derive(unit_scope(), "R").unwrap();
    let bays: Vec<_> = model
        .snap_planes
        .iter()
        .filter(|p| p.label == "bays")
        .collect();
    assert_eq!(bays.len(), 6, "should register all 6 face planes");
}

#[test]
fn reg_snap_world_planes_match_scope_faces() {
    let mut interp = Interpreter::new();
    interp.add_rule(
        "R",
        vec![
            ShapeOp::RegSnap("bays".to_string()),
            ShapeOp::I("Mass".to_string()),
        ],
    );
    let scope = Scope::new(Vec3::ZERO, Quat::IDENTITY, Vec3::new(10.0, 4.0, 6.0));
    let model = interp.derive(scope, "R").unwrap();
    // Find the +X face: normal +X, point.x = 10.
    let plus_x = model
        .snap_planes
        .iter()
        .find(|p| p.normal.x > 0.99)
        .expect("+X plane");
    assert!((plus_x.point.x - 10.0).abs() < 1e-9);
    // -Z face: normal -Z, point.z = 0.
    let minus_z = model
        .snap_planes
        .iter()
        .find(|p| p.normal.z < -0.99)
        .expect("-Z plane");
    assert!((minus_z.point.z).abs() < 1e-9);
}

#[test]
fn reg_snap_via_grammar() {
    let ops = parse_ops("RegSnap(\"bays\") I(\"Mass\")").unwrap();
    let mut interp = Interpreter::new();
    interp.add_rule("R", ops);
    let model = interp.derive(unit_scope(), "R").unwrap();
    assert!(model.snap_planes.iter().any(|p| p.label == "bays"));
}

// ── Snap-aware Split adjusts boundaries ──────────────────────────────────────

#[test]
fn snap_split_aligns_to_registered_plane() {
    // BFS-order note: a snap-aware Split sees only snap-planes registered by
    // rules already drained from the queue. To make the Upper snap consult
    // Ground-bay edges, we add one rule of indirection on Upper so its
    // snap-Split is one BFS step later than Ground's bay rules.
    let mut interp = Interpreter::new();
    interp.add_rule(
        "Lot",
        vec![ShapeOp::Split {
            axis: Axis::Y,
            slots: vec![
                SplitSlot {
                    size: SplitSize::Absolute(5.0),
                    rule: "Ground".to_string(),
                },
                SplitSlot {
                    size: SplitSize::Absolute(5.0),
                    rule: "Upper".to_string(),
                },
            ],
            snap: None,
        }],
    );
    interp.add_rule(
        "Ground",
        vec![ShapeOp::Split {
            axis: Axis::X,
            slots: vec![
                SplitSlot {
                    size: SplitSize::Absolute(5.0),
                    rule: "GroundLeft".to_string(),
                },
                SplitSlot {
                    size: SplitSize::Absolute(5.0),
                    rule: "GroundRight".to_string(),
                },
            ],
            snap: None,
        }],
    );
    interp.add_rule(
        "GroundLeft",
        vec![
            // Register face planes — includes the right-edge X=5 boundary.
            ShapeOp::RegSnap("bays".to_string()),
            ShapeOp::I("LeftFloor".to_string()),
        ],
    );
    interp.add_rule("GroundRight", vec![ShapeOp::I("RightFloor".to_string())]);
    // Indirection: Upper delegates to UpperImpl. This makes UpperImpl pop
    // one BFS step later, after the GroundLeft RegSnap has run.
    interp.add_rule("Upper", vec![ShapeOp::Rule("UpperImpl".to_string())]);
    interp.add_rule(
        "UpperImpl",
        vec![ShapeOp::Split {
            axis: Axis::X,
            slots: vec![
                // Without snap → boundary at X=4.5; with snap="bays" + tol=1
                // it must snap to X=5 (Ground bay's right edge).
                SplitSlot {
                    size: SplitSize::Absolute(4.5),
                    rule: "UpperBay".to_string(),
                },
                SplitSlot {
                    size: SplitSize::Absolute(5.5),
                    rule: "UpperBay".to_string(),
                },
            ],
            snap: Some(SnapBinding {
                label: "bays".to_string(),
                tolerance: Some(1.0),
            }),
        }],
    );
    interp.add_rule("UpperBay", vec![ShapeOp::I("Bay".to_string())]);

    let scope = Scope::new(Vec3::ZERO, Quat::IDENTITY, Vec3::new(10.0, 10.0, 10.0));
    let model = interp.derive(scope, "Lot").unwrap();

    let upper_bays: Vec<_> = model
        .terminals
        .iter()
        .filter(|t| t.mesh_id == "Bay")
        .collect();
    assert_eq!(upper_bays.len(), 2);
    let widths: Vec<f64> = upper_bays.iter().map(|t| t.scope.size.x).collect();
    assert!(
        (widths[0] - 5.0).abs() < 1e-6,
        "expected snapped width 5.0, got {}",
        widths[0]
    );
    assert!(
        (widths[1] - 5.0).abs() < 1e-6,
        "expected snapped width 5.0, got {}",
        widths[1]
    );
}

#[test]
fn snap_split_does_nothing_when_outside_tolerance() {
    // Same Lot/Ground/GL/GR/Upper/UpperImpl structure as the snap-success
    // test, but with tolerance = 0.1 (smaller than the 0.5 boundary offset)
    // — boundaries should remain at the unsnapped positions.
    let mut interp = Interpreter::new();
    interp.add_rule(
        "Lot",
        vec![ShapeOp::Split {
            axis: Axis::Y,
            slots: vec![
                SplitSlot {
                    size: SplitSize::Absolute(5.0),
                    rule: "Ground".to_string(),
                },
                SplitSlot {
                    size: SplitSize::Absolute(5.0),
                    rule: "Upper".to_string(),
                },
            ],
            snap: None,
        }],
    );
    interp.add_rule(
        "Ground",
        vec![ShapeOp::Split {
            axis: Axis::X,
            slots: vec![
                SplitSlot {
                    size: SplitSize::Absolute(5.0),
                    rule: "GL".to_string(),
                },
                SplitSlot {
                    size: SplitSize::Absolute(5.0),
                    rule: "GR".to_string(),
                },
            ],
            snap: None,
        }],
    );
    interp.add_rule(
        "GL",
        vec![
            ShapeOp::RegSnap("bays".to_string()),
            ShapeOp::I("LeftFloor".to_string()),
        ],
    );
    interp.add_rule("GR", vec![ShapeOp::I("RightFloor".to_string())]);
    interp.add_rule("Upper", vec![ShapeOp::Rule("UpperImpl".to_string())]);
    interp.add_rule(
        "UpperImpl",
        vec![ShapeOp::Split {
            axis: Axis::X,
            slots: vec![
                SplitSlot {
                    size: SplitSize::Absolute(4.5),
                    rule: "UB".to_string(),
                },
                SplitSlot {
                    size: SplitSize::Absolute(5.5),
                    rule: "UB".to_string(),
                },
            ],
            snap: Some(SnapBinding {
                label: "bays".to_string(),
                tolerance: Some(0.1),
            }),
        }],
    );
    interp.add_rule("UB", vec![ShapeOp::I("Bay".to_string())]);

    let scope = Scope::new(Vec3::ZERO, Quat::IDENTITY, Vec3::new(10.0, 10.0, 10.0));
    let model = interp.derive(scope, "Lot").unwrap();
    let bays: Vec<_> = model
        .terminals
        .iter()
        .filter(|t| t.mesh_id == "Bay")
        .collect();
    assert_eq!(bays.len(), 2);
    let widths: Vec<f64> = bays.iter().map(|t| t.scope.size.x).collect();
    assert!(
        (widths[0] - 4.5).abs() < 1e-6,
        "expected 4.5 (no snap), got {}",
        widths[0]
    );
    assert!(
        (widths[1] - 5.5).abs() < 1e-6,
        "expected 5.5 (no snap), got {}",
        widths[1]
    );
}

#[test]
fn snap_split_default_tolerance_is_5pct_of_axis() {
    // Without an explicit tolerance, `snap=` defaults to 5% of axis length.
    // For a 10m X axis the default tol = 0.5; a 0.4 offset should snap.
    // Same BFS-order setup as `snap_split_aligns_to_registered_plane`.
    let mut interp = Interpreter::new();
    interp.add_rule(
        "Lot",
        vec![ShapeOp::Split {
            axis: Axis::Y,
            slots: vec![
                SplitSlot {
                    size: SplitSize::Absolute(5.0),
                    rule: "Ground".to_string(),
                },
                SplitSlot {
                    size: SplitSize::Absolute(5.0),
                    rule: "Upper".to_string(),
                },
            ],
            snap: None,
        }],
    );
    interp.add_rule(
        "Ground",
        vec![ShapeOp::Split {
            axis: Axis::X,
            slots: vec![
                SplitSlot {
                    size: SplitSize::Absolute(5.0),
                    rule: "GL".to_string(),
                },
                SplitSlot {
                    size: SplitSize::Absolute(5.0),
                    rule: "GR".to_string(),
                },
            ],
            snap: None,
        }],
    );
    interp.add_rule(
        "GL",
        vec![
            ShapeOp::RegSnap("bays".to_string()),
            ShapeOp::I("LeftFloor".to_string()),
        ],
    );
    interp.add_rule("GR", vec![ShapeOp::I("RightFloor".to_string())]);
    interp.add_rule("Upper", vec![ShapeOp::Rule("UpperImpl".to_string())]);
    interp.add_rule(
        "UpperImpl",
        vec![ShapeOp::Split {
            axis: Axis::X,
            slots: vec![
                SplitSlot {
                    size: SplitSize::Absolute(4.6),
                    rule: "UB".to_string(),
                },
                SplitSlot {
                    size: SplitSize::Absolute(5.4),
                    rule: "UB".to_string(),
                },
            ],
            snap: Some(SnapBinding {
                label: "bays".to_string(),
                tolerance: None, // default = 5% of 10 = 0.5
            }),
        }],
    );
    interp.add_rule("UB", vec![ShapeOp::I("Bay".to_string())]);

    let scope = Scope::new(Vec3::ZERO, Quat::IDENTITY, Vec3::new(10.0, 10.0, 10.0));
    let model = interp.derive(scope, "Lot").unwrap();
    let bays: Vec<_> = model
        .terminals
        .iter()
        .filter(|t| t.mesh_id == "Bay")
        .collect();
    let widths: Vec<f64> = bays.iter().map(|t| t.scope.size.x).collect();
    // 0.4 offset (boundary at 4.6, plane at 5) is within default 0.5 → snaps.
    assert!(
        (widths[0] - 5.0).abs() < 1e-6,
        "expected snap to 5.0, got {}",
        widths[0]
    );
}

#[test]
fn snap_split_grammar_syntax() {
    let ops = parse_ops("Split(X, snap=\"bays\", tol=1.0) { 4.5: A | 5.5: B }").unwrap();
    let ShapeOp::Split { axis, slots, snap } = &ops[0] else {
        panic!("expected Split");
    };
    assert_eq!(*axis, Axis::X);
    assert_eq!(slots.len(), 2);
    let snap = snap.as_ref().expect("snap binding");
    assert_eq!(snap.label, "bays");
    assert_eq!(snap.tolerance, Some(1.0));
}

#[test]
fn snap_split_grammar_syntax_default_tol() {
    let ops = parse_ops("Split(X, snap=\"bays\") { 4.5: A | 5.5: B }").unwrap();
    let ShapeOp::Split { snap, .. } = &ops[0] else {
        panic!("expected Split");
    };
    let snap = snap.as_ref().expect("snap binding");
    assert_eq!(snap.label, "bays");
    assert_eq!(snap.tolerance, None);
}

#[test]
fn snap_split_with_no_matching_label_leaves_boundaries_intact() {
    let mut interp = Interpreter::new();
    interp.add_rule(
        "R",
        vec![
            // Register under a DIFFERENT label than the Split's snap binding.
            ShapeOp::RegSnap("ignored".to_string()),
            ShapeOp::Split {
                axis: Axis::X,
                slots: vec![
                    SplitSlot {
                        size: SplitSize::Absolute(4.5),
                        rule: "A".to_string(),
                    },
                    SplitSlot {
                        size: SplitSize::Absolute(5.5),
                        rule: "B".to_string(),
                    },
                ],
                snap: Some(SnapBinding {
                    label: "bays".to_string(),
                    tolerance: Some(1.0),
                }),
            },
        ],
    );
    let scope = Scope::new(Vec3::ZERO, Quat::IDENTITY, Vec3::new(10.0, 10.0, 10.0));
    let model = interp.derive(scope, "R").unwrap();
    assert!((model.terminals[0].scope.size.x - 4.5).abs() < 1e-6);
    assert!((model.terminals[1].scope.size.x - 5.5).abs() < 1e-6);
}

#[test]
fn snap_planes_cleared_between_derive_calls() {
    let mut interp = Interpreter::new();
    interp.add_rule(
        "R",
        vec![
            ShapeOp::RegSnap("once".to_string()),
            ShapeOp::I("Mass".to_string()),
        ],
    );
    let scope = Scope::new(Vec3::ZERO, Quat::IDENTITY, Vec3::new(1.0, 1.0, 1.0));
    let m1 = interp.derive(scope, "R").unwrap();
    let m2 = interp.derive(scope, "R").unwrap();
    // Each derivation gets a fresh registry — no accumulation across calls.
    assert_eq!(m1.snap_planes.len(), 6);
    assert_eq!(m2.snap_planes.len(), 6);
}

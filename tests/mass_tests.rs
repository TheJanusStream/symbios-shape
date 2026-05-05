//! Integration tests for mass properties (issue #29).
//!
//! Verifies mass / centroid / inertia computation across every `FaceProfile`,
//! that mass propagates through `Mat` + branching ops (`Split`, `Comp`, `Repeat`),
//! and that omitting density leaves `mass_properties = None`.

use symbios_shape::grammar::parse_ops;
use symbios_shape::model::{Material, compute_mass_properties};
use symbios_shape::ops::{Axis, ShapeOp, SplitSize, SplitSlot};
use symbios_shape::{FaceProfile, Interpreter, MassProperties, Quat, Scope, Vec3};

fn unit_box_scope() -> Scope {
    Scope::new(Vec3::ZERO, Quat::IDENTITY, Vec3::new(2.0, 3.0, 4.0))
}

// ── Material plumbing ────────────────────────────────────────────────────────

#[test]
fn mat_without_density_leaves_mass_properties_none() {
    let mut interp = Interpreter::new();
    interp.add_rule(
        "R",
        vec![
            ShapeOp::Mat(Material::new("Brick")),
            ShapeOp::I("Wall".to_string()),
        ],
    );
    let model = interp.derive(unit_box_scope(), "R").unwrap();
    assert!(model.terminals[0].mass_properties.is_none());
    assert_eq!(model.terminals[0].material, Some(Material::new("Brick")));
}

#[test]
fn mat_with_density_populates_mass_and_centroid() {
    let mut interp = Interpreter::new();
    interp.add_rule(
        "R",
        vec![
            ShapeOp::Mat(Material::with_density("Brick", 1800.0)),
            ShapeOp::I("Wall".to_string()),
        ],
    );
    let model = interp.derive(unit_box_scope(), "R").unwrap();
    let mp = model.terminals[0]
        .mass_properties
        .as_ref()
        .expect("mass properties present");
    // Volume = 2 × 3 × 4 = 24 m³; mass = 24 × 1800 = 43_200 kg.
    assert!((mp.mass - 43_200.0).abs() < 1e-6);
    // Centroid of a box at origin with that size = (1, 1.5, 2).
    assert!((mp.centroid - Vec3::new(1.0, 1.5, 2.0)).length() < 1e-9);
}

#[test]
fn mat_with_density_via_grammar() {
    let ops = parse_ops(r#"Mat("Brick", 1800) I("Wall")"#).unwrap();
    let mut interp = Interpreter::new();
    interp.add_rule("R", ops);
    let model = interp.derive(unit_box_scope(), "R").unwrap();
    let mp = model.terminals[0].mass_properties.as_ref().unwrap();
    assert!((mp.mass - 43_200.0).abs() < 1e-6);
}

// ── Box (Rectangle profile) inertia ───────────────────────────────────────────

#[test]
fn box_inertia_matches_closed_form() {
    let scope = Scope::new(Vec3::ZERO, Quat::IDENTITY, Vec3::new(2.0, 3.0, 4.0));
    let mp = compute_mass_properties(&scope, &FaceProfile::Rectangle, 1000.0).unwrap();
    let m = mp.mass;
    let inertia = mp.inertia.unwrap();
    let i_xx = m * (3.0 * 3.0 + 4.0 * 4.0) / 12.0;
    let i_yy = m * (2.0 * 2.0 + 4.0 * 4.0) / 12.0;
    let i_zz = m * (2.0 * 2.0 + 3.0 * 3.0) / 12.0;
    assert!((inertia.x_axis.x - i_xx).abs() < 1e-6);
    assert!((inertia.y_axis.y - i_yy).abs() < 1e-6);
    assert!((inertia.z_axis.z - i_zz).abs() < 1e-6);
    // Off-diagonals zero for an axis-aligned box.
    assert!(inertia.x_axis.y.abs() < 1e-9);
    assert!(inertia.x_axis.z.abs() < 1e-9);
}

#[test]
fn box_inertia_rotates_into_world_frame() {
    // 90° rotation about Y swaps the X and Z dimensions of the inertia tensor.
    let rot = Quat::from_axis_angle(Vec3::Y, std::f64::consts::FRAC_PI_2);
    let scope = Scope::new(Vec3::ZERO, rot, Vec3::new(2.0, 3.0, 4.0));
    let mp = compute_mass_properties(&scope, &FaceProfile::Rectangle, 1000.0).unwrap();
    let inertia = mp.inertia.unwrap();
    let m = mp.mass;
    // After rotation, world-X axis was the body's local Z, world-Z axis was -X.
    let i_local_zz = m * (2.0 * 2.0 + 3.0 * 3.0) / 12.0;
    let i_local_xx = m * (3.0 * 3.0 + 4.0 * 4.0) / 12.0;
    assert!(
        (inertia.x_axis.x - i_local_zz).abs() < 1e-6,
        "world I_xx should be body I_zz, got {}",
        inertia.x_axis.x
    );
    assert!(
        (inertia.z_axis.z - i_local_xx).abs() < 1e-6,
        "world I_zz should be body I_xx, got {}",
        inertia.z_axis.z
    );
}

// ── Triangle / Trapezoid / Polygon ────────────────────────────────────────────

#[test]
fn triangle_volume_is_half_box() {
    let scope = Scope::new(Vec3::ZERO, Quat::IDENTITY, Vec3::new(2.0, 3.0, 4.0));
    let profile = FaceProfile::Triangle { peak_offset: 0.5 };
    let mp = compute_mass_properties(&scope, &profile, 1000.0).unwrap();
    // Area = (1/2)·base·height = 0.5·2·3 = 3; volume = 3·4 = 12; mass = 12_000.
    assert!((mp.mass - 12_000.0).abs() < 1e-6);
    // Centroid of an isoceles triangle: x = sx/2 = 1.0, y = sy/3 = 1.0, z = sz/2 = 2.0.
    assert!((mp.centroid - Vec3::new(1.0, 1.0, 2.0)).length() < 1e-9);
    // Inertia is populated (closed-form is implemented).
    assert!(mp.inertia.is_some());
}

#[test]
fn triangle_inertia_diagonal_for_symmetric_triangle() {
    let scope = Scope::new(Vec3::ZERO, Quat::IDENTITY, Vec3::new(4.0, 6.0, 2.0));
    let profile = FaceProfile::Triangle { peak_offset: 0.5 };
    let mp = compute_mass_properties(&scope, &profile, 1.0).unwrap();
    let inertia = mp.inertia.unwrap();
    // Symmetric triangle has zero product of inertia about its own centroid.
    assert!(inertia.x_axis.y.abs() < 1e-9);
}

#[test]
fn trapezoid_volume_matches_average_width_formula() {
    let scope = Scope::new(Vec3::ZERO, Quat::IDENTITY, Vec3::new(2.0, 3.0, 4.0));
    // Symmetric trapezoid: top_width = 0.5, offset_x = 0.25.
    let profile = FaceProfile::Trapezoid {
        top_width: 0.5,
        offset_x: 0.25,
    };
    let mp = compute_mass_properties(&scope, &profile, 1000.0).unwrap();
    // Area = sx·sy·(1 + top_width)/2 = 2·3·0.75 = 4.5; volume = 4.5·4 = 18; mass = 18_000.
    assert!(
        (mp.mass - 18_000.0).abs() < 1e-6,
        "expected 18000, got {}",
        mp.mass
    );
}

#[test]
fn polygon_volume_matches_shoelace_area_times_height() {
    // Unit square footprint in XZ plane: vertices (0,0), (2,0), (2,2), (0,2).
    let verts = vec![
        glam::DVec2::new(0.0, 0.0),
        glam::DVec2::new(2.0, 0.0),
        glam::DVec2::new(2.0, 2.0),
        glam::DVec2::new(0.0, 2.0),
    ];
    let scope = Scope::new(Vec3::ZERO, Quat::IDENTITY, Vec3::new(0.0, 5.0, 0.0));
    let profile = FaceProfile::Polygon(verts);
    let mp = compute_mass_properties(&scope, &profile, 1000.0).unwrap();
    // Area = 4; volume = 4·5 = 20; mass = 20_000.
    assert!((mp.mass - 20_000.0).abs() < 1e-6);
    // Centroid at (1, 2.5, 1).
    assert!((mp.centroid - Vec3::new(1.0, 2.5, 1.0)).length() < 1e-9);
}

#[test]
fn polygon_l_shape_centroid_offset_from_bbox_center() {
    // L-shape: 6 vertices forming a 2×2 area (square with bite taken out).
    //   (0,0) → (2,0) → (2,1) → (1,1) → (1,2) → (0,2) → close
    // Should have area = 3 (full 2×2 = 4 minus 1×1 bite = 3).
    let verts = vec![
        glam::DVec2::new(0.0, 0.0),
        glam::DVec2::new(2.0, 0.0),
        glam::DVec2::new(2.0, 1.0),
        glam::DVec2::new(1.0, 1.0),
        glam::DVec2::new(1.0, 2.0),
        glam::DVec2::new(0.0, 2.0),
    ];
    let scope = Scope::new(Vec3::ZERO, Quat::IDENTITY, Vec3::new(0.0, 1.0, 0.0));
    let mp = compute_mass_properties(&scope, &FaceProfile::Polygon(verts), 1.0).unwrap();
    assert!((mp.mass - 3.0).abs() < 1e-9, "L-shape area should be 3");
    // L-shape centroid is offset from bbox-center (1, _, 1) toward the heavier side.
    assert!(mp.centroid.x < 1.0);
    assert!(mp.centroid.z < 1.0);
}

// ── Taper (mass + centroid only, no inertia) ─────────────────────────────────

#[test]
fn taper_full_pyramid_centroid_at_quarter_height() {
    let scope = Scope::new(Vec3::ZERO, Quat::IDENTITY, Vec3::new(2.0, 4.0, 2.0));
    let mp = compute_mass_properties(&scope, &FaceProfile::Taper(1.0), 1000.0).unwrap();
    // Pyramid centroid is at h/4 from the base. sy = 4 → centroid Y = 1.0.
    assert!(
        (mp.centroid.y - 1.0).abs() < 1e-6,
        "expected pyramid centroid Y = 1.0, got {}",
        mp.centroid.y
    );
    // Volume of a pyramid = (1/3)·base·height = (1/3)·2·2·4 = 16/3; mass ≈ 5333.33.
    assert!((mp.mass - 16.0 / 3.0 * 1000.0).abs() < 1e-6);
    // Inertia not implemented for Taper.
    assert!(mp.inertia.is_none());
}

#[test]
fn taper_zero_equals_box() {
    let scope = Scope::new(Vec3::ZERO, Quat::IDENTITY, Vec3::new(2.0, 3.0, 4.0));
    let box_mp = compute_mass_properties(&scope, &FaceProfile::Rectangle, 1000.0).unwrap();
    let taper_mp = compute_mass_properties(&scope, &FaceProfile::Taper(0.0), 1000.0).unwrap();
    // Taper(0) is geometrically a full box; mass and centroid must match exactly.
    assert!((box_mp.mass - taper_mp.mass).abs() < 1e-9);
    assert!((box_mp.centroid - taper_mp.centroid).length() < 1e-9);
}

// ── Propagation through branching ops ────────────────────────────────────────

#[test]
fn mass_propagates_through_split() {
    let mut interp = Interpreter::new();
    interp.add_rule(
        "R",
        vec![
            ShapeOp::Mat(Material::with_density("Brick", 2000.0)),
            ShapeOp::Split {
                axis: Axis::Y,
                slots: vec![
                    SplitSlot {
                        size: SplitSize::Absolute(2.0),
                        rule: "Floor".to_string(),
                    },
                    SplitSlot {
                        size: SplitSize::Absolute(3.0),
                        rule: "Floor".to_string(),
                    },
                ],
                snap: None,
            },
        ],
    );
    interp.add_rule("Floor", vec![ShapeOp::I("Slab".to_string())]);
    let scope = Scope::new(Vec3::ZERO, Quat::IDENTITY, Vec3::new(2.0, 5.0, 4.0));
    let model = interp.derive(scope, "R").unwrap();
    assert_eq!(model.len(), 2);
    let total_mass: f64 = model
        .terminals
        .iter()
        .map(|t| t.mass_properties.as_ref().unwrap().mass)
        .sum();
    // Two slabs together = original volume × density. 2·5·4 × 2000 = 80_000.
    assert!((total_mass - 80_000.0).abs() < 1e-6);
    // Each slab has its own mass.
    let m0 = model.terminals[0].mass_properties.as_ref().unwrap().mass;
    let m1 = model.terminals[1].mass_properties.as_ref().unwrap().mass;
    // Heights 2 and 3 → masses 32_000 and 48_000.
    assert!((m0 - 32_000.0).abs() < 1e-6);
    assert!((m1 - 48_000.0).abs() < 1e-6);
}

#[test]
fn mass_propagates_through_comp() {
    let mut interp = Interpreter::new();
    interp.add_rule(
        "R",
        parse_ops(r#"Mat("Stone", 2500) Comp(Faces) { All: Wall }"#).unwrap(),
    );
    interp.add_rule("Wall", vec![ShapeOp::I("Panel".to_string())]);
    let scope = Scope::new(Vec3::ZERO, Quat::IDENTITY, Vec3::new(4.0, 3.0, 2.0));
    let model = interp.derive(scope, "R").unwrap();
    // 6 face panels, all stamped with Stone (density 2500).
    assert_eq!(model.len(), 6);
    for term in &model.terminals {
        assert_eq!(
            term.material.as_ref().unwrap().id,
            "Stone",
            "material id must propagate to face panels"
        );
        // Face panels are flat (size.z = 0) → zero volume → no mass_properties.
        assert!(
            term.mass_properties.is_none(),
            "flat face panels should have no mass_properties (zero volume)",
        );
    }
}

#[test]
fn mass_propagates_through_repeat() {
    let mut interp = Interpreter::new();
    interp.add_rule(
        "R",
        vec![
            ShapeOp::Mat(Material::with_density("Wood", 700.0)),
            ShapeOp::Repeat {
                axis: Axis::X,
                tile_sizes: vec![1.0],
                rule: "Tile".to_string(),
            },
        ],
    );
    interp.add_rule("Tile", vec![ShapeOp::I("Plank".to_string())]);
    let scope = Scope::new(Vec3::ZERO, Quat::IDENTITY, Vec3::new(5.0, 1.0, 2.0));
    let model = interp.derive(scope, "R").unwrap();
    assert_eq!(model.len(), 5);
    let total_mass: f64 = model
        .terminals
        .iter()
        .map(|t| t.mass_properties.as_ref().unwrap().mass)
        .sum();
    // Total = 5·1·2 × 700 = 7000.
    assert!(
        (total_mass - 7_000.0).abs() < 1e-6,
        "tiled mass should sum to whole, got {total_mass}"
    );
}

// ── Sanity: zero-density material is rejected, degenerate scope returns None ─

#[test]
fn zero_density_returns_none() {
    let mp = compute_mass_properties(&unit_box_scope(), &FaceProfile::Rectangle, 0.0);
    assert!(mp.is_none());
}

#[test]
fn negative_density_returns_none() {
    let mp = compute_mass_properties(&unit_box_scope(), &FaceProfile::Rectangle, -100.0);
    assert!(mp.is_none());
}

#[test]
fn collinear_polygon_returns_none() {
    let verts = vec![
        glam::DVec2::new(0.0, 0.0),
        glam::DVec2::new(1.0, 0.0),
        glam::DVec2::new(2.0, 0.0),
    ];
    let scope = Scope::new(Vec3::ZERO, Quat::IDENTITY, Vec3::new(0.0, 1.0, 0.0));
    let mp = compute_mass_properties(&scope, &FaceProfile::Polygon(verts), 1.0);
    assert!(mp.is_none());
}

// ── Inertia tensor is symmetric (a basic sanity invariant) ───────────────────

#[test]
fn inertia_tensor_is_symmetric_for_all_supported_profiles() {
    let scope = Scope::new(
        Vec3::new(1.0, 2.0, 3.0),
        Quat::from_axis_angle(Vec3::Y, 0.7),
        Vec3::new(2.0, 3.0, 4.0),
    );
    let profiles = [
        FaceProfile::Rectangle,
        FaceProfile::Triangle { peak_offset: 0.4 },
        FaceProfile::Trapezoid {
            top_width: 0.6,
            offset_x: 0.2,
        },
        FaceProfile::Polygon(vec![
            glam::DVec2::new(0.0, 0.0),
            glam::DVec2::new(2.0, 0.0),
            glam::DVec2::new(2.0, 1.0),
            glam::DVec2::new(1.0, 1.0),
            glam::DVec2::new(1.0, 2.0),
            glam::DVec2::new(0.0, 2.0),
        ]),
    ];
    for profile in profiles {
        let mp: MassProperties = compute_mass_properties(&scope, &profile, 1234.0).unwrap();
        let i = mp.inertia.expect("should have inertia tensor");
        assert!(
            (i.x_axis.y - i.y_axis.x).abs() < 1e-6,
            "Ixy=Iyx for {profile:?}, got {} vs {}",
            i.x_axis.y,
            i.y_axis.x,
        );
        assert!((i.x_axis.z - i.z_axis.x).abs() < 1e-6);
        assert!((i.y_axis.z - i.z_axis.y).abs() < 1e-6);
    }
}

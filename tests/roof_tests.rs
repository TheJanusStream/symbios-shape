//! Integration tests for the `Roof` operator (issue #31, plus fascia from #26).
//!
//! Asserts panel counts, ridge orientations, fascia presence, and that all
//! 15 roof types interpret without error on a representative footprint.

use symbios_shape::grammar::parse_ops;
use symbios_shape::ops::{RoofCase, RoofConfig, RoofFaceSelector, RoofType, ShapeOp};
use symbios_shape::{Interpreter, Quat, Scope, Vec3};

fn footprint() -> Scope {
    // Y = roof height (the volume the roof grows on top of).
    Scope::new(Vec3::ZERO, Quat::IDENTITY, Vec3::new(10.0, 5.0, 6.0))
}

fn run_roof(config: RoofConfig, cases: Vec<RoofCase>) -> Vec<symbios_shape::model::Terminal> {
    let mut interp = Interpreter::new();
    interp.add_rule("Lot", vec![ShapeOp::Roof { config, cases }]);
    interp.derive(footprint(), "Lot").unwrap().terminals
}

fn case(selector: RoofFaceSelector, rule: &str) -> RoofCase {
    RoofCase {
        selector,
        rule: rule.to_string(),
    }
}

// ── Panel count per roof type (no fascia) ─────────────────────────────────────

#[test]
fn flat_roof_has_one_panel() {
    let t = run_roof(
        RoofConfig::new(RoofType::Flat, 30.0),
        vec![case(RoofFaceSelector::All, "Tile")],
    );
    assert_eq!(t.len(), 1);
}

#[test]
fn shed_roof_has_three_panels() {
    let t = run_roof(
        RoofConfig::new(RoofType::Shed, 30.0),
        vec![case(RoofFaceSelector::All, "Tile")],
    );
    // 1 slope + 2 triangular gable ends.
    assert_eq!(t.len(), 3);
}

#[test]
fn gable_has_two_slopes_and_two_ends() {
    let t = run_roof(
        RoofConfig::new(RoofType::Gable, 35.0),
        vec![case(RoofFaceSelector::All, "Tile")],
    );
    assert_eq!(t.len(), 4);
}

#[test]
fn open_gable_has_two_slopes_no_ends() {
    let t = run_roof(
        RoofConfig::new(RoofType::OpenGable, 35.0),
        vec![case(RoofFaceSelector::All, "Tile")],
    );
    assert_eq!(t.len(), 2);
}

#[test]
fn box_gable_has_two_slopes_two_rect_ends() {
    let t = run_roof(
        RoofConfig::new(RoofType::BoxGable, 35.0),
        vec![case(RoofFaceSelector::All, "Tile")],
    );
    assert_eq!(t.len(), 4);
}

#[test]
fn hip_roof_has_four_slopes() {
    let t = run_roof(
        RoofConfig::new(RoofType::Hip, 30.0),
        vec![case(RoofFaceSelector::Slope, "Tile")],
    );
    assert_eq!(t.len(), 4);
}

#[test]
fn pyramid_has_four_slopes() {
    let t = run_roof(
        RoofConfig::new(RoofType::Pyramid, 40.0),
        vec![case(RoofFaceSelector::Slope, "Tile")],
    );
    assert_eq!(t.len(), 4);
}

#[test]
fn pyramid_hip_has_four_slopes() {
    let t = run_roof(
        RoofConfig::new(RoofType::PyramidHip, 40.0),
        vec![case(RoofFaceSelector::Slope, "Tile")],
    );
    assert_eq!(t.len(), 4);
}

#[test]
fn butterfly_has_two_valley_slopes() {
    let t = run_roof(
        RoofConfig::new(RoofType::Butterfly, 25.0),
        vec![case(RoofFaceSelector::ValleySlope, "Tile")],
    );
    assert_eq!(t.len(), 2);
}

#[test]
fn m_shaped_has_outer_and_inner_slopes() {
    let t = run_roof(
        RoofConfig::new(RoofType::MShaped, 30.0),
        vec![
            case(RoofFaceSelector::OuterSlope, "Outer"),
            case(RoofFaceSelector::InnerSlope, "Inner"),
        ],
    );
    let outer = t.iter().filter(|x| x.mesh_id == "Outer").count();
    let inner = t.iter().filter(|x| x.mesh_id == "Inner").count();
    assert_eq!(outer, 2);
    assert_eq!(inner, 2);
}

#[test]
fn gambrel_has_lower_and_upper_slopes() {
    let mut cfg = RoofConfig::new(RoofType::Gambrel, 60.0);
    cfg.secondary_pitch = Some(25.0);
    let t = run_roof(
        cfg,
        vec![
            case(RoofFaceSelector::LowerSlope, "Lower"),
            case(RoofFaceSelector::UpperSlope, "Upper"),
            case(RoofFaceSelector::GableEnd, "End"),
        ],
    );
    assert_eq!(t.iter().filter(|x| x.mesh_id == "Lower").count(), 2);
    assert_eq!(t.iter().filter(|x| x.mesh_id == "Upper").count(), 2);
    assert!(t.iter().any(|x| x.mesh_id == "End"));
}

#[test]
fn mansard_has_eight_slopes() {
    let mut cfg = RoofConfig::new(RoofType::Mansard, 60.0);
    cfg.secondary_pitch = Some(20.0);
    let t = run_roof(
        cfg,
        vec![
            case(RoofFaceSelector::LowerSlope, "Lower"),
            case(RoofFaceSelector::UpperSlope, "Upper"),
        ],
    );
    assert_eq!(t.iter().filter(|x| x.mesh_id == "Lower").count(), 4);
    assert_eq!(t.iter().filter(|x| x.mesh_id == "Upper").count(), 4);
}

#[test]
fn saltbox_asymmetric_slopes() {
    let mut cfg = RoofConfig::new(RoofType::Saltbox, 45.0);
    cfg.ridge_offset = 0.3;
    let t = run_roof(
        cfg,
        vec![
            case(RoofFaceSelector::Slope, "Slope"),
            case(RoofFaceSelector::GableEnd, "End"),
        ],
    );
    assert_eq!(t.iter().filter(|x| x.mesh_id == "Slope").count(), 2);
    assert_eq!(t.iter().filter(|x| x.mesh_id == "End").count(), 2);
}

#[test]
fn jerkinhead_has_slopes_ends_and_hip_caps() {
    let mut cfg = RoofConfig::new(RoofType::Jerkinhead, 45.0);
    cfg.tier_height = Some(0.3);
    let t = run_roof(
        cfg,
        vec![
            case(RoofFaceSelector::Slope, "Slope"),
            case(RoofFaceSelector::GableEnd, "End"),
            case(RoofFaceSelector::HipEnd, "Hip"),
        ],
    );
    assert_eq!(t.iter().filter(|x| x.mesh_id == "Slope").count(), 2);
    assert_eq!(t.iter().filter(|x| x.mesh_id == "End").count(), 2);
    assert_eq!(t.iter().filter(|x| x.mesh_id == "Hip").count(), 2);
}

#[test]
fn dutch_gable_has_lower_hip_and_upper_gable() {
    let mut cfg = RoofConfig::new(RoofType::DutchGable, 45.0);
    cfg.tier_height = Some(0.6);
    let t = run_roof(
        cfg,
        vec![
            case(RoofFaceSelector::Slope, "Slope"),
            case(RoofFaceSelector::GableEnd, "End"),
        ],
    );
    // 4 lower hip slopes + 2 upper gable slopes = 6 slopes.
    assert_eq!(t.iter().filter(|x| x.mesh_id == "Slope").count(), 6);
    assert_eq!(t.iter().filter(|x| x.mesh_id == "End").count(), 2);
}

// ── Slope normals point outward (re-asserted for #31) ─────────────────────────

#[test]
fn gable_slope_normals_point_outward() {
    let t = run_roof(
        RoofConfig::new(RoofType::Gable, 30.0),
        vec![case(RoofFaceSelector::Slope, "Tile")],
    );
    // The two slope normals must have opposite Z signs (front + back).
    let normals: Vec<f64> = t
        .iter()
        .map(|term| (term.scope.rotation * Vec3::Z).z)
        .collect();
    assert_eq!(normals.len(), 2);
    assert!(
        normals[0] * normals[1] < 0.0,
        "slope normals should point in opposite Z directions, got {:?}",
        normals
    );
}

#[test]
fn hip_slope_normals_cover_all_four_directions() {
    let t = run_roof(
        RoofConfig::new(RoofType::Hip, 30.0),
        vec![case(RoofFaceSelector::Slope, "Tile")],
    );
    assert_eq!(t.len(), 4);
    let mut have_pos_x = false;
    let mut have_neg_x = false;
    let mut have_pos_z = false;
    let mut have_neg_z = false;
    for term in &t {
        let n = term.scope.rotation * Vec3::Z;
        if n.x > 0.1 {
            have_pos_x = true;
        }
        if n.x < -0.1 {
            have_neg_x = true;
        }
        if n.z > 0.1 {
            have_pos_z = true;
        }
        if n.z < -0.1 {
            have_neg_z = true;
        }
    }
    assert!(have_pos_x && have_neg_x && have_pos_z && have_neg_z);
}

// ── Fascia (#26) ──────────────────────────────────────────────────────────────

#[test]
fn fascia_zero_depth_emits_no_fascia_panels() {
    let cfg = RoofConfig::new(RoofType::Gable, 30.0);
    let t = run_roof(
        cfg,
        vec![
            case(RoofFaceSelector::Slope, "Tile"),
            case(RoofFaceSelector::Fascia, "Board"),
        ],
    );
    assert!(t.iter().all(|x| x.mesh_id != "Board"));
}

#[test]
fn fascia_set_emits_one_band_per_perimeter_slope_on_gable() {
    let mut cfg = RoofConfig::new(RoofType::Gable, 30.0);
    cfg.fascia_depth = 0.3;
    let t = run_roof(
        cfg,
        vec![
            case(RoofFaceSelector::Slope, "Tile"),
            case(RoofFaceSelector::Fascia, "Board"),
        ],
    );
    // Gable has 2 slopes → 2 fascia bands.
    let boards: Vec<&symbios_shape::model::Terminal> =
        t.iter().filter(|x| x.mesh_id == "Board").collect();
    assert_eq!(boards.len(), 2);
    for board in &boards {
        // Fascia is a vertical band: thin Y, zero Z (face).
        assert!((board.scope.size.y - 0.3).abs() < 1e-9);
        assert!((board.scope.size.z).abs() < 1e-9);
        // Fascia outward normal is purely horizontal (no Y component).
        let n = board.scope.rotation * Vec3::Z;
        assert!(
            n.y.abs() < 1e-6,
            "fascia normal should be horizontal, got {n:?}"
        );
        // Fascia top edge sits at the eave Y of a 0-degree-tilt eave (y = 0 with no overhang).
        assert!(
            (board.scope.position.y + 0.3).abs() < 1e-6,
            "fascia origin Y should be -fascia_depth, got {}",
            board.scope.position.y,
        );
    }
}

#[test]
fn fascia_emits_four_bands_on_hip() {
    let mut cfg = RoofConfig::new(RoofType::Hip, 30.0);
    cfg.fascia_depth = 0.25;
    let t = run_roof(
        cfg,
        vec![
            case(RoofFaceSelector::Slope, "Tile"),
            case(RoofFaceSelector::Fascia, "Board"),
        ],
    );
    assert_eq!(t.iter().filter(|x| x.mesh_id == "Board").count(), 4);
}

#[test]
fn fascia_skipped_on_butterfly() {
    let mut cfg = RoofConfig::new(RoofType::Butterfly, 25.0);
    cfg.fascia_depth = 0.4;
    let t = run_roof(
        cfg,
        vec![
            case(RoofFaceSelector::ValleySlope, "Tile"),
            case(RoofFaceSelector::Fascia, "Board"),
        ],
    );
    // Butterfly has no perimeter eave at the slope-panel level — no fascia generated.
    assert!(t.iter().all(|x| x.mesh_id != "Board"));
}

#[test]
fn fascia_skipped_on_flat() {
    let mut cfg = RoofConfig::new(RoofType::Flat, 30.0);
    cfg.fascia_depth = 0.4;
    let t = run_roof(
        cfg,
        vec![
            case(RoofFaceSelector::All, "Tile"),
            case(RoofFaceSelector::Fascia, "Board"),
        ],
    );
    // Flat panel normal is vertical → no horizontal fascia direction.
    assert!(t.iter().all(|x| x.mesh_id != "Board"));
}

#[test]
fn fascia_emits_two_bands_on_outer_slopes_of_m_shaped() {
    let mut cfg = RoofConfig::new(RoofType::MShaped, 30.0);
    cfg.fascia_depth = 0.2;
    let t = run_roof(
        cfg,
        vec![
            case(RoofFaceSelector::OuterSlope, "Outer"),
            case(RoofFaceSelector::InnerSlope, "Inner"),
            case(RoofFaceSelector::Fascia, "Board"),
        ],
    );
    // Only the two outer slopes have perimeter eaves.
    assert_eq!(t.iter().filter(|x| x.mesh_id == "Board").count(), 2);
}

#[test]
fn fascia_only_on_lower_slopes_of_gambrel() {
    let mut cfg = RoofConfig::new(RoofType::Gambrel, 60.0);
    cfg.secondary_pitch = Some(25.0);
    cfg.fascia_depth = 0.2;
    let t = run_roof(
        cfg,
        vec![
            case(RoofFaceSelector::LowerSlope, "Lower"),
            case(RoofFaceSelector::UpperSlope, "Upper"),
            case(RoofFaceSelector::GableEnd, "End"),
            case(RoofFaceSelector::Fascia, "Board"),
        ],
    );
    // Gambrel sx >= sz → 2 lower slopes (front + back), 2 fascia bands.
    assert_eq!(t.iter().filter(|x| x.mesh_id == "Board").count(), 2);
}

#[test]
fn fascia_via_grammar_named_param() {
    let ops = parse_ops("Roof(Hip, 30, fascia=0.3) { Slope: Tile | Fascia: Board }").unwrap();
    let mut interp = Interpreter::new();
    interp.add_rule("Lot", ops);
    let model = interp.derive(footprint(), "Lot").unwrap();
    assert_eq!(
        model
            .terminals
            .iter()
            .filter(|t| t.mesh_id == "Board")
            .count(),
        4
    );
}

//! Simple building example — demonstrates a classic CGA Shape Grammar workflow.
//!
//! Grammar:
//! ```text
//! Lot    --> Extrude(12) Split(Y) { 3: Ground | ~1: Floor | ~1: Floor | 2: Roof }
//! Ground --> Split(X) { 1.5: Column | ~1: ShopFront | 1.5: Column }
//! Floor  --> Repeat(X, 2.5) { Window }
//! Roof   --> Taper(0.75) I("RoofMesh")
//! ```
//!
//! Run with: `cargo run --example simple_building`

use symbios_shape::grammar::parse_ops;
use symbios_shape::{Interpreter, Quat, Scope, Vec3};

fn main() {
    let mut interp = Interpreter::new();

    // Vertical subdivision: ground (3m) + two floating floors + roof cap (2m)
    interp.add_rule(
        "Lot",
        parse_ops("Extrude(12) Split(Y) { 3: Ground | ~1: Floor | ~1: Floor | 2: Roof }")
            .expect("Lot rule"),
    );

    // Ground floor: flanking columns + wide shop window
    interp.add_rule(
        "Ground",
        parse_ops("Split(X) { 1.5: Column | ~1: ShopFront | 1.5: Column }").expect("Ground rule"),
    );

    // Upper floors: repeating window bays every 2.5m
    interp.add_rule(
        "Floor",
        parse_ops("Repeat(X, 2.5) { Window }").expect("Floor rule"),
    );

    // Roof: tapered cap
    interp.add_rule(
        "Roof",
        parse_ops(r#"Taper(0.75) I("RoofMesh")"#).expect("Roof rule"),
    );

    // Leaf rules resolve to implicit terminals (mesh_id == rule name)
    // Column, ShopFront, Window are all terminals.

    // A 15m × 15m footprint
    let footprint = Scope::new(Vec3::ZERO, Quat::IDENTITY, Vec3::new(15.0, 0.0, 15.0));

    let model = interp.derive(footprint, "Lot").expect("derivation failed");

    println!("Building generated: {} terminal nodes", model.len());
    println!();

    for (i, terminal) in model.terminals.iter().enumerate() {
        let s = &terminal.scope;
        print!(
            "  [{:>3}] mesh={:<14} pos=({:>6.2}, {:>6.2}, {:>6.2})  \
             size=({:>6.2} × {:>6.2} × {:>6.2})",
            i,
            terminal.mesh_id,
            s.position.x,
            s.position.y,
            s.position.z,
            s.size.x,
            s.size.y,
            s.size.z,
        );
        if terminal.taper > 0.0 {
            print!("  taper={:.2}", terminal.taper);
        }
        println!();
    }

    // Verify basic structure
    let ground_nodes: Vec<_> = model
        .terminals
        .iter()
        .filter(|t| t.mesh_id == "Column" || t.mesh_id == "ShopFront")
        .collect();
    let window_nodes: Vec<_> = model
        .terminals
        .iter()
        .filter(|t| t.mesh_id == "Window")
        .collect();
    let roof_nodes: Vec<_> = model
        .terminals
        .iter()
        .filter(|t| t.mesh_id == "RoofMesh")
        .collect();

    println!();
    println!(
        "Ground elements : {} (2 columns + 1 shopfront)",
        ground_nodes.len()
    );
    println!(
        "Window bays     : {} (floor(15/2.5) = 6 per floor × 2 floors)",
        window_nodes.len()
    );
    println!("Roof nodes      : {}", roof_nodes.len());

    assert_eq!(ground_nodes.len(), 3, "expected 2 columns + 1 shopfront");
    assert_eq!(window_nodes.len(), 12, "expected 6 windows × 2 floors");
    assert_eq!(roof_nodes.len(), 1, "expected 1 roof");
    assert!(
        (roof_nodes[0].taper - 0.75).abs() < 1e-9,
        "roof taper should be 0.75"
    );

    println!();
    println!("All assertions passed.");
}

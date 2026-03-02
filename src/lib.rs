//! # Symbios Shape
//!
//! **A Sovereign Derivation Engine for CGA Shape Grammars.**
//!
//! Symbios Shape is a pure-Rust engine for generating procedural geometry using
//! Computer-Generated Architecture (CGA) Shape Grammars, as popularised by
//! Esri CityEngine. It is designed for embedding in game engines (Bevy, Godot)
//! and offline procedural pipelines where reliability and determinism are paramount.
//!
//! ## Key Features
//!
//! - **Lightweight**: Depends only on `glam` (math), `nom` (parsing), and `rand` (stochastic rules) — no engine or runtime required.
//! - **CGA-Compatible Operations**: `Extrude`, `Split`, `Comp`, `Repeat`, `I`, and more.
//! - **Flexible Split Sizing**: Absolute, relative (`'`), and floating (`~`) modes.
//! - **Bevy-Ready Output**: `ShapeModel` containing `Terminal` nodes (scope + mesh_id).
//!
//! ## Example
//!
//! ```rust
//! use symbios_shape::{Interpreter, Scope, Vec3, Quat};
//! use symbios_shape::grammar::parse_ops;
//!
//! let mut interp = Interpreter::new();
//!
//! // A simple 3-storey building
//! interp.add_rule("Lot", parse_ops("Extrude(12) Split(Y) { 3: Ground | ~1: Upper | 2: Roof }").unwrap());
//! interp.add_rule("Ground", parse_ops(r#"I("GroundFloor")"#).unwrap());
//! interp.add_rule("Upper",  parse_ops(r#"I("Floor")"#).unwrap());
//! interp.add_rule("Roof",   parse_ops(r#"Taper(0.8) I("Roof")"#).unwrap());
//!
//! let footprint = Scope::new(Vec3::ZERO, Quat::IDENTITY, Vec3::new(10.0, 0.0, 10.0));
//! let model = interp.derive(footprint, "Lot").unwrap();
//!
//! assert_eq!(model.len(), 3);
//! assert_eq!(model.terminals[0].mesh_id, "GroundFloor");
//! assert_eq!(model.terminals[2].mesh_id, "Roof");
//! assert!(matches!(model.terminals[2].face_profile, symbios_shape::FaceProfile::Taper(t) if (t - 0.8).abs() < 1e-9));
//! ```

pub mod error;
pub mod genetics;
pub mod grammar;
pub mod interpreter;
pub mod model;
pub mod ops;
pub mod scope;

pub use error::ShapeError;
pub use interpreter::Interpreter;
pub use model::{FaceProfile, ShapeModel, Terminal};
pub use ops::{
    AttachCase, AttachSelector, Axis, CompTarget, FaceSelector, OffsetCase, OffsetSelector,
    RoofCase, RoofConfig, RoofFaceSelector, RoofType, ShapeOp, SplitSize, SplitSlot,
};
pub use scope::{Quat, Scope, Vec3};

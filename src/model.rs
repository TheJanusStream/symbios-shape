use serde::{Deserialize, Serialize};

use crate::scope::Scope;

/// A fully-resolved terminal node in the shape model.
///
/// Represents a concrete mesh instance placed at the given `scope`.
/// The `mesh_id` identifies which asset to spawn (e.g. `"Window"`, `"Door"`, `"Pillar"`).
/// `taper` ∈ `[0, 1]`: 0 = no taper (box), 1 = full pyramid (top collapses to a point).
/// `material`: optional material/texture identifier set by `Mat(...)` operations.
/// This is the "DOM" that Bevy (or any renderer) reads to spawn entities.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Terminal {
    pub scope: Scope,
    pub mesh_id: String,
    /// Pyramidal taper factor: 0.0 = box, 1.0 = full pyramid.
    pub taper: f64,
    /// Optional material identifier stamped by a `Mat("...")` operation.
    pub material: Option<String>,
}

impl Terminal {
    pub fn new(scope: Scope, mesh_id: impl Into<String>) -> Self {
        Self {
            scope,
            mesh_id: mesh_id.into(),
            taper: 0.0,
            material: None,
        }
    }

    pub fn new_with_taper(scope: Scope, mesh_id: impl Into<String>, taper: f64) -> Self {
        Self {
            scope,
            mesh_id: mesh_id.into(),
            taper,
            material: None,
        }
    }

    pub fn new_full(
        scope: Scope,
        mesh_id: impl Into<String>,
        taper: f64,
        material: Option<String>,
    ) -> Self {
        Self {
            scope,
            mesh_id: mesh_id.into(),
            taper,
            material,
        }
    }
}

/// The output of a shape grammar derivation.
///
/// Contains all terminal nodes produced by the grammar, ready for rendering.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ShapeModel {
    pub terminals: Vec<Terminal>,
}

impl ShapeModel {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn push(&mut self, terminal: Terminal) {
        self.terminals.push(terminal);
    }

    pub fn len(&self) -> usize {
        self.terminals.len()
    }

    pub fn is_empty(&self) -> bool {
        self.terminals.is_empty()
    }
}

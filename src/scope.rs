use serde::{Deserialize, Serialize};

use crate::error::ShapeError;

/// A 3-component double-precision vector. Re-exported from `glam`.
pub use glam::DVec3 as Vec3;

/// A double-precision unit quaternion. Re-exported from `glam`.
pub use glam::DQuat as Quat;

/// An Oriented Bounding Box (OBB) that defines a shape's coordinate frame.
///
/// Every CGA operation transforms a parent `Scope` into one or more child `Scope`s.
/// - `position`: world-space origin of the scope's corner (min point in local space).
/// - `rotation`: local-to-world orientation.
/// - `size`: extents along the local X, Y, Z axes.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Scope {
    pub position: Vec3,
    pub rotation: Quat,
    pub size: Vec3,
}

impl Scope {
    /// Creates a unit scope at the origin with identity rotation.
    pub fn unit() -> Self {
        Self {
            position: Vec3::ZERO,
            rotation: Quat::IDENTITY,
            size: Vec3::ONE,
        }
    }

    pub fn new(position: Vec3, rotation: Quat, size: Vec3) -> Self {
        Self {
            position,
            rotation,
            size,
        }
    }

    pub fn validate(&self) -> Result<(), ShapeError> {
        if !self.position.is_finite() || !self.rotation.is_finite() || !self.size.is_finite() {
            return Err(ShapeError::InvalidNumericValue);
        }
        Ok(())
    }

    /// Returns the world-space position of the local-space point `(u, v, w)`
    /// where each coordinate is in `[0, 1]` (relative to scope size).
    pub fn world_point(&self, u: f64, v: f64, w: f64) -> Vec3 {
        let local = Vec3::new(u * self.size.x, v * self.size.y, w * self.size.z);
        self.position + self.rotation * local
    }
}

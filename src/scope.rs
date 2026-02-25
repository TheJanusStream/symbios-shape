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

    /// Splits this scope along the Y axis at a relative offset `t ∈ [0, 1]`.
    /// Returns `(bottom, top)`.
    pub fn split_y(&self, t: f64) -> (Scope, Scope) {
        let bottom_height = self.size.y * t;
        let top_height = self.size.y * (1.0 - t);
        let top_pos = self.position + self.rotation * Vec3::new(0.0, bottom_height, 0.0);

        let bottom = Scope {
            position: self.position,
            rotation: self.rotation,
            size: Vec3::new(self.size.x, bottom_height, self.size.z),
        };
        let top = Scope {
            position: top_pos,
            rotation: self.rotation,
            size: Vec3::new(self.size.x, top_height, self.size.z),
        };
        (bottom, top)
    }

    /// Splits this scope along the X axis at a relative offset `t ∈ [0, 1]`.
    /// Returns `(left, right)`.
    pub fn split_x(&self, t: f64) -> (Scope, Scope) {
        let left_width = self.size.x * t;
        let right_width = self.size.x * (1.0 - t);
        let right_pos = self.position + self.rotation * Vec3::new(left_width, 0.0, 0.0);

        let left = Scope {
            position: self.position,
            rotation: self.rotation,
            size: Vec3::new(left_width, self.size.y, self.size.z),
        };
        let right = Scope {
            position: right_pos,
            rotation: self.rotation,
            size: Vec3::new(right_width, self.size.y, self.size.z),
        };
        (left, right)
    }

    /// Splits this scope along the Z axis at a relative offset `t ∈ [0, 1]`.
    /// Returns `(front, back)`.
    pub fn split_z(&self, t: f64) -> (Scope, Scope) {
        let front_depth = self.size.z * t;
        let back_depth = self.size.z * (1.0 - t);
        let back_pos = self.position + self.rotation * Vec3::new(0.0, 0.0, front_depth);

        let front = Scope {
            position: self.position,
            rotation: self.rotation,
            size: Vec3::new(self.size.x, self.size.y, front_depth),
        };
        let back = Scope {
            position: back_pos,
            rotation: self.rotation,
            size: Vec3::new(self.size.x, self.size.y, back_depth),
        };
        (front, back)
    }
}

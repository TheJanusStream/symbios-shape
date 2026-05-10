//! Error type returned by the parser and the interpreter.
//!
//! Every fallible API in this crate returns `Result<T, ShapeError>`. Variants
//! cover three broad failure classes:
//!
//! - **Parse failures** — `ParseError` carries a human-readable message from
//!   the nom parser, including the offending input span.
//! - **Numeric / structural validation** — `InvalidNumericValue`,
//!   `EmptySplit`, `InvalidFloatingSize`, `SplitOverflow`, `NoFloatingSlots`,
//!   `UnknownCompSelector`, `OffsetTooLarge`, `InvalidRoofAngle`,
//!   `InvalidAlignTarget` are raised when a parsed grammar would produce
//!   geometry that isn't well-defined.
//! - **DoS safety caps** — `CapacityOverflow` and `DepthLimitExceeded` are
//!   returned when a grammar exceeds the engine's bounded queue, terminal,
//!   op-count, identifier, or recursion limits rather than allowing
//!   unbounded resource use.

use thiserror::Error;

/// All errors produced by the parser and the interpreter.
///
/// See the module-level documentation for a categorised overview.
#[derive(Error, Debug, PartialEq)]
pub enum ShapeError {
    #[error("Parse error: {0}")]
    ParseError(String),
    #[error("Non-finite numeric value detected (NaN/Inf)")]
    InvalidNumericValue,
    #[error("Scope capacity overflow")]
    CapacityOverflow,
    #[error("Split sizes are empty")]
    EmptySplit,
    #[error("Split floating size must be positive: {0}")]
    InvalidFloatingSize(f64),
    #[error("Split absolute sizes exceed scope dimension {0}")]
    SplitOverflow(f64),
    #[error("No floating slots to absorb remainder in split")]
    NoFloatingSlots,
    #[error("Comp selector '{0}' not recognised")]
    UnknownCompSelector(String),
    #[error("Derivation depth limit {0} exceeded")]
    DepthLimitExceeded(usize),
    #[error("Internal error: {0}")]
    Internal(String),
    #[error("Offset inset distance exceeds scope dimension")]
    OffsetTooLarge,
    #[error("Roof angle must be in (0°, 90°): got {0}")]
    InvalidRoofAngle(f64),
    #[error("Align target vector must be non-zero")]
    InvalidAlignTarget,
}

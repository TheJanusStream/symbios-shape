use thiserror::Error;

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
}

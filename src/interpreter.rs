/// Queue-based CGA Shape Grammar interpreter.
///
/// The interpreter owns a set of named rules. Each rule has one or more
/// weighted variants — for deterministic rules there is exactly one variant.
/// Derivation starts from a root `Scope` and root rule name, expanding rules
/// breadth-first until every branch terminates with an `I(mesh)` terminal.
use std::collections::{HashMap, VecDeque};
use std::f64::consts::{FRAC_PI_2, PI};

use rand::SeedableRng;
use rand_pcg::Pcg64;

use crate::error::ShapeError;
use crate::model::{ShapeModel, Terminal};
use crate::ops::{Axis, CompTarget, FaceSelector, ShapeOp, SplitSize, SplitSlot};
use crate::scope::{Quat, Scope, Vec3};

/// Safety caps (DoS protection).
const MAX_DEPTH: usize = 64;
const MAX_QUEUE: usize = 100_000;
const MAX_TERMINALS: usize = 100_000;

// ── Weighted rule variant ─────────────────────────────────────────────────────

/// One alternative in a stochastic or deterministic rule.
pub struct WeightedVariant {
    /// Relative weight (need not sum to 1.0 across variants).
    pub weight: f64,
    pub ops: Vec<ShapeOp>,
}

// ── Work queue item ───────────────────────────────────────────────────────────

struct WorkItem {
    scope: Scope,
    rule: String,
    depth: usize,
    /// Accumulated taper from ancestor ops (propagated to the terminal).
    taper: f64,
    /// Material identifier set by `Mat("...")` ops; propagates to child scopes.
    material: Option<String>,
}

// ── Split size resolution ─────────────────────────────────────────────────────

/// Resolves `SplitSlot` sizes against `total_dim`, returning absolute sizes.
fn resolve_split_sizes(slots: &[SplitSlot], total_dim: f64) -> Result<Vec<f64>, ShapeError> {
    if slots.is_empty() {
        return Err(ShapeError::EmptySplit);
    }
    for slot in slots {
        if !slot.size.is_valid() {
            return match &slot.size {
                SplitSize::Floating(v) => Err(ShapeError::InvalidFloatingSize(*v)),
                _ => Err(ShapeError::InvalidNumericValue),
            };
        }
    }

    let mut fixed: Vec<Option<f64>> = Vec::with_capacity(slots.len());
    let mut used = 0.0_f64;
    let mut float_weight_total = 0.0_f64;

    for slot in slots {
        match slot.size {
            SplitSize::Absolute(v) => {
                fixed.push(Some(v));
                used += v;
            }
            SplitSize::Relative(t) => {
                let s = total_dim * t;
                fixed.push(Some(s));
                used += s;
            }
            SplitSize::Floating(w) => {
                fixed.push(None);
                float_weight_total += w;
            }
        }
    }

    if used > total_dim + 1e-9 {
        return Err(ShapeError::SplitOverflow(total_dim));
    }

    let remaining = (total_dim - used).max(0.0);

    let mut result = Vec::with_capacity(slots.len());
    for (i, slot) in slots.iter().enumerate() {
        match fixed[i] {
            Some(v) => result.push(v),
            None => {
                let w = match slot.size {
                    SplitSize::Floating(w) => w,
                    _ => unreachable!(),
                };
                if float_weight_total <= 0.0 {
                    return Err(ShapeError::NoFloatingSlots);
                }
                result.push(remaining * w / float_weight_total);
            }
        }
    }

    Ok(result)
}

// ── Scope slicing helpers ─────────────────────────────────────────────────────

/// Creates a child scope that is a sub-interval `[offset, offset+size]` of the
/// parent scope along `axis`. All measurements are in local (scope) units.
fn slice_scope(parent: &Scope, axis: Axis, offset: f64, size: f64) -> Scope {
    let offset_vec = match axis {
        Axis::X => Vec3::new(offset, 0.0, 0.0),
        Axis::Y => Vec3::new(0.0, offset, 0.0),
        Axis::Z => Vec3::new(0.0, 0.0, offset),
    };

    let child_position = parent.position + parent.rotation * offset_vec;

    let child_size = match axis {
        Axis::X => Vec3::new(size, parent.size.y, parent.size.z),
        Axis::Y => Vec3::new(parent.size.x, size, parent.size.z),
        Axis::Z => Vec3::new(parent.size.x, parent.size.y, size),
    };

    Scope::new(child_position, parent.rotation, child_size)
}

// ── Face decomposition ────────────────────────────────────────────────────────

/// All six canonical faces of an OBB, each with a proper outward-facing orientation.
///
/// Each entry: `(selector, local_offset, face_size, rotation_delta)`.
///
/// The `rotation_delta` is composed with the parent rotation so that the resulting
/// face scope has:
/// - Local **Z** = outward face normal
/// - Local **X** = face width tangent
/// - Local **Y** = face height tangent
/// - `size.z = 0` (the face is a 2-D canvas)
///
/// This allows rules like `Repeat(X) { Window }` to work naturally on any side
/// of a building after `Comp(Faces)`.
fn face_descs(scope_size: Vec3) -> [(FaceSelector, Vec3, Vec3, Quat); 6] {
    let sx = scope_size.x;
    let sy = scope_size.y;
    let sz = scope_size.z;

    [
        // Bottom: normal = -Y (at y = 0)
        // from_axis_angle(X, +π/2): local Z → world -Y
        (
            FaceSelector::Bottom,
            Vec3::ZERO,
            Vec3::new(sx, sz, 0.0),
            Quat::from_axis_angle(Vec3::X, FRAC_PI_2),
        ),
        // Top: normal = +Y (at y = sy)
        // from_axis_angle(X, -π/2): local Z → world +Y
        (
            FaceSelector::Top,
            Vec3::new(0.0, sy, 0.0),
            Vec3::new(sx, sz, 0.0),
            Quat::from_axis_angle(Vec3::X, -FRAC_PI_2),
        ),
        // Front: normal = -Z (at z = 0)
        // from_axis_angle(Y, π): local Z → world -Z
        (
            FaceSelector::Front,
            Vec3::ZERO,
            Vec3::new(sx, sy, 0.0),
            Quat::from_axis_angle(Vec3::Y, PI),
        ),
        // Back: normal = +Z (at z = sz)
        // identity: local Z → world +Z
        (
            FaceSelector::Back,
            Vec3::new(0.0, 0.0, sz),
            Vec3::new(sx, sy, 0.0),
            Quat::IDENTITY,
        ),
        // Left: normal = -X (at x = 0)
        // from_axis_angle(Y, -π/2): local Z → world -X
        (
            FaceSelector::Left,
            Vec3::ZERO,
            Vec3::new(sz, sy, 0.0),
            Quat::from_axis_angle(Vec3::Y, -FRAC_PI_2),
        ),
        // Right: normal = +X (at x = sx)
        // from_axis_angle(Y, +π/2): local Z → world +X
        (
            FaceSelector::Right,
            Vec3::new(sx, 0.0, 0.0),
            Vec3::new(sz, sy, 0.0),
            Quat::from_axis_angle(Vec3::Y, FRAC_PI_2),
        ),
    ]
}

/// Finds the rule to apply to a face given a list of `CompFaceCase`s.
fn find_face_rule(selector: FaceSelector, cases: &[crate::ops::CompFaceCase]) -> Option<&str> {
    for case in cases {
        if case.selector == selector {
            return Some(&case.rule);
        }
    }
    let is_side = matches!(
        selector,
        FaceSelector::Front | FaceSelector::Back | FaceSelector::Left | FaceSelector::Right
    );
    if is_side {
        for case in cases {
            if case.selector == FaceSelector::Side {
                return Some(&case.rule);
            }
        }
    }
    for case in cases {
        if case.selector == FaceSelector::All {
            return Some(&case.rule);
        }
    }
    None
}

// ── Stochastic selection ──────────────────────────────────────────────────────

fn select_variant<'a>(variants: &'a [WeightedVariant], rng: &mut Pcg64) -> &'a [ShapeOp] {
    if variants.len() == 1 {
        return &variants[0].ops;
    }
    let total: f64 = variants.iter().map(|v| v.weight).sum();
    use rand::Rng;
    let r: f64 = rng.random::<f64>() * total;
    let mut acc = 0.0;
    for v in variants {
        acc += v.weight;
        if r < acc {
            return &v.ops;
        }
    }
    &variants.last().unwrap().ops
}

// ── Interpreter ───────────────────────────────────────────────────────────────

/// The CGA Shape Grammar derivation engine.
///
/// Rules are registered by name, then `derive` is called with a root scope and
/// root rule name. The engine expands rules breadth-first until every branch
/// terminates with an `I(mesh)` terminal.
///
/// Stochastic rules with multiple weighted variants use the engine's `seed` for
/// reproducible randomness — the same seed always yields the same building.
pub struct Interpreter {
    rules: HashMap<String, Vec<WeightedVariant>>,
    pub max_depth: usize,
    pub max_terminals: usize,
    /// Seed for stochastic rule selection. Default 0.
    pub seed: u64,
}

impl Default for Interpreter {
    fn default() -> Self {
        Self::new()
    }
}

impl Interpreter {
    pub fn new() -> Self {
        Self {
            rules: HashMap::new(),
            max_depth: MAX_DEPTH,
            max_terminals: MAX_TERMINALS,
            seed: 0,
        }
    }

    /// Registers a deterministic production rule.
    pub fn add_rule(&mut self, name: impl Into<String>, ops: Vec<ShapeOp>) {
        self.rules
            .insert(name.into(), vec![WeightedVariant { weight: 1.0, ops }]);
    }

    /// Registers a stochastic rule with multiple weighted alternatives.
    ///
    /// `variants` is a list of `(relative_weight, ops)` pairs. Weights need not
    /// sum to 1.0 — they are normalised internally during selection.
    pub fn add_weighted_rules(
        &mut self,
        name: impl Into<String>,
        variants: Vec<(f64, Vec<ShapeOp>)>,
    ) {
        let wvs = variants
            .into_iter()
            .map(|(weight, ops)| WeightedVariant { weight, ops })
            .collect();
        self.rules.insert(name.into(), wvs);
    }

    /// Returns true if a rule with `name` is registered.
    pub fn has_rule(&self, name: &str) -> bool {
        self.rules.contains_key(name)
    }

    /// Derives the shape model starting from `root_scope` and `root_rule`.
    ///
    /// Uses a breadth-first work queue to expand rules until all branches
    /// terminate. A fresh RNG seeded from `self.seed` is created for each call,
    /// making derivations reproducible for the same `seed` value.
    pub fn derive(
        &self,
        root_scope: Scope,
        root_rule: impl Into<String>,
    ) -> Result<ShapeModel, ShapeError> {
        root_scope.validate()?;

        let mut model = ShapeModel::new();
        let mut queue: VecDeque<WorkItem> = VecDeque::new();
        let mut rng = Pcg64::seed_from_u64(self.seed);

        queue.push_back(WorkItem {
            scope: root_scope,
            rule: root_rule.into(),
            depth: 0,
            taper: 0.0,
            material: None,
        });

        while let Some(item) = queue.pop_front() {
            if queue.len() > MAX_QUEUE {
                return Err(ShapeError::CapacityOverflow);
            }
            if item.depth > self.max_depth {
                return Err(ShapeError::DepthLimitExceeded(self.max_depth));
            }

            let ops = match self.rules.get(&item.rule) {
                Some(variants) => select_variant(variants, &mut rng),
                None => {
                    // Unknown rule → implicit I(rule_name) terminal.
                    if model.len() >= self.max_terminals {
                        return Err(ShapeError::CapacityOverflow);
                    }
                    model.push(Terminal::new_full(
                        item.scope,
                        &item.rule,
                        item.taper,
                        item.material,
                    ));
                    continue;
                }
            };

            self.apply_ops(
                item.scope,
                item.taper,
                item.material,
                ops,
                item.depth,
                &mut queue,
                &mut model,
            )?;
        }

        Ok(model)
    }

    /// Processes the ops sequence for a single rule invocation.
    ///
    /// Transformation ops (`Extrude`, `Scale`, etc.) mutate `scope` in place.
    /// The first branching op (`Split`, `Comp`, `Repeat`) or terminal op
    /// (`I`, `Rule`) ends the sequence by pushing new work items.
    #[allow(clippy::too_many_arguments)]
    fn apply_ops(
        &self,
        initial_scope: Scope,
        initial_taper: f64,
        initial_material: Option<String>,
        ops: &[ShapeOp],
        depth: usize,
        queue: &mut VecDeque<WorkItem>,
        model: &mut ShapeModel,
    ) -> Result<(), ShapeError> {
        let mut scope = initial_scope;
        let mut taper = initial_taper;
        let mut material = initial_material;

        for op in ops {
            match op {
                // ── Transformations ───────────────────────────────────────
                ShapeOp::Extrude(h) => {
                    if !h.is_finite() || *h <= 0.0 {
                        return Err(ShapeError::InvalidNumericValue);
                    }
                    scope.size.y = *h;
                }

                ShapeOp::Taper(amount) => {
                    if !amount.is_finite() {
                        return Err(ShapeError::InvalidNumericValue);
                    }
                    taper = amount.clamp(0.0, 1.0);
                }

                ShapeOp::Rotate(q) => {
                    if !q.is_finite() {
                        return Err(ShapeError::InvalidNumericValue);
                    }
                    scope.rotation *= *q;
                }

                ShapeOp::Translate(v) => {
                    if !v.is_finite() {
                        return Err(ShapeError::InvalidNumericValue);
                    }
                    scope.position += scope.rotation * *v;
                }

                ShapeOp::Scale(v) => {
                    if !v.is_finite() {
                        return Err(ShapeError::InvalidNumericValue);
                    }
                    scope.size *= *v;
                }

                ShapeOp::Mat(mat_id) => {
                    material = Some(mat_id.clone());
                }

                // ── Branching: Split ──────────────────────────────────────
                ShapeOp::Split { axis, slots } => {
                    let total = match axis {
                        Axis::X => scope.size.x,
                        Axis::Y => scope.size.y,
                        Axis::Z => scope.size.z,
                    };
                    let sizes = resolve_split_sizes(slots, total)?;
                    let mut offset = 0.0;
                    for (slot, size) in slots.iter().zip(sizes.iter()) {
                        let child = slice_scope(&scope, *axis, offset, *size);
                        queue.push_back(WorkItem {
                            scope: child,
                            rule: slot.rule.clone(),
                            depth: depth + 1,
                            taper: 0.0,
                            material: material.clone(),
                        });
                        offset += size;
                    }
                    return Ok(());
                }

                // ── Branching: Repeat ─────────────────────────────────────
                //
                // Uses `floor()` for tile count (never fewer tiles than fit),
                // then stretches actual tile size to fill the scope with no gaps.
                // Example: 10.5m scope / 2m target → 5 tiles × 2.1m each.
                ShapeOp::Repeat {
                    axis,
                    tile_size,
                    rule,
                } => {
                    if !tile_size.is_finite() || *tile_size <= 0.0 {
                        return Err(ShapeError::InvalidNumericValue);
                    }
                    let total = match axis {
                        Axis::X => scope.size.x,
                        Axis::Y => scope.size.y,
                        Axis::Z => scope.size.z,
                    };
                    let n_tiles = (total / tile_size).floor() as usize;
                    if n_tiles > 0 {
                        let actual_size = total / n_tiles as f64;
                        for i in 0..n_tiles {
                            let offset = i as f64 * actual_size;
                            let child = slice_scope(&scope, *axis, offset, actual_size);
                            queue.push_back(WorkItem {
                                scope: child,
                                rule: rule.clone(),
                                depth: depth + 1,
                                taper: 0.0,
                                material: material.clone(),
                            });
                        }
                    }
                    return Ok(());
                }

                // ── Branching: Comp ───────────────────────────────────────
                //
                // Each face scope is properly oriented so that local Z points
                // along the outward face normal. Rules can then use Split(X/Y)
                // or Repeat(X) naturally on any face of the parent volume.
                ShapeOp::Comp(CompTarget::Faces(cases)) => {
                    for (selector, offset_local, face_size, rot_delta) in face_descs(scope.size) {
                        let rule = match find_face_rule(selector, cases) {
                            Some(r) => r,
                            None => continue,
                        };
                        let face_pos = scope.position + scope.rotation * offset_local;
                        let face_rotation = scope.rotation * rot_delta;
                        let face_scope = Scope::new(face_pos, face_rotation, face_size);
                        queue.push_back(WorkItem {
                            scope: face_scope,
                            rule: rule.to_string(),
                            depth: depth + 1,
                            taper: 0.0,
                            material: material.clone(),
                        });
                    }
                    return Ok(());
                }

                // ── Terminal: mesh instance ───────────────────────────────
                ShapeOp::I(mesh_id) => {
                    if model.len() >= self.max_terminals {
                        return Err(ShapeError::CapacityOverflow);
                    }
                    model.push(Terminal::new_full(scope, mesh_id, taper, material));
                    return Ok(());
                }

                // ── Delegate: named sub-rule ──────────────────────────────
                ShapeOp::Rule(name) => {
                    queue.push_back(WorkItem {
                        scope,
                        rule: name.clone(),
                        depth: depth + 1,
                        taper,
                        material,
                    });
                    return Ok(());
                }
            }
        }

        // Ops exhausted without a terminal — scope is silently discarded
        // (matches CGA "delete this shape" semantics for empty successors).
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::{Axis, SplitSize, SplitSlot};
    use crate::scope::{Quat, Vec3};

    fn slot(size: SplitSize, rule: &str) -> SplitSlot {
        SplitSlot {
            size,
            rule: rule.to_string(),
        }
    }

    #[test]
    fn test_resolve_split_absolute() {
        let slots = vec![
            slot(SplitSize::Absolute(3.0), "A"),
            slot(SplitSize::Absolute(7.0), "B"),
        ];
        let sizes = resolve_split_sizes(&slots, 10.0).unwrap();
        assert!((sizes[0] - 3.0).abs() < 1e-9);
        assert!((sizes[1] - 7.0).abs() < 1e-9);
    }

    #[test]
    fn test_resolve_split_floating_equal() {
        let slots = vec![
            slot(SplitSize::Floating(1.0), "A"),
            slot(SplitSize::Floating(1.0), "B"),
        ];
        let sizes = resolve_split_sizes(&slots, 10.0).unwrap();
        assert!((sizes[0] - 5.0).abs() < 1e-9);
        assert!((sizes[1] - 5.0).abs() < 1e-9);
    }

    #[test]
    fn test_resolve_split_mixed() {
        let slots = vec![
            slot(SplitSize::Absolute(2.0), "Base"),
            slot(SplitSize::Floating(1.0), "A"),
            slot(SplitSize::Floating(1.0), "B"),
        ];
        let sizes = resolve_split_sizes(&slots, 10.0).unwrap();
        assert!((sizes[0] - 2.0).abs() < 1e-9);
        assert!((sizes[1] - 4.0).abs() < 1e-9);
        assert!((sizes[2] - 4.0).abs() < 1e-9);
    }

    #[test]
    fn test_resolve_split_overflow_rejected() {
        let slots = vec![
            slot(SplitSize::Absolute(6.0), "A"),
            slot(SplitSize::Absolute(6.0), "B"),
        ];
        assert!(matches!(
            resolve_split_sizes(&slots, 10.0),
            Err(ShapeError::SplitOverflow(_))
        ));
    }

    #[test]
    fn test_derive_extrude_then_terminal() {
        let mut interp = Interpreter::new();
        interp.add_rule(
            "Lot",
            vec![ShapeOp::Extrude(10.0), ShapeOp::I("Building".to_string())],
        );
        let scope = Scope::unit();
        let model = interp.derive(scope, "Lot").unwrap();
        assert_eq!(model.len(), 1);
        assert_eq!(model.terminals[0].mesh_id, "Building");
        assert!((model.terminals[0].scope.size.y - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_derive_split_y_three_floors() {
        let mut interp = Interpreter::new();
        interp.add_rule(
            "Building",
            vec![ShapeOp::Split {
                axis: Axis::Y,
                slots: vec![
                    slot(SplitSize::Absolute(2.0), "Ground"),
                    slot(SplitSize::Floating(1.0), "Upper"),
                    slot(SplitSize::Absolute(1.5), "Roof"),
                ],
            }],
        );
        let scope = Scope::new(Vec3::ZERO, Quat::IDENTITY, Vec3::new(10.0, 10.0, 10.0));
        let model = interp.derive(scope, "Building").unwrap();
        assert_eq!(model.len(), 3);
        assert!((model.terminals[0].scope.size.y - 2.0).abs() < 1e-9);
        assert!((model.terminals[1].scope.size.y - 6.5).abs() < 1e-9);
        assert!((model.terminals[2].scope.size.y - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_derive_depth_limit() {
        let mut interp = Interpreter::new();
        interp.add_rule("A", vec![ShapeOp::Rule("A".to_string())]);
        interp.max_depth = 5;
        let model = interp.derive(Scope::unit(), "A");
        assert!(matches!(model, Err(ShapeError::DepthLimitExceeded(_))));
    }

    #[test]
    fn test_derive_comp_faces() {
        let mut interp = Interpreter::new();
        interp.add_rule(
            "Box",
            vec![ShapeOp::Comp(CompTarget::Faces(vec![
                crate::ops::CompFaceCase {
                    selector: FaceSelector::Top,
                    rule: "Roof".to_string(),
                },
                crate::ops::CompFaceCase {
                    selector: FaceSelector::Side,
                    rule: "Wall".to_string(),
                },
                crate::ops::CompFaceCase {
                    selector: FaceSelector::Bottom,
                    rule: "Base".to_string(),
                },
            ]))],
        );
        let scope = Scope::new(Vec3::ZERO, Quat::IDENTITY, Vec3::new(5.0, 3.0, 5.0));
        let model = interp.derive(scope, "Box").unwrap();
        assert_eq!(model.len(), 6);
    }

    #[test]
    fn test_derive_repeat() {
        let mut interp = Interpreter::new();
        interp.add_rule(
            "Facade",
            vec![ShapeOp::Repeat {
                axis: Axis::X,
                tile_size: 2.0,
                rule: "Window".to_string(),
            }],
        );
        let scope = Scope::new(Vec3::ZERO, Quat::IDENTITY, Vec3::new(10.0, 4.0, 0.0));
        let model = interp.derive(scope, "Facade").unwrap();
        // 10 / 2 = 5 tiles, each stretched to exactly 2.0m (no remainder here)
        assert_eq!(model.len(), 5);
    }

    #[test]
    fn test_derive_mat_propagates() {
        let mut interp = Interpreter::new();
        interp.add_rule(
            "R",
            vec![
                ShapeOp::Mat("Brick".to_string()),
                ShapeOp::I("Wall".to_string()),
            ],
        );
        let model = interp.derive(Scope::unit(), "R").unwrap();
        assert_eq!(model.terminals[0].material, Some("Brick".to_string()));
    }

    #[test]
    fn test_stochastic_rule_deterministic_with_seed() {
        let mut interp = Interpreter::new();
        interp.add_weighted_rules(
            "Facade",
            vec![
                (70.0, vec![ShapeOp::I("Brick".to_string())]),
                (30.0, vec![ShapeOp::I("Glass".to_string())]),
            ],
        );
        interp.seed = 42;
        // Same seed → same result
        let m1 = interp.derive(Scope::unit(), "Facade").unwrap();
        let m2 = interp.derive(Scope::unit(), "Facade").unwrap();
        assert_eq!(m1.terminals[0].mesh_id, m2.terminals[0].mesh_id);
    }

    #[test]
    fn test_face_comp_orientations() {
        // After Comp, each face scope should have local Z pointing along its outward normal.
        // We verify by checking the rotation: applying the face rotation to (0,0,1) should
        // give the expected world-space normal direction.
        let mut interp = Interpreter::new();
        interp.add_rule(
            "Box",
            vec![ShapeOp::Comp(CompTarget::Faces(vec![
                crate::ops::CompFaceCase {
                    selector: FaceSelector::All,
                    rule: "Face".to_string(),
                },
            ]))],
        );
        let scope = Scope::new(Vec3::ZERO, Quat::IDENTITY, Vec3::new(4.0, 3.0, 2.0));
        let model = interp.derive(scope, "Box").unwrap();
        assert_eq!(model.len(), 6);

        // Collect the outward normals by rotating (0,0,1) with each face's rotation
        let normals: Vec<Vec3> = model
            .terminals
            .iter()
            .map(|t| t.scope.rotation * Vec3::Z)
            .collect();

        // We expect exactly one terminal pointing in each of the 6 cardinal directions
        let expected = [
            Vec3::NEG_Y, // Bottom
            Vec3::Y,     // Top
            Vec3::NEG_Z, // Front
            Vec3::Z,     // Back
            Vec3::NEG_X, // Left
            Vec3::X,     // Right
        ];
        for exp in &expected {
            assert!(
                normals.iter().any(|n| (*n - *exp).length() < 1e-6),
                "missing normal {:?}, got {:?}",
                exp,
                normals
            );
        }
    }
}

//! Genetic evolution wrapper for CGA Shape Grammar interpreters.
//!
//! Provides [`ShapeGenotype`], a [`symbios_genetics::Genotype`]-compatible
//! wrapper around the grammar rule table. Plug it directly into any
//! `symbios-genetics` algorithm (SimpleGA, NSGA-II, MAP-Elites) to evolve
//! procedural building grammars interactively.
//!
//! # Example
//!
//! ```rust
//! use symbios_shape::{Interpreter, Scope, Vec3, Quat};
//! use symbios_shape::grammar::parse_ops;
//! use symbios_shape::genetics::ShapeGenotype;
//! use symbios_genetics::Genotype;
//! use rand::SeedableRng;
//! use rand_pcg::Pcg64;
//!
//! let mut interp = Interpreter::new();
//! interp.add_rule("Lot", parse_ops("Extrude(10) Split(Y) { 3: Floor | ~1: Roof }").unwrap());
//! interp.add_rule("Floor", parse_ops(r#"I("Floor")"#).unwrap());
//! interp.add_rule("Roof",  parse_ops(r#"Taper(0.8) I("Roof")"#).unwrap());
//!
//! let mut dna = ShapeGenotype::from_interpreter(&interp);
//! let mut rng = Pcg64::seed_from_u64(42);
//! dna.mutate(&mut rng, 0.3);
//!
//! let evolved = dna.to_interpreter();
//! let footprint = Scope::new(Vec3::ZERO, Quat::IDENTITY, Vec3::new(10.0, 0.0, 10.0));
//! let _model = evolved.derive(footprint, "Lot").unwrap();
//! ```

use std::collections::HashMap;

use rand::Rng;
use serde::{Deserialize, Serialize};
use symbios_genetics::Genotype;

use crate::interpreter::{Interpreter, WeightedVariant};
use crate::ops::{ShapeOp, SplitSize};
use crate::scope::Vec3;

// ── ShapeGenotype ─────────────────────────────────────────────────────────────

/// Genetic encoding of a CGA shape grammar.
///
/// Wraps the rule table of an [`Interpreter`] so that the grammar can be
/// evolved by `symbios-genetics` algorithms.  Parametric floats are mutated
/// with Gaussian jitter; crossover uses homologous BLX-α blending on rules
/// that share both name and op-sequence topology, or uniform crossover when
/// topologies differ.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ShapeGenotype {
    /// Grammar rules: rule name → weighted variants.
    pub rules: HashMap<String, Vec<WeightedVariant>>,
}

impl ShapeGenotype {
    /// Snapshot the rule table from a live interpreter.
    pub fn from_interpreter(interp: &Interpreter) -> Self {
        Self {
            rules: interp.rules().clone(),
        }
    }

    /// Build a fresh [`Interpreter`] from this genotype.
    ///
    /// Does **not** copy `seed`, `max_depth`, or `max_terminals` — set those
    /// on the returned interpreter if your grammar requires non-default limits.
    pub fn to_interpreter(&self) -> Interpreter {
        let mut interp = Interpreter::new();
        for (name, variants) in &self.rules {
            interp.set_variants(name.clone(), variants.clone());
        }
        interp
    }
}

// ── Genotype impl ─────────────────────────────────────────────────────────────

impl Genotype for ShapeGenotype {
    /// Perturb parametric floats throughout the grammar.
    ///
    /// Each mutable float is independently tested against `rate`.  When
    /// selected, Gaussian noise (Box-Muller) is applied and the result is
    /// clamped to keep the grammar structurally valid:
    ///
    /// | Op | Parameter | σ | Clamp |
    /// |---|---|---|---|
    /// | `Extrude(h)` | h | 0.5 | > 0.1 |
    /// | `Taper(t)` | t | 0.1 | [0, 1] |
    /// | `Scale(v)` | each component | 0.2 | > 0.1 |
    /// | `Translate(v)` | each component | 0.5 | none |
    /// | `Split` slot sizes | size value | 0.3 | > 0.1 (or [0.01, 1] for Relative) |
    /// | `Repeat` tile_size | tile_size | 0.3 | > 0.1 |
    /// | `Roof` angle | angle (°) | 5.0 | [1, 89] |
    /// | `Roof` overhang | overhang | 0.2 | [0, 2] |
    ///
    /// After per-slot jitter, each `Split` is repaired so that its absolute and
    /// relative slot sums do not exceed their pre-mutation totals — this prevents
    /// independent jitter from producing `SplitOverflow` errors at interpret time
    /// (issue #27).
    fn mutate<R: Rng>(&mut self, rng: &mut R, rate: f32) {
        // Iterate by sorted key so the per-op RNG draws are deterministic for
        // a given seed — `HashMap`'s default value iteration order is not.
        let mut keys: Vec<&String> = self.rules.keys().collect();
        keys.sort();
        let keys: Vec<String> = keys.into_iter().cloned().collect();
        for key in keys {
            if let Some(variants) = self.rules.get_mut(&key) {
                for variant in variants.iter_mut() {
                    for op in variant.ops.iter_mut() {
                        mutate_op(op, rng, rate);
                    }
                }
            }
        }
    }

    /// Homologous BLX-α crossover (α = 0.5).
    ///
    /// For each rule name shared by both parents:
    /// - If both parents have the **same number of variants** and each
    ///   variant pair has the **same op-sequence topology** (same variant
    ///   discriminants in the same order), every float parameter is blended
    ///   using BLX-α, producing offspring that explore slightly beyond the
    ///   parental range.
    /// - If the variant counts or topologies differ, the whole rule is
    ///   inherited uniformly at random from one parent (50 / 50).
    ///
    /// Rules present in only one parent are passed through to the child
    /// unchanged, so the child always has a complete, runnable grammar.
    fn crossover<R: Rng>(&self, other: &Self, rng: &mut R) -> Self {
        let mut child_rules = self.rules.clone();

        for (name, self_variants) in &self.rules {
            if let Some(other_variants) = other.rules.get(name) {
                if self_variants.len() == other_variants.len() {
                    let blended: Vec<WeightedVariant> = self_variants
                        .iter()
                        .zip(other_variants.iter())
                        .map(|(sv, ov)| crossover_variant(sv, ov, rng))
                        .collect();
                    child_rules.insert(name.clone(), blended);
                } else if rng.random::<f32>() < 0.5 {
                    child_rules.insert(name.clone(), other_variants.clone());
                }
                // else: keep self's rule (already cloned into child_rules)
            }
        }

        // Rules present only in `other` are added to the child.
        for (name, variants) in &other.rules {
            if !child_rules.contains_key(name) {
                child_rules.insert(name.clone(), variants.clone());
            }
        }

        ShapeGenotype { rules: child_rules }
    }
}

// ── Gaussian jitter ───────────────────────────────────────────────────────────

/// Returns `value + N(0, sigma)` with probability `rate`, else `value`.
///
/// Uses the Box-Muller transform for Gaussian sampling from two uniform draws.
fn jitter<R: Rng>(rng: &mut R, rate: f32, value: f64, sigma: f64) -> f64 {
    if rng.random::<f32>() < rate {
        let u1: f64 = rng.random::<f64>().max(1e-15); // avoid log(0)
        let u2: f64 = rng.random::<f64>();
        let gauss = (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos();
        value + sigma * gauss
    } else {
        value
    }
}

// ── Per-op mutation ───────────────────────────────────────────────────────────

fn mutate_op<R: Rng>(op: &mut ShapeOp, rng: &mut R, rate: f32) {
    match op {
        ShapeOp::Extrude(h) => {
            *h = jitter(rng, rate, *h, 0.5).max(0.1);
        }
        ShapeOp::Taper(t) => {
            *t = jitter(rng, rate, *t, 0.1).clamp(0.0, 1.0);
        }
        ShapeOp::Scale(v) => {
            *v = Vec3::new(
                jitter(rng, rate, v.x, 0.2).max(0.1),
                jitter(rng, rate, v.y, 0.2).max(0.1),
                jitter(rng, rate, v.z, 0.2).max(0.1),
            );
        }
        ShapeOp::Translate(v) => {
            *v = Vec3::new(
                jitter(rng, rate, v.x, 0.5),
                jitter(rng, rate, v.y, 0.5),
                jitter(rng, rate, v.z, 0.5),
            );
        }
        ShapeOp::Split { slots, .. } => {
            // Snapshot pre-mutation sums so we can prevent absolute / relative
            // sums from growing past their original totals. Without this guard
            // independent per-slot Gaussian jitter can push Σ(absolute) past
            // the scope dimension and trip `SplitOverflow` at interpret time.
            let original_abs_sum = sum_split_size(slots, |s| matches!(s, SplitSize::Absolute(_)));
            let original_rel_sum = sum_split_size(slots, |s| matches!(s, SplitSize::Relative(_)));
            for slot in slots.iter_mut() {
                mutate_split_size(&mut slot.size, rng, rate);
            }
            repair_split_sums(slots, original_abs_sum, original_rel_sum);
        }
        ShapeOp::Repeat { tile_sizes, .. } => {
            for ts in tile_sizes.iter_mut() {
                *ts = jitter(rng, rate, *ts, 0.3).max(0.1);
            }
        }
        ShapeOp::Roof { config, .. } => {
            config.pitch = jitter(rng, rate, config.pitch, 5.0).clamp(1.0, 89.0);
            config.overhang = jitter(rng, rate, config.overhang, 0.2).clamp(0.0, 2.0);
        }
        ShapeOp::Polygon(verts) => {
            for v in verts.iter_mut() {
                v.x = jitter(rng, rate, v.x, 0.2);
                v.y = jitter(rng, rate, v.y, 0.2);
            }
        }
        // Non-parametric ops have no float to jitter.
        ShapeOp::Rotate(_)
        | ShapeOp::Comp(_)
        | ShapeOp::Offset { .. }
        | ShapeOp::I(_)
        | ShapeOp::Mat(_)
        | ShapeOp::Rule(_)
        | ShapeOp::Align { .. }
        | ShapeOp::Attach { .. }
        | ShapeOp::RegSnap(_)
        | ShapeOp::IfClear { .. }
        | ShapeOp::IfOccluded { .. } => {}
    }
}

fn mutate_split_size<R: Rng>(size: &mut SplitSize, rng: &mut R, rate: f32) {
    match size {
        SplitSize::Absolute(v) => *v = jitter(rng, rate, *v, 0.3).max(0.1),
        SplitSize::Relative(v) => *v = jitter(rng, rate, *v, 0.05).clamp(0.01, 1.0),
        SplitSize::Floating(v) => *v = jitter(rng, rate, *v, 0.3).max(0.1),
    }
}

/// Sums size values of slots whose `SplitSize` matches `kind`.
fn sum_split_size<F>(slots: &[crate::ops::SplitSlot], kind: F) -> f64
where
    F: Fn(&SplitSize) -> bool,
{
    slots
        .iter()
        .filter(|s| kind(&s.size))
        .map(|s| match s.size {
            SplitSize::Absolute(v) | SplitSize::Relative(v) | SplitSize::Floating(v) => v,
        })
        .sum()
}

/// Prevents Split absolute / relative slot sums from growing past their
/// pre-mutation totals. Each affected slot is scaled down proportionally,
/// preserving the relative weights chosen by the mutation.
fn repair_split_sums(
    slots: &mut [crate::ops::SplitSlot],
    original_abs_sum: f64,
    original_rel_sum: f64,
) {
    let new_abs_sum = sum_split_size(slots, |s| matches!(s, SplitSize::Absolute(_)));
    if original_abs_sum > 1e-9 && new_abs_sum > original_abs_sum {
        let scale = original_abs_sum / new_abs_sum;
        for slot in slots.iter_mut() {
            if let SplitSize::Absolute(v) = &mut slot.size {
                *v = (*v * scale).max(0.1);
            }
        }
    }
    let new_rel_sum = sum_split_size(slots, |s| matches!(s, SplitSize::Relative(_)));
    if original_rel_sum > 1e-9 && new_rel_sum > original_rel_sum {
        let scale = original_rel_sum / new_rel_sum;
        for slot in slots.iter_mut() {
            if let SplitSize::Relative(v) = &mut slot.size {
                *v = (*v * scale).clamp(0.01, 1.0);
            }
        }
    }
}

// ── Per-variant crossover ─────────────────────────────────────────────────────

fn crossover_variant<R: Rng>(
    a: &WeightedVariant,
    b: &WeightedVariant,
    rng: &mut R,
) -> WeightedVariant {
    if same_structure(&a.ops, &b.ops) {
        let ops = a
            .ops
            .iter()
            .zip(b.ops.iter())
            .map(|(ao, bo)| blend_op(ao, bo, rng))
            .collect();
        WeightedVariant {
            weight: blx(a.weight, b.weight, 0.5, rng).max(0.0),
            ops,
        }
    } else {
        // Topologies differ — uniform crossover: pick one parent whole.
        if rng.random::<f32>() < 0.5 {
            a.clone()
        } else {
            b.clone()
        }
    }
}

/// Returns `true` when `a` and `b` have identical ShapeOp variant sequences.
fn same_structure(a: &[ShapeOp], b: &[ShapeOp]) -> bool {
    a.len() == b.len() && a.iter().zip(b.iter()).all(|(ao, bo)| same_op_kind(ao, bo))
}

fn same_op_kind(a: &ShapeOp, b: &ShapeOp) -> bool {
    use ShapeOp::*;
    matches!(
        (a, b),
        (Extrude(_), Extrude(_))
            | (Taper(_), Taper(_))
            | (Rotate(_), Rotate(_))
            | (Translate(_), Translate(_))
            | (Scale(_), Scale(_))
            | (Split { .. }, Split { .. })
            | (Repeat { .. }, Repeat { .. })
            | (Comp(_), Comp(_))
            | (I(_), I(_))
            | (Mat(_), Mat(_))
            | (Rule(_), Rule(_))
            | (Align { .. }, Align { .. })
            | (Offset { .. }, Offset { .. })
            | (Roof { .. }, Roof { .. })
            | (Attach { .. }, Attach { .. })
            | (Polygon(_), Polygon(_))
    )
}

// ── BLX-α blend ──────────────────────────────────────────────────────────────

/// BLX-α blend: samples uniformly from `[min − α·d, max + α·d]` where `d = max − min`.
///
/// With α = 0.0 this reduces to uniform crossover in the parental range.
/// With α = 0.5 (the default used here) it allows moderate exploration beyond parents.
fn blx<R: Rng>(a: f64, b: f64, alpha: f64, rng: &mut R) -> f64 {
    let lo = a.min(b);
    let hi = a.max(b);
    let d = (hi - lo) * alpha;
    let lo_ext = lo - d;
    let hi_ext = hi + d;
    if hi_ext <= lo_ext {
        (a + b) / 2.0
    } else {
        rng.random::<f64>() * (hi_ext - lo_ext) + lo_ext
    }
}

fn blend_op<R: Rng>(a: &ShapeOp, b: &ShapeOp, rng: &mut R) -> ShapeOp {
    match (a, b) {
        (ShapeOp::Extrude(ha), ShapeOp::Extrude(hb)) => {
            ShapeOp::Extrude(blx(*ha, *hb, 0.5, rng).max(0.1))
        }
        (ShapeOp::Taper(ta), ShapeOp::Taper(tb)) => {
            ShapeOp::Taper(blx(*ta, *tb, 0.5, rng).clamp(0.0, 1.0))
        }
        (ShapeOp::Scale(va), ShapeOp::Scale(vb)) => ShapeOp::Scale(Vec3::new(
            blx(va.x, vb.x, 0.5, rng).max(0.1),
            blx(va.y, vb.y, 0.5, rng).max(0.1),
            blx(va.z, vb.z, 0.5, rng).max(0.1),
        )),
        (ShapeOp::Translate(va), ShapeOp::Translate(vb)) => ShapeOp::Translate(Vec3::new(
            blx(va.x, vb.x, 0.5, rng),
            blx(va.y, vb.y, 0.5, rng),
            blx(va.z, vb.z, 0.5, rng),
        )),
        (
            ShapeOp::Split {
                axis,
                slots: slots_a,
                snap,
            },
            ShapeOp::Split { slots: slots_b, .. },
        ) => {
            // Blend slot sizes pairwise; keep rules, axis and snap binding from parent A.
            let slots = if slots_a.len() == slots_b.len() {
                slots_a
                    .iter()
                    .zip(slots_b.iter())
                    .map(|(sa, sb)| crate::ops::SplitSlot {
                        size: blend_split_size(&sa.size, &sb.size, rng),
                        rule: sa.rule.clone(),
                    })
                    .collect()
            } else {
                slots_a.clone()
            };
            ShapeOp::Split {
                axis: *axis,
                slots,
                snap: snap.clone(),
            }
        }
        (
            ShapeOp::Repeat {
                axis,
                tile_sizes: tsa,
                rule,
            },
            ShapeOp::Repeat {
                tile_sizes: tsb, ..
            },
        ) => {
            let blended: Vec<f64> = if tsa.len() == tsb.len() {
                tsa.iter()
                    .zip(tsb.iter())
                    .map(|(a, b)| blx(*a, *b, 0.5, rng).max(0.1))
                    .collect()
            } else {
                tsa.clone()
            };
            ShapeOp::Repeat {
                axis: *axis,
                tile_sizes: blended,
                rule: rule.clone(),
            }
        }
        (ShapeOp::Roof { config: ca, cases }, ShapeOp::Roof { config: cb, .. }) => {
            let mut config = ca.clone();
            config.pitch = blx(ca.pitch, cb.pitch, 0.5, rng).clamp(1.0, 89.0);
            config.overhang = blx(ca.overhang, cb.overhang, 0.5, rng).clamp(0.0, 2.0);
            ShapeOp::Roof {
                config,
                cases: cases.clone(),
            }
        }
        // Non-parametric or unblendable: use parent A unchanged.
        _ => a.clone(),
    }
}

fn blend_split_size<R: Rng>(a: &SplitSize, b: &SplitSize, rng: &mut R) -> SplitSize {
    match (a, b) {
        (SplitSize::Absolute(va), SplitSize::Absolute(vb)) => {
            SplitSize::Absolute(blx(*va, *vb, 0.5, rng).max(0.1))
        }
        (SplitSize::Relative(va), SplitSize::Relative(vb)) => {
            SplitSize::Relative(blx(*va, *vb, 0.5, rng).clamp(0.01, 1.0))
        }
        (SplitSize::Floating(va), SplitSize::Floating(vb)) => {
            SplitSize::Floating(blx(*va, *vb, 0.5, rng).max(0.1))
        }
        // Mixed SplitSize kinds: keep parent A's.
        _ => a.clone(),
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grammar::parse_ops;
    use rand::SeedableRng;
    use rand_pcg::Pcg64;

    fn build_interp() -> Interpreter {
        let mut interp = Interpreter::new();
        interp.add_rule(
            "Lot",
            parse_ops("Extrude(10) Split(Y) { 3: Floor | ~1: Top }").unwrap(),
        );
        interp.add_rule("Floor", parse_ops(r#"Taper(0.0) I("Floor")"#).unwrap());
        interp.add_rule("Top", parse_ops(r#"Taper(0.8) I("Roof")"#).unwrap());
        interp
    }

    #[test]
    fn test_round_trip() {
        let interp = build_interp();
        let dna = ShapeGenotype::from_interpreter(&interp);
        let interp2 = dna.to_interpreter();
        // Both should derive the same shape model.
        let footprint = crate::scope::Scope::new(
            Vec3::ZERO,
            crate::scope::Quat::IDENTITY,
            Vec3::new(10.0, 0.0, 10.0),
        );
        let m1 = interp.derive(footprint, "Lot").unwrap();
        let m2 = interp2.derive(footprint, "Lot").unwrap();
        assert_eq!(m1.len(), m2.len());
        assert_eq!(m1.terminals[0].mesh_id, m2.terminals[0].mesh_id);
    }

    #[test]
    fn test_mutate_preserves_validity() {
        let interp = build_interp();
        let mut dna = ShapeGenotype::from_interpreter(&interp);
        let mut rng = Pcg64::seed_from_u64(7);
        // High rate to exercise most code paths.
        dna.mutate(&mut rng, 1.0);
        let interp2 = dna.to_interpreter();
        let footprint = crate::scope::Scope::new(
            Vec3::ZERO,
            crate::scope::Quat::IDENTITY,
            Vec3::new(10.0, 0.0, 10.0),
        );
        // Should still derive without error.
        interp2.derive(footprint, "Lot").unwrap();
    }

    #[test]
    fn test_crossover_produces_valid_grammar() {
        let interp_a = build_interp();
        let mut interp_b = build_interp();
        // Give parent B different parameters.
        interp_b.add_rule(
            "Lot",
            parse_ops("Extrude(20) Split(Y) { 5: Floor | ~2: Top }").unwrap(),
        );

        let dna_a = ShapeGenotype::from_interpreter(&interp_a);
        let dna_b = ShapeGenotype::from_interpreter(&interp_b);
        let mut rng = Pcg64::seed_from_u64(99);
        let child = dna_a.crossover(&dna_b, &mut rng);
        let interp_child = child.to_interpreter();

        let footprint = crate::scope::Scope::new(
            Vec3::ZERO,
            crate::scope::Quat::IDENTITY,
            Vec3::new(10.0, 0.0, 10.0),
        );
        interp_child.derive(footprint, "Lot").unwrap();
    }

    #[test]
    fn test_crossover_with_disjoint_rules() {
        let mut interp_a = Interpreter::new();
        interp_a.add_rule("A", parse_ops(r#"Extrude(5) I("Mesh")"#).unwrap());

        let mut interp_b = Interpreter::new();
        interp_b.add_rule("B", parse_ops(r#"Extrude(8) I("Mesh")"#).unwrap());

        let dna_a = ShapeGenotype::from_interpreter(&interp_a);
        let dna_b = ShapeGenotype::from_interpreter(&interp_b);
        let mut rng = Pcg64::seed_from_u64(1);
        let child = dna_a.crossover(&dna_b, &mut rng);
        // Child must contain both disjoint rules.
        assert!(child.rules.contains_key("A"));
        assert!(child.rules.contains_key("B"));
    }

    #[test]
    fn test_mutate_extrude_clamp() {
        let mut interp = Interpreter::new();
        interp.add_rule("R", parse_ops("Extrude(0.11) I(M)").unwrap());
        let mut dna = ShapeGenotype::from_interpreter(&interp);
        let mut rng = Pcg64::seed_from_u64(0);
        // Mutate at rate 1.0 many times — Extrude must stay > 0.1.
        for _ in 0..500 {
            dna.mutate(&mut rng, 1.0);
            let h = match &dna.rules["R"][0].ops[0] {
                ShapeOp::Extrude(h) => *h,
                _ => panic!("expected Extrude"),
            };
            assert!(h >= 0.1, "Extrude height {h} < 0.1");
        }
    }

    #[test]
    fn test_blx_same_parents() {
        // When a == b, BLX-α with alpha > 0 still stays near the parental value
        // (d = 0, so lo_ext == hi_ext == a, result should be a).
        use rand::SeedableRng;
        let mut rng = Pcg64::seed_from_u64(42);
        let result = blx(5.0, 5.0, 0.5, &mut rng);
        assert!((result - 5.0).abs() < 1e-9);
    }

    /// Property test for issue #27: 1000 random heavy-mutation passes against a
    /// rich grammar (every parametric op kind) must never produce a genotype that
    /// fails to interpret. Each pass is also exercised against multiple footprints.
    #[test]
    fn test_mutate_property_1000_genotypes_all_interpret() {
        use crate::scope::{Quat, Scope, Vec3};
        use rand::SeedableRng;

        // Grammar that exercises every parametric op kind so the property test
        // covers Extrude, Taper, Scale, Translate, Split (all 3 SplitSize kinds),
        // Repeat, and Roof (pitch + overhang).
        let mut interp = Interpreter::new();
        interp.add_rule(
            "Lot",
            parse_ops("Extrude(8) Split(Y) { 3: Floor | ~1: Mid | '0.2: Cap | 1.5: Top }").unwrap(),
        );
        interp.add_rule("Floor", parse_ops("Repeat(X, 2.0) { Bay }").unwrap());
        interp.add_rule(
            "Bay",
            parse_ops(r#"Scale(0.9, 0.9, 0.9) Translate(0.1, 0, 0) I("Bay")"#).unwrap(),
        );
        interp.add_rule("Mid", parse_ops(r#"Taper(0.3) I("Mid")"#).unwrap());
        interp.add_rule("Cap", parse_ops(r#"I("Cap")"#).unwrap());
        interp.add_rule(
            "Top",
            parse_ops("Roof(Gable, 35) { Slope: Tile | GableEnd: Brick }").unwrap(),
        );

        let footprint = Scope::new(Vec3::ZERO, Quat::IDENTITY, Vec3::new(10.0, 0.0, 6.0));

        let dna_seed = ShapeGenotype::from_interpreter(&interp);
        let mut failures: Vec<(u64, String)> = Vec::new();

        for seed in 0u64..1000 {
            let mut dna = dna_seed.clone();
            let mut rng = Pcg64::seed_from_u64(seed);
            // High rate (1.0) and multiple passes guarantee every parametric float
            // is jittered many times, exercising the clamp boundaries hard.
            for _ in 0..3 {
                dna.mutate(&mut rng, 1.0);
            }
            let interp = dna.to_interpreter();
            match interp.derive(footprint, "Lot") {
                Ok(_) => {}
                Err(e) => failures.push((seed, format!("{e:?}"))),
            }
        }

        assert!(
            failures.is_empty(),
            "{} of 1000 mutated genotypes failed to interpret. First failures: {:?}",
            failures.len(),
            failures.iter().take(5).collect::<Vec<_>>(),
        );
    }
}

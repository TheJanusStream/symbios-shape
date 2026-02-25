/// CGA Shape Grammar text parser.
///
/// Parses a sequence of operations in a grammar rule body:
///
/// ```text
/// Extrude(10)
/// Split(Y) { ~1: Floor | ~1: Floor | 2: Roof }
/// Comp(Faces) { Top: Roof | Side: Facade | Bottom: Foundation }
/// I("Window")
/// Mat("Brick")
/// ```
///
/// Stochastic rules use `weight%` syntax:
/// ```text
/// Facade --> 70% BrickWall | 30% GlassCurtain
/// ```
use nom::{
    IResult, Parser,
    branch::alt,
    bytes::complete::{is_not, tag, take_until, take_while, take_while1},
    character::complete::{char as c_char, multispace1},
    combinator::{map, opt, verify},
    error::{Error, ErrorKind},
    multi::{many0, separated_list1},
    number::complete::double,
    sequence::{delimited, preceded, terminated},
};

use crate::error::ShapeError;
use crate::ops::{Axis, CompFaceCase, CompTarget, FaceSelector, ShapeOp, SplitSize, SplitSlot};
use crate::scope::{Quat, Vec3};

// Safety limits (DoS protection)
const MAX_SPLIT_SLOTS: usize = 256;
const MAX_COMP_CASES: usize = 32;
const MAX_OPS: usize = 1024;
const MAX_IDENTIFIER_LEN: usize = 64;

// ── Whitespace & comments ────────────────────────────────────────────────────

fn space_or_comment<'a, E: nom::error::ParseError<&'a str>>(
    input: &'a str,
) -> IResult<&'a str, (), E> {
    let comment = alt((
        preceded(tag("/*"), terminated(take_until("*/"), tag("*/"))),
        preceded(tag("//"), is_not("\n\r")),
    ));
    let mut p = many0(alt((map(multispace1, |_| ()), map(comment, |_| ()))));
    p.parse(input).map(|(i, _)| (i, ()))
}

fn ws<'a, F, O, E: nom::error::ParseError<&'a str>>(
    inner: F,
) -> impl Parser<&'a str, Output = O, Error = E>
where
    F: Parser<&'a str, Output = O, Error = E>,
{
    delimited(space_or_comment, inner, space_or_comment)
}

// ── Primitives ───────────────────────────────────────────────────────────────

fn finite_float(input: &str) -> IResult<&str, f64> {
    verify(double, |x: &f64| x.is_finite()).parse(input)
}

fn is_ident_start(c: char) -> bool {
    c.is_alphabetic() || c == '_'
}

fn is_ident_char(c: char) -> bool {
    c.is_alphanumeric() || c == '_'
}

fn identifier(input: &str) -> IResult<&str, &str> {
    let (input, s) = take_while1(is_ident_char).parse(input)?;
    if !s.chars().next().map(is_ident_start).unwrap_or(false) {
        return Err(nom::Err::Error(Error::new(input, ErrorKind::Alpha)));
    }
    if s.len() > MAX_IDENTIFIER_LEN {
        return Err(nom::Err::Failure(Error::new(input, ErrorKind::TooLarge)));
    }
    Ok((input, s))
}

/// Parses an identifier OR a quoted string literal `"name"`.
fn rule_name(input: &str) -> IResult<&str, String> {
    alt((
        map(
            delimited(c_char('"'), take_while(|c| c != '"'), c_char('"')),
            |s: &str| s.to_string(),
        ),
        map(ws(identifier), |s: &str| s.to_string()),
    ))
    .parse(input)
}

// ── Axis ─────────────────────────────────────────────────────────────────────

fn parse_axis(input: &str) -> IResult<&str, Axis> {
    alt((
        map(tag("X"), |_| Axis::X),
        map(tag("Y"), |_| Axis::Y),
        map(tag("Z"), |_| Axis::Z),
    ))
    .parse(input)
}

// ── Vec3 / Quat literals ──────────────────────────────────────────────────────

/// Parses `(x, y, z)`.
fn parse_vec3(input: &str) -> IResult<&str, Vec3> {
    let (input, _) = ws(c_char('(')).parse(input)?;
    let (input, x) = ws(finite_float).parse(input)?;
    let (input, _) = ws(c_char(',')).parse(input)?;
    let (input, y) = ws(finite_float).parse(input)?;
    let (input, _) = ws(c_char(',')).parse(input)?;
    let (input, z) = ws(finite_float).parse(input)?;
    let (input, _) = ws(c_char(')')).parse(input)?;
    Ok((input, Vec3::new(x, y, z)))
}

/// Parses `(w, x, y, z)` — grammar syntax uses (w,x,y,z); glam DQuat stores (x,y,z,w).
fn parse_quat(input: &str) -> IResult<&str, Quat> {
    let (input, _) = ws(c_char('(')).parse(input)?;
    let (input, w) = ws(finite_float).parse(input)?;
    let (input, _) = ws(c_char(',')).parse(input)?;
    let (input, x) = ws(finite_float).parse(input)?;
    let (input, _) = ws(c_char(',')).parse(input)?;
    let (input, y) = ws(finite_float).parse(input)?;
    let (input, _) = ws(c_char(',')).parse(input)?;
    let (input, z) = ws(finite_float).parse(input)?;
    let (input, _) = ws(c_char(')')).parse(input)?;
    // glam DQuat::from_xyzw takes (x, y, z, w)
    Ok((input, Quat::from_xyzw(x, y, z, w)))
}

// ── Split sizes ───────────────────────────────────────────────────────────────

/// `~2.5` → `Floating(2.5)`, `'0.5` → `Relative(0.5)`, `2.5` → `Absolute(2.5)`
fn parse_split_size(input: &str) -> IResult<&str, SplitSize> {
    alt((
        map(preceded(c_char('~'), ws(finite_float)), SplitSize::Floating),
        map(
            preceded(c_char('\''), ws(finite_float)),
            SplitSize::Relative,
        ),
        map(ws(finite_float), SplitSize::Absolute),
    ))
    .parse(input)
}

/// `~1.0: Floor` or `2.0: Roof`
fn parse_split_slot(input: &str) -> IResult<&str, SplitSlot> {
    let (input, size) = ws(parse_split_size).parse(input)?;
    let (input, _) = ws(c_char(':')).parse(input)?;
    let (input, rule) = ws(rule_name).parse(input)?;
    Ok((input, SplitSlot { size, rule }))
}

// ── Comp faces ────────────────────────────────────────────────────────────────

fn parse_comp_face_case(input: &str) -> IResult<&str, CompFaceCase> {
    let (input, sel_str) = ws(identifier).parse(input)?;
    let selector = FaceSelector::parse(sel_str)
        .ok_or_else(|| nom::Err::Failure(Error::new(input, ErrorKind::Tag)))?;
    let (input, _) = ws(c_char(':')).parse(input)?;
    let (input, rule) = ws(rule_name).parse(input)?;
    Ok((input, CompFaceCase { selector, rule }))
}

// ── Individual operations ─────────────────────────────────────────────────────

fn parse_extrude(input: &str) -> IResult<&str, ShapeOp> {
    let (input, _) = tag("Extrude").parse(input)?;
    let (input, h) = delimited(ws(c_char('(')), ws(finite_float), ws(c_char(')'))).parse(input)?;
    if h <= 0.0 {
        return Err(nom::Err::Failure(Error::new(input, ErrorKind::Verify)));
    }
    Ok((input, ShapeOp::Extrude(h)))
}

fn parse_taper(input: &str) -> IResult<&str, ShapeOp> {
    let (input, _) = tag("Taper").parse(input)?;
    let (input, amount) =
        delimited(ws(c_char('(')), ws(finite_float), ws(c_char(')'))).parse(input)?;
    if !(0.0..=1.0).contains(&amount) {
        return Err(nom::Err::Failure(Error::new(input, ErrorKind::Verify)));
    }
    Ok((input, ShapeOp::Taper(amount)))
}

fn parse_rotate(input: &str) -> IResult<&str, ShapeOp> {
    let (input, _) = tag("Rotate").parse(input)?;
    let (input, q) = ws(parse_quat).parse(input)?;
    Ok((input, ShapeOp::Rotate(q)))
}

fn parse_translate(input: &str) -> IResult<&str, ShapeOp> {
    let (input, _) = tag("Translate").parse(input)?;
    let (input, v) = ws(parse_vec3).parse(input)?;
    Ok((input, ShapeOp::Translate(v)))
}

fn parse_scale(input: &str) -> IResult<&str, ShapeOp> {
    let (input, _) = tag("Scale").parse(input)?;
    let (input, v) = ws(parse_vec3).parse(input)?;
    Ok((input, ShapeOp::Scale(v)))
}

fn parse_split(input: &str) -> IResult<&str, ShapeOp> {
    let (input, _) = tag("Split").parse(input)?;
    let (input, _) = ws(c_char('(')).parse(input)?;
    let (input, axis) = ws(parse_axis).parse(input)?;
    let (input, _) = ws(c_char(')')).parse(input)?;
    let (input, _) = ws(c_char('{')).parse(input)?;
    let (input, slots) = separated_list1(ws(c_char('|')), ws(parse_split_slot)).parse(input)?;
    let (input, _) = opt(ws(c_char('|'))).parse(input)?;
    let (input, _) = ws(c_char('}')).parse(input)?;
    if slots.len() > MAX_SPLIT_SLOTS {
        return Err(nom::Err::Failure(Error::new(input, ErrorKind::TooLarge)));
    }
    Ok((input, ShapeOp::Split { axis, slots }))
}

fn parse_repeat(input: &str) -> IResult<&str, ShapeOp> {
    let (input, _) = tag("Repeat").parse(input)?;
    let (input, _) = ws(c_char('(')).parse(input)?;
    let (input, axis) = ws(parse_axis).parse(input)?;
    let (input, _) = ws(c_char(',')).parse(input)?;
    let (input, tile_size) = ws(finite_float).parse(input)?;
    let (input, _) = ws(c_char(')')).parse(input)?;
    let (input, _) = ws(c_char('{')).parse(input)?;
    let (input, rule) = ws(rule_name).parse(input)?;
    let (input, _) = ws(c_char('}')).parse(input)?;
    if tile_size <= 0.0 {
        return Err(nom::Err::Failure(Error::new(input, ErrorKind::Verify)));
    }
    Ok((
        input,
        ShapeOp::Repeat {
            axis,
            tile_size,
            rule,
        },
    ))
}

fn parse_comp(input: &str) -> IResult<&str, ShapeOp> {
    let (input, _) = tag("Comp").parse(input)?;
    let (input, _) = ws(c_char('(')).parse(input)?;
    let (input, kind) = ws(tag("Faces")).parse(input)?;
    let _ = kind; // only Faces supported in v0.1
    let (input, _) = ws(c_char(')')).parse(input)?;
    let (input, _) = ws(c_char('{')).parse(input)?;
    let (input, cases) = separated_list1(ws(c_char('|')), ws(parse_comp_face_case)).parse(input)?;
    let (input, _) = opt(ws(c_char('|'))).parse(input)?;
    let (input, _) = ws(c_char('}')).parse(input)?;
    if cases.len() > MAX_COMP_CASES {
        return Err(nom::Err::Failure(Error::new(input, ErrorKind::TooLarge)));
    }
    Ok((input, ShapeOp::Comp(CompTarget::Faces(cases))))
}

fn parse_instance(input: &str) -> IResult<&str, ShapeOp> {
    let (input, _) = tag("I").parse(input)?;
    let (input, mesh_id) =
        delimited(ws(c_char('(')), ws(rule_name), ws(c_char(')'))).parse(input)?;
    Ok((input, ShapeOp::I(mesh_id)))
}

/// Parses `Mat("Brick")` or `Mat(Brick)` — sets material on the current work item.
fn parse_mat(input: &str) -> IResult<&str, ShapeOp> {
    let (input, _) = tag("Mat").parse(input)?;
    let (input, mat_id) =
        delimited(ws(c_char('(')), ws(rule_name), ws(c_char(')'))).parse(input)?;
    Ok((input, ShapeOp::Mat(mat_id)))
}

fn parse_rule_ref(input: &str) -> IResult<&str, ShapeOp> {
    map(ws(rule_name), ShapeOp::Rule).parse(input)
}

// ── Top-level op parser ───────────────────────────────────────────────────────

fn parse_op(input: &str) -> IResult<&str, ShapeOp> {
    // Order matters: longer/specific tags must come before the generic rule_name fallback.
    alt((
        parse_extrude,
        parse_taper,
        parse_rotate,
        parse_translate,
        parse_scale,
        parse_split,
        parse_repeat,
        parse_comp,
        parse_instance,
        parse_mat,
        parse_rule_ref,
    ))
    .parse(input)
}

// ── Public API ────────────────────────────────────────────────────────────────

/// Parses a sequence of CGA operations separated by optional whitespace/newlines.
///
/// # Example
/// ```
/// use symbios_shape::grammar::parse_ops;
///
/// let ops = parse_ops("Extrude(10) Split(Y) { ~1: Floor | 2: Roof }").unwrap();
/// assert_eq!(ops.len(), 2);
/// ```
pub fn parse_ops(input: &str) -> Result<Vec<ShapeOp>, ShapeError> {
    let mut remaining = input;
    let mut ops = Vec::new();

    loop {
        let (next, _) = space_or_comment::<Error<&str>>(remaining)
            .map_err(|e| ShapeError::ParseError(e.to_string()))?;
        remaining = next;

        if remaining.is_empty() {
            break;
        }

        if ops.len() >= MAX_OPS {
            return Err(ShapeError::CapacityOverflow);
        }

        let (next, op) = parse_op(remaining).map_err(|e| ShapeError::ParseError(e.to_string()))?;
        ops.push(op);
        remaining = next;
    }

    Ok(ops)
}

// ── Stochastic rule helpers ───────────────────────────────────────────────────

/// Splits `s` on `|` characters at brace depth 0.
/// This correctly skips `|` inside `Split { }` and `Comp { }` blocks.
fn split_top_level_pipe(s: &str) -> Vec<&str> {
    let mut depth: i32 = 0;
    let mut start = 0;
    let mut parts = Vec::new();
    for (i, c) in s.char_indices() {
        match c {
            '{' => depth += 1,
            '}' => depth -= 1,
            '|' if depth == 0 => {
                parts.push(s[start..i].trim());
                start = i + 1;
            }
            _ => {}
        }
    }
    parts.push(s[start..].trim());
    parts
}

fn parse_weight_prefix(input: &str) -> IResult<&str, f64> {
    let (input, w) = ws(double).parse(input)?;
    let (input, _) = ws(c_char('%')).parse(input)?;
    Ok((input, w))
}

/// Tries to parse `float%` from the start of `input`, returning `(weight_0_to_1, rest)`.
fn try_parse_weight(input: &str) -> Option<(f64, &str)> {
    let trimmed = input.trim_start();
    match parse_weight_prefix(trimmed) {
        Ok((rest, w)) if w.is_finite() && w >= 0.0 => Some((w / 100.0, rest)),
        _ => None,
    }
}

// ── Named grammar rules ───────────────────────────────────────────────────────

/// A named production rule in a shape grammar.
///
/// Each variant carries a relative weight and a sequence of ops.
/// For deterministic rules there is exactly one variant with weight `1.0`.
///
/// ```text
/// Lot --> Extrude(10) Split(Y) { ~1: Floor | 2: Roof }
/// Facade --> 70% BrickWall | 30% GlassCurtain
/// ```
#[derive(Debug, Clone)]
pub struct GrammarRule {
    pub name: String,
    /// `(weight, ops)` pairs. Weights are relative (not necessarily normalised to 1.0).
    pub variants: Vec<(f64, Vec<ShapeOp>)>,
}

impl GrammarRule {
    /// Convenience accessor for deterministic (single-variant) rules.
    pub fn ops(&self) -> &[ShapeOp] {
        &self.variants[0].1
    }
}

/// Parses a complete named grammar rule: `Name --> op op ...`
/// or stochastic: `Name --> 70% ops | 30% ops`
pub fn parse_rule(input: &str) -> Result<GrammarRule, ShapeError> {
    let (remaining, _) = space_or_comment::<Error<&str>>(input)
        .map_err(|e| ShapeError::ParseError(e.to_string()))?;
    let (remaining, name) = ws(identifier)
        .parse(remaining)
        .map_err(|e| ShapeError::ParseError(e.to_string()))?;
    let (remaining, _) = ws(tag::<_, _, Error<&str>>("-->"))
        .parse(remaining)
        .map_err(|e| ShapeError::ParseError(e.to_string()))?;

    // Split the body on top-level `|` to detect stochastic alternatives.
    let parts = split_top_level_pipe(remaining);

    let variants = if parts.len() > 1 {
        // Multiple alternatives: each must start with `weight%`
        parts
            .iter()
            .map(|part| match try_parse_weight(part) {
                Some((weight, rest)) => Ok((weight, parse_ops(rest)?)),
                None => Err(ShapeError::ParseError(format!(
                    "stochastic rule alternative missing 'weight%' prefix: {:?}",
                    part
                ))),
            })
            .collect::<Result<Vec<_>, _>>()?
    } else {
        // Single deterministic variant
        vec![(1.0, parse_ops(remaining)?)]
    };

    Ok(GrammarRule {
        name: name.to_string(),
        variants,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::{Axis, ShapeOp, SplitSize};

    #[test]
    fn test_parse_extrude() {
        let ops = parse_ops("Extrude(10)").unwrap();
        assert_eq!(ops.len(), 1);
        assert_eq!(ops[0], ShapeOp::Extrude(10.0));
    }

    #[test]
    fn test_parse_taper() {
        let ops = parse_ops("Taper(0.5)").unwrap();
        assert_eq!(ops[0], ShapeOp::Taper(0.5));
    }

    #[test]
    fn test_parse_split_y() {
        let ops = parse_ops("Split(Y) { ~1.0: Floor | ~1.0: Floor | 2.0: Roof }").unwrap();
        assert_eq!(ops.len(), 1);
        let ShapeOp::Split { axis, slots } = &ops[0] else {
            panic!("expected Split");
        };
        assert_eq!(*axis, Axis::Y);
        assert_eq!(slots.len(), 3);
        assert_eq!(slots[0].size, SplitSize::Floating(1.0));
        assert_eq!(slots[0].rule, "Floor");
        assert_eq!(slots[2].size, SplitSize::Absolute(2.0));
        assert_eq!(slots[2].rule, "Roof");
    }

    #[test]
    fn test_parse_split_relative() {
        let ops = parse_ops("Split(X) { '0.3: Left | '0.7: Right }").unwrap();
        let ShapeOp::Split { slots, .. } = &ops[0] else {
            panic!("expected Split");
        };
        assert_eq!(slots[0].size, SplitSize::Relative(0.3));
        assert_eq!(slots[1].size, SplitSize::Relative(0.7));
    }

    #[test]
    fn test_parse_comp_faces() {
        let ops =
            parse_ops("Comp(Faces) { Top: Roof | Side: Facade | Bottom: Foundation }").unwrap();
        assert_eq!(ops.len(), 1);
        let ShapeOp::Comp(CompTarget::Faces(cases)) = &ops[0] else {
            panic!("expected Comp(Faces)");
        };
        assert_eq!(cases.len(), 3);
        assert_eq!(cases[0].selector, FaceSelector::Top);
        assert_eq!(cases[0].rule, "Roof");
    }

    #[test]
    fn test_parse_instance() {
        let ops = parse_ops(r#"I("Window")"#).unwrap();
        assert_eq!(ops[0], ShapeOp::I("Window".to_string()));
    }

    #[test]
    fn test_parse_mat() {
        let ops = parse_ops(r#"Mat("Brick")"#).unwrap();
        assert_eq!(ops[0], ShapeOp::Mat("Brick".to_string()));
        let ops2 = parse_ops("Mat(Stone)").unwrap();
        assert_eq!(ops2[0], ShapeOp::Mat("Stone".to_string()));
    }

    #[test]
    fn test_parse_rule_ref() {
        let ops = parse_ops("Floor").unwrap();
        assert_eq!(ops[0], ShapeOp::Rule("Floor".to_string()));
    }

    #[test]
    fn test_parse_multiple_ops() {
        let ops =
            parse_ops(r#"Extrude(10) Split(Y) { 2.0: Ground | ~1.0: Upper | 3.0: Roof }"#).unwrap();
        assert_eq!(ops.len(), 2);
    }

    #[test]
    fn test_parse_grammar_rule_deterministic() {
        let rule = parse_rule("Lot --> Extrude(10) Split(Y) { ~1: Floor | 2: Roof }").unwrap();
        assert_eq!(rule.name, "Lot");
        assert_eq!(rule.variants.len(), 1);
        assert!((rule.variants[0].0 - 1.0).abs() < 1e-9);
        assert_eq!(rule.variants[0].1.len(), 2);
    }

    #[test]
    fn test_parse_grammar_rule_stochastic() {
        let rule = parse_rule("Facade --> 70% BrickWall | 30% GlassCurtain").unwrap();
        assert_eq!(rule.name, "Facade");
        assert_eq!(rule.variants.len(), 2);
        assert!((rule.variants[0].0 - 0.70).abs() < 1e-9);
        assert_eq!(
            rule.variants[0].1,
            vec![ShapeOp::Rule("BrickWall".to_string())]
        );
        assert!((rule.variants[1].0 - 0.30).abs() < 1e-9);
        assert_eq!(
            rule.variants[1].1,
            vec![ShapeOp::Rule("GlassCurtain".to_string())]
        );
    }

    #[test]
    fn test_parse_stochastic_with_complex_ops() {
        // `|` inside braces must not split the alternative
        let rule = parse_rule("R --> 50% Split(X) { ~1: A | ~1: B } | 50% I(Solid)").unwrap();
        assert_eq!(rule.variants.len(), 2);
        assert_eq!(rule.variants[0].1.len(), 1); // the Split op
        assert_eq!(rule.variants[1].1.len(), 1); // I(Solid)
    }

    #[test]
    fn test_extrude_zero_rejected() {
        assert!(parse_ops("Extrude(0)").is_err());
    }

    #[test]
    fn test_taper_out_of_range_rejected() {
        assert!(parse_ops("Taper(1.5)").is_err());
    }

    #[test]
    fn test_comments_ignored() {
        let ops = parse_ops("// comment\nExtrude(5) /* block */ Taper(0.2)").unwrap();
        assert_eq!(ops.len(), 2);
    }
}

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
    bytes::complete::{tag, take_until, take_while, take_while_m_n, take_while1},
    character::complete::{char as c_char, multispace1},
    combinator::{map, verify},
    error::{Error, ErrorKind},
    multi::many0,
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
        // take_while (unlike is_not) accepts empty input, so `// EOF` without a
        // trailing newline parses correctly instead of returning a fatal error.
        preceded(tag("//"), take_while(|c: char| c != '\n' && c != '\r')),
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
    let original = input;
    let (input, s) = take_while1(is_ident_char).parse(input)?;
    if !s.chars().next().map(is_ident_start).unwrap_or(false) {
        // Use `original` so the error span points to the start of the bad token,
        // not to the position after consuming it.
        return Err(nom::Err::Error(Error::new(original, ErrorKind::Alpha)));
    }
    if s.len() > MAX_IDENTIFIER_LEN {
        return Err(nom::Err::Failure(Error::new(input, ErrorKind::TooLarge)));
    }
    Ok((input, s))
}

/// Parses an identifier OR a quoted string literal `"name"`.
///
/// Quoted strings are limited to `MAX_IDENTIFIER_LEN` characters via
/// `take_while_m_n`: the scan stops at the limit, causing `c_char('"')` to
/// fail for oversized inputs before any allocation occurs.
fn rule_name(input: &str) -> IResult<&str, String> {
    alt((
        map(
            delimited(
                c_char('"'),
                take_while_m_n(0, MAX_IDENTIFIER_LEN, |c: char| c != '"'),
                c_char('"'),
            ),
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
    // glam requires unit quaternions; normalize the user-supplied values.
    // Reject degenerate (near-zero) inputs that cannot be normalized.
    let len_sq = x * x + y * y + z * z + w * w;
    if len_sq < 1e-12 {
        return Err(nom::Err::Failure(Error::new(input, ErrorKind::Verify)));
    }
    // glam DQuat::from_xyzw takes (x, y, z, w)
    Ok((input, Quat::from_xyzw(x, y, z, w).normalize()))
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
    if v.x <= 0.0 || v.y <= 0.0 || v.z <= 0.0 {
        return Err(nom::Err::Failure(Error::new(input, ErrorKind::Verify)));
    }
    Ok((input, ShapeOp::Scale(v)))
}

fn parse_split(input: &str) -> IResult<&str, ShapeOp> {
    let (input, _) = tag("Split").parse(input)?;
    let (input, _) = ws(c_char('(')).parse(input)?;
    let (input, axis) = ws(parse_axis).parse(input)?;
    let (input, _) = ws(c_char(')')).parse(input)?;
    let (input, _) = ws(c_char('{')).parse(input)?;
    // Bounded list: cap allocation before it happens (DoS guard).
    let (mut remaining, first) = ws(parse_split_slot).parse(input)?;
    let mut slots = vec![first];
    loop {
        if slots.len() >= MAX_SPLIT_SLOTS {
            return Err(nom::Err::Failure(Error::new(
                remaining,
                ErrorKind::TooLarge,
            )));
        }
        let Ok((after_sep, _)) = ws::<_, _, Error<&str>>(c_char('|')).parse(remaining) else {
            break;
        };
        match ws(parse_split_slot).parse(after_sep) {
            Ok((after_item, slot)) => {
                slots.push(slot);
                remaining = after_item;
            }
            Err(nom::Err::Failure(e)) => return Err(nom::Err::Failure(e)),
            Err(_) => {
                remaining = after_sep; // trailing `|` — consume separator, stop
                break;
            }
        }
    }
    let (remaining, _) = ws(c_char('}')).parse(remaining)?;
    Ok((remaining, ShapeOp::Split { axis, slots }))
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
    // Bounded list: cap allocation before it happens (DoS guard).
    let (mut remaining, first) = ws(parse_comp_face_case).parse(input)?;
    let mut cases = vec![first];
    loop {
        if cases.len() >= MAX_COMP_CASES {
            return Err(nom::Err::Failure(Error::new(
                remaining,
                ErrorKind::TooLarge,
            )));
        }
        let Ok((after_sep, _)) = ws::<_, _, Error<&str>>(c_char('|')).parse(remaining) else {
            break;
        };
        match ws(parse_comp_face_case).parse(after_sep) {
            Ok((after_item, case)) => {
                cases.push(case);
                remaining = after_item;
            }
            Err(nom::Err::Failure(e)) => return Err(nom::Err::Failure(e)),
            Err(_) => {
                remaining = after_sep; // trailing `|` — consume separator, stop
                break;
            }
        }
    }
    let (remaining, _) = ws(c_char('}')).parse(remaining)?;
    Ok((remaining, ShapeOp::Comp(CompTarget::Faces(cases))))
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

/// Returns true if `op` ends execution of a rule body — either by branching
/// (Split / Comp / Repeat) or by producing a terminal (I / Rule).
/// Any operations listed after such an op in the source are unreachable.
fn is_terminating_op(op: &ShapeOp) -> bool {
    matches!(
        op,
        ShapeOp::I(_)
            | ShapeOp::Rule(_)
            | ShapeOp::Split { .. }
            | ShapeOp::Comp(_)
            | ShapeOp::Repeat { .. }
    )
}

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
        let terminates = is_terminating_op(&op);
        ops.push(op);
        remaining = next;

        if terminates {
            // Any content after a terminal/branching op is unreachable. Catch it
            // here so the parser rejects such rules rather than silently dropping ops.
            let (after_ws, _) = space_or_comment::<Error<&str>>(remaining)
                .map_err(|e| ShapeError::ParseError(e.to_string()))?;
            if !after_ws.is_empty() {
                return Err(ShapeError::ParseError(
                    "unreachable operations after terminal or branching op".to_string(),
                ));
            }
            break;
        }
    }

    Ok(ops)
}

// ── Stochastic rule helpers ───────────────────────────────────────────────────

/// Splits `s` on `|` characters at brace depth 0, respecting quoted strings
/// and comments so that `|` inside `"Wall|Door"` or `/* { */` is ignored.
///
/// Contexts handled:
/// - `"..."` string literals — `|` and `{`/`}` inside are invisible.
/// - `/* ... */` block comments — same.
/// - `// ...` line comments — same (skipped until `\n` or `\r`).
fn split_top_level_pipe(s: &str) -> Vec<&str> {
    let mut depth: i32 = 0;
    let mut in_string = false;
    let mut in_block_comment = false;
    let mut in_line_comment = false;
    let mut start = 0;
    let mut parts = Vec::new();
    // Collect into a vec so we can peek ahead by index.
    let chars: Vec<(usize, char)> = s.char_indices().collect();
    let len = chars.len();
    let mut i = 0;

    while i < len {
        let (byte_pos, c) = chars[i];

        if in_line_comment {
            if c == '\n' || c == '\r' {
                in_line_comment = false;
            }
            i += 1;
            continue;
        }

        if in_block_comment {
            if c == '*' && i + 1 < len && chars[i + 1].1 == '/' {
                in_block_comment = false;
                i += 2;
            } else {
                i += 1;
            }
            continue;
        }

        if in_string {
            if c == '"' {
                in_string = false;
            }
            i += 1;
            continue;
        }

        // Normal (unquoted, uncommented) context — check for comment openers.
        if c == '/' && i + 1 < len {
            match chars[i + 1].1 {
                '/' => {
                    in_line_comment = true;
                    i += 2;
                    continue;
                }
                '*' => {
                    in_block_comment = true;
                    i += 2;
                    continue;
                }
                _ => {}
            }
        }

        match c {
            '"' => in_string = true,
            '{' => depth += 1,
            '}' => depth -= 1,
            '|' if depth == 0 => {
                parts.push(s[start..byte_pos].trim());
                start = byte_pos + c.len_utf8();
            }
            _ => {}
        }
        i += 1;
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
        // Single variant: may optionally carry a `weight%` prefix (e.g. `100% Extrude(10)`).
        // If present, strip it; otherwise treat as deterministic weight 1.0.
        let part = parts[0];
        match try_parse_weight(part) {
            Some((weight, rest)) => vec![(weight, parse_ops(rest)?)],
            None => vec![(1.0, parse_ops(remaining)?)],
        }
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

    // ── Issue 1: split_top_level_pipe must ignore `|` inside quoted strings ────

    #[test]
    fn test_pipe_in_quoted_mesh_id_not_split() {
        // `|` inside `"Wall|Door"` must NOT be treated as a stochastic separator.
        let rule = parse_rule(r#"Lot --> I("Wall|Door")"#).unwrap();
        assert_eq!(rule.variants.len(), 1);
        assert_eq!(
            rule.variants[0].1,
            vec![ShapeOp::I("Wall|Door".to_string())]
        );
    }

    #[test]
    fn test_block_comment_with_brace_does_not_confuse_depth() {
        // `/* { */` must not increment brace depth, so the top-level `|` is still found.
        let rule = parse_rule("Facade --> 50% Extrude(10) /* { */ | 50% I(Solid)").unwrap();
        assert_eq!(rule.variants.len(), 2);
        assert!((rule.variants[0].0 - 0.50).abs() < 1e-9);
        assert!((rule.variants[1].0 - 0.50).abs() < 1e-9);
    }

    // ── Issue 2: unreachable ops after terminal/branching op are rejected ──────

    #[test]
    fn test_ops_after_instance_rejected() {
        assert!(parse_ops(r#"I("Wall") Scale(2, 2, 2)"#).is_err());
    }

    #[test]
    fn test_ops_after_rule_ref_rejected() {
        assert!(parse_ops("Floor Scale(2, 2, 2)").is_err());
    }

    #[test]
    fn test_ops_after_split_rejected() {
        assert!(parse_ops("Split(Y) { ~1: A | ~1: B } Scale(1, 2, 1)").is_err());
    }

    // ── Issue 3: single-variant stochastic rule (100% prefix) ─────────────────

    #[test]
    fn test_single_variant_with_weight_prefix() {
        let rule = parse_rule("Lot --> 100% Extrude(10)").unwrap();
        assert_eq!(rule.variants.len(), 1);
        assert!((rule.variants[0].0 - 1.0).abs() < 1e-9);
        assert_eq!(rule.variants[0].1, vec![ShapeOp::Extrude(10.0)]);
    }

    // ── Issue 5: non-positive scale values are rejected ───────────────────────

    #[test]
    fn test_scale_negative_rejected() {
        assert!(parse_ops("Scale(-1, 1, 1)").is_err());
    }

    #[test]
    fn test_scale_zero_rejected() {
        assert!(parse_ops("Scale(0, 1, 1)").is_err());
    }

    #[test]
    fn test_scale_positive_accepted() {
        let ops = parse_ops("Scale(0.5, 2, 1)").unwrap();
        assert_eq!(ops.len(), 1);
    }
}

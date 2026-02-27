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
    combinator::{cut, map, opt, verify},
    error::{Error, ErrorKind},
    multi::many0,
    number::complete::double,
    sequence::{delimited, preceded, terminated},
};

use crate::error::ShapeError;
use crate::ops::{
    Axis, CompFaceCase, CompTarget, FaceSelector, OffsetCase, OffsetSelector, RoofCase,
    RoofFaceSelector, RoofType, ShapeOp, SplitSize, SplitSlot,
};
use crate::scope::{Quat, Vec3};

// Safety limits (DoS protection)
const MAX_SPLIT_SLOTS: usize = 256;
const MAX_COMP_CASES: usize = 32;
const MAX_OPS: usize = 1024;
const MAX_IDENTIFIER_LEN: usize = 64;
/// Maximum number of stochastic variants in a single rule body (`A | B | C…`).
/// Without this cap an attacker can force unbounded `Vec` allocations in
/// `split_top_level_pipe` before any derivation-time guards can fire.
pub const MAX_VARIANTS: usize = 64;

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
    // Also reject overflow: components like 1e160 are finite but len_sq = INFINITY.
    if !len_sq.is_finite() || len_sq < 1e-12 {
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
    let (input, h) = cut(delimited(
        ws(c_char('(')),
        ws(finite_float),
        ws(c_char(')')),
    ))
    .parse(input)?;
    if h <= 0.0 {
        return Err(nom::Err::Failure(Error::new(input, ErrorKind::Verify)));
    }
    Ok((input, ShapeOp::Extrude(h)))
}

fn parse_taper(input: &str) -> IResult<&str, ShapeOp> {
    let (input, _) = tag("Taper").parse(input)?;
    let (input, amount) = cut(delimited(
        ws(c_char('(')),
        ws(finite_float),
        ws(c_char(')')),
    ))
    .parse(input)?;
    if !(0.0..=1.0).contains(&amount) {
        return Err(nom::Err::Failure(Error::new(input, ErrorKind::Verify)));
    }
    Ok((input, ShapeOp::Taper(amount)))
}

fn parse_rotate(input: &str) -> IResult<&str, ShapeOp> {
    let (input, _) = tag("Rotate").parse(input)?;
    let (input, q) = cut(ws(parse_quat)).parse(input)?;
    Ok((input, ShapeOp::Rotate(q)))
}

fn parse_translate(input: &str) -> IResult<&str, ShapeOp> {
    let (input, _) = tag("Translate").parse(input)?;
    let (input, v) = cut(ws(parse_vec3)).parse(input)?;
    Ok((input, ShapeOp::Translate(v)))
}

fn parse_scale(input: &str) -> IResult<&str, ShapeOp> {
    let (input, _) = tag("Scale").parse(input)?;
    let (input, v) = cut(ws(parse_vec3)).parse(input)?;
    if v.x <= 0.0 || v.y <= 0.0 || v.z <= 0.0 {
        return Err(nom::Err::Failure(Error::new(input, ErrorKind::Verify)));
    }
    Ok((input, ShapeOp::Scale(v)))
}

fn parse_split(input: &str) -> IResult<&str, ShapeOp> {
    let (input, _) = tag("Split").parse(input)?;
    // After the "Split" keyword is consumed, any argument error is fatal.
    let (input, _) = cut(ws(c_char('('))).parse(input)?;
    let (input, axis) = cut(ws(parse_axis)).parse(input)?;
    let (input, _) = cut(ws(c_char(')'))).parse(input)?;
    let (input, _) = cut(ws(c_char('{'))).parse(input)?;
    // Bounded list: cap allocation before it happens (DoS guard).
    let (mut remaining, first) = cut(ws(parse_split_slot)).parse(input)?;
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
    // After the "Repeat" keyword is consumed, any argument error is fatal.
    let (input, _) = cut(ws(c_char('('))).parse(input)?;
    let (input, axis) = cut(ws(parse_axis)).parse(input)?;
    let (input, _) = cut(ws(c_char(','))).parse(input)?;
    let (input, tile_size) = cut(ws(finite_float)).parse(input)?;
    let (input, _) = cut(ws(c_char(')'))).parse(input)?;
    let (input, _) = cut(ws(c_char('{'))).parse(input)?;
    let (input, rule) = cut(ws(rule_name)).parse(input)?;
    let (input, _) = cut(ws(c_char('}'))).parse(input)?;
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
    // After the "Comp" keyword is consumed, any argument error is fatal.
    let (input, _) = cut(ws(c_char('('))).parse(input)?;
    let (input, kind) = cut(ws(tag("Faces"))).parse(input)?;
    let _ = kind; // only Faces supported in v0.1
    let (input, _) = cut(ws(c_char(')'))).parse(input)?;
    let (input, _) = cut(ws(c_char('{'))).parse(input)?;
    // Bounded list: cap allocation before it happens (DoS guard).
    let (mut remaining, first) = cut(ws(parse_comp_face_case)).parse(input)?;
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
        cut(delimited(ws(c_char('(')), ws(rule_name), ws(c_char(')')))).parse(input)?;
    Ok((input, ShapeOp::I(mesh_id)))
}

/// Parses `Mat("Brick")` or `Mat(Brick)` — sets material on the current work item.
fn parse_mat(input: &str) -> IResult<&str, ShapeOp> {
    let (input, _) = tag("Mat").parse(input)?;
    let (input, mat_id) =
        cut(delimited(ws(c_char('(')), ws(rule_name), ws(c_char(')')))).parse(input)?;
    Ok((input, ShapeOp::Mat(mat_id)))
}

/// Parses a named world-direction shorthand: `Up`, `Down`, `Right`, `Left`,
/// `Forward`, `Back` (also accepted with a `World.` prefix).
fn parse_align_target(input: &str) -> IResult<&str, Vec3> {
    let (input, _) = opt(terminated(tag("World"), c_char('.'))).parse(input)?;
    let (input, name) = identifier.parse(input)?;
    let v = match name {
        "Up" => Vec3::new(0.0, 1.0, 0.0),
        "Down" => Vec3::new(0.0, -1.0, 0.0),
        "Right" => Vec3::new(1.0, 0.0, 0.0),
        "Left" => Vec3::new(-1.0, 0.0, 0.0),
        "Forward" => Vec3::new(0.0, 0.0, -1.0),
        "Back" => Vec3::new(0.0, 0.0, 1.0),
        _ => return Err(nom::Err::Failure(Error::new(input, ErrorKind::Tag))),
    };
    Ok((input, v))
}

/// `Align(Y, Up)` or `Align(Z, World.Forward)`
fn parse_align(input: &str) -> IResult<&str, ShapeOp> {
    let (input, _) = tag("Align").parse(input)?;
    let (input, _) = cut(ws(c_char('('))).parse(input)?;
    let (input, local_axis) = cut(ws(parse_axis)).parse(input)?;
    let (input, _) = cut(ws(c_char(','))).parse(input)?;
    let (input, target) = cut(ws(parse_align_target)).parse(input)?;
    let (input, _) = cut(ws(c_char(')'))).parse(input)?;
    Ok((input, ShapeOp::Align { local_axis, target }))
}

fn parse_offset_case(input: &str) -> IResult<&str, OffsetCase> {
    let (input, sel_str) = ws(identifier).parse(input)?;
    let selector = OffsetSelector::parse(sel_str)
        .ok_or_else(|| nom::Err::Failure(Error::new(input, ErrorKind::Tag)))?;
    let (input, _) = ws(c_char(':')).parse(input)?;
    let (input, rule) = ws(rule_name).parse(input)?;
    Ok((input, OffsetCase { selector, rule }))
}

/// `Offset(-0.2) { Inside: Glass | Border: Frame }`
fn parse_offset(input: &str) -> IResult<&str, ShapeOp> {
    let (input, _) = tag("Offset").parse(input)?;
    let (input, distance) = cut(delimited(
        ws(c_char('(')),
        ws(finite_float),
        ws(c_char(')')),
    ))
    .parse(input)?;
    let (input, _) = cut(ws(c_char('{'))).parse(input)?;
    let (mut remaining, first) = cut(ws(parse_offset_case)).parse(input)?;
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
        match ws(parse_offset_case).parse(after_sep) {
            Ok((after_item, case)) => {
                cases.push(case);
                remaining = after_item;
            }
            Err(nom::Err::Failure(e)) => return Err(nom::Err::Failure(e)),
            Err(_) => {
                remaining = after_sep;
                break;
            }
        }
    }
    let (remaining, _) = ws(c_char('}')).parse(remaining)?;
    Ok((remaining, ShapeOp::Offset { distance, cases }))
}

fn parse_roof_case(input: &str) -> IResult<&str, RoofCase> {
    let (input, sel_str) = ws(identifier).parse(input)?;
    let selector = RoofFaceSelector::parse(sel_str)
        .ok_or_else(|| nom::Err::Failure(Error::new(input, ErrorKind::Tag)))?;
    let (input, _) = ws(c_char(':')).parse(input)?;
    let (input, rule) = ws(rule_name).parse(input)?;
    Ok((input, RoofCase { selector, rule }))
}

/// `Roof(Gable, 30) { Slope: Tiles | GableEnd: Bricks }`
/// `Roof(Hip, 30, 0.5) { Slope: Tiles }`
fn parse_roof(input: &str) -> IResult<&str, ShapeOp> {
    let (input, _) = tag("Roof").parse(input)?;
    let (input, _) = cut(ws(c_char('('))).parse(input)?;
    let (input, type_str) = cut(ws(identifier)).parse(input)?;
    let roof_type = RoofType::parse(type_str)
        .ok_or_else(|| nom::Err::Failure(Error::new(input, ErrorKind::Tag)))?;
    let (input, _) = cut(ws(c_char(','))).parse(input)?;
    let (input, angle) = cut(ws(finite_float)).parse(input)?;
    if angle <= 0.0 || angle >= 90.0 {
        return Err(nom::Err::Failure(Error::new(input, ErrorKind::Verify)));
    }
    // Optional overhang argument.
    let (input, overhang) = opt(preceded(ws(c_char(',')), ws(finite_float))).parse(input)?;
    let overhang = overhang.unwrap_or(0.0);
    if overhang < 0.0 {
        return Err(nom::Err::Failure(Error::new(input, ErrorKind::Verify)));
    }
    let (input, _) = cut(ws(c_char(')'))).parse(input)?;
    let (input, _) = cut(ws(c_char('{'))).parse(input)?;
    let (mut remaining, first) = cut(ws(parse_roof_case)).parse(input)?;
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
        match ws(parse_roof_case).parse(after_sep) {
            Ok((after_item, case)) => {
                cases.push(case);
                remaining = after_item;
            }
            Err(nom::Err::Failure(e)) => return Err(nom::Err::Failure(e)),
            Err(_) => {
                remaining = after_sep;
                break;
            }
        }
    }
    let (remaining, _) = ws(c_char('}')).parse(remaining)?;
    Ok((
        remaining,
        ShapeOp::Roof {
            roof_type,
            angle,
            overhang,
            cases,
        },
    ))
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
        parse_align,
        parse_offset,
        parse_roof,
        parse_rule_ref,
    ))
    .parse(input)
}

// ── Public API ────────────────────────────────────────────────────────────────

/// Returns true if `op` ends execution of a rule body — either by branching
/// (Split / Comp / Repeat / Offset / Roof) or by producing a terminal (I / Rule).
/// Any operations listed after such an op in the source are unreachable.
fn is_terminating_op(op: &ShapeOp) -> bool {
    matches!(
        op,
        ShapeOp::I(_)
            | ShapeOp::Rule(_)
            | ShapeOp::Split { .. }
            | ShapeOp::Comp(_)
            | ShapeOp::Repeat { .. }
            | ShapeOp::Offset { .. }
            | ShapeOp::Roof { .. }
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
/// Returns at most [`MAX_VARIANTS`] parts; exceeding that limit returns
/// `Err(CapacityOverflow)` before any further allocation occurs.
///
/// Contexts handled:
/// - `"..."` string literals — `|` and `{`/`}` inside are invisible.
/// - `/* ... */` block comments — same.
/// - `// ...` line comments — same (skipped until `\n` or `\r`).
fn split_top_level_pipe(s: &str) -> Result<Vec<&str>, ShapeError> {
    let mut depth: i32 = 0;
    let mut in_string = false;
    let mut in_block_comment = false;
    let mut in_line_comment = false;
    let mut start = 0;
    let mut parts = Vec::new();
    // Use a Peekable iterator so we never allocate the full char list — a
    // Vec<(usize,char)> of the whole input would be O(n) memory (16 bytes per
    // char on 64-bit) even before any safety limits can fire.
    let mut iter = s.char_indices().peekable();

    while let Some((byte_pos, c)) = iter.next() {
        if in_line_comment {
            if c == '\n' || c == '\r' {
                in_line_comment = false;
            }
            continue;
        }

        if in_block_comment {
            if c == '*' && matches!(iter.peek(), Some((_, '/'))) {
                iter.next(); // consume '/'
                in_block_comment = false;
            }
            continue;
        }

        if in_string {
            if c == '"' {
                in_string = false;
            }
            continue;
        }

        // Normal (unquoted, uncommented) context — check for comment openers.
        if c == '/' {
            match iter.peek() {
                Some(&(_, '/')) => {
                    iter.next();
                    in_line_comment = true;
                    continue;
                }
                Some(&(_, '*')) => {
                    iter.next();
                    in_block_comment = true;
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
                // After this push, `parts.len() + 1` parts exist; the loop's
                // final push adds one more, giving `parts.len() + 2` total.
                // Reject before the push when that would exceed MAX_VARIANTS.
                if parts.len() + 2 > MAX_VARIANTS {
                    return Err(ShapeError::CapacityOverflow);
                }
                parts.push(s[start..byte_pos].trim());
                start = byte_pos + c.len_utf8();
            }
            _ => {}
        }
    }

    parts.push(s[start..].trim());
    Ok(parts)
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
    /// Returns an empty slice if the rule has no variants.
    pub fn ops(&self) -> &[ShapeOp] {
        self.variants
            .first()
            .map(|(_, ops)| ops.as_slice())
            .unwrap_or(&[])
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
    let parts = split_top_level_pipe(remaining)?;

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

    // ── Issue 5 (review #10): GrammarRule::ops() safe on empty variants ──────

    #[test]
    fn test_grammar_rule_ops_empty_variants_returns_empty() {
        let rule = GrammarRule {
            name: "Empty".to_string(),
            variants: vec![],
        };
        assert_eq!(rule.ops(), &[] as &[ShapeOp]);
    }

    // ── Issue 2: quaternion overflow bypass ───────────────────────────────────

    #[test]
    fn test_rotate_overflow_components_rejected() {
        // Each component is finite, but squaring overflows to INFINITY.
        // The len_sq check must catch this and reject the quaternion.
        assert!(parse_ops("Rotate(1e160, 0, 0, 0)").is_err());
        assert!(parse_ops("Rotate(1, 1e200, 0, 0)").is_err());
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

    // ── Issue 3 (review #14): MAX_VARIANTS cap in stochastic rule parsing ───────

    #[test]
    fn test_too_many_variants_rejected() {
        // Build a rule body with MAX_VARIANTS + 1 alternatives.
        let variants: Vec<String> = (0..=MAX_VARIANTS).map(|i| format!("1% I(M{i})")).collect();
        let rule_str = format!("R --> {}", variants.join(" | "));
        assert!(matches!(
            parse_rule(&rule_str),
            Err(ShapeError::CapacityOverflow)
        ));
    }

    #[test]
    fn test_max_variants_boundary_accepted() {
        // Exactly MAX_VARIANTS alternatives must succeed.
        let variants: Vec<String> = (0..MAX_VARIANTS).map(|i| format!("1% I(M{i})")).collect();
        let rule_str = format!("R --> {}", variants.join(" | "));
        let rule = parse_rule(&rule_str).unwrap();
        assert_eq!(rule.variants.len(), MAX_VARIANTS);
    }

    // ── Issue 2 (review #14): cut prevents backtracking on malformed op args ───

    #[test]
    fn test_translate_wrong_arg_count_rejected() {
        // Before cut: "Translate" would backtrack to parse_rule_ref, producing a
        // misleading "unreachable operations after terminal" error.
        // After cut: immediately fatal once "Translate" tag is consumed.
        assert!(parse_ops("Translate(1.0, 2.0)").is_err());
    }

    #[test]
    fn test_extrude_missing_arg_rejected() {
        assert!(parse_ops("Extrude()").is_err());
    }

    #[test]
    fn test_split_missing_brace_rejected() {
        // "Split(Y)" without a body should be a fatal parse error, not a rule ref.
        assert!(parse_ops("Split(Y)").is_err());
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

    // ── Feature: Align ───────────────────────────────────────────────────────

    #[test]
    fn test_parse_align_basic() {
        let ops = parse_ops("Align(Y, Up)").unwrap();
        assert_eq!(ops.len(), 1);
        let ShapeOp::Align { local_axis, target } = &ops[0] else {
            panic!("expected Align");
        };
        assert_eq!(*local_axis, Axis::Y);
        assert!((*target - crate::scope::Vec3::new(0.0, 1.0, 0.0)).length() < 1e-9);
    }

    #[test]
    fn test_parse_align_world_prefix() {
        let ops = parse_ops("Align(Z, World.Forward)").unwrap();
        let ShapeOp::Align { local_axis, target } = &ops[0] else {
            panic!("expected Align");
        };
        assert_eq!(*local_axis, Axis::Z);
        assert!((*target - crate::scope::Vec3::new(0.0, 0.0, -1.0)).length() < 1e-9);
    }

    #[test]
    fn test_parse_align_unknown_target_rejected() {
        assert!(parse_ops("Align(Y, Sideways)").is_err());
    }

    // ── Feature: Offset ──────────────────────────────────────────────────────

    #[test]
    fn test_parse_offset_basic() {
        let ops = parse_ops("Offset(-0.2) { Inside: Glass | Border: Frame }").unwrap();
        assert_eq!(ops.len(), 1);
        let ShapeOp::Offset { distance, cases } = &ops[0] else {
            panic!("expected Offset");
        };
        assert!((*distance - (-0.2)).abs() < 1e-9);
        assert_eq!(cases.len(), 2);
        assert_eq!(cases[0].selector, OffsetSelector::Inside);
        assert_eq!(cases[0].rule, "Glass");
        assert_eq!(cases[1].selector, OffsetSelector::Border);
        assert_eq!(cases[1].rule, "Frame");
    }

    #[test]
    fn test_parse_offset_is_terminating() {
        assert!(parse_ops("Offset(-0.1) { Inside: A } Scale(1, 2, 1)").is_err());
    }

    // ── Feature: Roof ────────────────────────────────────────────────────────

    #[test]
    fn test_parse_roof_gable_no_overhang() {
        let ops = parse_ops("Roof(Gable, 30) { Slope: Tiles | GableEnd: Bricks }").unwrap();
        assert_eq!(ops.len(), 1);
        let ShapeOp::Roof {
            roof_type,
            angle,
            overhang,
            cases,
        } = &ops[0]
        else {
            panic!("expected Roof");
        };
        assert_eq!(*roof_type, RoofType::Gable);
        assert!((*angle - 30.0).abs() < 1e-9);
        assert!((*overhang).abs() < 1e-9);
        assert_eq!(cases.len(), 2);
        assert_eq!(cases[0].selector, RoofFaceSelector::Slope);
        assert_eq!(cases[1].selector, RoofFaceSelector::GableEnd);
    }

    #[test]
    fn test_parse_roof_hip_with_overhang() {
        let ops = parse_ops("Roof(Hip, 45, 0.5) { Slope: Tiles }").unwrap();
        let ShapeOp::Roof {
            roof_type,
            angle,
            overhang,
            ..
        } = &ops[0]
        else {
            panic!("expected Roof");
        };
        assert_eq!(*roof_type, RoofType::Hip);
        assert!((*angle - 45.0).abs() < 1e-9);
        assert!((*overhang - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_parse_roof_angle_out_of_range_rejected() {
        assert!(parse_ops("Roof(Gable, 0) { Slope: Tiles }").is_err());
        assert!(parse_ops("Roof(Gable, 90) { Slope: Tiles }").is_err());
    }

    #[test]
    fn test_parse_roof_is_terminating() {
        assert!(parse_ops("Roof(Shed, 30) { Slope: Tiles } Scale(1, 2, 1)").is_err());
    }
}

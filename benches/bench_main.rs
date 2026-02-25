use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};
use symbios_shape::grammar::parse_ops;
use symbios_shape::ops::{Axis, ShapeOp, SplitSize, SplitSlot};
use symbios_shape::{Interpreter, Quat, Scope, Vec3};

fn make_slot(size: SplitSize, rule: &str) -> SplitSlot {
    SplitSlot {
        size,
        rule: rule.to_string(),
    }
}

fn bench_simple_building(c: &mut Criterion) {
    let mut group = c.benchmark_group("Derivation");

    // 10-floor building: Ground + 8 Upper floors + Roof
    // Each floor is split into 5 window bays via Repeat.
    group.bench_function("10-floor building with facade repeat", |b| {
        b.iter(|| {
            let mut interp = Interpreter::new();

            interp.add_rule(
                "Lot",
                vec![ShapeOp::Split {
                    axis: Axis::Y,
                    slots: vec![
                        make_slot(SplitSize::Absolute(3.0), "Ground"),
                        make_slot(SplitSize::Floating(1.0), "Upper"),
                        make_slot(SplitSize::Absolute(2.0), "Roof"),
                    ],
                }],
            );
            interp.add_rule(
                "Ground",
                vec![ShapeOp::Repeat {
                    axis: Axis::X,
                    tile_size: 2.0,
                    rule: "ShopWindow".to_string(),
                }],
            );
            interp.add_rule(
                "Upper",
                vec![ShapeOp::Repeat {
                    axis: Axis::X,
                    tile_size: 2.0,
                    rule: "Window".to_string(),
                }],
            );

            let footprint = Scope::new(Vec3::ZERO, Quat::IDENTITY, Vec3::new(20.0, 30.0, 15.0));
            interp.derive(footprint, black_box("Lot")).unwrap()
        })
    });
    group.finish();
}

fn bench_deep_split(c: &mut Criterion) {
    let mut group = c.benchmark_group("Deep Split");

    // Stress test: recursive binary split creating a complete binary tree of depth 12
    group.bench_function("Binary split depth 12 (4096 terminals)", |b| {
        b.iter(|| {
            let mut interp = Interpreter::new();

            // Build rules: Split0..Split12, leaf = I("Tile")
            for d in 0..12_usize {
                let next = if d + 1 < 12 {
                    format!("Split{}", d + 1)
                } else {
                    "Tile".to_string()
                };
                interp.add_rule(
                    format!("Split{}", d),
                    vec![ShapeOp::Split {
                        axis: Axis::X,
                        slots: vec![
                            make_slot(SplitSize::Floating(1.0), &next),
                            make_slot(SplitSize::Floating(1.0), &next),
                        ],
                    }],
                );
            }

            let scope = Scope::new(Vec3::ZERO, Quat::IDENTITY, Vec3::new(4096.0, 1.0, 1.0));
            interp.derive(scope, black_box("Split0")).unwrap()
        })
    });
    group.finish();
}

fn bench_parse_ops(c: &mut Criterion) {
    let mut group = c.benchmark_group("Parser");

    let input = r#"
        Extrude(10)
        Split(Y) { 3.0: Ground | ~1.0: Upper | ~1.0: Upper | 2.0: Roof }
        Comp(Faces) { Top: Roof | Side: Facade | Bottom: Base }
        Repeat(X, 2.0) { Window }
    "#;

    group.bench_function("parse_ops (complex rule body)", |b| {
        b.iter(|| parse_ops(black_box(input)))
    });
    group.finish();
}

criterion_group!(
    benches,
    bench_simple_building,
    bench_deep_split,
    bench_parse_ops
);
criterion_main!(benches);

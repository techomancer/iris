#![cfg(feature = "rex-dirty-upload")]

#[path = "../src/dev/ng1/rex3_dirty.rs"]
mod rex3_dirty;

use rex3_dirty::{
    copy_u32_rects, plan_framebuffer_uploads, DamagePlane, DamageRect, DamageSnapshot,
    DamageTracker, FB_HEIGHT, FB_WIDTH, PLANE_BYTES, TILE_COLUMNS, TILE_HEIGHT, TILE_ROWS,
    TILE_WIDTH,
};

#[test]
fn clipped_rectangles_reject_empty_and_bound_the_plane() {
    assert_eq!(
        DamageRect::clipped(-20, -10, 70, 40),
        Some(DamageRect {
            x: 0,
            y: 0,
            width: 70,
            height: 40,
        })
    );
    assert_eq!(
        DamageRect::clipped(2000, 1000, 4000, 2000),
        Some(DamageRect {
            x: 2000,
            y: 1000,
            width: 48,
            height: 24,
        })
    );
    assert_eq!(DamageRect::clipped(-10, 0, 0, 20), None);
    assert_eq!(DamageRect::clipped(0, FB_HEIGHT as i64, 20, 2000), None);
}

#[test]
fn tracker_takes_planes_independently_and_clears_marks() {
    let tracker = DamageTracker::default();
    tracker.mark_rect(DamagePlane::Rgb, DamageRect::clipped(1, 1, 2, 2).unwrap());
    tracker.mark_rect(
        DamagePlane::Aux,
        DamageRect::clipped(80, 40, 81, 41).unwrap(),
    );

    assert!(tracker.has_damage());
    let first = tracker.take();
    assert_eq!(first.tile_count(DamagePlane::Rgb), 1);
    assert_eq!(first.tile_count(DamagePlane::Aux), 1);
    assert!(!tracker.has_damage());
    assert!(tracker.take().is_empty());

    tracker.mark_rect(
        DamagePlane::Rgb,
        DamageRect::clipped(100, 100, 101, 101).unwrap(),
    );
    let second = tracker.take();
    assert_eq!(second.tile_count(DamagePlane::Rgb), 1);
    assert!(second.plane_is_empty(DamagePlane::Aux));
}

#[test]
fn rectangles_coalesce_across_tile_rows() {
    let mut damage = DamageSnapshot::default();
    damage.mark_rect(
        DamagePlane::Rgb,
        DamageRect::clipped(10, 10, 130, 70).unwrap(),
    );
    let mut rectangles = Vec::new();
    damage.rectangles_into(DamagePlane::Rgb, &mut rectangles);
    assert_eq!(damage.tile_count(DamagePlane::Rgb), 9);
    assert_eq!(
        rectangles,
        vec![DamageRect {
            x: 0,
            y: 0,
            width: 3 * TILE_WIDTH,
            height: 3 * TILE_HEIGHT,
        }]
    );
}

#[test]
fn disjoint_runs_remain_disjoint() {
    let mut damage = DamageSnapshot::default();
    damage.mark_rect(DamagePlane::Aux, DamageRect::clipped(0, 0, 10, 64).unwrap());
    damage.mark_rect(
        DamagePlane::Aux,
        DamageRect::clipped(2 * TILE_WIDTH as i64, 0, 2 * TILE_WIDTH as i64 + 10, 64).unwrap(),
    );
    let mut rectangles = Vec::new();
    damage.rectangles_into(DamagePlane::Aux, &mut rectangles);
    assert_eq!(
        rectangles,
        vec![
            DamageRect {
                x: 0,
                y: 0,
                width: TILE_WIDTH,
                height: 2 * TILE_HEIGHT,
            },
            DamageRect {
                x: 2 * TILE_WIDTH,
                y: 0,
                width: TILE_WIDTH,
                height: 2 * TILE_HEIGHT,
            },
        ]
    );
}

#[test]
fn full_plane_is_one_rectangle_with_exact_bytes() {
    let mut damage = DamageSnapshot::default();
    damage.mark_full(DamagePlane::Rgb);
    let mut rectangles = Vec::new();
    damage.rectangles_into(DamagePlane::Rgb, &mut rectangles);
    assert_eq!(
        damage.tile_count(DamagePlane::Rgb),
        TILE_COLUMNS * TILE_ROWS
    );
    assert_eq!(rectangles, vec![DamageRect::full()]);
    assert_eq!(rectangles[0].bytes(), PLANE_BYTES);
}

#[test]
fn full_tracker_starts_with_both_planes_marked() {
    let damage = DamageTracker::new_full().take();
    assert_eq!(
        damage.tile_count(DamagePlane::Rgb),
        TILE_COLUMNS * TILE_ROWS
    );
    assert_eq!(
        damage.tile_count(DamagePlane::Aux),
        TILE_COLUMNS * TILE_ROWS
    );
}

#[test]
fn published_snapshot_unions_with_new_marks() {
    let mut pending = DamageSnapshot::default();
    pending.mark_rect(DamagePlane::Rgb, DamageRect::clipped(0, 0, 1, 1).unwrap());
    let tracker = DamageTracker::default();
    tracker.publish(&pending);
    tracker.mark_rect(
        DamagePlane::Rgb,
        DamageRect::clipped(TILE_WIDTH as i64, 0, TILE_WIDTH as i64 + 1, 1).unwrap(),
    );
    assert_eq!(tracker.take().tile_count(DamagePlane::Rgb), 2);
}

#[test]
fn partial_copy_matches_full_reference_in_dirty_regions() {
    let len = FB_WIDTH * FB_HEIGHT;
    let source: Vec<u32> = (0..len as u32).collect();
    let mut destination = vec![0xdead_beef; len];
    let untouched = destination.clone();
    let rectangles = [
        DamageRect::clipped(64, 32, 192, 96).unwrap(),
        DamageRect::clipped(512, 200, 576, 232).unwrap(),
    ];
    let bytes = copy_u32_rects(&mut destination, &source, &rectangles);
    assert_eq!(
        bytes,
        rectangles.iter().map(|rect| rect.bytes()).sum::<usize>()
    );

    for y in 0..FB_HEIGHT {
        for x in 0..FB_WIDTH {
            let copied = rectangles.iter().any(|rect| {
                x >= rect.x && x < rect.x + rect.width && y >= rect.y && y < rect.y + rect.height
            });
            let index = y * FB_WIDTH + x;
            assert_eq!(
                destination[index],
                if copied {
                    source[index]
                } else {
                    untouched[index]
                }
            );
        }
    }
}

#[test]
fn snapshots_merge_without_crossing_planes() {
    let mut first = DamageSnapshot::default();
    first.mark_rect(DamagePlane::Rgb, DamageRect::clipped(0, 0, 1, 1).unwrap());
    let mut second = DamageSnapshot::default();
    second.mark_rect(DamagePlane::Aux, DamageRect::clipped(0, 0, 1, 1).unwrap());
    first.merge(&second);
    assert_eq!(first.tile_count(DamagePlane::Rgb), 1);
    assert_eq!(first.tile_count(DamagePlane::Aux), 1);
}

#[test]
fn upload_planning_keeps_planes_independent_after_initialization() {
    let rgb = [DamageRect::clipped(64, 32, 128, 64).unwrap()];
    let aux = [DamageRect::clipped(256, 96, 320, 128).unwrap()];

    let (rgb_plan, aux_plan) = plan_framebuffer_uploads(false, false, &rgb, &[]);
    assert_eq!(rgb_plan, &[DamageRect::full()]);
    assert_eq!(aux_plan, &[DamageRect::full()]);

    let (rgb_plan, aux_plan) = plan_framebuffer_uploads(true, false, &rgb, &[]);
    assert_eq!(rgb_plan, &rgb);
    assert!(aux_plan.is_empty());

    let (rgb_plan, aux_plan) = plan_framebuffer_uploads(true, false, &[], &aux);
    assert!(rgb_plan.is_empty());
    assert_eq!(aux_plan, &aux);

    let (rgb_plan, aux_plan) = plan_framebuffer_uploads(true, true, &rgb, &[]);
    assert_eq!(rgb_plan, &[DamageRect::full()]);
    assert_eq!(aux_plan, &[DamageRect::full()]);
}

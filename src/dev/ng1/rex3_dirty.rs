use std::sync::atomic::{AtomicU32, Ordering};

pub(crate) const FB_WIDTH: usize = 2048;
pub(crate) const FB_HEIGHT: usize = 1024;
pub(crate) const TILE_WIDTH: usize = 64;
pub(crate) const TILE_HEIGHT: usize = 32;
pub(crate) const TILE_COLUMNS: usize = FB_WIDTH / TILE_WIDTH;
pub(crate) const TILE_ROWS: usize = FB_HEIGHT / TILE_HEIGHT;
pub(crate) const PLANE_BYTES: usize = FB_WIDTH * FB_HEIGHT * std::mem::size_of::<u32>();
const FULL_DAMAGE: [DamageRect; 1] = [DamageRect::full()];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum DamagePlane {
    Rgb,
    Aux,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct DamageRect {
    pub x: usize,
    pub y: usize,
    pub width: usize,
    pub height: usize,
}

impl DamageRect {
    pub(crate) const fn full() -> Self {
        Self {
            x: 0,
            y: 0,
            width: FB_WIDTH,
            height: FB_HEIGHT,
        }
    }

    pub(crate) fn clipped(x0: i64, y0: i64, x1: i64, y1: i64) -> Option<Self> {
        let x0 = x0.clamp(0, FB_WIDTH as i64);
        let y0 = y0.clamp(0, FB_HEIGHT as i64);
        let x1 = x1.clamp(0, FB_WIDTH as i64);
        let y1 = y1.clamp(0, FB_HEIGHT as i64);
        (x0 < x1 && y0 < y1).then_some(Self {
            x: x0 as usize,
            y: y0 as usize,
            width: (x1 - x0) as usize,
            height: (y1 - y0) as usize,
        })
    }

    pub(crate) const fn bytes(self) -> usize {
        self.width * self.height * std::mem::size_of::<u32>()
    }
}

pub(crate) fn plan_framebuffer_uploads<'a>(
    textures_initialized: bool,
    force_full: bool,
    rgb_damage: &'a [DamageRect],
    aux_damage: &'a [DamageRect],
) -> (&'a [DamageRect], &'a [DamageRect]) {
    if !textures_initialized || force_full {
        (&FULL_DAMAGE, &FULL_DAMAGE)
    } else {
        (rgb_damage, aux_damage)
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct DamageSnapshot {
    rgb: [u32; TILE_ROWS],
    aux: [u32; TILE_ROWS],
}

impl DamageSnapshot {
    pub(crate) fn is_empty(&self) -> bool {
        self.rgb.iter().chain(self.aux.iter()).all(|&row| row == 0)
    }

    pub(crate) fn plane_is_empty(&self, plane: DamagePlane) -> bool {
        self.rows(plane).iter().all(|&row| row == 0)
    }

    pub(crate) fn tile_count(&self, plane: DamagePlane) -> usize {
        self.rows(plane)
            .iter()
            .map(|row| row.count_ones() as usize)
            .sum()
    }

    pub(crate) fn mark_rect(&mut self, plane: DamagePlane, rect: DamageRect) {
        let first_col = rect.x / TILE_WIDTH;
        let last_col = (rect.x + rect.width - 1) / TILE_WIDTH;
        let first_row = rect.y / TILE_HEIGHT;
        let last_row = (rect.y + rect.height - 1) / TILE_HEIGHT;
        let mask = range_mask(first_col, last_col);
        for row in &mut self.rows_mut(plane)[first_row..=last_row] {
            *row |= mask;
        }
    }

    pub(crate) fn mark_full(&mut self, plane: DamagePlane) {
        self.rows_mut(plane).fill(u32::MAX);
    }

    pub(crate) fn merge(&mut self, other: &Self) {
        for plane in [DamagePlane::Rgb, DamagePlane::Aux] {
            for (dst, src) in self.rows_mut(plane).iter_mut().zip(other.rows(plane)) {
                *dst |= *src;
            }
        }
    }

    pub(crate) fn rectangles_into(&self, plane: DamagePlane, rectangles: &mut Vec<DamageRect>) {
        rectangles.clear();
        let mut active = [(0usize, 0usize, 0usize); TILE_COLUMNS / 2];
        let mut active_len = 0usize;

        for (tile_y, &row_mask) in self.rows(plane).iter().enumerate() {
            let y = tile_y * TILE_HEIGHT;
            let mut next = [(0usize, 0usize, 0usize); TILE_COLUMNS / 2];
            let mut next_len = 0usize;
            let mut mask = row_mask;

            while mask != 0 {
                let first_col = mask.trailing_zeros() as usize;
                let shifted = mask >> first_col;
                let run_len = shifted.trailing_ones() as usize;
                let last_col = first_col + run_len - 1;
                let run_mask = range_mask(first_col, last_col);
                mask &= !run_mask;

                let x = first_col * TILE_WIDTH;
                let width = run_len * TILE_WIDTH;
                let existing = active[..active_len]
                    .iter()
                    .find(|&&(active_x, active_width, _)| active_x == x && active_width == width)
                    .map(|&(_, _, index)| index);
                let index = if let Some(index) = existing {
                    rectangles[index].height += TILE_HEIGHT;
                    index
                } else {
                    rectangles.push(DamageRect {
                        x,
                        y,
                        width,
                        height: TILE_HEIGHT,
                    });
                    rectangles.len() - 1
                };
                next[next_len] = (x, width, index);
                next_len += 1;
            }

            active = next;
            active_len = next_len;
        }
    }

    fn rows(&self, plane: DamagePlane) -> &[u32; TILE_ROWS] {
        match plane {
            DamagePlane::Rgb => &self.rgb,
            DamagePlane::Aux => &self.aux,
        }
    }

    fn rows_mut(&mut self, plane: DamagePlane) -> &mut [u32; TILE_ROWS] {
        match plane {
            DamagePlane::Rgb => &mut self.rgb,
            DamagePlane::Aux => &mut self.aux,
        }
    }
}

pub(crate) struct DamageTracker {
    rgb: [AtomicU32; TILE_ROWS],
    aux: [AtomicU32; TILE_ROWS],
}

impl Default for DamageTracker {
    fn default() -> Self {
        Self {
            rgb: std::array::from_fn(|_| AtomicU32::new(0)),
            aux: std::array::from_fn(|_| AtomicU32::new(0)),
        }
    }
}

impl DamageTracker {
    pub(crate) fn new_full() -> Self {
        let tracker = Self::default();
        tracker.mark_full(DamagePlane::Rgb);
        tracker.mark_full(DamagePlane::Aux);
        tracker
    }

    pub(crate) fn has_damage(&self) -> bool {
        self.rgb
            .iter()
            .chain(self.aux.iter())
            .any(|row| row.load(Ordering::Acquire) != 0)
    }

    pub(crate) fn mark_rect(&self, plane: DamagePlane, rect: DamageRect) {
        let first_col = rect.x / TILE_WIDTH;
        let last_col = (rect.x + rect.width - 1) / TILE_WIDTH;
        let first_row = rect.y / TILE_HEIGHT;
        let last_row = (rect.y + rect.height - 1) / TILE_HEIGHT;
        let mask = range_mask(first_col, last_col);
        for row in &self.rows(plane)[first_row..=last_row] {
            row.fetch_or(mask, Ordering::Release);
        }
    }

    pub(crate) fn mark_full(&self, plane: DamagePlane) {
        for row in self.rows(plane) {
            row.fetch_or(u32::MAX, Ordering::Release);
        }
    }

    pub(crate) fn publish(&self, snapshot: &DamageSnapshot) {
        for plane in [DamagePlane::Rgb, DamagePlane::Aux] {
            for (dst, &src) in self.rows(plane).iter().zip(snapshot.rows(plane)) {
                if src != 0 {
                    dst.fetch_or(src, Ordering::Release);
                }
            }
        }
    }

    pub(crate) fn take(&self) -> DamageSnapshot {
        DamageSnapshot {
            rgb: std::array::from_fn(|i| self.rgb[i].swap(0, Ordering::Acquire)),
            aux: std::array::from_fn(|i| self.aux[i].swap(0, Ordering::Acquire)),
        }
    }

    fn rows(&self, plane: DamagePlane) -> &[AtomicU32; TILE_ROWS] {
        match plane {
            DamagePlane::Rgb => &self.rgb,
            DamagePlane::Aux => &self.aux,
        }
    }
}

pub(crate) fn copy_u32_rects(
    destination: &mut [u32],
    source: &[u32],
    rectangles: &[DamageRect],
) -> usize {
    debug_assert!(destination.len() >= FB_WIDTH * FB_HEIGHT);
    debug_assert!(source.len() >= FB_WIDTH * FB_HEIGHT);
    let mut bytes = 0usize;
    for rect in rectangles {
        for y in rect.y..rect.y + rect.height {
            let start = y * FB_WIDTH + rect.x;
            let end = start + rect.width;
            destination[start..end].copy_from_slice(&source[start..end]);
        }
        bytes += rect.bytes();
    }
    bytes
}

fn range_mask(first: usize, last: usize) -> u32 {
    debug_assert!(first <= last);
    debug_assert!(last < TILE_COLUMNS);
    (u32::MAX << first) & (u32::MAX >> (TILE_COLUMNS - 1 - last))
}

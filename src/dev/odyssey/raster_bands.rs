//! Drawing a primitive in bands of rows on a pool of host threads: the
//! band pool, the split of a draw into bands, and the switch that turns it
//! on. Each band repeats the primitive's setup and keeps only its rows.
//!
//! Ported from hw/display/sgi/rasterizer.c in jamesbraid/qemu-sgi-origami.

use super::*;

pub const SGI_RASTER_PARALLEL_MAX_BANDS: u32 = 16;

pub const SGI_RASTER_PARALLEL_DEFAULT_BANDS: u32 = 4;

pub const SGI_RASTER_PARALLEL_MIN_PIXELS: u64 = 2048;

pub const SGI_SOFT_BAND_ROWS_SHIFT: u32 = 3;

pub static SOFT_BANDS_WANTED: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);

pub static SOFT_BANDS_MIN_PIXELS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(SGI_RASTER_PARALLEL_MIN_PIXELS);

/// The host's processors, read once (soft_pool_init()): every draw and
/// clear chooses its bands. A later change to the host's processors is not
/// seen.
pub static SOFT_HOST_PROCESSORS: std::sync::OnceLock<u32> = std::sync::OnceLock::new();

pub type SoftBandFn = fn(Opaque, u32, u32);

/// The pool's shared state, under SOFT_POOL.lock. The job's opaque
/// pointer is a P, which is Send: a band's data is the caller's, which the
/// caller keeps alive until every band is done.
#[derive(Default)]
pub struct SoftPoolState {
    pub threads: u32,
    pub generation: u64,
    pub fn_: Option<SoftBandFn>,
    pub opaque: Opaque,
    pub bands: u32,
    pub pending: u32,
    /// A band panicked: the caller, once all are done, panics too.
    pub panicked: bool,
}

pub struct SoftPool {
    /// one band-parallel call at a time
    pub run: std::sync::Mutex<()>,
    pub lock: std::sync::Mutex<SoftPoolState>,
    pub start: std::sync::Condvar,
    pub done: std::sync::Condvar,
}

pub static SOFT_POOL: SoftPool = SoftPool {
    run: std::sync::Mutex::new(()),
    lock: std::sync::Mutex::new(SoftPoolState {
        threads: 0,
        generation: 0,
        fn_: None,
        opaque: P(std::ptr::null_mut()),
        bands: 0,
        pending: 0,
        panicked: false,
    }),
    start: std::sync::Condvar::new(),
    done: std::sync::Condvar::new(),
};

/// Count the host's processors, once, on the first draw or clear that
/// chooses its bands. Workers start only when a draw first runs in bands.
pub fn soft_pool_init() {
    SOFT_HOST_PROCESSORS.get_or_init(|| {
        std::cmp::max(std::thread::available_parallelism().map_or(1, |n| n.get() as u32), 1)
    });
}

/// A band worker: waits for each generation of the pool's job, runs its
/// band of it if it has one, and counts itself done.
pub fn soft_band_worker(band: u32) {
    let mut seen: u64 = 0;
    let mut g = SOFT_POOL.lock.lock().unwrap_or_else(|e| e.into_inner());

    loop {
        while g.generation == seen {
            g = SOFT_POOL.start.wait(g).unwrap_or_else(|e| e.into_inner());
        }
        seen = g.generation;
        if band >= g.bands {
            continue;
        }
        let fn_ = g.fn_.unwrap();
        let job = g.opaque;
        let bands = g.bands;
        drop(g);
        let ok = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| fn_(job, band, bands))).is_ok();
        g = SOFT_POOL.lock.lock().unwrap_or_else(|e| e.into_inner());
        if !ok {
            g.panicked = true;
        }
        g.pending -= 1;
        if g.pending == 0 {
            SOFT_POOL.done.notify_one();
        }
    }
}

pub fn soft_bands() -> u32 {
    soft_pool_init();
    let mut bands = SOFT_BANDS_WANTED.load(std::sync::atomic::Ordering::Relaxed);

    if bands == 0 {
        bands = std::cmp::min(SGI_RASTER_PARALLEL_DEFAULT_BANDS, *SOFT_HOST_PROCESSORS.get().unwrap());
    }
    std::cmp::min(bands, SGI_RASTER_PARALLEL_MAX_BANDS)
}

/// fn(opaque, band, bands) for each band, band 0 on this thread, the rest
/// on workers, returning when all have; false, having run none, when
/// another thread is using the workers.
pub fn soft_bands_run(f: SoftBandFn, opaque: Opaque, bands: u32) -> bool {
    soft_pool_init();
    let _run = match SOFT_POOL.run.try_lock() {
        Ok(g) => g,
        Err(std::sync::TryLockError::Poisoned(e)) => e.into_inner(),
        Err(std::sync::TryLockError::WouldBlock) => return false,
    };
    let mut g = SOFT_POOL.lock.lock().unwrap_or_else(|e| e.into_inner());
    while g.threads < bands - 1 {
        let n = g.threads + 1;

        let spawned = std::thread::Builder::new()
            .name(format!("sgi-raster-{}", n))
            .spawn(move || soft_band_worker(n));
        if spawned.is_err() {
            // No thread to run that band: draw on fewer threads is not an
            // option once the bands are chosen, so run none.
            return false;
        }
        g.threads += 1;
    }
    g.fn_ = Some(f);
    g.opaque = opaque;
    g.bands = bands;
    g.pending = bands - 1;
    g.panicked = false;
    g.generation += 1;
    SOFT_POOL.start.notify_all();
    drop(g);

    let own = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| f(opaque, 0, bands)));

    let mut g = SOFT_POOL.lock.lock().unwrap_or_else(|e| e.into_inner());
    while g.pending != 0 {
        g = SOFT_POOL.done.wait(g).unwrap_or_else(|e| e.into_inner());
    }
    let panicked = g.panicked;
    drop(g);
    drop(_run);
    if let Err(e) = own {
        std::panic::resume_unwind(e);
    }
    if panicked {
        panic!("sgi-raster: a band panicked");
    }
    true
}

impl RasterTarget {
    /// The pixel count from which a draw or clear into @t runs in bands.
    pub fn soft_band_min_pixels(&self) -> u64 {
        if self.band_min_pixels != 0 {
            self.band_min_pixels as u64
        } else {
            SOFT_BANDS_MIN_PIXELS.load(std::sync::atomic::Ordering::Relaxed)
        }
    }
}

impl SoftDraw {
    pub fn row_in_band(&self, py: i32) -> bool {
        self.bands <= 1 || ((py as u32) >> SGI_SOFT_BAND_ROWS_SHIFT) % self.bands == self.band
    }
}

pub fn raster_set_parallel(bands: u32, min_pixels: u64) {
    SOFT_BANDS_WANTED.store(bands, std::sync::atomic::Ordering::Relaxed);
    SOFT_BANDS_MIN_PIXELS.store(min_pixels, std::sync::atomic::Ordering::Relaxed);
}

/// A draw handed to the band pool: pointers to the caller's draw, which
/// stays alive until every band has returned.
pub struct SoftBandDraw {
    pub r: P<SoftDraw>,
    pub d: P<GlDraw>,
}

pub fn soft_band_draw(opaque: Opaque, band: u32, bands: u32) {
    let job: &SoftBandDraw = from_opaque(opaque);
    let mut r: SoftDraw = *job.r;

    r.band = band;
    r.bands = bands;
    if !r.texture.is_null() {
        r.texels = r.texture.soft_texel_cache_begin();
    }
    if !r.fixed.is_null() && !r.fixed.prefilled && !r.fixed_begin() {
        r.fixed = P::null();
    }
    SOFT_BAND_QUIET.with(|q| q.set(band > 0));
    r.draw_triangles(&job.d);
    SOFT_BAND_QUIET.with(|q| q.set(false));
}

impl SoftDraw {
    /// Draw the filled primitives in bands when the target allows it and they
    /// cover enough pixels to pay for repeating each primitive's setup in
    /// every band; false, having drawn nothing, otherwise.
    pub fn draw_parallel(&self, d: &GlDraw) -> bool {
        let bands = soft_bands();
        let min_pixels = self.t.soft_band_min_pixels();
        let mut job = SoftBandDraw { r: P::from_ref(self), d: P::from_ref(d) };

        if !self.t.concurrent_pixels || bands < 2 || self.draw_pixels(d, min_pixels) < min_pixels {
            return false;
        }
        soft_bands_run(soft_band_draw, P::new(&mut job).opaque(), bands)
    }
}

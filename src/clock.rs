//! Time, randomness and time zone behind one door, so the same code runs natively and in a browser.
//!
//! Native builds use `std`. A WebAssembly build (`wasm32`) cannot: `SystemTime::now()` and `Instant::now()` panic
//! there. Instead the page hands in four functions through the module's `env` imports (see `web/tinybot.js`):
//!
//! | import                 | the page provides        |
//! |------------------------|--------------------------|
//! | `host_now_ms`          | `Date.now()`             |
//! | `host_perf_ms`         | `performance.now()`      |
//! | `host_random`          | `Math.random()`          |
//! | `host_tz_offset_min`   | `-new Date().getTimezoneOffset()` |
//!
//! `--features host-clock` switches a native build to the same imports, which is how the tests exercise this
//! path without a browser (they define the four functions themselves).

#[cfg(any(target_arch = "wasm32", feature = "host-clock"))]
mod host {
    unsafe extern "C" {
        pub safe fn host_now_ms() -> f64;
        pub safe fn host_perf_ms() -> f64;
        pub safe fn host_random() -> f64;
        pub safe fn host_tz_offset_min() -> f64;
    }
}

/// Milliseconds since the Unix epoch (wall clock).
#[cfg(not(any(target_arch = "wasm32", feature = "host-clock")))]
pub fn unix_ms() -> i64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_millis() as i64).unwrap_or(0)
}

#[cfg(any(target_arch = "wasm32", feature = "host-clock"))]
pub fn unix_ms() -> i64 {
    host::host_now_ms() as i64
}

/// Microseconds on a monotonic clock. Only differences mean anything.
#[cfg(not(any(target_arch = "wasm32", feature = "host-clock")))]
pub fn mono_us() -> u64 {
    use std::sync::OnceLock;
    use std::time::Instant;
    static START: OnceLock<Instant> = OnceLock::new();
    START.get_or_init(Instant::now).elapsed().as_micros() as u64
}

#[cfg(any(target_arch = "wasm32", feature = "host-clock"))]
pub fn mono_us() -> u64 {
    (host::host_perf_ms() * 1000.0).max(0.0) as u64
}

/// A fresh 64-bit seed.
#[cfg(not(any(target_arch = "wasm32", feature = "host-clock")))]
pub fn seed() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_nanos() as u64).unwrap_or(1)
}

#[cfg(any(target_arch = "wasm32", feature = "host-clock"))]
pub fn seed() -> u64 {
    // two 53-bit draws cover all 64 bits; the clock keeps two loads in the same millisecond apart
    let a = (host::host_random() * 9_007_199_254_740_992.0) as u64;
    let b = (host::host_random() * 9_007_199_254_740_992.0) as u64;
    a ^ b.rotate_left(32) ^ (host::host_now_ms() as u64).rotate_left(11)
}

/// Minutes east of UTC according to the host, when the host can tell us (the browser can, a bare OS call can't).
#[cfg(not(any(target_arch = "wasm32", feature = "host-clock")))]
pub fn host_tz_offset_min() -> Option<i32> {
    None
}

#[cfg(any(target_arch = "wasm32", feature = "host-clock"))]
pub fn host_tz_offset_min() -> Option<i32> {
    let m = host::host_tz_offset_min();
    (m.is_finite() && (-720.0..=840.0).contains(&m)).then_some(m as i32)
}

/// Stand-ins for the four functions the page provides, so `cargo test --lib --features host-clock` can run the
/// browser code path (the same `extern "C"` calls the wasm build makes) without a browser. `--lib` matters:
/// the `tinybot` binary does not define these, so linking it with the feature on fails by design.
#[cfg(all(test, feature = "host-clock"))]
mod fake_host {
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::{Instant, SystemTime, UNIX_EPOCH};

    #[unsafe(no_mangle)]
    pub extern "C" fn host_now_ms() -> f64 {
        SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_millis() as f64).unwrap_or(0.0)
    }
    #[unsafe(no_mangle)]
    pub extern "C" fn host_perf_ms() -> f64 {
        static START: std::sync::OnceLock<Instant> = std::sync::OnceLock::new();
        START.get_or_init(Instant::now).elapsed().as_secs_f64() * 1000.0
    }
    #[unsafe(no_mangle)]
    pub extern "C" fn host_random() -> f64 {
        static S: AtomicU64 = AtomicU64::new(0x9E37_79B9_7F4A_7C15);
        let x = S.fetch_add(0x9E37_79B9_7F4A_7C15, Ordering::Relaxed);
        let z = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        ((z ^ (z >> 27)) >> 11) as f64 / (1u64 << 53) as f64
    }
    #[unsafe(no_mangle)]
    pub extern "C" fn host_tz_offset_min() -> f64 {
        330.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clocks_behave() {
        let a = unix_ms();
        assert!(a > 1_700_000_000_000, "{a}");
        let (m1, m2) = (mono_us(), mono_us());
        assert!(m2 >= m1);
        assert_ne!(seed(), seed());
        #[cfg(feature = "host-clock")]
        assert_eq!(host_tz_offset_min(), Some(330));
        #[cfg(not(feature = "host-clock"))]
        assert_eq!(host_tz_offset_min(), None);
    }
}

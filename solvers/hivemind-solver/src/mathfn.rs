// Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
//! The ONLY place where this crate calls a libm-style function (`sin`, `cos`, `sqrt`).
//!
//! The science families (pendulum, projectile with drag, Kepler) need these functions in their
//! right-hand sides. `sqrt` is correctly rounded by IEEE-754 and therefore identical on every
//! conforming CPU; `sin` and `cos` are NOT correctly rounded: they go to the system libm, whose
//! last-bit results can differ between libm versions. Routing every call through this module means
//! that a future switch to a pinned pure-Rust libm (e.g. the `libm` crate) is a change to these
//! three function bodies and nothing else.

/// Sine (system libm today).
#[inline]
pub fn sin(x: f64) -> f64 {
    x.sin()
}

/// Cosine (system libm today).
#[inline]
pub fn cos(x: f64) -> f64 {
    x.cos()
}

/// Square root (correctly rounded by IEEE-754).
#[inline]
pub fn sqrt(x: f64) -> f64 {
    x.sqrt()
}

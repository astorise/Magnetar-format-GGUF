#![no_main]

use libfuzzer_sys::fuzz_target;

// Offline/periodic fuzzing target (not run in per-PR CI: cargo-fuzz requires
// a nightly toolchain). Exercises the same public entry point
// `tests/corpus`'s regression suite replays. Any crash or hang this finds
// should have its input byte sequence checked into `tests/corpus/` as a new
// `.bin` file, growing that suite's permanent, stable-toolchain-runnable
// coverage.
fuzz_target!(|data: &[u8]| {
    let _ = magnetar_format_gguf::parse(data);
});

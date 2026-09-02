# magnetar-format-gguf

## Purpose

A GGUF Model Artifact parser for the
[Magnetar](https://github.com/astorise/Magnetar) local AI Runtime. It reads
GGUF-encoded weight files and produces Magnetar's generic, Provider-agnostic
types (`ModelTensorMetadata`, `ModelDType`, `ModelQuantization`/
`ModelQuantizationFormat`) at the boundary into `magnetar-runtime` --
`magnetar-runtime` SHALL have zero dependency on this crate or any
GGUF-specific type describing tensor data, per Magnetar's architectural
invariant that Providers and Format parsers own native/format-specific
implementation details while the Core stays generic. GGUF support does not
introduce a `GGUFProvider`.

## Status

**Real parser, not a template.** `parse(bytes: &[u8]) -> Result<GgufArtifact, GgufError>`
reads the full GGUF container structure -- magic, version, the typed
key-value metadata section (including recursively-nested arrays, depth
limited against adversarial nesting), and the tensor-info section -- and
normalizes tensors into `ModelTensorMetadata`. Structural details (header
layout, metadata value type enum, string/array encoding, alignment default
and rule) and quantized block layouts (`block_q4_K`, `block_q5_K`,
`block_q8_0` byte sizes and elements-per-block) were verified against the
upstream `ggml-org/ggml` specification and source, not recalled from memory
alone.

**Quantization scope, by design, not oversight:** only `ggml_type` values
with an existing `ModelDType`/`ModelQuantizationFormat` equivalent are
normalized -- unquantized `F32`/`F16`/`BF16`/`I8`/`I16`/`I32`/`I64`/`F64`,
and quantized `Q4_K`/`Q5_K`/`Q8_0`. GGUF defines roughly thirty `ggml_type`
values in total (`Q4_0`, `Q2_K`...`Q6_K`, the `IQ*` family, etc.); a tensor
declaring any of those is rejected with `GgufError::UnsupportedGgmlType`
naming the numeric type, never silently approximated. Full coverage is real
follow-up work. (Implementing this surfaced a real gap in
`magnetar-runtime` itself: `ModelDType::Q8` already existed, paired with
`ModelDType::parse`'s `"q8"`/`"q8_0"` strings, but `ModelQuantizationFormat`
had no matching `GgufQ8` variant to construct a `ModelQuantization` value
against -- added as a small, additive fix alongside this parser.)

Overflow and bounds safety: every offset/size computation uses checked
arithmetic, declared tensor byte ranges are validated against the file's
actual length, overlapping ranges are rejected, and unrecognized metadata
keys are preserved as opaque data rather than interpreted. 16 unit tests
plus a 22-entry checked-in malformed-input corpus (`tests/corpus/`) prove
every category of malformed input -- including a deliberately deeply
nested metadata array, to prove the recursive-descent parser's depth limit
actually bounds stack usage -- is rejected with a structured error, not a
panic. A `cargo-fuzz` target (`fuzz/fuzz_targets/parse.rs`) exercises the
same entry point for offline/periodic fuzzing (`cargo +nightly fuzz build`
verified to build; live execution requires a sanitizer-runtime-equipped
environment -- not exercised on the Windows machine this crate was
developed on, a real, documented gap rather than an implicit claim of full
local fuzzing coverage).

Deliberately not a full `ModelManifest`: assembling model identity/digest
from external context is the caller's job, not this parser's; see this
crate's governing OpenSpec capability's design.md for why.

## Governing contract

[`gguf-format`](https://github.com/astorise/Magnetar/blob/main/openspec/changes/implement-model-format-parsers/specs/gguf-format/spec.md)
in the main Magnetar repository's OpenSpec change set defines this crate's
requirements in full (type-boundary discipline, quantization scope,
overflow/panic safety, corpus/fuzz coverage). `model-format-roadmap`'s
"GGUF Support" requirement records that support as implemented for its
supported subset, not aspirational, as of this crate.

## Relationship to magnetar-runtime

This crate depends only on `magnetar-runtime`'s public `model` module types
(`formats/gguf -> magnetar-runtime`, never the reverse) -- `magnetar-runtime`
compiling and testing cleanly without this crate present, and never
appearing in its own dependency graph, is verified by a static CI guard
(`.github/workflows/quality.yml`'s `submodule-integration` job). It is
pinned into the main [Magnetar](https://github.com/astorise/Magnetar)
repository as a git submodule at `formats/gguf`.

# magnetar-format-gguf

## Purpose

A GGUF Model Artifact parser for the
[Magnetar](https://github.com/astorise/Magnetar) local AI Runtime. It reads
GGUF-encoded weight files and produces Magnetar's generic, Provider-agnostic
types (`ModelArtifact`, `TensorDescriptor`, `QuantizationDescriptor`,
normalized tokenizer/model metadata) at the boundary into `magnetar-runtime`
-- `magnetar-runtime` SHALL have zero dependency on this crate or any
GGUF-specific type, per Magnetar's architectural invariant that Providers
and Format parsers own native/format-specific implementation details while
the Core stays generic.

## Status

**Empty template.** This crate is currently a bare `cargo new --lib`
scaffold: no GGUF parsing exists here yet. `magnetar-runtime`'s E2E
conformance fixtures currently supply weights as a plain in-memory
`BTreeMap<String, HostTensor>` with no serialized byte format at all --
there is no real Model Artifact byte format anywhere in the Magnetar
workspace today. Building a real parser here is required before
`reach-architecture-freeze-1` task group 8 (Model Loading creates the exact
weight resources consumed by execution) can close its remaining `8.1`/`8.2`
tasks (build a real minimal Model Artifact; parse it through Model
Loading).

## Requirements once implemented

Per the audit that scoped this submodule (see `reach-architecture-freeze-1`
Correctif 16 in the main repository), a real parser here must:

- Produce only generic types across the boundary into `magnetar-runtime`
  (never leak GGUF-specific structs).
- Reject overflow, out-of-bounds offsets/tensor sizes, and overlapping or
  absurd dimensions -- GGUF files are untrusted input.
- Never panic on malformed input; carry a fuzzing and corpus regression
  suite proving this.

## Relationship to magnetar-runtime

`magnetar-runtime` has zero dependency on this crate today (verified by a
static guard in that repository's test suite) and is expected to keep it
that way even once this parser is real -- Model Loading consumes whatever
generic `ModelArtifact` this crate produces, never GGUF types directly. It
is pinned into the main [Magnetar](https://github.com/astorise/Magnetar)
repository as a git submodule at `formats/gguf`.

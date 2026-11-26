# gguf-format Specification

## Purpose
TBD - created by archiving change implement-model-format-parsers. Update Purpose after archive.
## Requirements
### Requirement: GGUF Parser Produces Only Generic Types
The GGUF parser SHALL normalize a parsed file's tensor-info section into `magnetar-runtime`'s existing generic types (`ModelTensorMetadata`, `ModelDType`, `ModelQuantization`/`ModelQuantizationFormat`) and SHALL NOT expose any GGUF-specific type across its public API boundary for tensor data. Unrecognized metadata key-value entries MAY be preserved in a format-local metadata value type, since no generic Model Artifact equivalent exists for arbitrary GGUF metadata yet.

#### Scenario: Tensor inventory normalized
- **WHEN** a well-formed GGUF file's tensor-info section is parsed
- **THEN** the result is a list of `ModelTensorMetadata` values (name, shape, storage dtype, byte offset, byte size, quantization metadata where applicable), with no GGUF-specific struct describing tensor data in the returned type.

#### Scenario: Unrecognized metadata is preserved, not interpreted
- **WHEN** the GGUF key-value metadata section contains a key the parser does not specifically recognize
- **THEN** the value is preserved as opaque source metadata and does not silently influence tensor normalization or Runtime policy.

### Requirement: GGUF Parser Scopes Quantization To Supported Types
The parser SHALL normalize tensors whose declared `ggml_type` maps to an existing `ModelDType`/`ModelQuantizationFormat` variant (`Q4K`, `Q5K`, `Q8`, and unquantized float/integer types), and SHALL reject any tensor with an unsupported `ggml_type` with a structured, named error rather than approximating, silently mis-mapping, or defining a new type.

#### Scenario: Supported quantized tensor
- **WHEN** a tensor declares a `ggml_type` corresponding to `Q4K`, `Q5K`, or `Q8`
- **THEN** it normalizes into `ModelTensorMetadata` with the matching `ModelQuantization`.

#### Scenario: Unsupported quantized tensor
- **WHEN** a tensor declares a `ggml_type` outside the supported subset (for example a `Q2_K` or `IQ`-family type)
- **THEN** the parser returns a structured `gguf-quantization-unsupported` error naming the unsupported type, rather than approximating it as a supported type.

### Requirement: GGUF Parser Rejects Overflow And Out-Of-Bounds Ranges
The parser SHALL use checked arithmetic for every size and offset computation derived from file bytes (including alignment padding), and SHALL reject a tensor whose declared byte range is out of bounds, overlapping with another tensor's range, or would overflow during size computation, before any data is read from that range.

#### Scenario: Declared range exceeds file length
- **WHEN** a tensor's declared offset and computed size extend past the end of the file
- **THEN** the parser returns a structured error and reads no tensor bytes.

#### Scenario: Overlapping tensor ranges
- **WHEN** two tensors' declared byte ranges overlap
- **THEN** the parser returns a structured error rather than accepting either tensor.

#### Scenario: Dimension count causes overflow
- **WHEN** a tensor's declared dimensions and dtype combination would overflow `u64` byte-size arithmetic
- **THEN** the parser returns a structured error instead of wrapping or panicking.

### Requirement: GGUF Parser Never Panics On Malformed Input
The parser SHALL return a structured error rather than panic for any malformed input, including wrong magic bytes, unsupported version, truncated key-value or tensor-info sections, invalid UTF-8 in string fields, and inconsistent tensor/metadata counts.

#### Scenario: Corpus regression suite
- **WHEN** the checked-in malformed-input corpus is replayed through the parser
- **THEN** every entry returns a structured error and none panics.

#### Scenario: Fuzz target exists
- **WHEN** the crate's fuzz target is run against arbitrary byte input
- **THEN** it exercises the same public parse entry point the corpus regression suite uses.

### Requirement: Magnetar-Runtime Has No GGUF Dependency
`magnetar-runtime` SHALL NOT depend on this crate or any GGUF-specific type, and SHALL NOT introduce a `GGUFProvider`.

#### Scenario: Static dependency guard
- **WHEN** `magnetar-runtime`'s dependency graph is inspected
- **THEN** the GGUF format crate does not appear in it.

#### Scenario: No GGUF-specific Provider
- **WHEN** GGUF support is implemented
- **THEN** no `GGUFProvider` or equivalent GGUF-specific Provider type is introduced anywhere in the workspace.

### Requirement: Parse Result Exposes The Tensor Data Section's Absolute Offset

The GGUF parser SHALL expose the absolute byte offset, from the start of the parsed file, at which the tensor data section begins, so a caller can compute each tensor's real absolute byte range from its already-returned data-section-relative `offset_bytes` without re-deriving the parser's own alignment and header-layout computation independently.

#### Scenario: The exposed offset addresses real tensor bytes

- **WHEN** a well-formed GGUF file is parsed
- **THEN** adding the exposed tensor data section offset to any tensor's own `offset_bytes` yields that tensor's real, correct absolute position in the original file


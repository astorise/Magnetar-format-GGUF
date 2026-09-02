//! Real GGUF file parsing, normalized into `magnetar-runtime`'s generic
//! Model Artifact types ([`ModelTensorMetadata`], [`ModelDType`],
//! [`ModelQuantization`]/[`ModelQuantizationFormat`]). No GGUF-specific
//! type describing tensor data crosses this crate's public boundary
//! (`gguf-format` capability); metadata key-value entries whose meaning is
//! GGUF-specific are preserved as opaque [`GgufMetadataValue`] instead,
//! since no generic Model Artifact equivalent exists for arbitrary GGUF
//! metadata yet.
//!
//! # Format
//!
//! `magic("GGUF") | version(u32) | tensor_count(u64) | metadata_kv_count(u64)`,
//! followed by `metadata_kv_count` key-value pairs (`key: string`,
//! `value_type: u32`, then a value shaped by that type -- including
//! recursively-typed arrays), followed by `tensor_count` tensor-info
//! entries (`name: string`, `n_dimensions: u32`, `dimensions: [u64; n]`,
//! `ggml_type: u32`, `offset: u64`), then `0x00` padding up to
//! `general.alignment` (default 32, from the metadata section already
//! parsed by that point), then the raw tensor data those offsets index
//! into (relative to the padded start).
//!
//! Structural details (magic, header layout, metadata value type enum,
//! string/array encoding, alignment default and rule) were verified
//! against the upstream `ggml-org/ggml` GGUF specification, not recalled
//! from memory alone. Quantized block layouts (`block_q4_K`, `block_q5_K`,
//! `block_q8_0` byte sizes and elements-per-block) were verified against
//! `ggml-common.h`.
//!
//! # Quantization scope
//!
//! Only the `ggml_type` values with an existing [`ModelDType`]/
//! [`ModelQuantizationFormat`] equivalent are normalized: unquantized
//! `F32`/`F16`/`BF16`/`I8`/`I16`/`I32`/`I64`/`F64`, and quantized
//! `Q4_K`/`Q5_K`/`Q8_0`. GGUF defines roughly thirty `ggml_type` values in
//! total (`Q4_0`, `Q2_K`...`Q6_K`, the `IQ*` family, etc.); a tensor
//! declaring any of those is rejected with
//! [`GgufError::UnsupportedGgmlType`] naming the numeric type, never
//! silently approximated. See this crate's governing OpenSpec capability
//! for why (`implement-model-format-parsers`'s design.md Non-Goals).
//!
//! # Safety discipline
//!
//! Every read is bounds-checked against the file's actual remaining bytes
//! before any slice indexing or allocation; every size/offset computation
//! uses checked arithmetic; nested metadata arrays are depth-limited
//! ([`MAX_ARRAY_NESTING_DEPTH`]) so a maliciously deep array-of-arrays
//! cannot exhaust the call stack in this recursive-descent parser; and no
//! collection is pre-allocated to an attacker-declared capacity (`tensor_count`,
//! `metadata_kv_count`, an array's declared length) before that many bytes
//! have actually been confirmed present in the file.

use magnetar_runtime::model::{
    ModelDType, ModelQuantization, ModelQuantizationFormat, ModelTensorMetadata,
};
use std::collections::BTreeMap;
use std::fmt;

const GGUF_MAGIC: u32 = 0x4655_4747; // "GGUF" read as a little-endian u32.
const MIN_SUPPORTED_VERSION: u32 = 2;
const MAX_SUPPORTED_VERSION: u32 = 3;
const DEFAULT_ALIGNMENT: u64 = 32;
const MAX_TENSOR_NAME_BYTES: usize = 64;
/// The GGUF spec says tensor dimension count is "currently max 4"; that is
/// current usage, not a documented MUST, so this crate accepts a generous
/// defensive ceiling above it rather than hard-rejecting a spec-valid file
/// from a future GGUF revision, while still bounding worst-case allocation
/// size for a maliciously large declared dimension count.
const MAX_DIMENSIONS: u32 = 8;
/// Recursion limit for nested GGUF metadata arrays (arrays may contain
/// arrays per spec). Bounds stack depth for this recursive-descent parser
/// against adversarial input; no real GGUF metadata nests anywhere near
/// this deep.
const MAX_ARRAY_NESTING_DEPTH: u32 = 8;

/// The normalized result of parsing a GGUF file: a tensor inventory plus
/// every metadata key-value entry (recognized or not). Deliberately not a
/// full `ModelManifest` -- assembling model identity/digest from external
/// context is the caller's responsibility, not this parser's; see this
/// crate's governing OpenSpec capability's design.md for why.
#[derive(Clone, Debug, PartialEq)]
pub struct GgufArtifact {
    pub tensors: Vec<ModelTensorMetadata>,
    pub metadata: BTreeMap<String, GgufMetadataValue>,
}

/// A GGUF metadata value, generic over GGUF's own typed key-value encoding.
/// Not a `magnetar-runtime` type: no generic Model Artifact equivalent
/// exists for arbitrary format-specific metadata yet.
#[derive(Clone, Debug, PartialEq)]
pub enum GgufMetadataValue {
    UInt8(u8),
    Int8(i8),
    UInt16(u16),
    Int16(i16),
    UInt32(u32),
    Int32(i32),
    Float32(f32),
    Bool(bool),
    String(String),
    UInt64(u64),
    Int64(i64),
    Float64(f64),
    Array(Vec<GgufMetadataValue>),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum GgufError {
    UnexpectedEof {
        at: u64,
        needed: u64,
    },
    BadMagic {
        found: u32,
    },
    UnsupportedVersion {
        found: u32,
    },
    StringNotUtf8,
    UnknownMetadataValueType {
        found: u32,
    },
    ArrayNestingTooDeep,
    InvalidAlignment {
        found: u32,
    },
    TensorNameTooLong {
        name_length: usize,
    },
    TooManyDimensions {
        name: String,
        found: u32,
    },
    UnsupportedGgmlType {
        name: String,
        ggml_type: u32,
    },
    ElementCountOverflow {
        name: String,
    },
    ElementCountNotBlockAligned {
        name: String,
        element_count: u64,
        block_elements: u64,
    },
    ByteSizeOverflow {
        name: String,
    },
    AlignmentOverflow,
    OffsetNotAligned {
        name: String,
        offset: u64,
        alignment: u64,
    },
    ByteRangeOutOfBounds {
        name: String,
        start: u64,
        end: u64,
    },
    OverlappingRanges {
        first: String,
        second: String,
    },
    DuplicateTensorName {
        name: String,
    },
}

impl fmt::Display for GgufError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnexpectedEof { at, needed } => {
                write!(
                    f,
                    "unexpected end of file at byte {at}, needed {needed} more bytes"
                )
            }
            Self::BadMagic { found } => {
                write!(f, "bad magic bytes: found {found:#010x}, expected \"GGUF\"")
            }
            Self::UnsupportedVersion { found } => write!(
                f,
                "unsupported GGUF version {found} (supported: {MIN_SUPPORTED_VERSION}-{MAX_SUPPORTED_VERSION})"
            ),
            Self::StringNotUtf8 => write!(f, "string field is not valid UTF-8"),
            Self::UnknownMetadataValueType { found } => {
                write!(f, "unknown metadata value type {found}")
            }
            Self::ArrayNestingTooDeep => {
                write!(f, "metadata array nesting exceeds the depth limit")
            }
            Self::InvalidAlignment { found } => write!(
                f,
                "'general.alignment' value {found} is invalid (must be a nonzero multiple of 8)"
            ),
            Self::TensorNameTooLong { name_length } => write!(
                f,
                "tensor name is {name_length} bytes, exceeding the {MAX_TENSOR_NAME_BYTES}-byte limit"
            ),
            Self::TooManyDimensions { name, found } => {
                write!(
                    f,
                    "tensor '{name}' declares {found} dimensions, exceeding the limit"
                )
            }
            Self::UnsupportedGgmlType { name, ggml_type } => write!(
                f,
                "tensor '{name}' declares unsupported ggml_type {ggml_type}"
            ),
            Self::ElementCountOverflow { name } => {
                write!(f, "tensor '{name}' element-count computation overflowed")
            }
            Self::ElementCountNotBlockAligned {
                name,
                element_count,
                block_elements,
            } => write!(
                f,
                "tensor '{name}' has {element_count} elements, not a multiple of its {block_elements}-element quantization block"
            ),
            Self::ByteSizeOverflow { name } => {
                write!(f, "tensor '{name}' byte-size computation overflowed")
            }
            Self::AlignmentOverflow => write!(f, "tensor-data alignment computation overflowed"),
            Self::OffsetNotAligned {
                name,
                offset,
                alignment,
            } => write!(
                f,
                "tensor '{name}' offset {offset} is not a multiple of the {alignment}-byte alignment"
            ),
            Self::ByteRangeOutOfBounds { name, start, end } => write!(
                f,
                "tensor '{name}' byte range [{start}, {end}) is out of bounds"
            ),
            Self::OverlappingRanges { first, second } => write!(
                f,
                "tensor '{first}' and tensor '{second}' declare overlapping byte ranges"
            ),
            Self::DuplicateTensorName { name } => {
                write!(f, "tensor name '{name}' is declared more than once")
            }
        }
    }
}

impl std::error::Error for GgufError {}

struct Cursor<'a> {
    bytes: &'a [u8],
    position: u64,
}

impl<'a> Cursor<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, position: 0 }
    }

    fn remaining(&self) -> u64 {
        self.bytes.len() as u64 - self.position
    }

    fn read_bytes(&mut self, count: u64) -> Result<&'a [u8], GgufError> {
        if count > self.remaining() {
            return Err(GgufError::UnexpectedEof {
                at: self.position,
                needed: count,
            });
        }
        let start = usize::try_from(self.position).map_err(|_| GgufError::UnexpectedEof {
            at: self.position,
            needed: count,
        })?;
        let end = usize::try_from(self.position + count).map_err(|_| GgufError::UnexpectedEof {
            at: self.position,
            needed: count,
        })?;
        self.position += count;
        Ok(&self.bytes[start..end])
    }

    fn read_u8(&mut self) -> Result<u8, GgufError> {
        Ok(self.read_bytes(1)?[0])
    }

    fn read_i8(&mut self) -> Result<i8, GgufError> {
        Ok(self.read_bytes(1)?[0] as i8)
    }

    fn read_u16_le(&mut self) -> Result<u16, GgufError> {
        let bytes: [u8; 2] = self
            .read_bytes(2)?
            .try_into()
            .expect("length checked above");
        Ok(u16::from_le_bytes(bytes))
    }

    fn read_i16_le(&mut self) -> Result<i16, GgufError> {
        Ok(self.read_u16_le()? as i16)
    }

    fn read_u32_le(&mut self) -> Result<u32, GgufError> {
        let bytes: [u8; 4] = self
            .read_bytes(4)?
            .try_into()
            .expect("length checked above");
        Ok(u32::from_le_bytes(bytes))
    }

    fn read_i32_le(&mut self) -> Result<i32, GgufError> {
        Ok(self.read_u32_le()? as i32)
    }

    fn read_f32_le(&mut self) -> Result<f32, GgufError> {
        Ok(f32::from_bits(self.read_u32_le()?))
    }

    fn read_u64_le(&mut self) -> Result<u64, GgufError> {
        let bytes: [u8; 8] = self
            .read_bytes(8)?
            .try_into()
            .expect("length checked above");
        Ok(u64::from_le_bytes(bytes))
    }

    fn read_i64_le(&mut self) -> Result<i64, GgufError> {
        Ok(self.read_u64_le()? as i64)
    }

    fn read_f64_le(&mut self) -> Result<f64, GgufError> {
        Ok(f64::from_bits(self.read_u64_le()?))
    }

    /// Reads a GGUF string: a `u64` byte length followed by that many UTF-8
    /// bytes. The length is bounds-checked against the file's actual
    /// remaining bytes by [`Self::read_bytes`] before any allocation, so an
    /// absurd declared length (up to `u64::MAX`) is rejected immediately
    /// rather than attempted.
    fn read_string(&mut self) -> Result<String, GgufError> {
        let length = self.read_u64_le()?;
        let bytes = self.read_bytes(length)?;
        String::from_utf8(bytes.to_vec()).map_err(|_| GgufError::StringNotUtf8)
    }
}

fn read_metadata_value(
    cursor: &mut Cursor<'_>,
    value_type: u32,
    depth: u32,
) -> Result<GgufMetadataValue, GgufError> {
    match value_type {
        0 => Ok(GgufMetadataValue::UInt8(cursor.read_u8()?)),
        1 => Ok(GgufMetadataValue::Int8(cursor.read_i8()?)),
        2 => Ok(GgufMetadataValue::UInt16(cursor.read_u16_le()?)),
        3 => Ok(GgufMetadataValue::Int16(cursor.read_i16_le()?)),
        4 => Ok(GgufMetadataValue::UInt32(cursor.read_u32_le()?)),
        5 => Ok(GgufMetadataValue::Int32(cursor.read_i32_le()?)),
        6 => Ok(GgufMetadataValue::Float32(cursor.read_f32_le()?)),
        7 => Ok(GgufMetadataValue::Bool(cursor.read_u8()? != 0)),
        8 => Ok(GgufMetadataValue::String(cursor.read_string()?)),
        9 => {
            if depth >= MAX_ARRAY_NESTING_DEPTH {
                return Err(GgufError::ArrayNestingTooDeep);
            }
            let element_type = cursor.read_u32_le()?;
            let length = cursor.read_u64_le()?;
            // Never pre-allocate to the declared length: it is untrusted
            // and unbounded. `read_metadata_value` fails fast on the first
            // element once the file's actual bytes run out, so a huge
            // declared length against a small file costs one failed read,
            // not an attempted huge allocation.
            let mut values = Vec::with_capacity(length.min(64) as usize);
            for _ in 0..length {
                values.push(read_metadata_value(cursor, element_type, depth + 1)?);
            }
            Ok(GgufMetadataValue::Array(values))
        }
        10 => Ok(GgufMetadataValue::UInt64(cursor.read_u64_le()?)),
        11 => Ok(GgufMetadataValue::Int64(cursor.read_i64_le()?)),
        12 => Ok(GgufMetadataValue::Float64(cursor.read_f64_le()?)),
        other => Err(GgufError::UnknownMetadataValueType { found: other }),
    }
}

/// The transport layout for one `ggml_type`: how many elements form one
/// quantization block, and how many bytes that block occupies. Unquantized
/// dtypes are modeled as one "block" per element (`block_elements: 1`,
/// `block_bytes` = the element's byte width), which lets tensor byte-size
/// computation use one uniform formula for both quantized and unquantized
/// tensors: `(element_count / block_elements) * block_bytes`, so long as
/// `element_count` is a multiple of `block_elements`.
struct DTypeLayout {
    dtype: ModelDType,
    quantization: Option<ModelQuantization>,
    block_elements: u64,
    block_bytes: u64,
}

fn dtype_layout_for_ggml_type(ggml_type: u32) -> Option<DTypeLayout> {
    let unquantized = |dtype: ModelDType, width: u64| DTypeLayout {
        dtype,
        quantization: None,
        block_elements: 1,
        block_bytes: width,
    };
    let quantized = |dtype: ModelDType,
                     format: ModelQuantizationFormat,
                     block_elements: u64,
                     block_bytes: u64| {
        DTypeLayout {
            dtype,
            quantization: Some(ModelQuantization {
                format,
                group_size: Some(block_elements as u32),
                block_size: Some(block_bytes as u32),
                scale_dtype: Some(ModelDType::F16),
                zero_point_dtype: None,
                per_channel: false,
                workspace_bytes: None,
                required_capabilities: Vec::new(),
            }),
            block_elements,
            block_bytes,
        }
    };
    match ggml_type {
        0 => Some(unquantized(ModelDType::F32, 4)),
        1 => Some(unquantized(ModelDType::F16, 2)),
        8 => Some(quantized(
            ModelDType::Q8,
            ModelQuantizationFormat::GgufQ8,
            32,
            34,
        )),
        12 => Some(quantized(
            ModelDType::Q4K,
            ModelQuantizationFormat::GgufQ4K,
            256,
            144,
        )),
        13 => Some(quantized(
            ModelDType::Q5K,
            ModelQuantizationFormat::GgufQ5K,
            256,
            176,
        )),
        24 => Some(unquantized(ModelDType::I8, 1)),
        25 => Some(unquantized(ModelDType::I16, 2)),
        26 => Some(unquantized(ModelDType::I32, 4)),
        27 => Some(unquantized(ModelDType::I64, 8)),
        28 => Some(unquantized(ModelDType::F64, 8)),
        30 => Some(unquantized(ModelDType::Bf16, 2)),
        _ => None,
    }
}

fn align_up(position: u64, alignment: u64) -> Result<u64, GgufError> {
    let remainder = position % alignment;
    if remainder == 0 {
        return Ok(position);
    }
    position
        .checked_add(alignment - remainder)
        .ok_or(GgufError::AlignmentOverflow)
}

/// Parses a complete GGUF file already held in memory. See the module
/// documentation for the format and safety discipline.
pub fn parse(bytes: &[u8]) -> Result<GgufArtifact, GgufError> {
    let mut cursor = Cursor::new(bytes);

    let magic = cursor.read_u32_le()?;
    if magic != GGUF_MAGIC {
        return Err(GgufError::BadMagic { found: magic });
    }
    let version = cursor.read_u32_le()?;
    if !(MIN_SUPPORTED_VERSION..=MAX_SUPPORTED_VERSION).contains(&version) {
        return Err(GgufError::UnsupportedVersion { found: version });
    }
    let tensor_count = cursor.read_u64_le()?;
    let metadata_kv_count = cursor.read_u64_le()?;

    let mut metadata = BTreeMap::new();
    for _ in 0..metadata_kv_count {
        let key = cursor.read_string()?;
        let value_type = cursor.read_u32_le()?;
        let value = read_metadata_value(&mut cursor, value_type, 0)?;
        metadata.insert(key, value);
    }

    let alignment = match metadata.get("general.alignment") {
        Some(GgufMetadataValue::UInt32(value)) if *value != 0 && value % 8 == 0 => *value as u64,
        Some(GgufMetadataValue::UInt32(value)) => {
            return Err(GgufError::InvalidAlignment { found: *value });
        }
        Some(_) => return Err(GgufError::InvalidAlignment { found: 0 }),
        None => DEFAULT_ALIGNMENT,
    };

    struct TensorInfo {
        name: String,
        dimensions: Vec<u64>,
        ggml_type: u32,
        offset: u64,
    }

    let mut tensor_infos = Vec::with_capacity(tensor_count.min(4096) as usize);
    for _ in 0..tensor_count {
        let name = cursor.read_string()?;
        if name.len() > MAX_TENSOR_NAME_BYTES {
            return Err(GgufError::TensorNameTooLong {
                name_length: name.len(),
            });
        }
        let n_dimensions = cursor.read_u32_le()?;
        if n_dimensions > MAX_DIMENSIONS {
            return Err(GgufError::TooManyDimensions {
                name,
                found: n_dimensions,
            });
        }
        let mut dimensions = Vec::with_capacity(n_dimensions as usize);
        for _ in 0..n_dimensions {
            dimensions.push(cursor.read_u64_le()?);
        }
        let ggml_type = cursor.read_u32_le()?;
        let offset = cursor.read_u64_le()?;
        tensor_infos.push(TensorInfo {
            name,
            dimensions,
            ggml_type,
            offset,
        });
    }

    let tensor_data_start = align_up(cursor.position, alignment)?;
    let file_length = bytes.len() as u64;

    let mut tensors = Vec::with_capacity(tensor_infos.len());
    let mut ranges: Vec<(u64, u64, String)> = Vec::with_capacity(tensor_infos.len());
    let mut seen_names = std::collections::BTreeSet::new();

    for info in tensor_infos {
        if !seen_names.insert(info.name.clone()) {
            return Err(GgufError::DuplicateTensorName { name: info.name });
        }
        if info.offset % alignment != 0 {
            return Err(GgufError::OffsetNotAligned {
                name: info.name,
                offset: info.offset,
                alignment,
            });
        }

        let layout = dtype_layout_for_ggml_type(info.ggml_type).ok_or_else(|| {
            GgufError::UnsupportedGgmlType {
                name: info.name.clone(),
                ggml_type: info.ggml_type,
            }
        })?;

        let element_count = info
            .dimensions
            .iter()
            .try_fold(1_u64, |count, &dimension| count.checked_mul(dimension))
            .ok_or_else(|| GgufError::ElementCountOverflow {
                name: info.name.clone(),
            })?;
        if element_count % layout.block_elements != 0 {
            return Err(GgufError::ElementCountNotBlockAligned {
                name: info.name,
                element_count,
                block_elements: layout.block_elements,
            });
        }
        let block_count = element_count / layout.block_elements;
        let byte_size = block_count.checked_mul(layout.block_bytes).ok_or_else(|| {
            GgufError::ByteSizeOverflow {
                name: info.name.clone(),
            }
        })?;

        let start = tensor_data_start.checked_add(info.offset).ok_or_else(|| {
            GgufError::ByteSizeOverflow {
                name: info.name.clone(),
            }
        })?;
        let end = start
            .checked_add(byte_size)
            .ok_or_else(|| GgufError::ByteSizeOverflow {
                name: info.name.clone(),
            })?;
        if end > file_length {
            return Err(GgufError::ByteRangeOutOfBounds {
                name: info.name,
                start,
                end,
            });
        }

        ranges.push((start, end, info.name.clone()));
        tensors.push(ModelTensorMetadata {
            name: info.name,
            shape: info.dimensions,
            storage_dtype: layout.dtype,
            layout: None,
            shard: None,
            offset_bytes: Some(info.offset),
            size_bytes: Some(byte_size),
            quantization: layout.quantization,
            expected_compute_dtype: None,
        });
    }

    ranges.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.2.cmp(&b.2)));
    for pair in ranges.windows(2) {
        let (_, first_end, first_name) = &pair[0];
        let (second_start, _, second_name) = &pair[1];
        if second_start < first_end {
            return Err(GgufError::OverlappingRanges {
                first: first_name.clone(),
                second: second_name.clone(),
            });
        }
    }

    tensors.sort_by(|a, b| a.name.cmp(&b.name));
    Ok(GgufArtifact { tensors, metadata })
}

#[cfg(test)]
mod tests;

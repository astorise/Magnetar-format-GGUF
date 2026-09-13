use super::*;

fn le_u32(value: u32) -> Vec<u8> {
    value.to_le_bytes().to_vec()
}

fn le_u64(value: u64) -> Vec<u8> {
    value.to_le_bytes().to_vec()
}

fn gguf_string(value: &str) -> Vec<u8> {
    let mut out = le_u64(value.len() as u64);
    out.extend_from_slice(value.as_bytes());
    out
}

fn kv_uint32(key: &str, value: u32) -> Vec<u8> {
    let mut out = gguf_string(key);
    out.extend(le_u32(4)); // GGUF_METADATA_VALUE_TYPE_UINT32
    out.extend(le_u32(value));
    out
}

fn kv_string(key: &str, value: &str) -> Vec<u8> {
    let mut out = gguf_string(key);
    out.extend(le_u32(8)); // GGUF_METADATA_VALUE_TYPE_STRING
    out.extend(gguf_string(value));
    out
}

struct TestTensor {
    name: &'static str,
    dimensions: Vec<u64>,
    ggml_type: u32,
    data: Vec<u8>,
}

/// Assembles a complete, well-formed GGUF byte buffer: header, metadata KV
/// section, tensor-info section, alignment padding, and tensor data -- with
/// every offset computed the same way `parse` expects to read them back,
/// so these fixtures exercise the real encoding, not a simplified stand-in.
fn build_gguf(
    version: u32,
    kv_entries: &[Vec<u8>],
    tensors: &[TestTensor],
    alignment: u64,
) -> Vec<u8> {
    let mut file = le_u32(GGUF_MAGIC);
    file.extend(le_u32(version));
    file.extend(le_u64(tensors.len() as u64));
    file.extend(le_u64(kv_entries.len() as u64));
    for entry in kv_entries {
        file.extend(entry);
    }

    let mut tensor_info_bytes = Vec::new();
    let mut data_section = Vec::new();
    let mut next_offset = 0_u64;
    for tensor in tensors {
        let padding = (alignment - (next_offset % alignment)) % alignment;
        data_section.extend(std::iter::repeat_n(0u8, padding as usize));
        next_offset += padding;
        let offset = next_offset;

        tensor_info_bytes.extend(gguf_string(tensor.name));
        tensor_info_bytes.extend(le_u32(tensor.dimensions.len() as u32));
        for dimension in &tensor.dimensions {
            tensor_info_bytes.extend(le_u64(*dimension));
        }
        tensor_info_bytes.extend(le_u32(tensor.ggml_type));
        tensor_info_bytes.extend(le_u64(offset));

        data_section.extend(&tensor.data);
        next_offset += tensor.data.len() as u64;
    }
    file.extend(tensor_info_bytes);

    let start_padding = (alignment - (file.len() as u64 % alignment)) % alignment;
    file.extend(std::iter::repeat_n(0u8, start_padding as usize));
    file.extend(data_section);
    file
}

#[test]
fn well_formed_file_parses_expected_tensors_and_metadata() {
    let kvs = vec![
        kv_string("general.name", "test-model"),
        kv_uint32("general.alignment", 32),
    ];
    let tensors = [
        TestTensor {
            name: "token_embedding",
            dimensions: vec![4, 8],
            ggml_type: 0, // F32
            data: vec![0u8; 4 * 8 * 4],
        },
        TestTensor {
            name: "layers.0.self_attn.q_proj",
            dimensions: vec![2, 2],
            ggml_type: 1, // F16
            data: vec![0u8; 2 * 2 * 2],
        },
    ];
    let file = build_gguf(3, &kvs, &tensors, 32);

    let artifact = parse(&file).expect("well-formed GGUF file parses");
    assert_eq!(
        artifact.metadata.get("general.name"),
        Some(&GgufMetadataValue::String("test-model".to_string()))
    );
    assert_eq!(artifact.tensors.len(), 2);
    let embedding = artifact
        .tensors
        .iter()
        .find(|t| t.name == "token_embedding")
        .unwrap();
    assert_eq!(embedding.shape, vec![4, 8]);
    assert_eq!(embedding.storage_dtype, ModelDType::F32);
    assert_eq!(embedding.size_bytes, Some(4 * 8 * 4));

    // `tensor_data_start` plus a tensor's own (data-section-relative)
    // `offset_bytes` must land on that tensor's real bytes in the file --
    // proven here by re-slicing the raw file at that absolute position
    // and comparing against the tensor's own all-zero payload, not merely
    // asserting the field is present.
    let embedding_offset = embedding.offset_bytes.expect("offset_bytes is set");
    let absolute_start = (artifact.tensor_data_start + embedding_offset) as usize;
    let absolute_end = absolute_start + embedding.size_bytes.unwrap() as usize;
    assert_eq!(
        &file[absolute_start..absolute_end],
        vec![0u8; 4 * 8 * 4].as_slice(),
        "tensor_data_start + offset_bytes must address token_embedding's real bytes in the file"
    );
}

/// `bind-materialized-weight-content-to-model-artifact-digests`: an F32
/// tensor's `ModelTensorMetadata.digest` is a real content digest, not a
/// placeholder -- it verifies against that exact tensor's real bytes as
/// parsed from the file, and correctly rejects tampered bytes.
#[test]
fn f32_tensor_digest_verifies_against_its_real_bytes() {
    let data: Vec<u8> = (0..16u8).collect();
    let tensors = [TestTensor {
        name: "weight",
        dimensions: vec![4],
        ggml_type: 0, // F32
        data: data.clone(),
    }];
    let file = build_gguf(3, &[], &tensors, 32);

    let artifact = parse(&file).expect("well-formed GGUF file parses");
    let tensor = artifact
        .tensors
        .iter()
        .find(|t| t.name == "weight")
        .unwrap();
    let digest = tensor
        .digest
        .as_ref()
        .expect("an unquantized F32 tensor should have a computed content digest");

    digest
        .verify_bytes(&data)
        .expect("digest verifies against the tensor's own real bytes");

    let mut tampered = data;
    tampered[0] ^= 0xFF;
    assert!(
        digest.verify_bytes(&tampered).is_err(),
        "digest must reject tampered content"
    );
}

/// Quantized tensors cannot be materialized into a `HostTensor` without
/// dequantization this crate does not perform, so their content cannot be
/// meaningfully digested here -- `digest: None` is correct, not an
/// oversight.
#[test]
fn quantized_tensor_has_no_content_digest() {
    let tensors = [TestTensor {
        name: "t",
        dimensions: vec![256],
        ggml_type: 12, // Q4_K
        data: vec![0u8; 144],
    }];
    let file = build_gguf(3, &[], &tensors, 32);

    let artifact = parse(&file).expect("well-formed GGUF file parses");
    assert!(artifact.tensors[0].digest.is_none());
}

#[test]
fn every_supported_ggml_type_round_trips() {
    let cases: &[(u32, ModelDType, u64, Vec<u64>)] = &[
        (0, ModelDType::F32, 4 * 4, vec![4]),
        (1, ModelDType::F16, 4 * 2, vec![4]),
        (24, ModelDType::I8, 4, vec![4]),
        (25, ModelDType::I16, 8, vec![4]),
        (26, ModelDType::I32, 16, vec![4]),
        (27, ModelDType::I64, 32, vec![4]),
        (28, ModelDType::F64, 32, vec![4]),
        (30, ModelDType::Bf16, 8, vec![4]),
        // Quantized: element count must be a multiple of the block size.
        (8, ModelDType::Q8, 34, vec![32]),
        (12, ModelDType::Q4K, 144, vec![256]),
        (13, ModelDType::Q5K, 176, vec![256]),
    ];
    for (ggml_type, expected_dtype, byte_size, dimensions) in cases {
        let tensors = [TestTensor {
            name: "t",
            dimensions: dimensions.clone(),
            ggml_type: *ggml_type,
            data: vec![0u8; *byte_size as usize],
        }];
        let file = build_gguf(3, &[], &tensors, 32);
        let artifact = parse(&file)
            .unwrap_or_else(|error| panic!("ggml_type {ggml_type} expected to parse, got {error}"));
        assert_eq!(artifact.tensors[0].storage_dtype, *expected_dtype);
        assert_eq!(artifact.tensors[0].size_bytes, Some(*byte_size));
    }
}

#[test]
fn quantized_types_carry_quantization_metadata() {
    let tensors = [TestTensor {
        name: "t",
        dimensions: vec![256],
        ggml_type: 12, // Q4_K
        data: vec![0u8; 144],
    }];
    let file = build_gguf(3, &[], &tensors, 32);
    let artifact = parse(&file).unwrap();
    let quantization = artifact.tensors[0]
        .quantization
        .as_ref()
        .expect("Q4_K carries quantization metadata");
    assert_eq!(quantization.format, ModelQuantizationFormat::GgufQ4K);
    assert_eq!(quantization.group_size, Some(256));
    assert_eq!(quantization.block_size, Some(144));
}

#[test]
fn unsupported_ggml_type_is_rejected_structurally() {
    // ggml_type 2 = Q4_0, outside this crate's supported subset.
    let tensors = [TestTensor {
        name: "t",
        dimensions: vec![32],
        ggml_type: 2,
        data: vec![0u8; 18],
    }];
    let file = build_gguf(3, &[], &tensors, 32);
    assert!(matches!(
        parse(&file),
        Err(GgufError::UnsupportedGgmlType { ggml_type: 2, .. })
    ));
}

#[test]
fn unrecognized_metadata_key_is_preserved_unchanged() {
    let kvs = vec![kv_string("some.unknown.vendor.key", "opaque-value")];
    let file = build_gguf(3, &kvs, &[], 32);
    let artifact = parse(&file).unwrap();
    assert_eq!(
        artifact.metadata.get("some.unknown.vendor.key"),
        Some(&GgufMetadataValue::String("opaque-value".to_string()))
    );
}

#[test]
fn bad_magic_is_rejected() {
    let mut file = le_u32(0xDEAD_BEEF);
    file.extend(le_u32(3));
    file.extend(le_u64(0));
    file.extend(le_u64(0));
    assert!(matches!(parse(&file), Err(GgufError::BadMagic { .. })));
}

#[test]
fn unsupported_version_is_rejected() {
    let file = build_gguf(99, &[], &[], 32);
    assert!(matches!(
        parse(&file),
        Err(GgufError::UnsupportedVersion { found: 99 })
    ));
}

#[test]
fn overlapping_tensor_ranges_are_rejected() {
    // Hand-construct two tensors whose offsets are each individually a
    // valid multiple of the alignment (so `OffsetNotAligned` cannot fire
    // first), but whose declared sizes make their byte ranges overlap:
    // "a" at offset 0 with 40 I8 elements covers [0, 40); "b" at offset 32
    // (the next aligned offset) covers [32, 36) -- overlapping at [32, 40).
    // `build_gguf` always lays tensors out non-overlapping, so this
    // bypasses that helper.
    let mut file = le_u32(GGUF_MAGIC);
    file.extend(le_u32(3));
    file.extend(le_u64(2)); // tensor_count
    file.extend(le_u64(0)); // metadata_kv_count
    for (name, dimension, offset) in [("a", 40u64, 0u64), ("b", 4u64, 32u64)] {
        file.extend(gguf_string(name));
        file.extend(le_u32(1)); // n_dimensions
        file.extend(le_u64(dimension));
        file.extend(le_u32(24)); // I8
        file.extend(le_u64(offset));
    }
    let start_padding = (32 - (file.len() as u64 % 32)) % 32;
    file.extend(std::iter::repeat_n(0u8, start_padding as usize));
    file.extend(vec![0u8; 40]); // enough data for both overlapping ranges
    assert!(matches!(
        parse(&file),
        Err(GgufError::OverlappingRanges { .. })
    ));
}

#[test]
fn out_of_range_byte_size_is_rejected() {
    let tensors = [TestTensor {
        name: "t",
        dimensions: vec![1000],
        ggml_type: 0,       // F32, 4000 bytes declared
        data: vec![0u8; 4], // but only 4 bytes actually present
    }];
    // Bypass build_gguf's automatic sizing so the declared shape disagrees
    // with the actual data present.
    let mut file = le_u32(GGUF_MAGIC);
    file.extend(le_u32(3));
    file.extend(le_u64(1));
    file.extend(le_u64(0));
    file.extend(gguf_string(tensors[0].name));
    file.extend(le_u32(1));
    file.extend(le_u64(1000));
    file.extend(le_u32(0));
    file.extend(le_u64(0));
    let start_padding = (32 - (file.len() as u64 % 32)) % 32;
    file.extend(std::iter::repeat_n(0u8, start_padding as usize));
    file.extend(vec![0u8; 4]);
    assert!(matches!(
        parse(&file),
        Err(GgufError::ByteRangeOutOfBounds { .. })
    ));
}

#[test]
fn duplicate_tensor_name_is_rejected() {
    let tensors = [
        TestTensor {
            name: "dup",
            dimensions: vec![4],
            ggml_type: 24,
            data: vec![0u8; 4],
        },
        TestTensor {
            name: "dup",
            dimensions: vec![4],
            ggml_type: 24,
            data: vec![0u8; 4],
        },
    ];
    let file = build_gguf(3, &[], &tensors, 32);
    assert!(matches!(
        parse(&file),
        Err(GgufError::DuplicateTensorName { .. })
    ));
}

#[test]
fn misaligned_offset_is_rejected() {
    let mut file = le_u32(GGUF_MAGIC);
    file.extend(le_u32(3));
    file.extend(le_u64(1));
    file.extend(le_u64(0));
    file.extend(gguf_string("t"));
    file.extend(le_u32(1));
    file.extend(le_u64(4));
    file.extend(le_u32(24)); // I8
    file.extend(le_u64(3)); // offset not a multiple of alignment (32)
    let start_padding = (32 - (file.len() as u64 % 32)) % 32;
    file.extend(std::iter::repeat_n(0u8, start_padding as usize));
    file.extend(vec![0u8; 8]);
    assert!(matches!(
        parse(&file),
        Err(GgufError::OffsetNotAligned { .. })
    ));
}

#[test]
fn element_count_not_block_aligned_is_rejected() {
    // Q4_K requires a multiple of 256 elements; 100 is not.
    let tensors = [TestTensor {
        name: "t",
        dimensions: vec![100],
        ggml_type: 12,
        data: vec![],
    }];
    let file = build_gguf(3, &[], &tensors, 32);
    assert!(matches!(
        parse(&file),
        Err(GgufError::ElementCountNotBlockAligned { .. })
    ));
}

#[test]
fn deeply_nested_array_is_rejected_not_stack_overflowing() {
    let mut file = le_u32(GGUF_MAGIC);
    file.extend(le_u32(3));
    file.extend(le_u64(0)); // tensor_count
    file.extend(le_u64(1)); // metadata_kv_count
    file.extend(gguf_string("nested"));
    file.extend(le_u32(9)); // ARRAY
    // Nest arrays-of-arrays deeper than MAX_ARRAY_NESTING_DEPTH.
    let mut body = Vec::new();
    for _ in 0..(MAX_ARRAY_NESTING_DEPTH + 4) {
        body.extend(le_u32(9)); // element type: ARRAY
        body.extend(le_u64(1)); // length: 1
    }
    file.extend(body);
    assert!(matches!(parse(&file), Err(GgufError::ArrayNestingTooDeep)));
}

#[test]
fn empty_file_is_rejected_not_panicking() {
    assert!(parse(&[]).is_err());
}

#[test]
fn huge_declared_counts_against_a_tiny_file_fail_fast_not_panicking() {
    let mut file = le_u32(GGUF_MAGIC);
    file.extend(le_u32(3));
    file.extend(le_u64(u64::MAX)); // tensor_count
    file.extend(le_u64(u64::MAX)); // metadata_kv_count
    assert!(matches!(parse(&file), Err(GgufError::UnexpectedEof { .. })));
}

/// Corpus regression suite (`gguf-format`'s "Never Panics On Malformed
/// Input" requirement): every checked-in malformed byte sequence must be
/// rejected with a structured error, never a panic. Runs under plain
/// `cargo test`, no `cargo-fuzz`/nightly requirement -- see `fuzz/` for the
/// offline fuzz target that grows this corpus over time.
#[test]
fn malformed_input_corpus_never_panics() {
    let corpus_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/corpus");
    let mut checked = 0;
    for entry in std::fs::read_dir(&corpus_dir).expect("corpus directory exists") {
        let entry = entry.expect("readable corpus directory entry");
        let path = entry.path();
        if path.extension().and_then(|extension| extension.to_str()) != Some("bin") {
            continue;
        }
        let bytes = std::fs::read(&path).expect("readable corpus file");
        let result = std::panic::catch_unwind(|| parse(&bytes));
        assert!(
            result.is_ok(),
            "parsing corpus file {path:?} must not panic"
        );
        assert!(
            result.unwrap().is_err(),
            "corpus file {path:?} is expected to be malformed (parse should fail, not succeed)"
        );
        checked += 1;
    }
    assert!(checked > 0, "corpus directory must not be empty");
}

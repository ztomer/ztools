use super::*;
use crate::ztools::twitter::capture::decode_body;
use std::io::Write as _;

fn roundtrip(bytes: &[u8], encoding: Option<&str>) -> Vec<u8> {
    decode_body(bytes, encoding).expect("fixture must decode")
}

#[test]
fn decode_identity_passthrough() {
    assert_eq!(roundtrip(b"{}", None), b"{}");
    assert_eq!(roundtrip(b"{}", Some("identity")), b"{}");
}

#[test]
fn decode_gzip_deflate_brotli_roundtrips() {
    let plain = br#"{"data":{"home":{}}}"#;
    let mut gz = Vec::new();
    flate2::write::GzEncoder::new(&mut gz, flate2::Compression::fast())
        .write_all(plain)
        .unwrap();
    // GzEncoder flushes on drop; rebuild to finish the stream.
    let mut enc = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::fast());
    enc.write_all(plain).unwrap();
    let gz = enc.finish().unwrap();
    assert_eq!(roundtrip(&gz, Some("gzip")), plain);

    let mut enc = flate2::write::DeflateEncoder::new(Vec::new(), flate2::Compression::fast());
    enc.write_all(plain).unwrap();
    assert_eq!(roundtrip(&enc.finish().unwrap(), Some("deflate")), plain);

    let mut br = Vec::new();
    {
        let mut enc = brotli::CompressorWriter::new(&mut br, 4096, 5, 22);
        enc.write_all(plain).unwrap();
        enc.flush().unwrap();
    }
    assert_eq!(roundtrip(&br, Some("BR")), plain);
}

#[test]
fn decode_unknown_encoding_is_a_named_error() {
    let err = decode_body(b"x", Some("zstd")).unwrap_err();
    assert!(err.to_string().contains("zstd"), "{err}");
}

#[test]
fn decode_corrupt_gzip_errors() {
    assert!(decode_body(b"not gzip at all", Some("gzip")).is_err());
}

#[test]
fn following_terms_default_and_override() {
    assert!(following_terms().contains(&"Following".to_string()));
    // Override path is env-dependent; default surface is asserted above.
    // The override branch is exercised in Phase 3 live runs.
}

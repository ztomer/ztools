//! The four filters on hand-built inputs, at their boundaries.

use super::*;
use serde_json::json;

fn qwen27b() -> Value {
    let types: Vec<&str> = (0..64)
        .map(|i| {
            if i % 4 == 3 {
                "full_attention"
            } else {
                "linear_attention"
            }
        })
        .collect();
    json!({"model_type": "qwen3_5", "text_config": {
        "num_hidden_layers": 64, "num_key_value_heads": 4, "head_dim": 256, "layer_types": types}})
}

fn files(list: &[(&str, u64)]) -> Vec<File> {
    list.iter()
        .map(|(p, s)| File {
            path: (*p).to_string(),
            size: *s,
        })
        .collect()
}

fn served() -> BTreeSet<String> {
    ["qwen3_5".to_string()].into()
}

#[test]
fn the_27b_kv_at_32k_is_the_designs_2_gib() {
    assert_eq!(kv_bytes(&qwen27b(), CTX_TOKENS), Ok(2 * GIB));
    let by_interval = json!({"num_hidden_layers": 64, "num_key_value_heads": 4,
        "head_dim": 256, "full_attention_interval": 4});
    assert_eq!(kv_bytes(&by_interval, CTX_TOKENS), Ok(2 * GIB));
}

#[test]
fn sliding_layers_count_their_window_and_global_layers_their_own_shape() {
    let mut types = vec!["sliding_attention"; 40];
    types.extend(["full_attention"; 8]);
    let gemma = json!({"num_hidden_layers": 48, "num_key_value_heads": 8, "head_dim": 256,
        "num_global_key_value_heads": 4, "global_head_dim": 256, "sliding_window": 1024,
        "layer_types": types});
    let full = 8 * 2 * 4 * 256 * 2 * CTX_TOKENS;
    let sliding = 40 * 2 * 8 * 256 * 2 * 1024;
    assert_eq!(kv_bytes(&gemma, CTX_TOKENS), Ok(full + sliding));
    let derived = json!({"num_hidden_layers": 2, "num_key_value_heads": 2,
        "hidden_size": 1024, "num_attention_heads": 8});
    assert_eq!(kv_bytes(&derived, 1), Ok(2 * 2 * 2 * 128 * 2));
}

#[test]
fn a_config_without_an_attention_shape_reads_unverified() {
    let c = fit(GIB, &json!({"model_type": "x"}), Some(48 * GIB));
    assert_eq!(
        c,
        Check::Unverified("config.json has no num_hidden_layers".into())
    );
    assert!(matches!(fit(GIB, &qwen27b(), None), Check::Unverified(_)));
}

#[test]
fn fit_holds_at_the_line_and_fails_one_byte_over() {
    let budget = 48 * GIB;
    let line = budget / 100 * FIT_SHARE_PERCENT;
    let weights = line - 2 * GIB - RUNTIME_BYTES;
    assert!(fit(weights, &qwen27b(), Some(budget)).passed());
    assert!(matches!(
        fit(weights + 1, &qwen27b(), Some(budget)),
        Check::Fail(_)
    ));
}

#[test]
fn disk_holds_at_the_floor_and_fails_under_it_or_with_one_staged() {
    let free = 35 * GIB;
    assert!(disk(free - DISK_FLOOR_BYTES, Some(free), 0).passed());
    assert!(matches!(
        disk(free - DISK_FLOOR_BYTES + 1, Some(free), 0),
        Check::Fail(_)
    ));
    assert!(matches!(disk(free + 1, Some(free), 0), Check::Fail(_)));
    assert!(matches!(disk(GIB, Some(free), 1), Check::Fail(_)));
    assert!(matches!(disk(GIB, None, 0), Check::Unverified(_)));
}

#[test]
fn provenance_is_the_exact_allowlist() {
    let names: BTreeSet<String> = ["qwen3.8-27b-jang_6d-crack".to_string()].into();
    assert!(provenance("dealignai/X", &names).passed());
    assert!(matches!(provenance("DealignAI/X", &names), Check::Fail(_)));
    let mirror = provenance("someone/Qwen3.8-27B-JANG_6D-CRACK", &names);
    assert!(mirror.text().starts_with("mirror"));
    assert!(
        provenance("someone/Other", &names)
            .text()
            .contains("not on the publisher allowlist")
    );
}

#[test]
fn format_names_what_it_accepts_and_why_it_drops() {
    let w = files(&[("model.safetensors", 10)]);
    let jang = files(&[("model.safetensors", 10), ("jang_config.json", 1)]);
    let optiq = files(&[("model.safetensors", 10), ("optiq_metadata.json", 1)]);
    let q = |mode: &str, bits: u64| json!({"model_type": "qwen3_5", "quantization": {"mode": mode, "bits": bits}});
    assert_eq!(
        format(&w, &q("affine", 4), &served()).text(),
        "MLX 4-bit qwen3_5"
    );
    assert_eq!(
        format(&jang, &q("mxfp8", 8), &served()).text(),
        "JANG mxfp8 qwen3_5"
    );
    assert_eq!(
        format(&optiq, &q("affine", 4), &served()).text(),
        "OptiQ 4-bit qwen3_5"
    );
    assert_eq!(
        format(&w, &json!({"model_type": "qwen3_5"}), &served()).text(),
        "unquantized qwen3_5"
    );
    assert!(matches!(
        format(&w, &q("nvfp4", 4), &served()),
        Check::Fail(_)
    ));
    let gptq = json!({"model_type": "qwen3_5", "quantization_config": {"quant_method": "gptq"}});
    assert_eq!(
        format(&w, &gptq, &served()),
        Check::Fail("gptq quantisation".into())
    );
    assert_eq!(
        format(&files(&[("m.gguf", 9)]), &q("affine", 4), &served()),
        Check::Fail("GGUF".into())
    );
    assert!(
        format(&files(&[("README.md", 1)]), &json!({}), &served())
            .text()
            .contains("no safetensors")
    );
    assert!(matches!(
        format(&w, &json!({}), &served()),
        Check::Unverified(_)
    ));
    let other = json!({"model_type": "llama"});
    assert!(
        format(&w, &other, &served())
            .text()
            .contains("not one Osaurus is shown to load")
    );
}

#[test]
fn sizes_count_weights_and_downloads_apart() {
    let f = files(&[
        ("a.safetensors", 3),
        ("b.safetensors", 4),
        ("tokenizer.json", 5),
    ]);
    assert_eq!(weights_bytes(&f), 7);
    assert_eq!(download_bytes(&f), 12);
    assert_eq!(gib(3 * GIB / 2), "1.5");
    assert_eq!(
        (author("a/b/c"), name("a/b/c"), author("solo")),
        ("a", "b/c", "solo")
    );
    assert_eq!(
        [
            Check::Pass(String::new()).glyph(),
            Check::Fail(String::new()).glyph()
        ],
        ["✓", "✗"]
    );
}

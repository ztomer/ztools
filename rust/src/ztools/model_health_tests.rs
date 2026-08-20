#[cfg(test)]
mod tests {
    use crate::ztools::model_health::*;
    use std::fs;
    use std::path::Path;
    use tempfile::tempdir;

    #[test]
    fn test_clean_model_has_no_defects() {
        let temp = tempdir().unwrap();
        let model_dir = temp.path().join("CleanModel");
        fs::create_dir_all(&model_dir).unwrap();

        fs::write(model_dir.join("config.json"), "{}").unwrap();
        fs::write(model_dir.join("model-00001.safetensors"), b"weights_data").unwrap();

        let defects = probe_model_dir_defects(&model_dir);
        assert!(defects.is_empty(), "Clean model should have no defects");

        let res = assess_viability("CleanModel", Some(15.0), Some(temp.path()));
        assert!(res.is_ok());
    }

    #[test]
    fn test_nested_org_model_discovery() {
        let temp = tempdir().unwrap();
        let org_dir = temp.path().join("OsaurusAI");
        let model_dir = org_dir.join("DeepNestedModel");
        fs::create_dir_all(&model_dir).unwrap();

        let found = find_model_dir("DeepNestedModel", Some(temp.path()));
        assert!(found.is_some());
        assert_eq!(found.unwrap(), model_dir);

        let not_found = find_model_dir("NonExistent", Some(temp.path()));
        assert!(not_found.is_none());

        let non_existent_root = find_model_dir("Any", Some(Path::new("/non/existent/path")));
        assert!(non_existent_root.is_none());
    }

    #[test]
    fn test_unsupported_mtp_shards_with_runtime_false_detected() {
        let temp = tempdir().unwrap();
        let model_dir = temp.path().join("BrokenMTPModel");
        fs::create_dir_all(&model_dir).unwrap();

        fs::write(model_dir.join("config.json"), "{}").unwrap();
        fs::write(
            model_dir.join("jang_config.json"),
            r#"{"runtime_available": false, "mtp_mode": "preserved_enabled"}"#,
        )
        .unwrap();
        fs::write(
            model_dir.join("model-mtp-00001.safetensors"),
            b"mtp_weights",
        )
        .unwrap();

        let defects = probe_model_dir_defects(&model_dir);
        assert_eq!(defects.len(), 1);
        assert!(defects[0].contains("unsupported MTP speculative drafting"));

        let res = assess_viability("BrokenMTPModel", Some(0.1), Some(temp.path()));
        assert!(res.is_err());
        assert!(res.unwrap_err().contains("broken:"));
    }

    #[test]
    fn test_unintegrated_standalone_mtp_detected() {
        let temp = tempdir().unwrap();
        let model_dir = temp.path().join("StandaloneMTP");
        fs::create_dir_all(&model_dir).unwrap();

        fs::write(
            model_dir.join("model-mtp-00001.safetensors"),
            b"mtp_weights",
        )
        .unwrap();

        let defects = probe_model_dir_defects(&model_dir);
        assert_eq!(defects.len(), 1);
        assert!(defects[0].contains("unintegrated MTP speculative shard(s)"));
    }

    #[test]
    fn test_missing_safetensor_shards_detected() {
        let temp = tempdir().unwrap();
        let model_dir = temp.path().join("MissingShardsModel");
        fs::create_dir_all(&model_dir).unwrap();

        let index_json = r#"{
            "weight_map": {
                "layer.0": "model-00001-of-00002.safetensors",
                "layer.1": "model-00002-of-00002.safetensors"
            }
        }"#;
        fs::write(model_dir.join("model.safetensors.index.json"), index_json).unwrap();
        fs::write(
            model_dir.join("model-00001-of-00002.safetensors"),
            b"shard1",
        )
        .unwrap();

        let defects = probe_model_dir_defects(&model_dir);
        assert_eq!(defects.len(), 1);
        assert!(defects[0].contains("missing 1 safetensor shard(s)"));
        assert!(defects[0].contains("model-00002-of-00002.safetensors"));
    }

    #[test]
    fn test_incomplete_download_artifacts_detected() {
        let temp = tempdir().unwrap();
        let model_dir = temp.path().join("HalfDownloadedModel");
        let cache_dir = model_dir.join(".cache");
        fs::create_dir_all(&cache_dir).unwrap();

        fs::write(
            model_dir.join("model-00001.safetensors.incomplete"),
            b"partial",
        )
        .unwrap();
        fs::write(cache_dir.join("download.incomplete"), b"partial_cache").unwrap();

        let defects = probe_model_dir_defects(&model_dir);
        assert_eq!(defects.len(), 1);
        assert!(defects[0].contains("incomplete download artifacts (2 .incomplete"));
    }

    #[test]
    fn test_decode_thrashing_rate_fails_viability() {
        let temp = tempdir().unwrap();
        let model_dir = temp.path().join("SlowModel");
        fs::create_dir_all(&model_dir).unwrap();

        let res = assess_viability("SlowModel", Some(0.2), Some(temp.path()));
        assert!(res.is_err());
        assert!(res.unwrap_err().contains("thrashing"));
    }

    #[test]
    fn test_probe_model_defects_non_existent_returns_empty() {
        let defects = probe_model_defects("NonExistentModel999", None);
        assert!(defects.is_empty());
    }
}

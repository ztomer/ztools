//! Model-specific prompt modifications.
//!
//! Ported from `lib/llm/quirks.py` + the quirk constants in
//! `lib/llm/constants.py`, bug-for-bug: a substitute model is a different
//! family, and the eval measures what production actually sends, so this port
//! must reproduce the Python behaviour EXACTLY -- including its dead branches.
//!
//! Known oddity carried over deliberately: `GEMMA4_FAMILY` is `"gemma4"` but
//! `_get_model_family` matches against `MODEL_FAMILIES` which contains
//! `"gemma"`, so a model named `gemma-4-e2b` resolves to family `"gemma"` and
//! the gemma4 system-prompt quirk never fires. The Python eval has run this way
//! in production; "fixing" it here would make the Rust eval measure different
//! prompts than the reference.

use serde_json::{json, Value};

pub const MODEL_FAMILIES: &[&str] = &[
    "qwopus",
    "qwen",
    "gemma",
    "nemotron",
    "laguna",
    "foundation",
];
const QWEN_FAMILY: &str = "qwen";
const GEMMA4_FAMILY: &str = "gemma4";

const QWEN_TRIGGER_PREFIX: &str = "Output JSON now";
const QWEN_TRIGGER_TEXT: &str = "Output JSON now.\n\n";
const NO_JSON_KW: &str = "no json";
const PLAIN_TEXT_KW: &str = "plain text";
const GEMMA4_PREFIX_KW: &str = "JSON";
const GEMMA4_PREFIX_IMPORTANT: &str = "IMPORTANT";
const GEMMA4_TRIGGER_TEXT: &str = "IMPORTANT: This is DATA EXTRACTION. Output JSON only. ";

const USER_REPLACE_EXECUTE: &str = "execute";
const USER_REPLACE_CONTEXT: &str = "context";
const REPLACE_SRC_CONTEXT: &str = "Current Context";
const REPLACE_TGT_CONTEXT: &str = "Data";
const REPLACE_SRC_TASK: &str = "Execute the task";
const REPLACE_TGT_TASK: &str = "Extract to JSON";
const REPLACE_SRC_TASK_BASED: &str = "Execute the task based on";
const REPLACE_TGT_TASK_BASED: &str = "Extract";

/// Extract model family from full model name.
pub fn get_model_family(model: &str) -> &'static str {
    let model_lower = model.to_lowercase();
    MODEL_FAMILIES
        .iter()
        .copied()
        .find(|family| model_lower.contains(family))
        .unwrap_or("default")
}

/// Apply model-specific prompt modifications to chat messages.
///
/// Multimodal messages carry a LIST of content parts rather than prose; every
/// rewrite below is a string operation, so non-string content passes through
/// untouched.
pub fn apply_model_quirks(messages: &[Value], model: &str) -> Vec<Value> {
    let family = get_model_family(model);

    messages
        .iter()
        .map(|msg| {
            let Some(content) = msg.get("content").and_then(|c| c.as_str()) else {
                return msg.clone();
            };
            let role = msg.get("role").and_then(|r| r.as_str()).unwrap_or("user");
            let mut content = content.to_string();

            if family == QWEN_FAMILY && role == "system" {
                // Prepend JSON trigger to prevent thinking output; skip when
                // the prompt already opts out of JSON or asks for plain text.
                if !content.is_empty() && !content.starts_with(QWEN_TRIGGER_PREFIX) {
                    let lower = content.to_lowercase();
                    if !lower.contains(NO_JSON_KW) && !lower.contains(PLAIN_TEXT_KW) {
                        content = format!("{QWEN_TRIGGER_TEXT}{content}");
                    }
                }
            } else if family == GEMMA4_FAMILY && role == "system" {
                // Gemma4 needs extraction framing. See module doc: currently
                // unreachable for models named like `gemma-4-...`.
                if content.to_uppercase().contains(GEMMA4_PREFIX_KW)
                    && !content.starts_with(GEMMA4_PREFIX_IMPORTANT)
                {
                    content = format!("{GEMMA4_TRIGGER_TEXT}{content}");
                }
            }

            if role == "user" {
                // Models respond badly to "Execute"/"Context"; use Data/Extract.
                let lower = content.to_lowercase();
                if lower.contains(USER_REPLACE_EXECUTE) || lower.contains(USER_REPLACE_CONTEXT) {
                    content = content
                        .replace(REPLACE_SRC_CONTEXT, REPLACE_TGT_CONTEXT)
                        .replace(REPLACE_SRC_TASK, REPLACE_TGT_TASK)
                        .replace(REPLACE_SRC_TASK_BASED, REPLACE_TGT_TASK_BASED);
                }
            }

            let mut updated = msg.clone();
            updated["content"] = json!(content);
            updated
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn msgs(items: &[(&str, &str)]) -> Vec<Value> {
        items
            .iter()
            .map(|(role, content)| json!({"role": role, "content": content}))
            .collect()
    }

    #[test]
    fn qwen_system_prompts_get_the_json_trigger() {
        let out = apply_model_quirks(&msgs(&[("system", "Extract events.")]), "qwen3.8-27b-8bit");
        assert!(
            out[0]["content"]
                .as_str()
                .unwrap()
                .starts_with("Output JSON now.\n\n"),
            "{out:?}"
        );
        // Non-system messages are untouched.
        let out = apply_model_quirks(&msgs(&[("user", "Extract events.")]), "qwen3.8-27b-8bit");
        assert_eq!(out[0]["content"], json!("Extract events."));
    }

    #[test]
    fn qwen_trigger_skipped_when_prompt_opts_out() {
        for content in ["Answer with no json output.", "Give plain text."] {
            let out = apply_model_quirks(&msgs(&[("system", content)]), "qwen3.8-27b");
            assert_eq!(out[0]["content"], json!(content), "{content}");
        }
        // Already-triggered prompts are not double-prefixed.
        let out = apply_model_quirks(
            &msgs(&[("system", "Output JSON now\nDo it.")]),
            "qwen3.8-27b",
        );
        assert_eq!(out[0]["content"], json!("Output JSON now\nDo it."));
    }

    #[test]
    fn user_role_word_swaps_apply_to_any_family() {
        let out = apply_model_quirks(
            &msgs(&[(
                "user",
                "Current Context: ... Execute the task based on the text.",
            )]),
            "gemma-4-e2b-it-8bit",
        );
        // Replace order matches Python: "Execute the task" fires before
        // "Execute the task based on", so the longer rule never sees its match.
        assert_eq!(
            out[0]["content"],
            json!("Data: ... Extract to JSON based on the text.")
        );
    }

    #[test]
    fn multimodal_content_passes_through_untouched() {
        let multimodal = vec![json!({
            "role": "user",
            "content": [
                {"type": "text", "text": "Execute the task"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,x"}}
            ]
        })];
        let out = apply_model_quirks(&multimodal, "qwen3.8-27b");
        assert_eq!(out[0], multimodal[0]);
    }

    #[test]
    fn gemma4_system_quirk_is_dead_for_gemma_dash_4_names() {
        // Carried over from Python bug-for-bug: family resolution matches
        // "gemma", not "gemma4", so this branch never fires in production.
        // The test PINS that behaviour so a silent change is visible.
        let out = apply_model_quirks(
            &msgs(&[("system", "JSON extraction task.")]),
            "gemma-4-e2b-it-8bit",
        );
        assert_eq!(out[0]["content"], json!("JSON extraction task."));
    }
}

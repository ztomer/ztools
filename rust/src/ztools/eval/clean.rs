//! Model-output cleaning: thinking-block removal, stats-token stripping and
//! markdown-fence handling. Ported from `lib/content_processing.py` so the Rust
//! eval path parses the same cleaned text the Python eval does.

use regex::Regex;

const THINK_RE: &str = r"(?s)<think>.*?</think>";
const GEMMA_THOUGHT_MATCH_RE: &str = r"(?s)<\|channel>thought.*?<channel\|>";
const GEMMA_THOUGHT_UNMATCH_RE: &str = r"(?s)<channel\|>thought\b[^<]*";
const GEMMA_CHANNEL_RE: &str = r"(?s)<\|channel>.*";
const CHANNEL_RE: &str = r"<channel\|>";
const THINKING_MARKER_RE: &str = r"<\|.*?\|>";
const GEMMA_INTERNAL_RE: &str = r"<\|channel\|[^|]*\|>";
const OUTPUT_MARKER_RE: &str = r"(?i)(?:Output Generation|Output|Final Answer|Response|Proceeds|I will now generate|I'll now generate|Let's draft|Draft)\s*[\.\:]\s*";
const SELF_CORRECTION_RE: &str = r"(?s)\n?\*?\[?\(?[Ss]elf-[Cc]orrection.*";
const JSON_START_RE: &str = r"[\[{]";
const TRAILING_STATS_RE: &str = r"\n*stats:\d+([;.]\d+)?\s*$";

const GEMMA_CORRECTION_LOOP_RE: &str = r"(\s*Let'?s? pick [^\n]+\? No\.){3,}";
const BLANK_JSON_START_RE: &str = r"\n\s*\n\s*([\[{])";

const STATS1_RE: &str = r"stats:\d+;[\d.]+";
const STATS2_RE: &str = r"\d+;[\d.]+$";
const STATS3_RE: &str = r"stats:.*$";

const MD_BLOCK1_RE: &str = r"```(?:\w+)?\s*\n?";
const MD_BLOCK2_RE: &str = r"```\s*";

const CODE_BLOCK_RE: &str = r"(?s)```(?:\w+)?\s*(.*?)```";

const QWEN_THINKING_MARKERS: &[&str] = &[
    "Here's a thinking process:",
    "Thinking Process:",
    "Here is my thinking process:",
    "Let me think",
    "Let me carefully",
    "Let me analyze",
];

const JSON_SEARCH_PREVIEW_LIMIT: usize = 2000;
const UNICODE_REPLACEMENT_CHAR: char = '\u{FFFE}';
const THINKING_END_TAG: &str = "</think>";
const THINKING_PREFIX_MARKER: &str = "Think:";

fn compile(re: &str) -> Regex {
    Regex::new(re).expect("static regex")
}

/// Remove ` thinking... response`, Gemma channel-thought loops and other
/// model thinking artifacts. Port of `remove_thinking_blocks`.
pub fn remove_thinking_blocks(content: &str) -> String {
    if content.is_empty() {
        return String::new();
    }
    let mut c = content.to_string();

    c = compile(THINK_RE).replace_all(&c, "").into_owned();
    c = compile(GEMMA_THOUGHT_MATCH_RE).replace_all(&c, "").into_owned();
    c = compile(GEMMA_THOUGHT_UNMATCH_RE).replace_all(&c, "").into_owned();
    c = compile(GEMMA_CHANNEL_RE).replace_all(&c, "").into_owned();
    c = compile(CHANNEL_RE).replace_all(&c, "").into_owned();
    c = compile(THINKING_MARKER_RE).replace_all(&c, "").into_owned();
    c = compile(GEMMA_INTERNAL_RE).replace_all(&c, "").into_owned();

    for marker in QWEN_THINKING_MARKERS {
        let Some(marker_idx) = c.find(marker) else {
            continue;
        };
        if let Some(output_match) = compile(OUTPUT_MARKER_RE).find(&c) {
            c = c[output_match.end()..].to_string();
            c = compile(SELF_CORRECTION_RE).replace_all(&c, "").into_owned();
        } else if let Some(json_match) = compile(JSON_START_RE).find(&c[marker_idx..]) {
            c = c[marker_idx + json_match.start()..].to_string();
        }
        break;
    }

    if c.contains(THINKING_END_TAG) {
        c = c.split(THINKING_END_TAG).last().unwrap_or("").to_string();
    } else if c.contains(THINKING_PREFIX_MARKER) {
        c = c
            .split(THINKING_PREFIX_MARKER)
            .last()
            .unwrap_or("")
            .to_string();
    }

    c = compile(TRAILING_STATS_RE).replace_all(&c, "").into_owned();
    c.trim().to_string()
}

/// Remove verbose inline chain-of-thought that precedes a JSON/plain answer.
/// Port of `remove_inline_thinking`.
pub fn remove_inline_thinking(content: &str) -> String {
    if content.is_empty() {
        return content.to_string();
    }
    let mut c = compile(GEMMA_CORRECTION_LOOP_RE)
        .replace_all(content, " [reasoning truncated]")
        .into_owned();

    let first_json = match (c.find('['), c.find('{')) {
        (Some(a), Some(b)) => a.min(b),
        (Some(a), None) => a,
        (None, Some(b)) => b,
        (None, None) => usize::MAX,
    };
    if first_json > JSON_SEARCH_PREVIEW_LIMIT {
        if let Some(caps) = compile(BLANK_JSON_START_RE).captures(&c) {
            if let Some(group) = caps.get(1) {
                c = c[group.start()..].to_string();
            }
        }
    }
    c.trim().to_string()
}

/// Strip `stats:...` tokens and control characters. Port of `remove_stats_tokens`.
pub fn remove_stats_tokens(content: &str) -> String {
    if content.is_empty() {
        return String::new();
    }
    let mut c = content.to_string();
    c = compile(STATS1_RE).replace_all(&c, "").into_owned();
    c = compile(STATS2_RE).replace_all(&c, "").into_owned();
    c = compile(STATS3_RE).replace_all(&c, "").into_owned();
    c = c.replace(UNICODE_REPLACEMENT_CHAR, "");
    c.trim().to_string()
}

/// Remove markdown code-block fence markers. Port of `remove_markdown_blocks`.
pub fn remove_markdown_blocks(content: &str) -> String {
    if content.is_empty() {
        return String::new();
    }
    let mut c = content.to_string();
    c = compile(MD_BLOCK1_RE).replace_all(&c, "").into_owned();
    c = compile(MD_BLOCK2_RE).replace_all(&c, "").into_owned();
    c.trim().to_string()
}

/// Extract content from markdown code blocks if present (last block wins).
pub fn extract_content_from_code_blocks(content: &str) -> Option<String> {
    if content.is_empty() {
        return None;
    }
    compile(CODE_BLOCK_RE)
        .captures_iter(content)
        .last()
        .and_then(|caps| caps.get(1))
        .map(|m| m.as_str().trim().to_string())
}

/// Comprehensive cleanup in the Python order: thinking blocks, inline
/// reasoning, stats tokens, markdown fences.
pub fn clean_model_output(content: &str) -> String {
    if content.is_empty() {
        return String::new();
    }
    let mut c = remove_thinking_blocks(content);
    c = remove_inline_thinking(&c);
    c = remove_stats_tokens(&c);
    c = remove_markdown_blocks(&c);
    c.trim().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn removes_think_tags() {
        assert_eq!(
            remove_thinking_blocks("prefix <think>inner</think> suffix"),
            "prefix  suffix"
        );
    }

    #[test]
    fn qwen_marker_with_output_tag() {
        assert_eq!(
            remove_thinking_blocks(
                "Here's a thinking process: I should analyze this. Output: The answer is 42."
            ),
            "The answer is 42."
        );
    }

    #[test]
    fn qwen_marker_without_output_tag_finds_json() {
        assert_eq!(
            remove_thinking_blocks(
                "Here's a thinking process: Let me think about this. {\"key\": \"value\"}"
            ),
            "{\"key\": \"value\"}"
        );
    }

    #[test]
    fn unmatched_end_think_tag() {
        assert_eq!(remove_thinking_blocks("keep this</think>discard this"), "discard this");
    }

    #[test]
    fn think_colon_split() {
        assert_eq!(remove_thinking_blocks("keep\nThink: discard"), "discard");
    }

    #[test]
    fn gemma_channel_blocks_removed() {
        assert_eq!(
            remove_thinking_blocks("hello <|channel>thought inside<channel|> world"),
            "hello  world"
        );
        assert_eq!(
            remove_thinking_blocks("hello <|channel|some_token|> world"),
            "hello  world"
        );
    }

    #[test]
    fn stats_tokens_removed() {
        assert_eq!(remove_stats_tokens("content\nstats:2114;97.2952"), "content");
        assert_eq!(remove_stats_tokens("stats:123;45.67"), "");
        assert_eq!(remove_stats_tokens("text\u{fffe}text"), "texttext");
        assert_eq!(remove_stats_tokens(""), "");
    }

    #[test]
    fn gemma_self_correction_collapsed() {
        let text = "Let's pick Zoo? No. Let's pick Park? No. Let's pick Museum? No. final";
        let result = remove_inline_thinking(text);
        assert!(result.contains("reasoning truncated"));
        assert!(!result.contains("Zoo"));
    }

    #[test]
    fn qwen_long_preamble_truncated_to_json() {
        let preamble = "A".repeat(2500);
        let text = format!("{preamble}\n\n{{\"key\": \"value\"}}");
        let result = remove_inline_thinking(&text);
        assert!(result.starts_with('{'));
    }

    #[test]
    fn short_preamble_untouched() {
        let text = "short preamble {\"key\": \"value\"}";
        assert_eq!(remove_inline_thinking(text), text);
    }

    #[test]
    fn code_block_extracted() {
        assert_eq!(
            extract_content_from_code_blocks("```json\n{\"key\": \"value\"}\n```"),
            Some("{\"key\": \"value\"}".to_string())
        );
        assert_eq!(extract_content_from_code_blocks(""), None);
    }

    #[test]
    fn full_pipeline() {
        assert_eq!(
            clean_model_output("<think>skip</think> body\nstats:100;50"),
            "body"
        );
        assert_eq!(clean_model_output(""), "");
    }
}
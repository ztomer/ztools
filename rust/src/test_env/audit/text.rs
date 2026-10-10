//! The DETECTORS: every question these gates ask is answered here, over text.
//!
//! Split out of `mod.rs` for the house 500-line cap, along the seam the file
//! already had. The split is a real one and not a size dodge: `mod.rs` holds
//! the GATES, which say what must be true of the crate, and this file holds
//! the ANSWERS, which are the only things that could be wrong about a snippet
//! of Rust. Keeping each detector beside the snippet that proves it can fire is
//! the point -- a detector and its calibration change together, and a
//! calibration that lives in another file is one that rots alone.
//!
//! Every detector here is a free function over TEXT, because a detector is only
//! evidence when it has been run against something that SHOULD trip it. None of
//! them parses Rust and none of them needs to: the question is always "does this
//! string contain this shape". The one place structure does matter -- where a
//! test body ends -- is [`test_bodies`], which is why that one blanks literals
//! rather than counting every brace it can see.

/// `env::var`, `env::var_os`, `set_var`, `remove_var` and `env_u64` called with
/// a literal name, plus the `*_ENV` seams those calls use indirectly.
pub(super) fn env_vars_read_by(text: &str) -> Vec<String> {
    let calls = ["var_os(", "var(", "set_var(", "remove_var(", "env_u64("];
    let mut found: Vec<String> = Vec::new();
    for line in text.lines() {
        let trimmed = line.trim();
        if calls.iter().any(|call| trimmed.contains(call)) {
            for capture in quoted_literals(trimmed) {
                if is_env_name(capture) {
                    push_unique(&mut found, capture.to_string());
                }
            }
        }
        // A `*_ENV` const exists only to NAME an environment variable, so its
        // declaration is a read the gate must know about. Named by suffix
        // rather than pattern-matched wholesale, because the crate also has
        // consts whose values are merely uppercase (`FAIL_PARSE`, `JSON`) and
        // those are not variables.
        if let Some(name) = declared_env_const(trimmed) {
            push_unique(&mut found, name.to_string());
        }
    }
    found
}

pub(super) fn quoted_literals(line: &str) -> Vec<&str> {
    line.split('"').skip(1).step_by(2).collect()
}

pub(super) fn is_env_name(candidate: &str) -> bool {
    !candidate.is_empty()
        && candidate
            .chars()
            .all(|c| c.is_ascii_uppercase() || c.is_ascii_digit() || c == '_')
        && !candidate.starts_with(|c: char| c.is_ascii_digit() || c == '_')
}

/// `pub const SOME_ENV: &str = "NAME";` → `Some("NAME")`.
pub(super) fn declared_env_const(line: &str) -> Option<&str> {
    let rest = line
        .strip_prefix("pub const ")
        .or_else(|| line.strip_prefix("const "))?;
    let (name, value) = rest.split_once(": &str = \"")?;
    let _ = name;
    if !name.ends_with("_ENV") {
        return None;
    }
    let value = value.split('"').next()?;
    is_env_name(value).then_some(value)
}

pub(super) fn push_unique(found: &mut Vec<String>, name: String) {
    if !found.contains(&name) {
        found.push(name);
    }
}

/// One `#[test]`'s body, in two forms.
///
/// `code` is the body with every literal and comment blanked (see
/// [`blank_out_literals`]), and is what a hazard is looked for IN; `source` is
/// the body as written, and is only ever used to name the test in a failure
/// message. Both halves earn their place. A scan over `source` fires on a doc
/// comment that MENTIONS `ZtoolsConfig {` -- which is exactly how this crate
/// documents what the gate looks for, in the files it watches -- and a message
/// written from `code` is a wall of spaces with the test name missing.
pub(super) struct Body {
    pub source: String,
    pub code: String,
}

/// One `#[test]` function's body: the attribute line through the brace that
/// closes the `fn`.
///
/// Counts braces over [`blank_out_literals`] rather than over the raw text, and
/// that is not a refinement -- it is the difference between the gate working and
/// the gate not working. A brace inside a string is not structure: a test
/// holding a JSON fixture such as `r#"{"choices":[...]}"#` unbalances the raw
/// count, the body "closes" at the end of that line, and every hazard BELOW it
/// is attributed to the test above. Measured on 2026-10-05: four unguarded
/// hazard tests in `weekend_phases_tests.rs` were invisible to this gate, and
/// one already-guarded test was reported as unguarded. A detector that cannot
/// see a test is worse than no detector, because it reports the count as zero.
pub(super) fn test_bodies(text: &str) -> Vec<Body> {
    let blanked = blank_out_literals(text);
    let blanked_lines: Vec<&str> = blanked.lines().collect();
    let source_lines: Vec<&str> = text.lines().collect();
    let mut bodies = Vec::new();
    let mut i = 0;
    while i < blanked_lines.len() {
        if blanked_lines[i].trim() != "#[test]" {
            i += 1;
            continue;
        }
        let start = i;
        let mut depth = 0i64;
        let mut seen_open = false;
        while i < blanked_lines.len() {
            for ch in blanked_lines[i].chars() {
                match ch {
                    '{' => {
                        depth += 1;
                        seen_open = true;
                    }
                    '}' => depth -= 1,
                    _ => {}
                }
            }
            i += 1;
            if seen_open && depth <= 0 {
                break;
            }
        }
        bodies.push(Body {
            source: source_lines[start..i].join("\n"),
            code: blanked_lines[start..i].join("\n"),
        });
    }
    bodies
}

/// The same text with every comment, string and char literal replaced by
/// spaces, newlines kept.
///
/// Blanking rather than deleting is what keeps the line numbering aligned with
/// the original, which is the only reason [`test_bodies`] can hand back the two
/// forms of the same body.
pub(super) fn blank_out_literals(text: &str) -> String {
    let chars: Vec<char> = text.chars().collect();
    let mut out = String::with_capacity(text.len());
    let mut state = Scan::Code;
    let mut i = 0;
    while i < chars.len() {
        let step = step(&chars, i, &mut state);
        let end = (i + step.consumed).min(chars.len());
        // A newline stays a newline even inside a blanked literal: the two forms
        // of a test body are sliced out of THIS text by line number, so a
        // newline that became a space would shift every boundary below it.
        for ch in &chars[i..end] {
            out.push(if step.blank && *ch != '\n' { ' ' } else { *ch });
        }
        i = end;
    }
    out
}

/// Where the scanner is: in code, or inside one of the three things whose
/// braces are not structure.
#[derive(PartialEq, Eq)]
enum Scan {
    Code,
    LineComment,
    BlockComment(usize),
    Str(char),
    Raw(usize),
    Char,
}

/// One step of the scanner: how many characters it consumed, and whether they
/// are code to keep or a literal to blank.
struct Step {
    consumed: usize,
    blank: bool,
}

impl Step {
    /// Code: copied through untouched.
    const fn keep() -> Self {
        Self {
            consumed: 1,
            blank: false,
        }
    }

    /// A literal or a comment: blanked, newlines excepted.
    const fn blank(consumed: usize) -> Self {
        Self {
            consumed,
            blank: true,
        }
    }
}

/// Advance one construct from `i`, blanking it into `state`'s buffer.
///
/// Every form a `.rs` file here can actually contain is handled: `//`, `/* */`
/// (nested, since Rust allows it), `"..."`, `'c'`, and the raw forms `r"..."` /
/// `r#"..."#` whose hash count is arbitrary. A detector that mis-read
/// `r##"..."##` would be the same bug one level down.
fn step(chars: &[char], i: usize, state: &mut Scan) -> Step {
    let c = chars[i];
    let next = chars.get(i + 1).copied();
    match state {
        Scan::Code => {
            if c == '/' && next == Some('/') {
                *state = Scan::LineComment;
                Step::blank(2)
            } else if c == '/' && next == Some('*') {
                *state = Scan::BlockComment(1);
                Step::blank(2)
            } else if let Some(hashes) = raw_hashes(chars, i) {
                *state = Scan::Raw(hashes);
                Step::blank(1 + hashes)
            } else if c == '"' {
                *state = Scan::Str('"');
                Step::blank(1)
            } else if c == '\'' && is_char_literal(chars, i) {
                *state = Scan::Char;
                Step::blank(1)
            } else {
                Step::keep()
            }
        }
        Scan::LineComment => {
            if c == '\n' {
                *state = Scan::Code;
            }
            Step::blank(1)
        }
        Scan::BlockComment(depth) => {
            if c == '/' && next == Some('*') {
                *depth += 1;
                Step::blank(2)
            } else if c == '*' && next == Some('/') {
                *depth -= 1;
                if *depth == 0 {
                    *state = Scan::Code;
                }
                Step::blank(2)
            } else {
                Step::blank(1)
            }
        }
        Scan::Str(quote) => {
            let escaped = c == '\\';
            let closes = !escaped && c == *quote;
            if closes {
                *state = Scan::Code;
            }
            Step::blank(if escaped { 2 } else { 1 })
        }
        Scan::Raw(hashes) => {
            let count = *hashes;
            if c == '"' && ends_raw(chars, i, count) {
                *state = Scan::Code;
                Step::blank(1 + count)
            } else {
                Step::blank(1)
            }
        }
        Scan::Char => {
            let escaped = c == '\\';
            if c == '\'' && !escaped {
                *state = Scan::Code;
            }
            Step::blank(if escaped { 2 } else { 1 })
        }
    }
}

/// The hash count of a raw string opening at `i`, or `None` when `r` here is
/// just an identifier — `result`, `r`, `r#` — and there is no `r#"…"#` at all.
fn raw_hashes(chars: &[char], i: usize) -> Option<usize> {
    if chars.get(i).copied() != Some('r') {
        return None;
    }
    let hashes = chars[i + 1..].iter().take_while(|ch| **ch == '#').count();
    (hashes > 0 || chars.get(i + 1) == Some(&'"')).then_some(hashes)
}

/// Whether a char literal — not a lifetime, not a label — opens at `i`.
///
/// `Option<&str>` where the string type has no room for a character; `&char` has
/// no room for the empty literal. That is the whole of the distinction and it is
/// why this is asked of the CHARS and not of a `&str` slice.
fn is_char_literal(chars: &[char], i: usize) -> bool {
    match (chars.get(i + 1).copied(), chars.get(i + 2).copied()) {
        // `''`: the "one thing" IS the closing quote. `'x'` and `'\n'`: quote,
        // one thing, closing quote. Either way the literal is complete within
        // three chars, and a lifetime (`'a:`, `'static`) cannot satisfy the
        // closing-quote position.
        (Some('\''), _) | (_, Some('\'')) => true,
        _ => false,
    }
}

/// Whether a `"` at `i` closes a raw string opened with `hashes` hashes.
fn ends_raw(chars: &[char], i: usize, hashes: usize) -> bool {
    i + hashes < chars.len() && chars[i + 1..=i + hashes].iter().all(|ch| *ch == '#')
}

/// What makes a test touch the operator's disk whether it means to or not.
///
/// `..Default::default()` is deliberately NOT a hazard: `RunnerConfig` and
/// `SearchHealth` and friends default to no filesystem location at all, so
/// matching the syntax flagged nine tests that cannot touch anything. What
/// makes a config dangerous is that ITS defaults name `~/`, so the two
/// spellings of `ZtoolsConfig` are the hazards and the syntax of struct
/// update is not.
pub(super) const HAZARDS: &[&str] = &[
    // Turns on the eval's signal recording, which WRITES to the store and the
    // output directory. This is the variable set `model_resolve_http.rs` got
    // half right.
    "record_signals: true",
    "ZtoolsConfig::default()",
    "ZtoolsConfig {",
    // The OTHER routes to the same defaults (2026-10-06). Deserialising a config
    // runs every `#[serde(default = ...)]` the file omits, and the checkout-derived
    // ones start from the running executable: `load_config` and a typed
    // `toml::from_str` reach exactly what `ZtoolsConfig::default()` reaches, and the
    // three `manifest` entry points ARE what it reaches. Found by a CI simulation in a
    // fresh clone, where a `load_config` test saw the clone's own `conf/` -- invisible
    // in the operator's checkout, whose derived root happens to equal the historical
    // default and is de-duplicated away.
    "load_config(",
    "::<ZtoolsConfig>",
    "config_paths(&",
    "checkout_roots()",
    "first_checkout_root()",
    // The functions that RESOLVE the operator's home (2026-10-10). A test that
    // calls one without the guard gets the real `~`, and the only thing keeping
    // it from writing there is that this particular test happens not to. A
    // retired test left `{"screen_name":"u","text":"t"}` in the real
    // `~/.twitter_summary_debug_cache.json` that way, and a fixture run's
    // summary became what the dashboard showed.
    "dirs::home_dir()",
    "twitter_store_dir()",
    "twitter_output_dir()",
    "weekend_store_dir()",
    "weekend_output_dir()",
    "debug_cache_path()",
    "settings_path()",
    "settings::retain(",
    // The LIVE file-summary render resolves its checkout from `ZTOOLS_EXE`,
    // `$HOME` and then the build's own checkout (`file_summary::live_root`).
    // Unguarded, it reads the home checkout while no peer sandbox is live and
    // this checkout while one is, so in a worktree whose listed files differ
    // from the home checkout's, two renders in one test disagreed (2026-10-10).
    "live_root()",
    "excerpts()",
    "render(FILE_SUMMARY_PROMPT",
    "render_both(FILE_SUMMARY_PROMPT",
];

pub(super) fn hazard_in(body: &str) -> Option<&'static str> {
    HAZARDS.iter().copied().find(|h| body.contains(h))
}

/// A `temp_dir().join(...)` whose argument is a fixed literal, so two
/// concurrent `cargo test` runs — or two agent sessions on one Mac — write the
/// same directory. `format!` with a pid suffix is the existing workaround and
/// is still flagged: `tempfile` is the fix.
pub(super) fn fixed_temp_paths(text: &str) -> Vec<(usize, String)> {
    text.lines()
        .enumerate()
        .filter(|(_, line)| line.contains("temp_dir()"))
        // A comment NAMING the old pattern is documentation, not a site: this
        // detector must not be silenced by rewriting prose, and it must not
        // fire on prose either.
        .filter(|(_, line)| !line.trim_start().starts_with("//"))
        .filter(|(_, line)| {
            let Some(start) = line.find("temp_dir().join(") else {
                return false;
            };
            let rest = &line[start + "temp_dir().join(".len()..];
            let Some(end) = rest.find(')') else {
                return false;
            };
            !rest[..end].contains("process::id")
        })
        .map(|(n, line)| (n + 1, line.trim().to_string()))
        .collect()
}

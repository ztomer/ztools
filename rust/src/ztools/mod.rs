//! Native Rust port of ztools subsystems: Twitter summarizer and Weekend planner.

pub mod embeddings;
pub mod image_renamer;
pub mod model_eval;
pub mod model_health;
pub mod twitter;
pub mod weekend;
pub mod weekend_cache;

#[cfg(test)]
mod model_health_tests;

use std::io::Write;
use std::process::{Command, Stdio};

/// Print formatted markdown to stdout, using `bat` if available for rich terminal rendering.
pub fn print_formatted_markdown(content: &str) {
    if let Ok(mut child) = Command::new("bat")
        .args(["-l", "md", "--color", "always", "--style", "plain"])
        .stdin(Stdio::piped())
        .spawn()
    {
        if let Some(mut stdin) = child.stdin.take() {
            if stdin.write_all(content.as_bytes()).is_ok() {
                drop(stdin);
                if child.wait().is_ok() {
                    return;
                }
            }
        }
    }
    print!("{}", content);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn print_formatted_markdown_does_not_panic() {
        // The function should not panic regardless of whether `bat` is present.
        print_formatted_markdown("# Hello world\n");
        print_formatted_markdown("");
    }
}

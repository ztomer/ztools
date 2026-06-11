#!/usr/bin/env python3
# /// script
# dependencies = ["playwright", "requests", "cryptography", "mlx-lm @ git+https://github.com/ml-explore/mlx-lm.git", "transformers", "pyyaml"]
# ///
"""twitter_summarizer — thin CLI entry point."""

from twitter.cli import main

if __name__ == "__main__":
    main()

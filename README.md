# ZTools

High-performance native Rust local LLM tools for [Osaurus](https://osaurus.ai/).

ZTools provides standalone command-line tools for Twitter timeline summarization, family weekend planning, image renaming, and model evaluation benchmarks—running entirely locally on Apple Silicon.

Detailed architectural deep-dive is available in [ARCHITECTURE.md](ARCHITECTURE.md).

---

## Prerequisites

| Requirement | Notes |
|------------|-------|
| **[Osaurus](https://osaurus.ai/)** server | **Hard runtime dependency.** ZTools communicates with an Osaurus (or OpenAI-compatible) server at `http://localhost:1337`. Start it via `osaurus serve &` or the Osaurus macOS menu app. |
| **Rust toolchain** | Required when building from source (`cargo` / `rustc`). |

---

## Installation (Homebrew)

```bash
# 1. Start Osaurus server (macOS 15+, Apple Silicon)
brew install --cask osaurus
osaurus serve &>/dev/null &

# 2. Install ZTools via Homebrew tap
brew tap ztomer/tap
brew install ztomer/tap/ztools
```

Installs native Rust binaries directly onto your `PATH`:

| Command | Alias | Description |
|---------|-------|-------------|
| `twitter` | `twitter-summarize` | Scrapes following timeline & generates executive summary |
| `weekend` | `weekend-plan` | Curates family weekend plans with weather & seasonal events |
| `rename_images` | `image-renamer` | Renames screenshots/photos using OCR and Vision LLMs |
| `oeval` | `model-eval` | Benchmarks local LLM performance & accuracy across 30 tasks |

### Local install door (`./install.sh`)

`brew install` is the released, network-verified path. To install the tree you
have checked out right now instead — a local build, no network, no tap — run:

```bash
./install.sh
```

Builds the Rust release and installs it (with the same subcommand aliases)
straight into the Homebrew bin, `$(brew --prefix)/bin` (`/opt/homebrew/bin`
here). This overwrites the prefix entry until the next `brew upgrade`;
`ZTOOLS_INSTALL_DIR=/elsewhere ./install.sh` targets a custom dir. These two
are the only install doors; for an uninstalled dev build use
`cargo run --manifest-path rust/Cargo.toml -- <subcommand>`.
| `ztools` | — | Unified native binary dispatcher for all subcommands |

From a checkout, `bin/ab_test` is the smoke + parity harness over the *installed* binary (every subcommand answers, the Rust suite, the collect-parity comparator).

---

## The Tools

### 1. Twitter Summarizer (`twitter`)

```bash
twitter                  # Scrape live timeline & summarize
twitter --use-cache      # Summarize cached tweets from last run
twitter --since 24h      # Fetch tweets from the last 24 hours
twitter --fetch-only     # Collect and cache tweets without summarizing
twitter --login          # Open browser window to sign in to x.com
twitter --debug          # Show browser window and verbose output
```

- **Session Discovery**: Automatically extracts authenticated `auth_token` and `ct0` cookies from Zen Browser, Firefox, LibreWolf, or Chrome.
- **Anti-Detect Headless Scraping**: Launches Camoufox to scroll the Following timeline and capture live GraphQL tweets.
- **Semantic Clustering**: Clusters related tweets via local embeddings before prompt synthesis to create structured topic categories.

### 2. Weekend Planner (`weekend`)

```bash
weekend
weekend --location "Vaughan/Toronto" --ages "13,10,6"
weekend --md-out ~/Documents/weekend_plan.md
```

- **Weather-Aware**: Fetches 3-day forecasts from Open-Meteo REST API.
- **Dual-Source Scraping**: Scrapes seasonal festivals and activities from DuckDuckGo (with automatic fallback to DuckDuckGo Lite).
- **4-Phase LLM Pipeline**: Condenses weather → Extracts candidate snippets → Drafts itinerary → Structures validated JSON.
- **Rule Enforcement**: Drops unsourced rows, ensures in-window dates, and applies venue exclusions from `conf/weekend.toml`.

### 3. Image Renamer (`rename_images`)

```bash
rename_images ~/Desktop/screenshots
rename_images /path/to/photos --dry-run
rename_images /path/to/photos --vlm-model qwen3.8-27b-8bit
```

- **Prompt Injection Defense**: Wraps OCR text in `<<<BEGIN_UNTRUSTED_DOCUMENT` delimiters to prevent prompt hijacking.
- **Vision Fallback**: Uses OpenAI-compatible base64 data URIs for images lacking OCR text.
- **Sanitization**: Generates clean, descriptive snake_case filenames.

### 4. Model Evaluator (`oeval`)

```bash
oeval                    # Run full model benchmark
oeval --model qwen3.8-27b-8bit
```

**Tasks:** `weekend_transient`, `weekend_fixed`, `summarize`, `filename`, `file_summary`, `taxes_anomalies`, `taxes_audit_readiness`, `taxes_synthesis`, `taxes_yoy_narrative`, `taxes_qa`, `taxes_slip_qa`, plus the mixed-signal, adversarial and vision variants — 30 in total; oeval prints the count it loaded.

- Evaluates models against 30 automated task suites (financial analysis, adversarial resistance, JSON extraction, entity grounding).
- Uses kernel-level GPU locking (`gpu_lock`) to synchronize access across concurrent tasks.

---

## Testing & Quality Gates

The implementation is Rust — the only runtime. The Python reference tree was retired
2026-09-13 once every behaviour was either ported (with the Python verdicts frozen as
goldens) or explicitly retired in `docs/PORT_PARITY.md`. No `.venv`, no `uv`, no
interpreter on the product path; `tools/*.py` are dev gates.

### Rust

```bash
cd rust
cargo test                          # 750 tests: 677 unit + 73 integration
cargo clippy --all-targets -- -D warnings
make coverage                       # the house coverage gate, floor 95% lines
```

Key suites:
- `tests/model_resolve_http.rs` — missing-model substitution over the wire (dead tag → roster → one retry, surfaced reason, quirk re-derivation)
- `tests/reasoning_retry.rs` — a REASONING overrun retries with a raised budget against a mock that only answers once it is raised
- `tests/transport_http.rs`, `tests/signals_prefill.rs` — wire format, stream guard, learned timeouts, capability recording
- `tests/validator_parity.rs`, `tests/weekend_parity.rs` — the Rust validators and corpus pipeline reproduce, byte for byte, the verdicts the Python implementation produced on its last run (frozen under `tests/fixtures/*/expected_python_*.json`)
- `src/ztools/eval/tasks_tests.rs`, `validators/mixed_text_tests.rs` — the eval task table and the mixed-signal scorers pinned to the Python table and verdicts
- `tests/gpu_lock_shell_parity.rs` — the Rust GPU lock and `tools/gpu_lock.sh` read each other's owner files

**Coverage**: floor **95% lines** (the house floor), measured **95.65%** by the house gate (`coverage_gate.sh`, one lcov export per test target, merged — a plain `cargo llvm-cov` reads ~92% because it counts generic instantiations separately). The residual uncovered code is live-process spawning (`login_live`, `collect_tweets_live` — a real Camoufox browser), the model-eval run loop (it takes the machine-wide GPU lock and needs a live model server), environment-absent branches, and assertion panic-format arms. The floor may only move up; re-baselining requires a stated reason in the diff.

### Shell tooling (dev)

```bash
python3 -m pytest tools/tests -q     # tools/gpu_lock.sh, the bash half of the GPU lock
```

### The gate is ONE list (`.gatesrc`)

`make ci` and `tools/gate.sh --full` — the pre-push hook — both delegate to the
same runner over the step list declared once in `.gatesrc`, so the hook can
never be weaker than CI. `tools/release.sh` runs it too, because it pushes with
`--no-verify` and would otherwise tag something nothing had checked.

The steps: the house Rust gate (fmt, clippy `-D warnings`, no `#[allow]`)
· `cargo audit` · the structural gate (native `goh`: emoji, 500-line cap,
conflict markers, shell lint, secrets; `vendor/` exempt) · `cargo test` · the
tools pytest · coverage at the 95% floor.

- **Pre-commit** (`.githooks/`): the structural gate over the staged files only.

GitHub Actions CI is disabled — the local gate is the gate of record.

Eval results live in `~/.config/ztools/` (`eval_results.csv`, `eval_history.json`, `eval_signals.json`, and the raw-answer archive under `outputs/`). To track:

```bash
git add -f ~/.config/ztools/eval_results.json ~/.config/ztools/eval_history.json
```

---

## Release

```bash
tools/release.sh            # bump patch from the latest tag (v2.1.7 -> v2.1.8)
tools/release.sh 2.2.0      # explicit version
```

The script runs `make ci`, syncs `rust/Cargo.toml` to the version, builds the tree from `git archive HEAD` (the bytes the tap will compile — an untracked path dependency fails here, not on the user's machine), runs `./install.sh` and checks the binary on PATH reports the version with every subcommand answering, tags HEAD with the `CHANGELOG.md` section for that version as the tag message (no section, no tag), pushes, computes the GitHub tarball's SHA256, and updates the Homebrew tap formula in `ztomer/homebrew-tap`. Requires `gh` authenticated.
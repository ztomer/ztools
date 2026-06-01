# Quality Eval Full Suite Findings

Run: 2026-05-31
Method: `lib/quality.py` — dimension-based quality scoring (3-4 dimensions per task, each 0-100, weighted composite)

## Models Tested

| # | Model | Config | Status |
|---|-------|--------|--------|
| 1 | foundation | full (3/3 prompts) | ✅ complete |
| 2 | laguna-xs.2-mxfp4 | full | ✅ complete |
| 3 | nemotron-3-nano-omni-30b-a3b-mxfp4 | full | ✅ complete |
| 4 | qwen3.6-27b-mxfp4 | full | ✅ complete |
| 5 | qwen3.6-27b-mxfp8-mtp | full | ✅ complete |
| 6 | qwen3.6-35b-a3b-mxfp4 | full | ✅ complete |
| 7 | qwopus3.6-27b-v2-mlx-4bit | full | ✅ complete |
| 8 | qwen3.6-35b-a3b-mxfp8-mtp | full | ⚠️ partial |
| 9 | minimax-m2.7-small-jangtq | full | ❌ unusable |
| 10 | gemma-4-e4b-it-4bit | partial (filename+summarize only) | ❌ unusable |
| 11 | gemma-4-e4b-it-8bit | partial (filename+summarize only) | ❌ didn't start |
| 12 | gemma-4-26b-a4b-it-mxfp4 | no configs | ⏭ skipped |

## Results by Model

### foundation
| Task | Score | Avg Time | Notes |
|------|-------|----------|-------|
| filename | **97.0%** | 0.6s | Near-perfect. Slight verbosity on long inputs |
| summarize | **88.0%** | 3.8s | Weak on Synthesis (52%) — no relationship/connecting language |
| file_summary | **91.6%** | 1.5s | All models tied at 91.6% (accuracy scorer penalizes extra detail) |

### laguna-xs.2-mxfp4
| Task | Score | Avg Time | Notes |
|------|-------|----------|-------|
| filename | **98.0%** | 3.1s | Excellent. Minor verbosity |
| summarize | **92.0%** | 4.1s | Best Synthesis of non-qwopus models |
| file_summary | **91.6%** | 1.1s | |

### nemotron-3-nano-omni-30b-a3b-mxfp4
| Task | Score | Avg Time | Notes |
|------|-------|----------|-------|
| filename | **84.0%** | 3.3s | **Leaks instruction text into output** (e.g. `'Here is the filename: login_error.png'`). Format penalty |
| summarize | **89.5%** | 8.1s | Weak Synthesis (58%) |
| file_summary | **91.6%** | 3.2s | |

### qwen3.6-27b-mxfp4
| Task | Score | Avg Time | Notes |
|------|-------|----------|-------|
| filename | **93.8%** | 4.0s | Filename too long for Quote case (68 chars) |
| summarize | **100%** | 25.3s | Perfect! Good Synthesis, good user mentions |
| file_summary | **91.6%** | 7.5s | |

### qwen3.6-27b-mxfp8-mtp
| Task | Score | Avg Time | Notes |
|------|-------|----------|-------|
| filename | **99.0%** | 6.4s | Excellent. Only minor verbosity on Event case |
| summarize | **100%** | 29.6s | Perfect! Best balance of quality and cost |
| file_summary | **91.6%** | 8.5s | |

### qwen3.6-35b-a3b-mxfp4
| Task | Score | Avg Time | Notes |
|------|-------|----------|-------|
| filename | **93.8%** | 4.5s | Same verbosity issue on Quote case |
| summarize | **94.0%** | 20.3s | Slight synthesis weakness on timeline |
| file_summary | **91.6%** | 5.6s | |

### qwopus3.6-27b-v2-mlx-4bit
| Task | Score | Avg Time | Notes |
|------|-------|----------|-------|
| filename | **98.2%** | ~40s | Best quality. BUT crashes 2/5 on cold start (40% failure rate) |
| summarize | **98.5%** | ~90s | Best Synthesis (94%). Rich connecting narrative |
| file_summary | **91.6%** | ~220s | **Extremely slow** |

### qwen3.6-35b-a3b-mxfp8-mtp (partial)
| Task | Score | Avg Time | Notes |
|------|-------|----------|-------|
| filename | **93.8%** | 44.8s | Same verbosity. Much slower than 27b variant |
| summarize | **0%** | — | **Consistent crashes** — returns empty output for all summarization tasks |
| file_summary | **0%** | — | **Consistent crash** — returns empty output |

### minimax-m2.7-small-jangtq
| Task | Score | Avg Time | Notes |
|------|-------|----------|-------|
| filename | **~25%** | ~400s | 3/5 generic outputs, 2/5 barely relevant (63%). **400s per call** |
| summarize | **0%** | — | Models crashes on complex tasks |
| file_summary | **0%** | 600s+ | Timed out after 600s |

### gemma-4-e4b-it-4bit / 8bit
Both models produce 0% on filename (empty responses). MLX backend may not support these models. gemma-4-e4b-it-8bit never produced a single valid response within timeout.

### gemma-4-26b-a4b-it-mxfp4
No model config prompts exist — model not yet integrated.

## Cross-Model Comparison (Fully Working Models)

| Model | Filename | Summarize | FileSum | Avg Speed | Reliable? |
|-------|----------|-----------|---------|-----------|-----------|
| **laguna-xs.2-mxfp4** | 98.0% | 92.0% | 91.6% | **2.8s** | ✅ 0 failures |
| **qwen3.6-27b-mxfp8-mtp** | 99.0% | 100.0% | 91.6% | **14.8s** | ✅ 0 failures |
| **foundation** | 97.0% | 88.0% | 91.6% | **1.5s** | ✅ 0 failures |
| **qwen3.6-27b-mxfp4** | 93.8% | 100.0% | 91.6% | 12.3s | ✅ 0 failures |
| **qwen3.6-35b-a3b-mxfp4** | 93.8% | 94.0% | 91.6% | 10.1s | ✅ 0 failures |
| nemotron-3-nano-omni | 84.0% | 89.5% | 91.6% | 4.4s | ⚠️ leaks instruction text |
| qwopus3.6-27b-v2-mlx | 98.2% | 98.5% | 91.6% | 52.1s | ❌ 40% crash rate |

## Key Findings

### Synthesis is the hardest dimension
All non-qwopus models score 52-58% on Synthesis. Qwopus scores 94%. This is the #1 quality gap. Synthesis requires:
- TL;DR summarization
- Narrative connecting words ("in response to", "the discussion shifted to")
- Relationship awareness between items

### file_summary Accuracy at 71.9% is a scorer artifact
ALL models produce 71.9% Accuracy because they add extra descriptive detail beyond the reference. The token-overlap scorer penalizes models for being MORE descriptive. If anything, models are adding value.

### Speed vs Quality Tradeoff
- **Fastest**: foundation (1.5s avg) — good filename (97%), okay summarize (88%)
- **Best balance**: qwen3.6-27b-mxfp8-mtp (14.8s avg) — near-perfect across all tasks
- **Best quality, but slow**: laguna-xs.2-mxfp4 (2.8s) — excellent filename+summarize balance
- **Avoid**: qwopus (52s avg, 40% crash), minimax (400s, 100% generic), nemotron (instruction leak)

### Model Reliability Issues
- **nemotron**: Leaks instruction text into filename output (e.g. "Here is the filename: ...")
- **qwopus**: 40% failure rate on cold start (empty output). 35% slower on second run (not cached)
- **qwen3.6-35b-a3b-mxfp8-mtp**: Consistently crashes on non-trivial tasks (summarize, file_summary return empty)
- **minimax**: Unusable — 400s+ per call, all outputs generic or empty
- **gemma-***: Unrecognized by MLX backend — all fail immediately

### Outstanding Questions
1. Why does qwen3.6-35b-a3b-mxfp8-mtp crash on summarize but not filename? Architecture issue (MoE layer handling?)
2. Is nemotron's instruction leak fixable with prompt change (like "Output ONLY the filename, no explanation")?
3. Should we add a "first-call reliability" metric for models used in interactive tools?

## Recommendations

### Interactive tools (ztools defaults)
Use **foundation** (1.5s avg, 97% filename, reliable).

### Batch/offline processing
Use **qwen3.6-27b-mxfp8-mtp** (14.8s avg, 99% filename, 100% summarize, 0 failures).

### Quality-critical applications
Use **laguna-xs.2-mxfp4** (2.8s avg, 98% filename, 92% summarize, no failures).

### Never use
minimax, gemma variants, qwen3.6-35b-a3b-mxfp8-mtp (for complex tasks), qwopus (unless quality is the only priority and you accept crashes).

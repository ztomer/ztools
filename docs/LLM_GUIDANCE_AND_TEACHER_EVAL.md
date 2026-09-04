# LLM Guidance & Teacher Evaluation for `ztools`

This document provides a deep, qualitative evaluation of `ztools` outputs across local models and establishes prompt distillation guidelines for smaller on-device LLMs (e.g. `gemma-4-e4b`, `qwen3.6-27b`, `foundation`) in `weekend` (`wk`) and `twitter` (`tw`).

---

## 1. Deep Qualitative Evaluation

### Failure of Automated Heuristics
The legacy evaluation scripts (`eval/validate.py`, `eval/report.py`) suffer from severe parser fragility:
- **Array Unwrapping Defect**: When models output a valid JSON array `[{"name": "..."}, ...]`, the script expects a dictionary wrapper `{"items": [...]}` or `{"events": [...]}`. It logs `Parsed N items... no items found` and assigns a **0% score** to 100% compliant outputs.
- **Cross-Domain Schema Leak**: The script evaluates event schema fields (`location`, `price`, `day`) against file summary outputs, marking comprehensive 29-file summaries as `partial (80%)`.

### Real Model Performance (Human / Teacher Inspection)

| Task | Smaller Model Behavior | Key Defect / Weakness | Teacher Prompt Remedy |
| :--- | :--- | :--- | :--- |
| **`weekend_transient`** | Generates 8-11 valid local GTA activities. | Tends to echo prompt constants (`$18-35`, `2-3 hours`) when snippet lacks price/duration. | Mandate empty string `""` for missing fields; supply 1 few-shot JSON example. |
| **`weekend_fixed`** | Emits real Toronto/Vaughan venues (Aga Khan Museum, McMichael Collection, Markham Museum). | Weather labels occasionally marked `indoor/outdoor` instead of clean single token `indoor` or `outdoor`. | Constrain `weather` enum values strictly to `"indoor"`, `"outdoor"`, or `"both"`. |
| **`twitter_summarize`** | Grouping under topic headers is strong. | Sometimes drops author attribution `(@username | Date Time)` or emits bare prose `at 14:30`. | Require bracket attribution `(@user \| Mon DD HH:MM)` on every summary bullet point. |
| **`filename`** | Concise, snake_case output (`financial_results_board_minutes`). | High fidelity; no structural issues. | Maintain current system prompt. |

---

## 2. Distillation & Guidance for Smaller Local LLMs

To maximize signal extraction and eliminate hallucinations on small local models, apply these 4 core prompt constraints in `conf/models/*.toml`:

### A. `weekend` (Event & Venue Extraction)
1. **Explicit Null Rule**:
   > *"If the source snippet does NOT state a price, age group, or duration, output `""` (empty string). NEVER invent `$18-35`, `$20-30`, or `2-3 hours`."*
2. **Schema Anchor**:
   > *"Output JSON matching EXACT schema: `{"transient_events": [{"name": "str", "location": "str", "target_ages": "str", "price": "str", "start_date": "str", "end_date": "str", "duration": "str", "weather": "str", "day": "str"}]}`."*
3. **Regional & Specificity Filter**:
   > *"Only extract SPECIFIC events at physical locations in the Greater Toronto Area (Vaughan, Toronto, Markham, Mississauga, Richmond Hill). Skip directory pages, guides, blog posts, and foreign venues."*

### B. `twitter` (Timeline Summarization)
1. **Uniform Attribution**:
   > *"Conclude every bullet point with the author and timestamp in exact bracket format: `(@username | Mon DD HH:MM)`."*
2. **Narrative Grouping**:
   > *"Group related tweets chronologically under `## Topic` headers. Use explicit narrative verbs ('announced', 'responded to', 'released') showing how tweets connect."*

---

## 3. Recommended Evaluation Strategy (LLM-as-a-Judge)

Replace regex-based string validators with a two-phase evaluation pipeline:
1. **Structural Validator (`lib/validators/json_validator.py`)**:
   - Normalize top-level JSON arrays automatically (`isinstance(data, list) -> {"items": data}`).
   - Verify non-emptiness (`len(items) > 0`).
2. **Teacher Judge (Gemini Evaluation)**:
   - Perform qualitative spot-checks evaluating:
     - **Fidelity**: No invented facts or fake dates.
     - **Constraint Adherence**: Location within target region, age appropriateness.
     - **Format Stability**: Bracket attribution format consistency.

# Mandatory Evaluation & Prompt Improvement Workflow

## Directive: Gemini Qualitative Evaluation Required on Every Run

Automated evaluation scripts (`eval/validate.py`, `eval/report.py`) provide rough structural checks, but suffer from rigid regex/schema assumptions. Therefore, **on every subsequent eval run, Gemini MUST perform a direct qualitative analysis of the raw model outputs and document the comparison against Gemini's production standards.**

---

## 1. Required Evaluation Steps for Future Runs

Whenever an evaluation run is conducted:
1. **Inspect Raw Artifacts**: Read `eval_results.json` directly to review exact text/JSON emitted by local LLMs across tasks (`weekend_transient`, `weekend_fixed`, `twitter_summarize`, `filename`, `file_summary`).
2. **Perform Line-by-Line Qualitative Comparison**:
   - Compare local model outputs against Gemini's ideal production standards on 4 key dimensions:
     - **Truthfulness & Provenance**: No invented names (e.g. Osaurus = dinosaurs), no fake prices/dates.
     - **Attribution Format**: Presence of bracket attributions `(@username | Mon DD HH:MM)`.
     - **Geographic & Semantic Accuracy**: Real GTA locations, correct indoor/outdoor weather classification.
     - **Sentinel Honesty**: Use of `""` / `"—"` for missing values instead of prompt-ordered constant placeholders (`$18-35`, `2-3 hours`).
3. **Record Findings**: Document the comparative evaluation in [`docs/DEEP_MODEL_VS_GEMINI_EVALUATION.md`](file:///Users/ztomer/Projects/ztools/docs/DEEP_MODEL_VS_GEMINI_EVALUATION.md) and update [`walkthrough.md`](file:///Users/ztomer/.gemini/antigravity/brain/77973c1a-8269-410d-813d-7dc536d0c853/walkthrough.md).

---

## 2. Actionable Blueprint: Steering Local LLMs to Maximum Quality

Based on qualitative evaluation of historic and recent model runs, use these 4 targeted interventions:

### Action 1: Eliminate Name-Based Hallucinations (Context Bounding)
- **Problem**: Small local models use token associations from file names (e.g. "Osaurus" -> "dinosaurs") when context is short.
- **Intervention**: Wrap input text in `<context>` XML tags and inject the prompt rule:
  > *"Rely ONLY on the provided `<context>` text. Do NOT infer functionality or domain from file names, words, or puns."*

### Action 2: Prevent Indoor/Outdoor Weather Label Inversions
- **Problem**: Models occasionally label High Park or Nature Walks as `"indoor"`.
- **Intervention**:
  - **Prompt Anchor**: Emphasize `"Venues with park, trail, nature, walk, garden, or festival in their name MUST be labeled 'outdoor'."`
  - **Code Enforcement**: `correct_weather_labels` in [`weekend/enforce.py`](file:///Users/ztomer/Projects/ztools/weekend/enforce.py#L172) checks both `INDOOR_MARKERS` and `OUTDOOR_MARKERS` to automatically correct inverted labels.

### Action 3: Prevent Vague Placeholders in Location Cells
- **Problem**: Models emit generic strings (`"Indoor venue"`, `"Outdoor venue"`, `"Park"`) when street addresses are omitted.
- **Intervention**: Prompt rule:
  > *"Copy the street address if present; if absent, output the city name (e.g. 'Toronto, ON' or 'Vaughan, ON'). NEVER output generic placeholders like 'Indoor venue' or 'Outdoor venue'."*

### Action 4: Guarantee Tweet Provenance & Timestamp Attributions
- **Problem**: Timeline summaries lose handles and timestamps when free-form text is generated.
- **Intervention**:
  - **Few-Shot Anchor**: Provide 2 explicit few-shot examples in `twitter/prompts.py` showing exact bracket format: `- @author description (@author | Jul 30 14:20).`
  - **Degraded Output Banner**: Trigger `DEGRADED OUTPUT` banner in `twitter/output.py` if attributions are missing.

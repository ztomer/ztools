# Weekend Planner - FINAL STATUS

## Working Configuration

### Results
- **Fixed: 10 items** ✅
- **Transient: 10 items** ✅ (model_eval shows 100%)
- **Time: 2-30 minutes** (normal, scraping is slow)

### Best Model
- **qwen3.6-35b-a3b-mxfp4** for both weekend_fixed and weekend_transient

## Lessons Learned

### 1. Gemma Weather Bug
All gemma models output weather data instead of events - NOT FIXABLE via prompts

### 2. "Infinite Loop" Misunderstanding
The script appeared stuck but was actually doing review scraping (~5s/item)
This is EXPECTED behavior for production mode

### 3. Version-Specific Configs
Different gemma versions need different top_keys due to different output structures
Created gemma_versions.yaml for per-model settings

### 4. Model Eval Updated
- Task names: `weekend_transient`, `weekend_fixed` (not json/detailed_json)
- Results cached in eval_results.json
- qwen3.6-35b-a3b-mxfp4: 100% on both tasks

## Files Fixed
- weekend_planner.py: Re-enabled scraper, fixed display labels
- model_eval.py: Renamed tasks to weekend_transient/weekend_fixed  
- MODEL_QUIRKS.md: Added known issues section
- lib/config.py: Version-specific config loading
- conf/models/gemma_versions.yaml: New version configs
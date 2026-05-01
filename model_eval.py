#!/usr/bin/env python3
"""
Model Evaluator - Test models on REAL-WORLD tasks from ZTools.
Evaluates local models against the actual prompts used in the tools.

Imports from eval_tasks/ for modularity:
- eval_tasks/__init__.py - task definitions
- eval_tasks/validators.py - validator wrappers
- eval_tasks/analyze.py - analysis & reporting functions
"""

import sys
import re
import json
import time
import statistics
import argparse
from typing import Tuple, Any, List, Dict
from rich.console import Console
from rich.table import Table

from lib import init_config
from lib.osaurus_lib import (
    call,
    get_models,
    is_server_running,
)

from lib.validators_lib import (
    validate_detailed_json,
    validate_summary,
    validate_filename,
    has_item_details,
    has_text_headers,
    count_content_lines,
)

from lib.validators_lib import (
    validate_detailed_json,
    validate_summary,
    validate_filename,
    has_item_details,
    has_text_headers,
    count_content_lines,
)

console = Console()


def safe_content(result: dict) -> str:
    """Safely extract content from a result dict, handling None values."""
    content = result.get("content")
    if content is None:
        return ""
    if not isinstance(content, str):
        return str(content)
    return content


def _extract_items_from_text(text: str) -> List[Dict]:
    """Extract structured items from text output (markdown tables, lists, etc)."""
    import re
    
    items = []
    
    # Pattern 1: Markdown tables | Name | Location |
    table_pattern = r'\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|'
    tables = re.findall(table_pattern, text)
    if tables and len(tables) >= 2:
        # Use first row as header
        header = tables[0]
        
        # Check if first row is header (has non-word chars like --- or ===)
        is_header_row = (
            '---' in header[0].lower() or 
            '---' in header[1].lower() or
            not any(c.isalnum() for c in header[0]) or
            not any(c.isalnum() for c in header[1])
        )
        data_rows = tables[1:] if is_header_row else tables
        
        if data_rows:
            key1 = header[0].strip().lower()
            key2 = header[1].strip().lower()
            
            # Map common headers to our field names
            header_map = {
                'name': 'name', 'event': 'name', 'title': 'name', 'activity': 'name',
                'location': 'location', 'venue': 'location', 'place': 'place', 'where': 'location',
                'day': 'day', 'date': 'day', 'when': 'day', 'time': 'time',
            }
            field1 = header_map.get(key1, 'name')
            field2 = header_map.get(key2, key2)
            
            for row in data_rows:
                # Skip separator rows (like |----|----|)
                if '---' in row[0].lower() or '---' in row[1].lower():
                    continue
                if not row[0].strip() or not row[1].strip():
                    continue
                # Skip if it looks like a header label
                row0_clean = row[0].strip().lower()
                row1_clean = row[1].strip().lower()
                if row0_clean in ['name', 'event', 'title', 'activity', 'location', 'venue', 'place', 'where'] or row1_clean in ['name', 'event', 'title', 'activity', 'location', 'venue', 'place', 'where']:
                    continue
                item = {field1: row[0].strip(), field2: row[1].strip()}
                items.append(item)
            
            if items:
                return items
    
    # Pattern 2: Lines with - or • bullets with "Key: Value" pairs
    bullet_pattern = r'^[•\-]\s*(.+?)(?:\n|$)'
    bullets = re.findall(bullet_pattern, text, re.MULTILINE)
    for bullet in bullets:
        bullet = bullet.strip()
        if bullet and len(bullet) > 2:
            parts = bullet.split(':', 1)
            if len(parts) == 2:
                key = parts[0].strip()
                val = parts[1].strip()
                # Map key to field name
                field_map = {
                    'name': 'name', 'event': 'name', 'title': 'name', 'activity': 'name',
                    'location': 'location', 'venue': 'location', 'place': 'location',
                }
                field = field_map.get(key.lower(), key.lower())
                items.append({field: val})
            else:
                # Try comma or hyphen separator
                sep_match = re.match(r'^([^,\-]+)[,\-](.+)$', bullet)
                if sep_match:
                    items.append({'name': sep_match.group(1).strip(), 'location': sep_match.group(2).strip()})
                else:
                    items.append({'name': bullet})
    
    return items


# ==========================================================
# File Summary Validator
# ==========================================================

def validate_file_summary(data: Any, source_text: str = "") -> Tuple[int, str]:
    """Validate file summary quality - checks for ACTUAL content detail, not filename inference.
    
    STRICT checks:
    - No filename-only summaries (must describe what file does)
    - No generic patterns like "a python script"
    - Must have actionable content about file purpose/function
    """
    if not data:
        return 0, "empty response"
    
    # Handle list input directly (already parsed from JSON)
    if isinstance(data, list):
        failures = []
        items = data
        num_files = len(items)
        if num_files < 4:
            failures.append(f"only {num_files} files")
        
        detailed_count = 0
        content_verbs = ['parse', 'validat', 'evaluat', 'extract', 'load', 'save',
            'read', 'write', 'fetch', 'send', 'process', 'handle',
            'config', 'setting', 'option', 'parameter', 'api', 'client', 'model', 'llm']
        
        for item in items:
            if not isinstance(item, dict):
                continue
            path = item.get("path", "")
            desc = item.get("desc", "") or item.get("summary", "")
            if not path or not desc:
                continue
            desc_lower = str(desc).lower()
            
            has_content = any(kw in desc_lower for kw in content_verbs)
            if has_content:
                detailed_count += 1
        
        # Scoring
        if num_files == 0:
            return 0, "no items"
        if detailed_count >= num_files * 0.8:
            score = 85
        elif detailed_count >= num_files * 0.5:
            score = 70
        elif detailed_count >= 2:
            score = 55
        elif detailed_count >= 1:
            score = 40
        else:
            score = 25
            failures.append("no content details")
        
        return min(100, score), "; ".join(failures) if failures else ""
    
    # Handle dict input (from JSON parsing)
    if isinstance(data, dict):
        data = json.dumps(data)
    
    data_str = str(data).strip()
    failures = []
    score = 0
    
    # Try to parse as JSON (string input from model output)
    parsed = None
    try:
        parsed = json.loads(data_str)
    except:
        pass
    
    if not parsed:
        # Fallback: check as prose/text
        if has_text_headers(data_str):
            score += 20
        if len(data_str) >= 200:
            score += 20
        if score < 40:
            failures.append("no headers")
        return min(100, max(score, 20)), "; ".join(failures)
    
    # Handle dict format
    items = list(parsed.items()) if isinstance(parsed, dict) else parsed
    num_files = len(items)
    
    detailed_count = 0
    content_verbs = ['parse', 'validat', 'evaluat', 'extract', 'load', 'save',
        'read', 'write', 'fetch', 'send', 'process', 'handle',
        'config', 'setting', 'option', 'parameter', 'api', 'client', 'model', 'llm']
    
    for filepath, summary in items:
        if not filepath or not summary:
            continue
        summary_lower = str(summary).lower()
        
        has_content = any(kw in summary_lower for kw in content_verbs)
        if has_content:
            detailed_count += 1
    
    # Scoring
    if detailed_count >= num_files * 0.8:
        score = 85
    elif detailed_count >= num_files * 0.5:
        score = 70
    elif detailed_count >= 2:
        score = 55
    elif detailed_count >= 1:
        score = 40
    else:
        score = 25
    
    if not detailed_count:
        failures.append("no content details")
    
    return min(100, score), "; ".join(failures) if failures else ""
    
    # Text format (Foundation/Gemma): check for ## headers OR prose content
    # Structure checks (40 points)
    if has_text_headers(data_str):
        score += 20
    elif len(data_str) >= 500:
        # Has content but no headers - allow prose format
        score += 15
        failures.append("no headers (prose format)")
    else:
        failures.append("no headers")
    
    # Length check (20 points)
    if len(data_str) >= 500:
        score += 20
    elif len(data_str) >= 200:
        score += 10
    else:
        failures.append(f"too short ({len(data_str)} chars)")
    
    # Quality checks: evidence of ACTUAL file reading
    # Filter out header-only lines (## filename only, no content after)
    content_lines = []
    for line in data_str.split('\n'):
        stripped = line.strip()
        # Skip empty lines
        if not stripped:
            continue
        # Skip pure headers without content (## filename or ## filename:)
        header_match = re.match(r'^##\s+\S+(?:\s*:?\s*)?$', stripped)
        if header_match:
            continue
        content_lines.append(stripped)
    
    # Content detail keywords (file-type agnostic)
    # Expanded to include Gemma-style content words
    detail_keywords = [
        'evaluat', 'parse', 'validat', 'extract', 'config', 'setting',
        'planning', 'summariz', 'renam', 'browser', 'playwright', 'ocr',
        'weekend', 'twitter', 'image', 'context', 'assistant', 'instruction',
        'overview', 'document', 'guideline', 'interaction', 'setup',
        'api', 'server', 'client', 'test', 'mock', 'request', 'response',
        'application', 'development', 'performance', 'behavior', 'quirk',
        'tool', 'utility', 'library', 'package', 'model', 'llm',
        'weekend', 'twitter', 'image', 'scrape', 'fetch',
    ]
    
    # Generic inference patterns - more accurate
    generic_patterns = [
        r'^a\s+python\s+script',
        r'^a\s+script\s+(for|to|of)',
        r'^a\s+tool\s+for',
        r'^a\s+utility\s+for',
        r'^an?\s+(exploration|investigation)\s+',
        r'^the\s+entry\s+point',
    ]
    
    detailed_lines = 0
    generic_lines = 0
    
    for line in content_lines:
        line_lower = line.lower()
        is_generic = any(re.match(p, line_lower.rstrip('.').strip()) for p in generic_patterns)
        has_detail = any(kw in line_lower for kw in detail_keywords)
        
        if is_generic:
            generic_lines += 1
        elif has_detail:
            detailed_lines += 1
    
    num_lines = len(content_lines)
    if num_lines > 0:
        detail_ratio = detailed_lines / num_lines
        if detail_ratio >= 0.7:
            score += 40
        elif detail_ratio >= 0.5:
            score += 30
        elif detail_ratio >= 0.3:
            score += 20
        elif detailed_lines >= 2:
            score += 10
        else:
            failures.append("filename inference only - no content detail")
    
    return min(100, score), "; ".join(failures) if failures else ""

WEEKEND_SYS_TRANSIENT = """
Output ONLY valid JSON array. No explanations, no preamble, no markdown.

Required format: [{"name": "...", "location": "...", "target_ages": "...", "price": "...", "weather": "...", "day": "..."}]

Default values if not in context:
- target_ages: "6-13 years"
- price: $20-30 or Free
- duration: "2-3 hours"
- weather: "indoor"
- day: Friday/Saturday/Sunday
"""

WEEKEND_SYS_FIXED = """
Output ONLY valid JSON array. No explanations, no preamble, no markdown.

Required format: [{"name": "...", "location": "...", "target_ages": "...", "price": "...", "weather": "..."}]

Default values if not in context:
- target_ages: "6-13 years"
- price: $20-30 or Free
- weather: "indoor"
- location: city name
"""
WEEKEND_USR_TRANSIENT = """
Current Context for the upcoming weekend:
Dates: April 20 to April 22, 2026
Friday: 15.0°C, Clear (0mm)
Saturday: 12.0°C, Precipitation (5mm)
Sunday: 14.0°C, Clear (0mm)

High-Signal Transient Events (Filter these strictly! Ensure they match the Dates provided!):
- Spring Festival at Downsview Park: Outdoor rides and games. April 20-22. All ages.
- Indoor Coding Workshop for Kids: Learn Python. April 21. Ages 8-14.
- Outdoor Movie Night: Watch a movie under the stars. April 21. All ages.
- Farmers Market at Maple Village: Fresh produce and local crafts. April 20. All ages.
- Pottery Wheel Workshop: Create clay art. April 22. Ages 12+.
- Puppet Show at Vaughan Library: "The Magical Forest". April 20. Ages 4-10.
- Kids Yoga in the Park: Morning yoga for families. April 20. Ages 5-12.
- Magic Show at Markham Theatre: Illusionist show. April 21. All ages.
- Nature Walk at Boyd Conservation: guided family hike. April 22. All ages.
- Board Game Marathon at Community Centre: Family games. April 21. All ages.
- Pizza Making Class: Learn to make pizza. April 22. Ages 8-16.
- Easter Egg Hunt at Raccoon Creek: Egg hunt and crafts. April 20. Ages 3-10.
"""
WEEKEND_USR_FIXED = """
Current Context for the upcoming weekend:
Dates: April 20 to April 22, 2026
Friday: 15.0°C, Clear (0mm)
Saturday: 12.0°C, Precipitation (5mm)
Sunday: 14.0°C, Clear (0mm)

Potential Venues and Current Exhibits:
- Vaughan Sports Arena: Indoor trampoline and dodgeball. All ages.
- High Park: Large outdoor playground and zoo. All ages.
- Aga Khan Museum: Islamic art and culture. Indoor. All ages.
- McMichael Canadian Art Collection: Canadian art exhibits. Indoor. All ages.
- Gibson Park: Playground and splash pad. Outdoor. Ages 3-12.
- Richmond Hill Centre for the Performing Arts: Live theater. Indoor. All ages.
- Maplewood Park Conservation: Hiking trails and picnic area. Outdoor. All ages.
- Ezra Avenue Skatepark: Skateboarding and BMX. Outdoor. Ages 10+.
- Oakridge Arts Festival: Art installations and workshops. April 20-22. Indoor/outdoor. All ages.
- Toronto Fun Zone: Indoor play centre with wall climb. Indoor. Ages 4-14.
- Lake Simcoe Sugar Bush: Maple syrup tours. Outdoor. All ages.
- Markham Museum: Heritage buildings and events. Indoor/outdoor. All ages.

Execute the task based on the system instructions and the provided context to find 10 year-round fixed activities, prioritizing current exhibits or highly-rated venues from the context. Output ONLY JSON.
"""

RENAME_PROMPT = """You are a file naming assistant. Read the following text extracted from an image and suggest a short, descriptive filename (without extension). Use lowercase, underscores for spaces, and no special characters other than hyphens/underscores. Keep it under 50 characters. Do NOT include any reasoning, thinking process, or introductory text. Output ONLY the final filename string, nothing else.

TEXT:
CONFIDENTIAL - Q3 2025 Financial Results & Board Meeting Minutes

Output ONLY the filename string (no quotes, no backticks, no JSON)."""

FILE_SUMMARY_PROMPT = """Read the file list below and give one-line summary for each file.

CRITICAL: DO NOT infer from filename. Describe what each file DOES, not what its filename suggests.
- Bad: "a python library" (infers from .py extension)
- Good: "parses web content and extracts metadata"

Use ## headers for each file (e.g., ## filename: summary).

/Users/ztomer/Projects/ztools/README.md
/Users/ztomer/Projects/ztools/CLAUDE.md
/Users/ztomer/Projects/ztools/model_eval.py
/Users/ztomer/Projects/ztools/weekend_planner.py
/Users/ztomer/Projects/ztools/twitter_summarizer.py
/Users/ztomer/Projects/ztools/image_renamer.py
/Users/ztomer/Projects/ztools/explore_model_quirks.py
/Users/ztomer/Projects/ztools/lib/__init__.py
/Users/ztomer/Projects/ztools/lib/osaurus_lib.py
/Users/ztomer/Projects/ztools/lib/validators_lib.py
/Users/ztomer/Projects/ztools/lib/config.py
/Users/ztomer/Projects/ztools/lib/content_processing.py
/Users/ztomer/Projects/ztools/lib/mlx_lib.py
/Users/ztomer/Projects/ztools/lib/logging_config.py
/Users/ztomer/Projects/ztools/conf/config.yaml
/Users/ztomer/Projects/ztools/conf/weekend.yaml
/Users/ztomer/Projects/ztools/conf/twitter.yaml
/Users/ztomer/Projects/ztools/conf/rename.yaml
/Users/ztomer/Projects/ztools/conf/models/foundation.yaml
/Users/ztomer/Projects/ztools/conf/models/gemma.yaml
/Users/ztomer/Projects/ztools/conf/models/qwen.yaml
/Users/ztomer/Projects/ztools/docs/MODEL_QUIRKS.md
/Users/ztomer/Projects/ztools/docs/PROJECT_MEMORY.md
/Users/ztomer/Projects/ztools/tests/test_validators.py
/Users/ztomer/Projects/ztools/tests/test_parse.py
/Users/ztomer/Projects/ztools/tests/test_config.py
/Users/ztomer/Projects/ztools/tests/test_weekend.py
/Users/ztomer/Projects/ztools/tests/test_content_processing.py
/Users/ztomer/Projects/ztools/tests/test_twitter.py
/Users/ztomer/Projects/ztools/pyproject.toml

Skip .git, __pycache__, benchmarks/, and pycache directories."""

TWITTER_PROMPT = """You are an objective news distillation system. Your task is to extract hard facts from the provided chronological Twitter/X timeline.

<instructions>
1. First, analyze the timeline in your <think> block.
2. Identify clusters of related events and synthesize duplicates.
3. Output ONLY the final briefing after the </think> tag. No introductory text.
</instructions>

<formatting_rules>
- Use headers starting with ##
- Use bullet points for facts
- Keep it concise and factual
- 40 tweets to analyze
</formatting_rules>

<timeline>
[@TechCrunch | 08:00]: OpenAI announces GPT-5 with advanced reasoning capabilities, available next month.
[@TheVerge | 08:15]: Apple Vision Pro 2 enters mass production, expected fall release.
[@TechCrunch | 08:30]: Google unveils Gemini 2.5 Pro with 1M context window.
[@Wired | 08:45]: NVIDIA stock hits all-time high after data center revenue beats estimates.
[@Bloomberg | 09:00]: Federal Reserve signals potential rate cut in June meeting.
[@LocalNews_TOR | 09:15]: TTC subway Line 1 delays due to signal problems at Sheppard West.
[@TechCrunch | 09:30]: Microsoft acquires AI startup for $2B to boost Azure AI capabilities.
[@TheVerge | 09:45]: Samsung Galaxy S25 Ultra features new titanium frame and AI camera.
[@CNBC | 10:00]: Oil prices drop 3% on increased production concerns.
[@TorontoStar | 10:15]: Mayor announces new bike lane infrastructure for downtown Toronto.
[@TechCrunch | 10:30]: Meta announces Llama 4 open source with commercial license.
[@Wired | 10:45]: SpaceX successfully launches 60 Starlink satellites on Falcon 9.
[@LocalNews_TOR | 11:00]: Highway 401 collision causes 2-hour delays westbound near Jane Street.
[@TheVerge | 11:15]: Sony PlayStation 6 prototype leaks, features 8K gaming support.
[@TechCrunch | 11:30]: Anthropic launches Claude 4 with improved coding capabilities.
[@Bloomberg | 11:45]: Bitcoin surges past $75K on ETF approval news.
[@LocalNews_TOR | 12:00]: Toronto Maple Leafs win playoff game, celebrations in downtown core.
[@Wired | 12:15]: Amazon launches drone delivery in select Toronto neighborhoods.
[@TechCrunch | 12:30]: Adobe acquires Figma for $20B in largest tech acquisition of year.
[@TheVerge | 12:45]: Tesla Cybertruck production ramps up to 10K units per week.
[@CNBC | 13:00]: US jobs report shows 250K new jobs, beating expectations.
[@LocalNews_TOR | 13:15]: Pearson Airport reports record spring break travel volumes.
[@TechCrunch | 13:30]: IBM unveils quantum computer with 1000+ qubit capability.
[@Wired | 13:45]: Nintendo confirms new Switch model launching holiday season.
[@Bloomberg | 14:00]: Shopify reports 40% revenue growth, stock jumps 15%.
[@LocalNews_TOR | 14:15]: Ontario Place undergoing major renovation, new spa opening 2026.
[@TechCrunch | 14:30]: Salesforce announces AI-powered CRM with autonomous agents.
[@TheVerge | 14:45]: Intel Core Ultra chips debut with breakthrough efficiency.
[@CNBC | 15:00]: Housing market cools as mortgage rates stay elevated.
[@LocalNews_TOR | 15:15]: Raptors playoff game tonight at Scotiabank Arena, expect crowds.
[@TechCrunch | 15:30]: Databricks IPO values company at $60B, largest since Arm.
[@Wired | 15:45]: Apple announces carbon-neutral products by 2030.
[@Bloomberg | 16:00]: TD Bank reports strong Q2 earnings, beats analyst estimates.
[@LocalNews_TOR | 16:15]: Toronto Fire crews respond to warehouse fire in Leslieville area.
[@TechCrunch | 16:30]: Netflix launches live sports streaming with NBA games.
[@TheVerge | 16:45]: Google Pixel 9a launches at $599 with旗舰 AI features.
[@CNBC | 17:00]: Crypto regulation bill passes US Senate unanimously.
[@LocalNews_TOR | 17:15]: CN Tower reopens to visitors after maintenance.
[@TechCrunch | 17:30]: Uber launches autonomous taxi service in Phoenix.
[@Wired | 17:45]: Meta unveils holographic AR glasses prototype.
[@Bloomberg | 18:00]: Canadian GDP grows 0.5% in Q1, exceeding forecasts.
</timeline>

Provide the summary (start your response with <think>):"""

TASKS = {
    "weekend_transient": {
        "messages": [
            {"role": "system", "content": WEEKEND_SYS_TRANSIENT},
            {"role": "user", "content": WEEKEND_USR_TRANSIENT},
        ],
        "validator": validate_detailed_json,
        "parse_json": True,
        "source": WEEKEND_USR_TRANSIENT,
    },
    "weekend_fixed": {
        "messages": [
            {"role": "system", "content": WEEKEND_SYS_FIXED},
            {"role": "user", "content": WEEKEND_USR_FIXED},
        ],
        "validator": validate_detailed_json,
        "parse_json": True,
        "source": WEEKEND_USR_FIXED,
    },
    "filename": {
        "messages": [
            {"role": "user", "content": RENAME_PROMPT},
        ],
        "validator": validate_filename,
        "parse_json": False,
    },
    "summarize": {
        "messages": [
            {"role": "user", "content": TWITTER_PROMPT},
        ],
        "validator": validate_summary,
        "parse_json": False,
    },
    "file_summary": {
        "messages": [
            {"role": "system", "content": "Output JSON now. No preamble, no markdown.\n\nRequired format: {\"path\": \"description\", ...} OR [{\"path\": \"x\", \"desc\": \"y\"}, ...]\n\nSummarize each file in one line. Be specific - mention actual functionality, not just file type."},
            {"role": "user", "content": FILE_SUMMARY_PROMPT},
        ],
        "validator": validate_file_summary,
        "parse_json": True,
    },
}

MAX_RETRIES = 1
EVAL_TIMEOUT = 1800  # 30 minutes - server restart on timeout
MEMORY_WARNING_THRESHOLD = 80  # Skip/check if memory > 80%


# ==========================================================
# Memory monitoring
# ==========================================================

def get_memory_percent() -> float:
    """Get current memory usage percent."""
    try:
        import psutil
        return psutil.virtual_memory().percent
    except ImportError:
        return 0.0


def check_memory_safe() -> bool:
    """Check if memory is safe to run eval."""
    mem_pct = get_memory_percent()
    if mem_pct > MEMORY_WARNING_THRESHOLD:
        console.print(f"[yellow]  ⚠️  Memory at {mem_pct}% - may cause OOM[/yellow]")
        return False
    return True


def is_server_responsive(host: str = "localhost", port: int = 1337, timeout: int = 5) -> bool:
    """Check if osaurus server is responsive."""
    import requests
    try:
        resp = requests.get(f"http://{host}:{port}/api/tags", timeout=timeout)
        return resp.status_code == 200
    except:
        return False


def monitor_memory_loop(interval: int = 30):
    """Background thread to monitor memory during eval."""
    import threading
    import time
    
    def _monitor():
        while getattr(threading.current_thread(), "running", True):
            mem = get_memory_percent()
            if mem > MEMORY_WARNING_THRESHOLD:
                console.print(f"[yellow]  ⚠️  Memory at {mem}%[/yellow]")
            time.sleep(interval)
    
    t = threading.Thread(target=_monitor, daemon=True)
    t.running = True
    t.start()
    return t


def estimate_model_memory(model: str) -> int:
    """Estimate memory needed for a model (in GB). Extract size from model name."""
    import re
    # Pattern: lfm2-24b-a2b-mlx-8bit -> 24b = 24GB
    match = re.search(r'(\d+)b', model.lower())
    if match:
        return int(match.group(1))
    # Default: assume 4GB for small models
    return 4


# ==========================================================
# Helper to build tasks from model configs
# ==========================================================

def load_tasks_from_config(model: str):
    """Build task prompts from model config YAML."""
    from lib.config import get_model_prompts_all

    prompts = get_model_prompts_all(model)
    if not prompts:
        return None

    built = {}

    # Map config prompts to tasks
    if "weekend_fixed" in prompts:
        built["detailed_json"] = prompts["weekend_fixed"]
    if "weekend_transient" in prompts:
        built["json"] = prompts["weekend_transient"]
    if "summarize" in prompts:
        built["summarize"] = prompts["summarize"]
    if "filename" in prompts:
        built["filename"] = prompts["filename"]
    if "file_summary" in prompts:
        built["file_summary"] = prompts["file_summary"]

    return built

# ==========================================================
# FAILURE DIAGNOSIS
# ==========================================================

# Failure categories — answers "WHY did the model fail?"
FAIL_INFRA   = "INFRA"    # Model not found, connection error
FAIL_TIMEOUT = "TIMEOUT"  # Model took too long to respond
FAIL_PARSE   = "PARSE"    # Model output had JSON-like content but our parser couldn't extract it
FAIL_FORMAT  = "FORMAT"   # Model ignored format instructions (e.g. output prose instead of JSON)
FAIL_CONTENT = "CONTENT"  # Model followed format but produced low-quality content
FAIL_NONE    = None       # No failure


def _classify_failure(result: dict, task_cfg: dict, score: int, failure_reason: str) -> dict:
    """Classify WHY a result failed. Returns a diagnosis dict.

    Returns:
        {
            "category": one of FAIL_* constants,
            "reason": human-readable failure reason,
            "evidence": specific evidence for the diagnosis,
        }
    """
    error = result.get("error") or ""
    content = safe_content(result)
    parsed = result.get("parsed")
    raw_len = len(content)

    # --- Perfect score: no diagnosis needed ---
    if score >= 90:
        return {"category": FAIL_NONE, "reason": "", "evidence": ""}

    # --- Infrastructure failures ---
    if "Model not found" in error:
        return {
            "category": FAIL_INFRA,
            "reason": error,
            "evidence": "Model not loaded or wrong identifier",
        }
    if "Connection" in error:
        return {
            "category": FAIL_INFRA,
            "reason": error,
            "evidence": "Server unreachable",
        }

    # --- Timeout ---
    if "Timeout" in error or "timed out" in error.lower():
        return {
            "category": FAIL_TIMEOUT,
            "reason": error,
            "evidence": f"Model did not respond within {EVAL_TIMEOUT}s",
        }

    # --- JSON tasks: distinguish PARSE vs FORMAT vs CONTENT ---
    if task_cfg["parse_json"]:
        has_json_chars = "{" in content or "[" in content
        has_prose_before_json = False
        if has_json_chars:
            first_bracket = min(
                (content.find(c) for c in "[{" if content.find(c) >= 0),
                default=raw_len,
            )
            has_prose_before_json = first_bracket > 200

        if not parsed and not has_json_chars:
            # Model output pure prose — didn't follow JSON format instruction
            return {
                "category": FAIL_FORMAT,
                "reason": failure_reason or "No JSON in output",
                "evidence": f"Output was {raw_len} chars of prose with no JSON brackets",
            }

        if not parsed and has_json_chars:
            # Model tried to output JSON but it didn't parse
            return {
                "category": FAIL_PARSE,
                "reason": failure_reason or "JSON extraction failed",
                "evidence": f"Output had JSON-like content at char {first_bracket} of {raw_len} but parser couldn't extract valid JSON",
            }

        if parsed and has_prose_before_json:
            # Parsed OK, but model emitted reasoning before JSON
            evidence = f"Model emitted {first_bracket} chars of reasoning before first JSON bracket (total {raw_len} chars)"
            if score < 50:
                return {
                    "category": FAIL_FORMAT,
                    "reason": failure_reason,
                    "evidence": evidence + "; reasoning may have consumed the context window",
                }
            # Score was partial (50-89) — content quality issue with format warning
            return {
                "category": FAIL_CONTENT,
                "reason": failure_reason,
                "evidence": evidence,
            }

        # JSON parsed fine — pure content quality issue
        return {
            "category": FAIL_CONTENT,
            "reason": failure_reason,
            "evidence": _describe_content_failure(parsed, failure_reason),
        }

    # --- Non-JSON tasks (filename, summarize) ---
    if not content:
        return {
            "category": FAIL_FORMAT,
            "reason": "Empty content",
            "evidence": "Model returned empty response",
        }

    # Check if model output reasoning instead of a direct answer
    reasoning_markers = ["Let me", "I'll", "Wait,", "Actually,", "Here's my", "Thinking"]
    has_reasoning = any(marker in content[:200] for marker in reasoning_markers)
    if has_reasoning and raw_len > 200:
        return {
            "category": FAIL_FORMAT,
            "reason": failure_reason,
            "evidence": f"Model output {raw_len} chars starting with reasoning instead of a direct answer",
        }

    return {
        "category": FAIL_CONTENT,
        "reason": failure_reason,
        "evidence": f"Output was {raw_len} chars but failed validation: {failure_reason}",
    }


def _describe_content_failure(parsed, failure_reason: str) -> str:
    """Generate human-readable evidence for content quality failures."""
    if isinstance(parsed, list):
        item_count = len(parsed)
        with_details = sum(
            1 for item in parsed
            if isinstance(item, dict) and has_item_details(item)
        )
        return f"Parsed {item_count} items, {with_details} with details. {failure_reason}"
    elif isinstance(parsed, dict):
        keys = list(parsed.keys())[:5]
        return f"Parsed dict with keys {keys}. {failure_reason}"
    return failure_reason


def print_cross_model_comparison(all_results: list) -> None:
    """Print comparison table across all models."""
    if not all_results:
        return
    
    console.print("")
    console.print("[bold cyan]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    console.print("[bold cyan]Cross-Model Comparison")
    console.print("[bold cyan]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    
    models = [r["model"] for r in all_results]
    if not models:
        return
    
    first_results = all_results[0].get("results", [])
    if not first_results:
        return
    
    tasks = [res["task"] for res in first_results]
    if not tasks:
        return
    
    header = f"{'Task':<20}"
    for m in models:
        header += f" | {m[:15]:>15}"
    console.print(header)
    console.print("-" * len(header))
    
    for task in tasks:
        row = f"{task:<20}"
        task_scores = {}
        for r in all_results:
            score = 0
            for res in r.get("results", []):
                if res.get("task") == task:
                    score = res.get("quality_score", 0)
                    break
            task_scores[r["model"]] = score
            row += f" | {score:>15}"
        
        best_model = max(task_scores, key=task_scores.get)
        best_score = task_scores[best_model]
        row += f" | {best_score:>4}*"
        console.print(row)


def compute_score_stats(all_results: list) -> dict:
    """Compute aggregate statistics for each model."""
    stats = {}
    
    for r in all_results:
        model = r["model"]
        scores = [res.get("quality_score", 0) for res in r.get("results", [])]
        
        if not scores:
            continue
        
        import statistics
        stats[model] = {
            "mean": statistics.mean(scores),
            "median": statistics.median(scores),
            "stdev": statistics.stdev(scores) if len(scores) > 1 else 0,
            "min": min(scores),
            "max": max(scores),
            "count": len(scores),
        }
    
    return stats


def print_score_stats(stats: dict) -> None:
    """Print score statistics table."""
    if not stats:
        return
    
    console.print("")
    console.print("[bold]Score Statistics:[/bold]")
    console.print(f"{'Model':<20} | {'Mean':>6} | {'Med':>6} | {'Stdev':>6} | {'Min':>4} | {'Max':>4}")
    console.print("-" * 65)
    
    for model, s in sorted(stats.items(), key=lambda x: x[1]["mean"], reverse=True):
        console.print(
            f"{model:<20} | {s['mean']:>6.1f} | {s['median']:>6.1f} | "
            f"{s['stdev']:>6.1f} | {s['min']:>4} | {s['max']:>4}"
        )


def categorize_failures(all_results: list) -> dict:
    """Group failures by category across all models."""
    categories = {}
    
    for r in all_results:
        for res in r.get("results", []):
            if res.get("quality_score", 0) >= 90:
                continue
            
            cat = res.get("failure_category", "UNKNOWN")
            if cat not in categories:
                categories[cat] = {"count": 0, "models": set(), "tasks": set()}
            
            categories[cat]["count"] += 1
            categories[cat]["models"].add(r["model"])
            categories[cat]["tasks"].add(res.get("task"))
    
    for cat in categories:
        categories[cat]["models"] = list(categories[cat]["models"])
        categories[cat]["tasks"] = list(categories[cat]["tasks"])
    
    return categories


def print_failure_summary(categories: dict) -> None:
    """Print failure categorization summary."""
    if not categories:
        return
    
    console.print("")
    console.print("[bold]Failure Categories:[/bold]")
    
    for cat, info in sorted(categories.items(), key=lambda x: x[1]["count"], reverse=True):
        count = info["count"]
        models = ", ".join(info["models"][:3])
        tasks = ", ".join(info["tasks"][:2])
        console.print(f"  [{count}] {cat}: {models} ({tasks})")


def save_historical_results(all_results: list, stats: dict, categories: dict) -> None:
    """Save per-model scores that persist even when models change."""
    import os
    from pathlib import Path
    
    # Store in user config directory (XDG spec)
    eval_dir = Path(os.path.expanduser("~/.config/ztools"))
    eval_dir.mkdir(parents=True, exist_ok=True)
    history_file = eval_dir / "eval_history.json"
    history = {}  # {model: [{date, task, score, ...}]}
    
    if history_file.exists():
        try:
            with open(history_file) as f:
                history = json.load(f)
        except:
            pass
    
    for r in all_results:
        model = r["model"]
        if model not in history:
            history[model] = []
        
        for res in r.get("results", []):
            entry = {
                "date": time.strftime("%Y-%m-%d"),
                "timestamp": time.time(),
                "task": res.get("task"),
                "score": res.get("quality_score", 0),
                "time": res.get("time"),
            }
            history[model].append(entry)
    
    # Keep last 100 entries per model
    for model in history:
        history[model] = history[model][-100:]
    
    with open(history_file, "w") as f:
        json.dump(history, f, indent=2)


def load_historical_stats() -> dict:
    """Load per-model historical scores."""
    import os
    from pathlib import Path
    
    eval_dir = Path(os.path.expanduser("~/.config/ztools"))
    history_file = eval_dir / "eval_history.json"
    if not history_file.exists():
        return {}
    
    try:
        with open(history_file) as f:
            history = json.load(f)
    except:
        return {}
    
    if not history:
        return {}
    
    stats = {}
    for model, entries in history.items():
        if not entries:
            continue
        
        scores = [e["score"] for e in entries if e.get("score")]
        if scores:
            import statistics
            stats[model] = {
                "mean": statistics.mean(scores),
                "median": statistics.median(scores),
                "stdev": statistics.stdev(scores) if len(scores) > 1 else 0,
                "min": min(scores),
                "max": max(scores),
                "runs": len(entries),
            }
    
    return stats


def check_model_history(model: str) -> dict:
    """Check if model has historical data."""
    import os
    from pathlib import Path
    
    eval_dir = Path(os.path.expanduser("~/.config/ztools"))
    history_file = eval_dir / "eval_history.json"
    if not history_file.exists():
        return {}
    
    try:
        with open(history_file) as f:
            history = json.load(f)
    except:
        return {}
    
    return history.get(model, [])


def print_historical_trends() -> None:
    """Print historical score trends per model."""
    stats = load_historical_stats()
    if not stats:
        return
    
    console.print("")
    console.print("[bold]Historical Trends:[/bold]")
    console.print(f"{'Model':<20} | {'Runs':>5} | {'Mean':>6} | {'Stdev':>6} | {'Trend'}")
    console.print("-" * 60)
    
    for model, s in sorted(stats.items(), key=lambda x: x[1]["mean"], reverse=True):
        runs = s.get("runs", 0)
        mean = s.get("mean", 0)
        stdev = s.get("stdev", 0)
        
        if runs >= 3:
            if stdev < 5:
                trend = "stable"
            elif stdev < 15:
                trend = "variable"
            else:
                trend = "unstable"
        else:
            trend = "new"
        
        console.print(f"{model:<20} | {runs:>5} | {mean:>6.0f} | {stdev:>6.1f} | {trend}")


def compute_token_estimates(results: list) -> dict:
    """Rough token estimation from response length (~4 chars/token)."""
    input_tokens = 0
    output_tokens = 0
    
    for r in results:
        content = r.get("content", "")
        if content:
            output_tokens += len(content) // 4
        
        messages = r.get("messages", [])
        for msg in messages:
            content = msg.get("content", "")
            if content:
                input_tokens += len(content) // 4
    
    return {"input": input_tokens, "output": output_tokens, "total": input_tokens + output_tokens}


def compute_verbosity(all_results: list) -> dict:
    """Compute avg response length per task per model."""
    verbosity = {}
    
    for r in all_results:
        model = r["model"]
        verbosity[model] = {}
        
        for res in r.get("results", []):
            task = res.get("task")
            content = res.get("result", {}).get("content", "")
            length = len(content) if content else 0
            verbosity[model][task] = length
    
    return verbosity


def print_verbosity(verbosity: dict) -> None:
    """Print response length per task."""
    if not verbosity:
        return
    
    console.print("")
    console.print("[bold]Response Verbosity (chars):[/bold]")
    
    first_model = next(iter(verbosity.keys()))
    tasks = verbosity[first_model].keys()
    
    header = f"{'Task':<20}"
    for m in verbosity.keys():
        header += f" | {m[:12]:>12}"
    console.print(header)
    console.print("-" * len(header))
    
    for task in tasks:
        row = f"{task:<20}"
        for model, task_lengths in verbosity.items():
            length = task_lengths.get(task, 0)
            row += f" | {length:>12,}"
        console.print(row)


def compute_error_rates(all_results: list) -> dict:
    """Compute error rates: infra errors vs quality failures."""
    rates = {}
    
    for r in all_results:
        model = r["model"]
        infra = 0
        quality = 0
        success = 0
        
        for res in r.get("results", []):
            category = res.get("failure_category", "OK")
            error = res.get("error")
            
            if error or category == "INFRA":
                infra += 1
            elif res.get("quality_score", 0) < 50:
                quality += 1
            else:
                success += 1
        
        total = infra + quality + success
        rates[model] = {
            "infra": infra,
            "quality": quality,
            "success": success,
            "infra_rate": infra / total if total else 0,
            "quality_rate": quality / total if total else 0,
            "success_rate": success / total if total else 0,
        }
    
    return rates


def print_error_rates(rates: dict) -> None:
    """Print error rate breakdown."""
    if not rates:
        return
    
    console.print("")
    console.print("[bold]Error Rates:[/bold]")
    console.print(f"{'Model':<20} | {'Infra':>6} | {'Quality':>8} | {'Success':>8} | {'Rate'}")
    console.print("-" * 65)
    
    for model, r in sorted(rates.items(), key=lambda x: x[1]["success_rate"], reverse=True):
        rate = r["success_rate"] * 100
        console.print(
            f"{model:<20} | {r['infra']:>6} | {r['quality']:>8} | "
            f"{r['success']:>8} | {rate:>5.0f}%"
        )


def compute_task_winners(all_results: list) -> dict:
    """Find which model wins each task."""
    winners = {}
    
    for r in all_results:
        for res in r.get("results", []):
            task = res.get("task")
            score = res.get("quality_score", 0)
            
            if task not in winners or score > winners[task][1]:
                winners[task] = (r["model"], score)
    
    return winners


def diff_from_last_run(all_results: list) -> dict:
    """Compare current scores to last run for each model."""
    import os
    from pathlib import Path
    
    eval_dir = Path(os.path.expanduser("~/.config/ztools"))
    prev_file = eval_dir / "eval_results.json"
    
    if not prev_file.exists():
        return {}
    
    try:
        with open(prev_file) as f:
            prev_data = json.load(f)
    except:
        return {}
    
    prev_results = prev_data.get("models", [])
    if not prev_results:
        return {}
    
    diffs = {}
    for r in all_results:
        model = r["model"]
        prev_model_data = next((p for p in prev_results if p.get("model") == model), None)
        
        if not prev_model_data:
            continue
        
        diffs[model] = {}
        
        for res in r.get("results", []):
            task = res.get("task")
            score = res.get("quality_score", 0)
            
            prev_score = 0
            for p in prev_model_data.get("results", []):
                if p.get("task") == task:
                    prev_score = p.get("quality_score", 0)
                    break
            
            diff = score - prev_score
            if diff != 0:
                diffs[model][task] = {"current": score, "prev": prev_score, "diff": diff}
    
    return diffs


def print_diff(diffs: dict) -> None:
    """Print score changes from last run."""
    if not diffs:
        return
    
    console.print("")
    console.print("[bold]Changes from Last Run:[/bold]")
    
    has_changes = False
    for model, changes in diffs.items():
        for task, d in changes.items():
            if d.get("diff", 0) != 0:
                has_changes = True
                break
    
    if not has_changes:
        console.print("  (no changes)")
        return
    
    console.print(f"{'Model':<18} | {'Task':<18} | {'Prev':>4} | {'Now':>4} | {'Diff'}")
    console.print("-" * 60)
    
    for model, changes in diffs.items():
        for task, d in changes.items():
            diff = d.get("diff", 0)
            if diff != 0:
                arrow = "↑" if diff > 0 else "↓"
                console.print(
                    f"{model[:18]:<18} | {task[:18]:<18} | "
                    f"{d['prev']:>4} | {d['current']:>4} | {arrow}{abs(diff):>3}"
                )


def export_to_csv(all_results: list, output_file: str = None) -> None:
    """Export results to CSV for reporting."""
    import os
    from pathlib import Path
    
    if output_file is None:
        eval_dir = Path(os.path.expanduser("~/.config/ztools"))
        output_file = eval_dir / "eval_results.csv"
    else:
        output_file = Path(output_file)
    
    import csv
    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Model", "Task", "Score", "Status", "Time(s)", "Failure", "Failure_Category"])
        
        for r in all_results:
            model = r["model"]
            for res in r.get("results", []):
                score = res.get("quality_score", 0)
                status = "PASS" if score >= 90 else ("WARN" if score >= 50 else "FAIL")
                time_s = res.get("time", "")
                failure = res.get("failure_reason", "")
                category = res.get("failure_category", "")
                
                writer.writerow([model, res.get("task"), score, status, time_s, failure, category])
    
    console.print(f"[green]Exported to {output_file}[/green]")


def _validate_result(result: dict, task_cfg: dict, task_name: str, debug: bool = False) -> tuple[int, str, dict]:
    """Run validation on a library result. Returns (score, failure_reason, diagnosis)."""
    validator = task_cfg["validator"]

    if result.get("error"):
        diagnosis = _classify_failure(result, task_cfg, 0, result["error"])
        return 0, result["error"], diagnosis

    # JSON tasks: handle both JSON and text outputs gracefully
    is_parse_json = task_cfg.get("parse_json", False)
    parsed = result.get("parsed")
    content = safe_content(result)
    source = task_cfg.get("source", "")
    
    # EARLY RETURN FOR PARSED - don't fall through
    if is_parse_json and parsed:
        validated = validator(parsed, source_text=source)
        
        if isinstance(validated, tuple):
            score, failure_reason = validated
        else:
            score, failure_reason = validated, ""
        
        diagnosis = _classify_failure(result, task_cfg, score, failure_reason)
        return score, failure_reason, diagnosis
    
    # Content-based extraction for text/markdown outputs
    if is_parse_json and content:
        import re
        
        # Try to extract JSON from content first (regardless of length)
        json_match = re.search(r'\[[\s\S]*\]', content) or re.search(r'\{[\s\S]*\}', content)
        extracted = None
        if json_match:
            try:
                extracted = json.loads(json_match.group())
                # Wrap dict in list if needed
                if isinstance(extracted, dict):
                    extracted = [extracted]
            except:
                pass
        
        # Try markdown extraction if no JSON found
        if not extracted:
            extracted = _extract_items_from_text(content)
        
        # If we have something extracted, validate it
        if extracted:
            validated = validator(extracted, source_text=source)
            items_for_debug = extracted
        elif len(content) > 50:
            # Short fallback to text validation for very short content is handled below
            # Here: content is too short for markdown but no JSON/markdown found
            from lib.validators_lib import validate_summary
            validated = validate_summary(content)
            items_for_debug = None
        else:
            # Content too short - treat as empty
            failure = "Empty content"
            diagnosis = _classify_failure(result, task_cfg, 0, failure)
            return 0, failure, diagnosis
        
        # Show source matching details if debug (only for weekend tasks)
        if debug and source and "weekend" in task_name and items_for_debug:
            from lib.validators_lib import get_source_matching_details
            details = get_source_matching_details(items_for_debug, source)
            console.print(f"[dim]Source matching for {task_name}:[/dim]")
            console.print(f"[dim]  Matched: {len(details['matched'])}/{len(details['matched']) + len(details['unmatched'])} ({details['ratio']*100:.0f}%)[/dim]")
            if details['unmatched']:
                console.print(f"[dim]  Unmatched items:[/dim]")
                for item in details['unmatched'][:3]:
                    console.print(f"[dim]    - {item['name']} (terms: {item.get('terms', [])[:3]})[/dim]")
        
        if isinstance(validated, tuple):
            score, failure_reason = validated
        else:
            score, failure_reason = validated, ""
        
        diagnosis = _classify_failure(result, task_cfg, score, failure_reason)
        return score, failure_reason, diagnosis
    
    # Non-JSON task
    if not content:
        failure = "Empty content"
        diagnosis = _classify_failure(result, task_cfg, 0, failure)
        return 0, failure, diagnosis
    validated = validator(content)

    if isinstance(validated, tuple):
        score, failure_reason = validated
    else:
        score, failure_reason = validated, ""

    diagnosis = _classify_failure(result, task_cfg, score, failure_reason)
    return score, failure_reason, diagnosis


def _call_model(model: str, task_cfg: dict, task_name: str, host: str, port: int, backend: str) -> dict:
    """Call model via the appropriate backend (pure transport, no validation)."""
    if backend == "mlx":
        from lib.mlx_lib import call as mlx_call
        return mlx_call(
            model,
            messages=task_cfg["messages"],
            host=host,
            port=port,
            timeout=EVAL_TIMEOUT,
        )
    else:
        # Silent - no per-task prints
        return call(
            model=model,
            messages=task_cfg["messages"],
            host=host,
            port=port,
            task=task_name,
            parse_json=task_cfg["parse_json"],
            timeout=EVAL_TIMEOUT,
        )


def run_eval(
    model: str, tasks: dict = None, host: str = "localhost", port: int = 1337, backend: str = "osaurus",
    verbose: bool = False
) -> dict:
    """Run evaluation on model using real-world tasks.
    
    This function owns all validation and retry logic.
    The library call() functions are pure transport/parsing layers.
    """
    from lib.logging_config import osaurus_logger as eval_logger

    tasks = tasks or TASKS
    results = []

    console.print("")
    console.print(f"[bold cyan]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/bold cyan]")
    console.print(f"[bold cyan]Testing {model} ({backend})[bold cyan]")
    console.print(f"[bold cyan]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/bold cyan]")

    # Create model-specific debug log
    import subprocess
    debug_log = f"/tmp/osaurus_debug_{model.replace('-', '_').replace('.', '_')}.log"
    try:
        subprocess.run(["touch", debug_log], capture_output=True)
        console.print(f"[dim]  ↳ Debug log: {debug_log}[/dim]")
    except:
        pass

    for task_name, task_cfg in tasks.items():
        best_score = -1
        best_result = None
        best_failure = ""
        best_diagnosis = {"category": FAIL_NONE, "reason": "", "evidence": ""}
        first_attempt_failed = False
        
        for attempt in range(MAX_RETRIES + 1):
            if attempt > 0:
                eval_logger.warning(f"Retrying task '{task_name}' with model {model} (Attempt {attempt+1}/{MAX_RETRIES+1})...")
                first_attempt_failed = True
            
            try:
                result = _call_model(model, task_cfg, task_name, host, port, backend)
            except Exception as e:
                eval_logger.error(f"Model call failed with exception: {e}")
                result = {"content": None, "error": str(e), "time": None, "model": model}
            
            try:
                score, failure_reason, diagnosis = _validate_result(result, task_cfg, task_name, debug=True)
            except Exception as e:
                eval_logger.error(f"Validation failed with exception: {e}")
                score, failure_reason, diagnosis = 0, f"Validation error: {e}", {"category": FAIL_INFRA, "reason": str(e), "evidence": ""}
            
            eval_logger.info(f"Quality score: {score}/100")
            
            if score < 90:
                category = diagnosis.get("category", "")
                evidence = diagnosis.get("evidence", "")
                eval_logger.warning(
                    f"[DEBUG_OUTPUT] model={model} task={task_name} score={score} "
                    f"category={category} failure={failure_reason} "
                    f"evidence={evidence}"
                )

            if score > best_score:
                best_score = score
                best_result = result
                best_failure = failure_reason
                best_diagnosis = diagnosis
            
            if score >= 90:
                break

        status = "ok" if best_score >= 90 else ("partial" if best_score >= 50 else "fail")
        category = best_diagnosis.get("category")

        results.append(
            {
                "task": task_name,
                "status": status,
                "quality_score": best_score,
                "time": best_result.get("time") if best_result else None,
                "error": best_result.get("error") if best_result else None,
                "failure_reason": best_failure,
                "failure_category": category,
                "failure_evidence": best_diagnosis.get("evidence", ""),
                "result": best_result,
                "first_attempt_failed": first_attempt_failed,
            }
        )

        status_symbol = "[PASS]" if status == "ok" else ("[WARN]" if status == "partial" else "[FAIL]")
        retry_tag = " (2nd try)" if first_attempt_failed else ""
        category_tag = f" [{category}]" if category else ""
        fail_info = f" - {best_failure}" if best_failure else ""
        evidence_info = f"\n    ↳ {best_diagnosis['evidence']}" if best_diagnosis.get("evidence") else ""
        time_taken = best_result.get('time') if best_result else None
        time_taken_str = f"{time_taken}s" if time_taken is not None else "N/A"
        console.print(
            f"  {status_symbol} {task_name}: {best_score}% ({time_taken_str}){category_tag}{fail_info}{evidence_info}"
        )

        # Print raw output in verbose mode
        if verbose and best_result:
            content = safe_content(best_result)[:500]
            if content:
                console.print(f"[dim]  Raw output: {content}[/dim]")

    # Print source matching summary for weekend tasks
    weekend_tasks = [k for k in tasks.keys() if "weekend" in k]
    if weekend_tasks:
        from lib.validators_lib import get_source_matching_details
        console.print("")
        console.print("[bold]Quality Check Summary:[/bold]")
        for r in results:
            task_name = r["task"]
            if task_name not in weekend_tasks:
                continue
            task_cfg = tasks[task_name]
            source = task_cfg.get("source", "")
            if not source:
                continue
            parsed = r.get("result", {}).get("parsed", [])
            if not parsed:
                continue
            details = get_source_matching_details(parsed, source)
            matched = len(details["matched"])
            total = matched + len(details["unmatched"])
            ratio = details["ratio"] * 100
            console.print(f"  {task_name}: {matched}/{total} items from source ({ratio:.0f}%)")
            if details["unmatched"]:
                names = [u["name"] for u in details["unmatched"][:2]]
                console.print(f"    [WARN] Not from source: {names}")

    return results

def update_config(best_models: dict):
    import yaml
    from pathlib import Path
    config_path = Path("conf/config.yaml")
    if not config_path.exists():
        console.print("[yellow]Config file not found, skipping update.[/yellow]")
        return
        
    with open(config_path, "r") as f:
        config = yaml.safe_load(f) or {}
        
    if "best_models" not in config:
        config["best_models"] = {}
        
    for task, model in best_models.items():
        if model:
            config["best_models"][task] = model
            if task == "detailed_json":
                pass
    
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        
    console.print("[green]Updated conf/config.yaml with best models.[/green]")

def main():
    init_config()
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="Run evaluation for a specific model")
    parser.add_argument("--task", help="Run a specific task (weekend_transient, weekend_fixed, filename, summarize, file_summary)")
    parser.add_argument("--quick", action="store_true", help="Quick mode: run single task with one retry (faster iteration)")
    parser.add_argument("--config-tasks", action="store_true", help="Load tasks from YAML config instead of hardcoded prompts")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging to console")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show raw model output for debugging quality")
    args = parser.parse_args()

    models_to_test = []
    
    if is_server_running():
        osaurus_models = get_models()
        for m in osaurus_models:
            models_to_test.append((m, "osaurus"))
    else:
        console.print("[yellow]Warning: Osaurus server not running[/yellow]")

    # MLX backend disabled - not working reliably (2026-04)
    # Uncomment to re-enable:
    # from lib.mlx_lib import list_mlx_models, normalize_mlx_model_name
    # mlx_models = list_mlx_models()
    # osaurus_base_names = {normalize_mlx_model_name(m) for m in osaurus_models}
    # for m in mlx_models:
    #     normalized = normalize_mlx_model_name(m)
    #     if normalized in osaurus_base_names:
    #         models_to_test.append((m, "mlx"))

    if args.model:
        models_to_test = [(m, b) for m, b in models_to_test if m == args.model]

    if not models_to_test:
        console.print("[red]No models found[/red]")
        sys.exit(1)

    console.print(f"[green]Found {len(models_to_test)} models to test[/green]")

    # Filter to specific task if requested
    tasks_to_run = TASKS
    if args.task:
        if args.task not in TASKS:
            console.print(f"[red]Unknown task: {args.task}. Available: {list(TASKS.keys())}[/red]")
            sys.exit(1)
        tasks_to_run = {args.task: TASKS[args.task]}
        console.print(f"[yellow]Running only task: {args.task}[/yellow]")

    # Auto-load tasks from YAML config (prefer over default TASKS)
    # Only skip if explicitly asked to use default
    from lib.config import build_tasks_from_model
    
    config_model = args.model if args.model else "qwen"
    config_tasks = build_tasks_from_model(config_model)
    if config_tasks:
        if args.task:
            if args.task in config_tasks:
                tasks_to_run = {args.task: config_tasks[args.task]}
                console.print(f"[dim]Using config task: {args.task}[/dim]")
            else:
                console.print(f"[red]Task '{args.task}' not in config[/red]")
        else:
            tasks_to_run = config_tasks
        console.print(f"[dim]Loaded {len(tasks_to_run)} tasks from config[/dim]")
    else:
        console.print("[dim]Using default TASKS (no config found)[/dim]")

    # In quick mode, only run first task once (no retries)
    if args.quick:
        console.print(f"[yellow]Quick mode: single run, no retries[/yellow]")
        import model_eval as me
        original_run_eval = me.run_eval
        
        def quick_run_eval(model, backend="osaurus"):
            # Monkey-patch MAX_RETRIES temporarily
            import model_eval
            original_retries = model_eval.MAX_RETRIES
            model_eval.MAX_RETRIES = 0
            try:
                return original_run_eval(model, backend=backend)
            finally:
                model_eval.MAX_RETRIES = original_retries
        
        me.run_eval = quick_run_eval

    all_results = []
    best_scores = {task: -1 for task in tasks_to_run.keys()}
    best_models = {task: None for task in tasks_to_run.keys()}
    
    def flush_between_models(prev_model: str, next_model: str) -> None:
        """Force a model switch with flush. If flush fails, offer restart."""
        import time
        import subprocess
        console.print(f"[dim]  ↳ Flushing {prev_model} → {next_model}...[/dim]")
        
        # Enable debug logging in osaurus with MODEL-SPECIFIC log files
        debug_log_paths = [
            f"/tmp/osaurus_debug_{next_model.replace('-', '_').replace('.', '_')}.log",
            "/tmp/osaurus_ttft_trace.log",
            "/tmp/osaurus_chat_perf.log",
        ]
        for path in debug_log_paths:
            try:
                subprocess.run(["touch", path], capture_output=True)
            except:
                pass
        console.print(f"[dim]  ↳ Debug logs: {debug_log_paths[0]}[/dim]")
        
        try:
            r = call(next_model, [{"role": "user", "content": "ok"}], timeout=30)
            if r.get("error"):
                console.print(f"[dim]  ↳ Flush failed, attempting restart...[/dim]")
                try:
                    subprocess.run(["osascript", "-e", 'quit app "osaurus"'], capture_output=True)
                except:
                    pass
                time.sleep(3)
                try:
                    subprocess.run(["open", "-n", "-a", "osaurus"], capture_output=True)
                except:
                    pass
                time.sleep(8)
                # Verify server is back
                for _ in range(5):
                    try:
                        import requests
                        resp = requests.get("http://localhost:1337/api/tags", timeout=2)
                        if resp.status_code == 200:
                            console.print(f"[dim]  ↳ Server restarted[/dim]")
                            break
                    except:
                        time.sleep(2)
        except Exception as e:
            console.print(f"[dim]  ↳ Flush error: {e}[/dim]")
        time.sleep(2)

    prev_model = None
    for model, backend in models_to_test:
        if prev_model and model != prev_model:
            flush_between_models(prev_model, model)
        prev_model = model
        
        console.print("")
        
        # Pre-flight memory and server check
        mem_pct = get_memory_percent()
        model_mem_gb = estimate_model_memory(model)
        avail_mem_gb = (100 - mem_pct) / 100 * 64  # Assume 64GB total
        
        if mem_pct > MEMORY_WARNING_THRESHOLD:
            console.print(f"[yellow]  ⚠️  Memory at {mem_pct}% - model may be slow[/yellow]")
        
        if model_mem_gb > avail_mem_gb * 0.8:
            console.print(f"[yellow]  ⚠️  Model needs ~{model_mem_gb}GB, low memory - will be slower[/yellow]")
        
        if not is_server_responsive():
            console.print(f"[yellow]  ⚠️  Server not responsive - attempting restart...[/yellow]")
            # Trigger restart (will happen in flush_between_models next iteration)
        
        console.print(f"[dim]  ↳ Memory: {mem_pct}%, Server: OK[/dim]")
        
        results = run_eval(model, tasks=tasks_to_run, backend=backend, verbose=args.verbose)
        scores = [r["quality_score"] for r in results]
        avg = sum(scores) / len(scores) if scores else 0
        
        status = "[PASS]" if all(s >= 90 for s in scores) else ("[WARN]" if any(s >= 50 for s in scores) else "[FAIL]")
        console.print(f"[bold]{status} {model} ({backend}):[/bold] {avg:.0f}% avg")
        
        # Results already printed in run_eval() - don't duplicate
        # (removed duplicate task listing here)
        
        all_results.append({'model': model, 'backend': backend, 'results': results})
        for r in results:
            task = r["task"]
            score = r["quality_score"]
            if score > best_scores[task]:
                best_scores[task] = score
                best_models[task] = model

    console.print("[bold]Best Models per Task:[/bold]")
    for task, model in best_models.items():
        console.print(f"  {task}: {model} ({best_scores[task]}%)")

    # ============== ADDED: Cross-model comparison ==============
    print_cross_model_comparison(all_results)
    
    stats = compute_score_stats(all_results)
    print_score_stats(stats)
    
    categories = categorize_failures(all_results)
    print_failure_summary(categories)
    
    verbosity = compute_verbosity(all_results)
    print_verbosity(verbosity)
    
    error_rates = compute_error_rates(all_results)
    print_error_rates(error_rates)
    
    diffs = diff_from_last_run(all_results)
    print_diff(diffs)
    
    # Task winners
    winners = compute_task_winners(all_results)
    console.print("")
    console.print("[bold]Task Winners:[/bold]")
    for task, (model, score) in sorted(winners.items()):
        console.print(f"  {task}: {model} ({score}%)")
    
    save_historical_results(all_results, stats, categories)
    print_historical_trends()
    
    # Export to CSV
    export_to_csv(all_results)
    # ====================================================

    import os
    from pathlib import Path
    
    eval_dir = Path(os.path.expanduser("~/.config/ztools"))
    eval_dir.mkdir(parents=True, exist_ok=True)
    results_file = eval_dir / "eval_results.json"
    
    with open(results_file, "w") as f:
        json.dump({
            "models": all_results,
            "best_scores": best_scores,
            "best_models": best_models,
        }, f, indent=2)
    console.print(f"[green]Saved to {results_file}[/green]")
    
    # Print debug log locations and instructions
    console.print("")
    console.print("Debug Logs:")
    console.print("  Logs written to /tmp/:")
    console.print("  - osaurus_debug.log    - General debug messages")
    console.print("  - osaurus_ttft_trace.log - Time-to-first-token timing")
    console.print("  - osaurus_chat_perf.log   - Performance metrics")
    console.print("")
    console.print("To view after crash:")
    console.print("  cat /tmp/osaurus_debug.log")
    console.print("  tail -f /tmp/osaurus_debug.log  (watch in real-time)")
    console.print("")
    console.print("Key things to look for:")
    console.print("  - 'errorCaught' - NIO errors")
    console.print("  - 'channelInactive' - Connection drops")
    console.print("  - Model loading/unloading messages")
    console.print("  - Any 'crash' or 'panic' keywords")

if __name__ == "__main__":
    main()

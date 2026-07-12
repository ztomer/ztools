#!/usr/bin/env python3
"""
Evaluation tasks and test cases for model evaluation.
Defines prompts, validators, and test data for all evaluation tasks.
"""

import re
import json
from pathlib import Path
from typing import List, Dict
from lib.validators.json_validator import validate_detailed_json, validate_json, validate_mixed_signal
from lib.validators.text_validator import (
    validate_filename, validate_summary, validate_file_summary,
    validate_mixed_summary, validate_mixed_file_summary, validate_mixed_filename,
    validate_no_leak, validate_strict_schema, validate_no_contradiction,
)

_TC_YEAR = "2026"  # check-ok: year


# ============================================================
# WEEKEND PLANNER PROMPTS
# ============================================================

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

WEEKEND_USR_TRANSIENT = f"""
Current Context for the upcoming weekend:
Dates: April 20 to April 22, {_TC_YEAR}
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

WEEKEND_USR_FIXED = f"""
Current Context for the upcoming weekend:
Dates: April 20 to April 22, {_TC_YEAR}
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

def _extract_items_from_text(text: str) -> List[Dict]:
    """Extract structured items from text output (markdown tables, lists, etc)."""
    items = []

    table_pattern = r'\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|'
    tables = re.findall(table_pattern, text)
    if tables and len(tables) >= 2:
        header = tables[0]

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

            header_map = {
                'name': 'name', 'event': 'name', 'title': 'name', 'activity': 'name',
                'location': 'location', 'venue': 'location', 'place': 'place', 'where': 'location',
                'day': 'day', 'date': 'day', 'when': 'day', 'time': 'time',
            }
            field1 = header_map.get(key1, 'name')
            field2 = header_map.get(key2, key2)

            for row in data_rows:
                if '---' in row[0].lower() or '---' in row[1].lower():
                    continue
                if not row[0].strip() or not row[1].strip():
                    continue
                row0_clean = row[0].strip().lower()
                row1_clean = row[1].strip().lower()
                if row0_clean in ['name', 'event', 'title', 'activity', 'location', 'venue', 'place', 'where'] or row1_clean in ['name', 'event', 'title', 'activity', 'location', 'venue', 'place', 'where']:
                    continue
                item = {field1: row[0].strip(), field2: row[1].strip()}
                items.append(item)

            if items:
                return items

    bullet_pattern = r'^[•\-]\s*(.+?)(?:\n|$)'
    bullets = re.findall(bullet_pattern, text, re.MULTILINE)
    for bullet in bullets:
        bullet = bullet.strip()
        if bullet and len(bullet) > 2:
            parts = bullet.split(':', 1)
            if len(parts) == 2:
                key = parts[0].strip()
                val = parts[1].strip()
                field_map = {
                    'name': 'name', 'event': 'name', 'title': 'name', 'activity': 'name',
                    'location': 'location', 'venue': 'location', 'place': 'location',
                }
                field = field_map.get(key.lower(), key.lower())
                items.append({field: val})
            else:
                sep_match = re.match(r'^([^,\-]+)[,\-](.+)$', bullet)
                if sep_match:
                    items.append({'name': sep_match.group(1).strip(), 'location': sep_match.group(2).strip()})
                else:
                    items.append({'name': bullet})

    return items

# --- MIXED PROMPTS (Base + Noise) ---
# These test the model's ability to filter signal from noise

WEEKEND_USR_TRANSIENT_MIXED = WEEKEND_USR_TRANSIENT + """

NOISE (Ignore these - test your filtering):
- Random text: asdfghjkl qwertyuiop zxcvbnm
- Lorem ipsum dolor sit amet consectetur adipiscing elit sed do eiusmod tempor
- Unrelated event: Underwater basket weaving in Atlantis. April 35. All ages.
- Malformed: Event without date or location or age
- Spam: BUY NOW! LIMITED TIME OFFER! CLICK HERE!
- Nonsense: The quick brown fox jumps over the lazy dog 1234567890
- Hallucinated venue: Mars Colony Indoor Playground. Indoor. Ages 0-100.
- Irrelevant: Stock market tips for weekend trading
"""

WEEKEND_USR_FIXED_MIXED = WEEKEND_USR_FIXED + """

NOISE (Ignore these - test your filtering):
- Random text: asdfghjkl qwertyuiop zxcvbnm
- Lorem ipsum dolor sit amet consectetur adipiscing elit sed do eiusmod tempor
- Unrelated venue: Moon Base Alpha Resort. Indoor. All ages.
- Malformed: Venue without location or age info
- Spam: BUY NOW! LIMITED TIME OFFER! CLICK HERE!
- Nonsense: The quick brown fox jumps over the lazy dog 1234567890
- Hallucinated: Atlantis Underwater Theme Park. Outdoor. Ages 5-99.
- Irrelevant: Cryptocurrency mining rigs for home use
"""


# ============================================================
# RENAME / FILENAME PROMPTS
# ============================================================

RENAME_PROMPT = """Give a short 2-4 word summary of: {text}

Output ONLY the filename string, lowercase with underscores. Max 35 characters."""

RENAME_PROMPT_MIXED = """Rename each text snippet below to a short 2-4 word filename, lowercase with underscores, max 35 chars.

Output a JSON array of filenames in the SAME ORDER as the snippets.

SNIPPETS:
1. How To Manage Your Underperformers
2. Scott Adams essays
3. 10 powerful sentences by Scott Adams navigating failure, ambition, the absurdities of life
4. 15 years of business lessons in under 500 words
5. Be delusional. Believe that you have the ability to make it work no matter what
6. How To Prioritize Like A Pro - Noemi Kis
7. elon musk: how to win at founding
8. context engineering template - comprehensive guide for AI prompts

NOISE (Ignore - do NOT produce filenames for these):
- Random noise: asdfghjkl
- Spam text: BUY NOW CLICK HERE!!!
- Nonsense: lorem ipsum dolor sit amet
- Malformed: incomplete text without meaning
"""

IMAGE_RENAME_PROMPT = """Convert each OCR text to a short descriptive filename, lowercase with underscores, max 35 chars.

Output a JSON array of filenames in the SAME ORDER.

TEXTS:
1. How To Manage Your Underperformers
2. Scott Adams essays
3. 10 powerful sentences by Scott Adams navigating failure, ambition, the absurdities of life
4. 15 years of business lessons in under 500 words: Marrying well is the biggest life hack of all
5. Be delusional. Believe that you have the ability to make it work no matter what
6. How To Prioritize Like A Pro - Noemi Kis: Understand Your Values First
7. elon musk: how to win at founding - taking risk if things don't work out
8. context engineering template - comprehensive guide for AI prompts
"""

IMAGE_RENAME_PROMPT_MIXED = """Convert each OCR text to a short descriptive filename, lowercase with underscores, max 35 chars.

Output a JSON array of filenames in the SAME ORDER.

TEXTS:
1. How To Manage Your Underperformers
2. Scott Adams essays
3. 10 powerful sentences by Scott Adams navigating failure, ambition, the absurdities of life
4. 15 years of business lessons in under 500 words: Marrying well is the biggest life hack of all
5. Be delusional. Believe that you have the ability to make it work no matter what
6. How To Prioritize Like A Pro - Noemi Kis: Understand Your Values First
7. elon musk: how to win at founding - taking risk if things don't work out
8. context engineering template - comprehensive guide for AI prompts

NOISE (Ignore - do NOT produce filenames for these):
- Random noise: asdfghjkl noise
- Spam text: BUY NOW CLICK HERE SPAM
- Nonsense: lorem ipsum dolor sit amet
- Malformed: incomplete without meaning
"""


# ============================================================
# FILE SUMMARY PROMPTS
# ============================================================

_PROJECT_ROOT = Path(__file__).parent.parent
_FILE_SUMMARY_FILES = [
    "README.md", "CLAUDE.md", "model_eval.py", "weekend_planner.py",
    "twitter_summarizer.py", "image_renamer.py", "explore_model_quirks.py",
    "lib/__init__.py", "lib/osaurus_lib.py", "lib/validators_lib.py",
    "lib/config.py", "lib/content_processing.py", "lib/mlx_lib.py",
    "lib/logging_config.py", "conf/config.yaml", "conf/weekend.yaml",
    "conf/twitter.yaml", "conf/rename.yaml", "conf/models/foundation.yaml",
    "conf/models/gemma.yaml", "conf/models/qwen.yaml",
    "docs/MODEL_QUIRKS.md", "docs/PROJECT_MEMORY.md",
    "tests/test_validators.py", "tests/test_parse.py", "tests/test_config.py",
    "tests/test_weekend.py", "tests/test_content_processing.py",
    "tests/test_twitter.py", "pyproject.toml",
]
FILE_SUMMARY_FILE_LIST = "\n".join(str(_PROJECT_ROOT / f) for f in _FILE_SUMMARY_FILES)
FILE_SUMMARY_PROMPT = f"""Read the file list below and give one-line summary for each file.

CRITICAL: DO NOT infer from filename. Describe what each file DOES, not what its filename suggests.
- Bad: "a python library" (infers from .py extension)
- Good: "parses web content and extracts metadata"

Use ## headers for each file (e.g., ## filename: summary).

{FILE_SUMMARY_FILE_LIST}

Skip .git, __pycache__, benchmarks/, and pycache directories."""

FILE_SUMMARY_PROMPT_MIXED = FILE_SUMMARY_PROMPT + """

NOISE FILES (Ignore - test your filtering):
/fake/path/nonexistent_file.txt
/totally/made/up/directory/garbage.log
/random/asdfghjkl/qwertyuiop.zxc
/spam/buy_now/click_here.exe
/irrelevant/crypto_price_predictions.md
/hallucinated/alien_landing_report.pdf
"""


# ============================================================
# TWITTER SUMMARIZER PROMPTS
# ============================================================

TWITTER_PROMPT = """You are an objective news distillation system. Your task is to extract hard facts from the provided chronological Twitter/X timeline.

<instructions>
1. First, analyze the timeline in block.
2. Identify clusters of related events and synthesize duplicates.
3. Output ONLY the final briefing after the  tag. No introductory text.
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
[@LocalNews_TOR | 14:15]: Ontario Place undergoing major renovation, new spa opening 2026.  # check-ok: year
[@TechCrunch | 14:30]: Salesforce announces AI-powered CRM with autonomous agents.
[@TheVerge | 14:45]: Intel Core Ultra chips debut with breakthrough efficiency.
[@CNBC | 15:00]: Housing market cools as mortgage rates stay elevated.
[@LocalNews_TOR | 15:15]: Raptors playoff game tonight at Scotiabank Arena, expect crowds.
[@TechCrunch | 15:30]: Databricks IPO values company at $60B, largest since Arm.
[@Wired | 15:45]: Apple announces carbon-neutral products by 2030.
[@Bloomberg | 16:00]: TD Bank reports strong Q2 earnings, beats analyst estimates.
[@LocalNews_TOR | 16:15]: Toronto Fire crews respond to warehouse fire in Leslieville area.
[@TechCrunch | 16:30]: Netflix launches live sports streaming with NBA games.
[@TheVerge | 16:45]: Google Pixel 9a launches at $599 with flagship AI features.
[@CNBC | 17:00]: Crypto regulation bill passes US Senate unanimously.
[@LocalNews_TOR | 17:15]: CN Tower reopens to visitors after maintenance.
[@TechCrunch | 17:30]: Uber launches autonomous taxi service in Phoenix.
[@Wired | 17:45]: Meta unveils holographic AR glasses prototype.
[@Bloomberg | 18:00]: Canadian GDP grows 0.5% in Q1, exceeding forecasts.
</timeline>

Provide the summary (start your response):"""

TWITTER_PROMPT_MIXED = TWITTER_PROMPT + """

NOISE (Ignore these - test your filtering):
- Random text: asdfghjkl qwertyuiop zxcvbnm
- Lorem ipsum dolor sit amet consectetur adipiscing elit sed do eiusmod tempor
- SPAM: BUY NOW! LIMITED TIME OFFER! CLICK HERE!
- Nonsense: The quick brown fox jumps over the lazy dog 1234567890
- Irrelevant: Cryptocurrency price prediction for next week
- Hallucinated: [@FakeNews | 12:00]: Aliens landed in Central Park!
- Malformed: incomplete tweet without proper format
- Off-topic: Best pizza recipes for Sunday dinner
"""

# Faithfulness probe: a clearly-fabricated "BREAKING" tweet injected into the
# timeline. A faithful model summarizes the REAL signal and ignores this
# planted falsehood. CONTRADICTION_PHRASE is the sentinel the validator checks.
CONTRADICTION_PHRASE = "quantum giraffes of Manitoba won the Stanley Cup"
TWITTER_PROMPT_CONTRADICTION = TWITTER_PROMPT.replace(
    "<timeline>",
    "<timeline>\n[@FakeNews | 07:30]: BREAKING: quantum giraffes of Manitoba won the "
    "Stanley Cup, defeating the Toronto Maple Leafs in a surprise finals sweep.",
)


# ============================================================
# TASK DEFINITIONS
# ============================================================

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
    "weekend_transient_mixed": {
        "messages": [
            {"role": "system", "content": WEEKEND_SYS_TRANSIENT},
            {"role": "user", "content": WEEKEND_USR_TRANSIENT_MIXED},
        ],
        "validator": validate_mixed_signal,
        "parse_json": True,
        "source": WEEKEND_USR_TRANSIENT_MIXED,
    },
    "weekend_fixed_mixed": {
        "messages": [
            {"role": "system", "content": WEEKEND_SYS_FIXED},
            {"role": "user", "content": WEEKEND_USR_FIXED_MIXED},
        ],
        "validator": validate_mixed_signal,
        "parse_json": True,
        "source": WEEKEND_USR_FIXED_MIXED,
    },
    "filename": {
        "messages": [
            {"role": "user", "content": RENAME_PROMPT},
        ],
        "validator": validate_filename,
        "parse_json": False,
    },
    "image_rename": {
        "messages": [
            {"role": "user", "content": IMAGE_RENAME_PROMPT},
        ],
        "validator": validate_mixed_filename,
        "parse_json": True,
        "source": IMAGE_RENAME_PROMPT,
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
    # --- MIXED VARIANTS ---
    "rename_mixed": {
        "messages": [
            {"role": "user", "content": RENAME_PROMPT_MIXED},
        ],
        "validator": validate_mixed_filename,
        "parse_json": True,
        "source": RENAME_PROMPT_MIXED,
    },
    "summarize_mixed": {
        "messages": [
            {"role": "user", "content": TWITTER_PROMPT_MIXED},
        ],
        "validator": validate_mixed_summary,
        "parse_json": False,
        "source": TWITTER_PROMPT_MIXED,
    },
    "file_summary_mixed": {
        "messages": [
            {"role": "system", "content": "Output JSON now. No preamble, no markdown.\n\nRequired format: {\"path\": \"description\", ...} OR [{\"path\": \"x\", \"desc\": \"y\"}, ...]\n\nSummarize each file in one line. Be specific - mention actual functionality, not just file type."},
            {"role": "user", "content": FILE_SUMMARY_PROMPT_MIXED},
        ],
        "validator": validate_mixed_file_summary,
        "parse_json": False,
        "source": FILE_SUMMARY_PROMPT_MIXED,
    },
    "filename_mixed": {
        "messages": [
            {"role": "user", "content": RENAME_PROMPT_MIXED},
        ],
        "validator": validate_mixed_filename,
        "parse_json": True,
        "source": RENAME_PROMPT_MIXED,
    },
    "image_rename_mixed": {
        "messages": [
            {"role": "user", "content": IMAGE_RENAME_PROMPT_MIXED},
        ],
        "validator": validate_mixed_filename,
        "parse_json": True,
        "source": IMAGE_RENAME_PROMPT_MIXED,
    },
    # --- FAITHFULNESS / SCHEMA / LEAK TESTS (Round 1-2) ---
    "weekend_transient_schema": {
        "messages": [
            {"role": "system", "content": WEEKEND_SYS_TRANSIENT},
            {"role": "user", "content": WEEKEND_USR_TRANSIENT},
        ],
        "validator": validate_strict_schema,
        "parse_json": False,
        "validator_kwargs": {"kind": "json"},
    },
    "summarize_contradiction": {
        "messages": [
            {"role": "user", "content": TWITTER_PROMPT_CONTRADICTION},
        ],
        "validator": validate_no_contradiction,
        "parse_json": False,
        "validator_kwargs": {"contradiction_phrase": CONTRADICTION_PHRASE},
    },
    "filename_leak": {
        "messages": [
            {"role": "user", "content": RENAME_PROMPT},
        ],
        "validator": validate_no_leak,
        "parse_json": False,
    },
}

TASKS["json"] = dict(TASKS["weekend_transient"])
TASKS["detailed_json"] = dict(TASKS["weekend_fixed"])
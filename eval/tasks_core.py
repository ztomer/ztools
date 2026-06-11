#!/usr/bin/env python3
"""
Core task definitions for model evaluation.
Contains all hardcoded prompts and the TASKS dict.
"""

import re
from typing import Dict, List

from lib.validators_lib import validate_detailed_json, validate_summary, validate_filename
from eval.validate import validate_file_summary


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

RENAME_PROMPT = """Give a short 2-4 word summary of: {text}

Output ONLY the filename string, lowercase with underscores. Max 35 characters."""

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
[@TheVerge | 16:45]: Google Pixel 9a launches at $599 with flagship AI features.
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
    "image_rename": {
        "description": "Rename images from OCR text - test multiple cases",
        "test_cases": [
            ("How To Manage Your Underperformers", "how_to_manage_underperformers"),
            ("Scott Adams essays", "scott_adams_essays"),
            ("10 powerful sentences by Scott Adams navigating failure, ambition, the absurdities of life: 1. Creativity is allowing yourself to make mistakes...", "scott_adams_powerful_sentences"),
            ("15 years of business lessons in <500 words: 1. Marrying well is the biggest life hack of all. 2. Be the type of person who reaches out to others...", "business_lessons"),
            ("Be delusional. Believe that you have the ability to make it work no matter what. Believe that regardless of what happens...", "delusional_belief"),
            ("How To Prioritize Like A Pro - Noemi Kis: Understand Your Values First Time Block Your Schedule Execute Through Habits...", "prioritize_pro"),
            ("elon musk: how to win at founding - taking risk if things don't work out", "musk_founding_tips"),
            ("context engineering template - comprehensive guide for AI prompts", "context_engineering"),
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

TASKS["json"] = dict(TASKS["weekend_transient"])
TASKS["detailed_json"] = dict(TASKS["weekend_fixed"])

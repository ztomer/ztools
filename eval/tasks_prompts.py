"""Prompt data for eval tasks — moved from tasks_core.py for file-size compliance.

Canonical source for all eval prompt strings (mixed variants, rename, file
summary, twitter). Imported by eval/tasks_core.py which re-exports these
together with the TASKS dict and _extract_items_from_text helper.
"""

from pathlib import Path

from lib.eval_data import WEEKEND_USR_FIXED, WEEKEND_USR_TRANSIENT

WEEKEND_USR_TRANSIENT_MIXED = (
    WEEKEND_USR_TRANSIENT
    + """

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
)

WEEKEND_USR_FIXED_MIXED = (
    WEEKEND_USR_FIXED
    + """

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
)

RENAME_PROMPT = """Give a short 2-4 word summary of: {text}

Output ONLY the filename string, lowercase with underscores. Max 35 characters."""

RENAME_PROMPT_MIXED = """\
Rename each text snippet below to a short 2-4 word filename, lowercase with
underscores, max 35 chars.

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

IMAGE_RENAME_PROMPT = """\
Convert each OCR text to a short descriptive filename, lowercase with
underscores, max 35 chars.

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

IMAGE_RENAME_PROMPT_MIXED = """\
Convert each OCR text to a short descriptive filename, lowercase with
underscores, max 35 chars.

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

_PROJECT_ROOT = Path(__file__).parent.parent
_FILE_SUMMARY_FILES = [
    "README.md",
    "CLAUDE.md",
    "model_eval.py",
    "weekend_planner.py",
    "twitter_summarizer.py",
    "image_renamer.py",
    "explore_model_quirks.py",
    "lib/__init__.py",
    "lib/osaurus_lib.py",
    "lib/validators_lib.py",
    "lib/config.py",
    "lib/content_processing.py",
    "lib/mlx_lib.py",
    "lib/logging_config.py",
    "conf/config.toml",
    "conf/weekend.toml",
    "conf/twitter.toml",
    "conf/rename.toml",
    "conf/models/foundation.toml",
    "conf/models/gemma.toml",
    "conf/models/qwen.toml",
    "docs/MODEL_QUIRKS.md",
    "docs/PROJECT_MEMORY.md",
    "tests/test_validators.py",
    "tests/test_parse.py",
    "tests/test_config.py",
    "tests/test_weekend.py",
    "tests/test_content_processing.py",
    "tests/test_twitter.py",
    "pyproject.toml",
]
FILE_SUMMARY_FILE_LIST = "\n".join(str(_PROJECT_ROOT / f) for f in _FILE_SUMMARY_FILES)
FILE_SUMMARY_PROMPT = f"""Read the file list below and give one-line summary for each file.

CRITICAL: DO NOT infer from filename. Describe what each file DOES, not what its filename suggests.
- Bad: "a python library" (infers from .py extension)
- Good: "parses web content and extracts metadata"

Use ## headers for each file (e.g., ## filename: summary).

{FILE_SUMMARY_FILE_LIST}

Skip .git, __pycache__, benchmarks/, and pycache directories."""

FILE_SUMMARY_PROMPT_MIXED = (
    FILE_SUMMARY_PROMPT
    + """

NOISE FILES (Ignore - test your filtering):
/fake/path/nonexistent_file.txt
/totally/made/up/directory/garbage.log
/random/asdfghjkl/qwertyuiop.zxc
/spam/buy_now/click_here.exe
/irrelevant/crypto_price_predictions.md
/hallucinated/alien_landing_report.pdf
"""
)

TWITTER_PROMPT = """\
You are an objective news distillation system. Your task is to extract hard
facts from the provided chronological Twitter/X timeline.

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
[@TechCrunch | 08:00]: OpenAI announces GPT-5 with advanced reasoning
capabilities, available next month.
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
[@LocalNews_TOR | 14:15]: Ontario Place undergoing major renovation, new spa
opening 2026.  # check-ok: year
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

TWITTER_PROMPT_MIXED = (
    TWITTER_PROMPT
    + """

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
)

CONTRADICTION_PHRASE = "quantum giraffes of Manitoba won the Stanley Cup"
TWITTER_PROMPT_CONTRADICTION = TWITTER_PROMPT.replace(
    "<timeline>",
    "<timeline>\n[@FakeNews | 07:30]: BREAKING: quantum giraffes of Manitoba won the "
    "Stanley Cup, defeating the Toronto Maple Leafs in a surprise finals sweep.",
)

FALSEHOOD_TWEET_1 = (
    "[@FakeNews | 07:30]: BREAKING: quantum giraffes of Manitoba won the "
    "Stanley Cup, defeating the Toronto Maple Leafs in a surprise finals sweep."
)
FALSEHOOD_TWEET_2 = (
    "[@BSNews | 14:45]: BREAKING: NVIDIA CEO Jensen Huang arrested for insider "
    "trading, stock plunges 30% after hours."
)
FALSEHOOD_TWEET_3 = (
    "[@HoaxWire | 06:00]: BREAKING: Google announces layoffs of 100,000 "
    "employees, entire search division restructured."
)

TWITTER_PROMPT_ACCURACY = TWITTER_PROMPT.replace(
    "<timeline>",
    f"<timeline>\n{FALSEHOOD_TWEET_1}\n{FALSEHOOD_TWEET_2}\n{FALSEHOOD_TWEET_3}",
)

FALSEHOOD_PHRASES = [
    CONTRADICTION_PHRASE,
    "CEO Jensen Huang arrested for insider trading",
    "layoffs of 100,000 employees",
]

KEY_FACTS = [
    "GPT-5",
    "Apple Vision Pro 2",
    "Gemini 2.5 Pro",
    "NVIDIA stock hits all-time high",
    "Microsoft acquires AI startup for $2B",
    "Samsung Galaxy S25 Ultra",
    "Meta announces Llama 4",
    "IBM unveils quantum computer with 1000+ qubit",
    "Shopify reports 40% revenue growth",
    "Bitcoin surges past $75K",
    "Databricks IPO values company at $60B",
    "Amazon launches drone delivery in Toronto",
    "Uber launches autonomous taxi service in Phoenix",
    "Canadian GDP grows 0.5%",
    "TD Bank reports strong Q2 earnings",
    "Adobe acquires Figma for $20B",
    "Intel Core Ultra chips debut",
    "Tesla Cybertruck production ramps up",
]

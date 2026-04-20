import json
from lib.osaurus_lib import _extract_json_only

with open("llm_raw_dump.txt", "r") as f:
    text = f.read()

# find all sections
sections = text.split("=== RES ===")
for i, sec in enumerate(sections):
    if not sec.strip(): continue
    raw = sec.split("=== END RES ===")[0]
    res = _extract_json_only(raw)
    print(f"Section {i} extracted length: {len(res) if res else 'None'}")
    if res:
        print("Extracted content:")
        print(res[:200])
        print("...")
        print(res[-200:])
        print("-" * 50)

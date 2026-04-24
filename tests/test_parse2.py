from lib.osaurus_lib import _extract_json_only
import json

text = """
<think>
[Final Check of the Prompt]: "Output JSON now. Use EXACT schema:
{"fixed_activities": [{"name": "str", "location": "str", "target_ages": "str", "price": "str", "weather": "str"}]}" ()
</think>
{"fixed_activities": [{"name": "The Works Museum", "location": "Vaughan", "target_ages": "6-13", "price": "$18-35 per child or free", "weather": "indoor"}]}
"""

res = _extract_json_only(text)
print("EXTRACTED:")
print(res)

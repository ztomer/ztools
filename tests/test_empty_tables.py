import json
with open("llm_req_dump.txt", "w") as f: f.write("")
with open("llm_raw_dump.txt", "w") as f: f.write("")

import weekend_planner
orig_call_llm_api = weekend_planner.call_llm_api
def call_llm_api_wrapped(*args, **kwargs):
    res = orig_call_llm_api(*args, **kwargs)
    with open("llm_raw_dump.txt", "a") as f:
        f.write("=== RES ===\n")
        f.write(res.get("content", ""))
        f.write("\n=== END RES ===\n")
    return res

weekend_planner.call_llm_api = call_llm_api_wrapped

# Just test the formatting
import lib.osaurus_lib
orig_extract = lib.osaurus_lib._extract_json_only
def _extract_json_only_wrapped(content):
    res = orig_extract(content)
    with open("llm_raw_dump.txt", "a") as f:
        f.write("=== EXTRACTED ===\n")
        f.write(str(res))
        f.write("\n=== END EXTRACTED ===\n")
    return res
lib.osaurus_lib._extract_json_only = _extract_json_only_wrapped

import sys
sys.argv = ["weekend_planner.py", "--skip-web", "--model", "qwen3.6-35b-a3b-mxfp4"]
weekend_planner.main(weekend_planner.parse_args())

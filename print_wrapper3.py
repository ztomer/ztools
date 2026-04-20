import weekend_planner
import json
orig_call_llm_api = weekend_planner.call_llm_api
def call_llm_api_wrapped(*args, **kwargs):
    with open("llm_req_dump.txt", "a") as f:
        f.write("=== REQ ===\n")
        f.write(json.dumps(kwargs.get("messages", args[2] if len(args)>2 else []), indent=2))
        f.write("\n=== END REQ ===\n")
    return orig_call_llm_api(*args, **kwargs)
weekend_planner.call_llm_api = call_llm_api_wrapped

import sys
sys.argv = ["weekend_planner.py", "--skip-web", "--model", "qwen3.6-35b-a3b-mxfp4"]
weekend_planner.main(weekend_planner.parse_args())

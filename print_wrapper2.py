import weekend_planner
orig_call_llm_api = weekend_planner.call_llm_api
def call_llm_api_wrapped(*args, **kwargs):
    res = orig_call_llm_api(*args, **kwargs)
    with open("llm_raw_dump.txt", "a") as f:
        f.write("=== RAW ===\n")
        f.write(res.get("content", ""))
        f.write("\n=== END RAW ===\n")
    return res
weekend_planner.call_llm_api = call_llm_api_wrapped

import sys
sys.argv = ["weekend_planner.py", "--skip-web", "--model", "qwen3.6-35b-a3b-mxfp4"]
weekend_planner.main(weekend_planner.parse_args())

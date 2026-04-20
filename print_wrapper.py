import json
import sys

def inject_print():
    import weekend_planner
    orig_get_llm_json = weekend_planner.get_llm_json
    def get_llm_json_wrapped(*args, **kwargs):
        res = orig_get_llm_json(*args, **kwargs)
        with open("llm_output_dump.txt", "a") as f:
            f.write(json.dumps(res, indent=2))
            f.write("\n---\n")
        return res
    weekend_planner.get_llm_json = get_llm_json_wrapped

inject_print()
import weekend_planner
sys.argv = ["weekend_planner.py", "--skip-web", "--model", "qwen3.6-35b-a3b-mxfp4"]
weekend_planner.main(weekend_planner.parse_args())

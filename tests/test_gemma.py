from lib.osaurus_lib import call as osaurus_call
from model_eval import TASKS

task = TASKS["json"]
model = "gemma-4-26b-a4b-it-4bit"
print("Calling model...")
result = osaurus_call(
    model=model,
    messages=task["messages"],
    task="json",
    parse_json=True,
    validator=task["validator"],
    max_retries=0,
)
print("RAW CONTENT:")
print(repr(result.get("content", "")))
print("PARSED:")
print(repr(result.get("parsed", "")))

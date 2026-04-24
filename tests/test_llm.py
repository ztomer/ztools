import json
from lib.config import get_best_model, Task
from lib.osaurus_lib import call_llm_api

model = get_best_model(Task.JSON)
print(f"Model: {model}")
sys_prompt = "Output JSON now. Use EXACT schema: {\"fixed_activities\": [{\"name\": \"str\", \"location\": \"str\", \"target_ages\": \"str\", \"price\": \"str\", \"weather\": \"str\"}]}"
usr_prompt = "Extract popular Vaughan/Toronto venues for families with kids ages 6-13. MANDATORY default values: target_ages: \"6-13\", price: $18-35 per child or free, weather: \"indoor\". Never leave any field empty."

res = call_llm_api(
    "http://localhost:1337",
    model,
    [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": usr_prompt}
    ],
    temperature=0.1,
    timeout=600,
    parse_json=False
)

content = res.get("content", "")
print("RAW CONTENT:")
print("---")
print(content)
print("---")

